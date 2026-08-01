package analysis

import (
	"context"
	"crypto/tls"
	"errors"
	"fmt"
	"strings"
	"time"

	analysisv1 "github.com/HeavenAQ/nstc-linebot-2025/api/analysis/v1"
	"github.com/HeavenAQ/nstc-linebot-2025/commons"
	"google.golang.org/grpc"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/credentials"
	insecurecredentials "google.golang.org/grpc/credentials/insecure"
	"google.golang.org/grpc/metadata"
	"google.golang.org/grpc/status"
)

const chunkSize = 1024 * 1024

var ErrNoMatchingExpert = errors.New("no same-handed expert is available")

type Client struct {
	connection *grpc.ClientConn
	service    analysisv1.BadmintonAnalysisClient
	apiKey     string
}

func NewClient(target, apiKey string, useInsecure bool) (*Client, error) {
	target = strings.TrimPrefix(strings.TrimPrefix(target, "https://"), "http://")
	if !strings.Contains(target, ":") {
		if useInsecure {
			target += ":80"
		} else {
			target += ":443"
		}
	}
	var transport credentials.TransportCredentials
	if useInsecure {
		transport = insecurecredentials.NewCredentials()
	} else {
		host := strings.Split(target, ":")[0]
		transport = credentials.NewTLS(&tls.Config{MinVersion: tls.VersionTLS12, ServerName: host})
	}
	connection, err := grpc.NewClient(
		target,
		grpc.WithTransportCredentials(transport),
		grpc.WithDefaultCallOptions(
			grpc.MaxCallRecvMsgSize(8*1024*1024),
			grpc.MaxCallSendMsgSize(chunkSize+1024),
		),
	)
	if err != nil {
		return nil, fmt.Errorf("connect to analysis service: %w", err)
	}
	return &Client{
		connection: connection,
		service:    analysisv1.NewBadmintonAnalysisClient(connection),
		apiKey:     apiKey,
	}, nil
}

func (c *Client) Close() error { return c.connection.Close() }

func (c *Client) authorizedContext(ctx context.Context) context.Context {
	return metadata.AppendToOutgoingContext(ctx, "x-api-key", c.apiKey)
}

func skillValue(skill string) (analysisv1.Skill, error) {
	switch skill {
	case "serve":
		return analysisv1.Skill_SKILL_SERVE, nil
	case "lift":
		return analysisv1.Skill_SKILL_LIFT, nil
	case "clear":
		return analysisv1.Skill_SKILL_CLEAR, nil
	case "smash":
		return analysisv1.Skill_SKILL_SMASH, nil
	default:
		return analysisv1.Skill_SKILL_UNSPECIFIED, fmt.Errorf("unsupported skill: %s", skill)
	}
}

func handednessValue(handedness string) (analysisv1.Handedness, error) {
	switch handedness {
	case "auto":
		return analysisv1.Handedness_HANDEDNESS_AUTO, nil
	case "right":
		return analysisv1.Handedness_HANDEDNESS_RIGHT, nil
	case "left":
		return analysisv1.Handedness_HANDEDNESS_LEFT, nil
	default:
		return analysisv1.Handedness_HANDEDNESS_UNSPECIFIED, fmt.Errorf("unsupported handedness: %s", handedness)
	}
}

func (c *Client) AnalyzeVideo(
	ctx context.Context,
	requestID, userID, filename, skill, handedness string,
	video []byte,
) (*commons.AnalysisOutcome, error) {
	if len(video) == 0 {
		return nil, fmt.Errorf("video is empty")
	}
	skillEnum, err := skillValue(skill)
	if err != nil {
		return nil, err
	}
	handednessEnum, err := handednessValue(handedness)
	if err != nil {
		return nil, err
	}
	ctx, cancel := context.WithTimeout(c.authorizedContext(ctx), 30*time.Minute)
	defer cancel()
	stream, err := c.service.AnalyzeVideo(ctx)
	if err != nil {
		return nil, fmt.Errorf("start analysis stream: %w", err)
	}
	if err := stream.Send(&analysisv1.AnalyzeVideoChunk{
		Payload: &analysisv1.AnalyzeVideoChunk_Header{Header: &analysisv1.AnalyzeVideoHeader{
			RequestId: requestID, UserId: userID, Filename: filename,
			Skill: skillEnum, Handedness: handednessEnum,
		}},
	}); err != nil {
		return nil, fmt.Errorf("send analysis header: %w", err)
	}
	for start := 0; start < len(video); start += chunkSize {
		end := min(start+chunkSize, len(video))
		if err := stream.Send(&analysisv1.AnalyzeVideoChunk{
			Payload: &analysisv1.AnalyzeVideoChunk_Data{Data: video[start:end]},
		}); err != nil {
			return nil, fmt.Errorf("stream video: %w", err)
		}
	}
	response, err := stream.CloseAndRecv()
	if err != nil {
		if status.Code(err) == codes.FailedPrecondition &&
			strings.Contains(status.Convert(err).Message(), "expert reference") {
			return nil, fmt.Errorf("%w: %s", ErrNoMatchingExpert, status.Convert(err).Message())
		}
		return nil, fmt.Errorf("receive analysis: %w", err)
	}
	return outcome(response), nil
}

func media(value *analysisv1.StoredVideo) commons.MediaRef {
	if value == nil {
		return commons.MediaRef{}
	}
	return commons.MediaRef{
		ObjectPath: value.ObjectPath, GCSURI: value.GcsUri, SignedURL: value.SignedUrl,
		SignedURLExpires: value.SignedUrlExpiresAtUnix, DurationSeconds: value.DurationSeconds,
		FPS: value.Fps, Width: value.Width, Height: value.Height,
	}
}

func outcome(response *analysisv1.AnalyzeVideoResponse) *commons.AnalysisOutcome {
	details := make([]commons.GradingDetail, 0, len(response.Grade.GradingDetails))
	for _, value := range response.Grade.GradingDetails {
		details = append(details, commons.GradingDetail{CriterionID: value.CriterionId, Description: value.Description, Grade: value.Grade, Maximum: value.Maximum})
	}
	timeline := make([]commons.PhaseMarker, 0, len(response.Timeline))
	for _, value := range response.Timeline {
		timeline = append(timeline, commons.PhaseMarker{ID: value.Id, Label: value.Label, NormalizedFrame: value.NormalizedFrame, NormalizedPosition: value.NormalizedPosition, TimestampSeconds: value.TimestampSeconds})
	}
	cues := make([]commons.CoachingCue, 0, len(response.CoachingCues))
	for _, value := range response.CoachingCues {
		cues = append(cues, commons.CoachingCue{Title: value.Title, Feedback: value.Feedback, NormalizedFrame: value.NormalizedFrame, NormalizedPosition: value.NormalizedPosition, StudentTimestampSeconds: value.StudentTimestampSeconds, PauseDurationSeconds: value.PauseDurationSeconds, JointIDs: value.JointIds})
	}
	diagnostics := make(map[string]float64, len(response.Diagnostics))
	for _, value := range response.Diagnostics {
		diagnostics[value.Key] = value.Value
	}
	return &commons.AnalysisOutcome{
		AnalysisID:   response.AnalysisId,
		Skill:        strings.TrimPrefix(strings.ToLower(response.Skill.String()), "skill_"),
		Handedness:   strings.TrimPrefix(strings.ToLower(response.Handedness.String()), "handedness_"),
		Grade:        commons.GradingOutcome{TotalGrade: response.Grade.TotalGrade, GradingDetails: details, ScoreStatus: response.Grade.ScoreStatus},
		StudentVideo: media(response.StudentVideo),
		Expert:       commons.ExpertMatch{ExpertID: response.Expert.ExpertId, DisplayName: response.Expert.DisplayName, CorrectionDistance: response.Expert.CorrectionDistance, Video: media(response.Expert.Video), MotionStartSeconds: response.Expert.MotionStartSeconds, MotionEndSeconds: response.Expert.MotionEndSeconds},
		Timeline:     timeline, CoachingCues: cues, OverallFeedback: response.OverallFeedback,
		Diagnostics: diagnostics,
	}
}

func (c *Client) RefreshPlaybackURLs(ctx context.Context, objectPaths ...string) ([]commons.MediaRef, error) {
	ctx, cancel := context.WithTimeout(c.authorizedContext(ctx), 15*time.Second)
	defer cancel()
	response, err := c.service.RefreshPlaybackUrls(ctx, &analysisv1.RefreshPlaybackUrlsRequest{ObjectPaths: objectPaths})
	if err != nil {
		return nil, fmt.Errorf("refresh playback URLs: %w", err)
	}
	values := make([]commons.MediaRef, 0, len(response.Videos))
	for _, value := range response.Videos {
		values = append(values, media(value))
	}
	return values, nil
}

func (c *Client) Health(ctx context.Context) error {
	ctx, cancel := context.WithTimeout(ctx, 10*time.Second)
	defer cancel()
	response, err := c.service.Health(ctx, &analysisv1.HealthRequest{})
	if err != nil {
		return err
	}
	if response.Status != "serving" {
		return fmt.Errorf("analysis service status: %s", response.Status)
	}
	return nil
}
