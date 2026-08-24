package analysis

import (
	"context"
	"crypto/tls"
	"errors"
	"fmt"
	"os"
	"strings"
	"time"

	analysisv1 "github.com/HeavenAQ/nstc-linebot-2025/api/analysis/v1"
	"github.com/HeavenAQ/nstc-linebot-2025/commons"
	"golang.org/x/oauth2"
	"google.golang.org/api/idtoken"
	"google.golang.org/grpc"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/credentials"
	insecurecredentials "google.golang.org/grpc/credentials/insecure"
	"google.golang.org/grpc/credentials/oauth"
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
	options := []grpc.DialOption{
		grpc.WithDefaultCallOptions(
			grpc.MaxCallRecvMsgSize(8*1024*1024),
			grpc.MaxCallSendMsgSize(chunkSize+1024),
		),
	}
	if useInsecure {
		options = append(options, grpc.WithTransportCredentials(insecurecredentials.NewCredentials()))
	} else {
		host := strings.Split(target, ":")[0]
		options = append(options, grpc.WithTransportCredentials(
			credentials.NewTLS(&tls.Config{MinVersion: tls.VersionTLS12, ServerName: host}),
		))
		// The analysis service runs on a GPU and is reachable on the public
		// internet, so it admits only callers Cloud Run IAM recognises as
		// invokers. Every call carries an OIDC identity token for this service
		// account, audience-bound to the analysis service so a token minted for
		// anything else is refused. The API key stays on top of it.
		tokens, err := identityTokens("https://" + host)
		if err != nil {
			return nil, fmt.Errorf("obtain analysis identity token source: %w", err)
		}
		options = append(options, grpc.WithPerRPCCredentials(oauth.TokenSource{TokenSource: tokens}))
	}
	connection, err := grpc.NewClient(target, options...)
	if err != nil {
		return nil, fmt.Errorf("connect to analysis service: %w", err)
	}
	return &Client{
		connection: connection,
		service:    analysisv1.NewBadmintonAnalysisClient(connection),
		apiKey:     apiKey,
	}, nil
}

// identityTokens yields OIDC tokens naming the analysis service as audience.
//
// On Cloud Run the metadata server mints them from the attached service
// account. Elsewhere -- a CI runner authenticated by workload identity
// federation, where minting an audience-bound token from ambient credentials
// does not work -- ANALYSIS_IDENTITY_TOKEN supplies one that was obtained out
// of band.
func identityTokens(audience string) (oauth2.TokenSource, error) {
	if token := strings.TrimSpace(os.Getenv("ANALYSIS_IDENTITY_TOKEN")); token != "" {
		return oauth2.StaticTokenSource(&oauth2.Token{AccessToken: token, TokenType: "Bearer"}), nil
	}
	return idtoken.NewTokenSource(context.Background(), audience)
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

func phaseMarkers(values []*analysisv1.PhaseMarker) []commons.PhaseMarker {
	markers := make([]commons.PhaseMarker, 0, len(values))
	for _, value := range values {
		markers = append(markers, commons.PhaseMarker{ID: value.Id, Label: value.Label, NormalizedFrame: value.NormalizedFrame, NormalizedPosition: value.NormalizedPosition, TimestampSeconds: value.TimestampSeconds})
	}
	return markers
}

func alignmentSamples(values []*analysisv1.AlignmentSample) []commons.AlignmentSample {
	samples := make([]commons.AlignmentSample, 0, len(values))
	for _, value := range values {
		samples = append(samples, commons.AlignmentSample{NormalizedPosition: value.NormalizedPosition, ExpertSeconds: value.ExpertSeconds})
	}
	return samples
}

func outcome(response *analysisv1.AnalyzeVideoResponse) *commons.AnalysisOutcome {
	details := make([]commons.GradingDetail, 0, len(response.Grade.GradingDetails))
	for _, value := range response.Grade.GradingDetails {
		details = append(details, commons.GradingDetail{CriterionID: value.CriterionId, Description: value.Description, Grade: value.Grade, Maximum: value.Maximum})
	}
	timeline := phaseMarkers(response.Timeline)
	expertTimeline := phaseMarkers(response.Expert.Timeline)
	expertAlignment := alignmentSamples(response.Expert.Alignment)
	cues := make([]commons.CoachingCue, 0, len(response.CoachingCues))
	for _, value := range response.CoachingCues {
		cues = append(cues, commons.CoachingCue{Title: value.Title, Feedback: value.Feedback, NormalizedFrame: value.NormalizedFrame, NormalizedPosition: value.NormalizedPosition, StudentTimestampSeconds: value.StudentTimestampSeconds, PauseDurationSeconds: value.PauseDurationSeconds, JointIDs: value.JointIds})
	}
	diagnostics := make(map[string]float64, len(response.Diagnostics))
	for _, value := range response.Diagnostics {
		diagnostics[value.Key] = value.Value
	}
	studentVideo := media(response.StudentVideo)
	feedbackVideo := media(response.FeedbackVideo)
	if feedbackVideo.ObjectPath == "" {
		feedbackVideo = studentVideo
	}
	return &commons.AnalysisOutcome{
		AnalysisID:           response.AnalysisId,
		Skill:                strings.TrimPrefix(strings.ToLower(response.Skill.String()), "skill_"),
		Handedness:           strings.TrimPrefix(strings.ToLower(response.Handedness.String()), "handedness_"),
		Grade:                commons.GradingOutcome{TotalGrade: response.Grade.TotalGrade, GradingDetails: details, ScoreStatus: response.Grade.ScoreStatus},
		StudentVideo:         studentVideo,
		FeedbackVideo:        feedbackVideo,
		SkeletonOverlayVideo: media(response.SkeletonOverlayVideo),
		Expert:               commons.ExpertMatch{ExpertID: response.Expert.ExpertId, DisplayName: response.Expert.DisplayName, CorrectionDistance: response.Expert.CorrectionDistance, Video: media(response.Expert.Video), MotionStartSeconds: response.Expert.MotionStartSeconds, MotionEndSeconds: response.Expert.MotionEndSeconds, Timeline: expertTimeline, Alignment: expertAlignment},
		Timeline:             timeline, CoachingCues: cues, OverallFeedback: response.OverallFeedback,
		Diagnostics: diagnostics,
	}
}

// refreshTimeout has to survive a cold start. The analysis service scales to
// zero, so the first playback request after an idle period waits for an L4 to
// boot and load its pose engines. It stays under the HTTP server's 100s write
// timeout, which is the real ceiling on this path.
const refreshTimeout = 90 * time.Second

func (c *Client) RefreshPlaybackURLs(ctx context.Context, objectPaths ...string) ([]commons.MediaRef, error) {
	ctx, cancel := context.WithTimeout(c.authorizedContext(ctx), refreshTimeout)
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
	// Also sized for a cold start: a health check that fires while the service
	// is scaling up from zero should wait for it, not report it down.
	ctx, cancel := context.WithTimeout(ctx, 60*time.Second)
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
