package analysis

import (
	"context"
	"io"
	"net"
	"testing"

	analysisv1 "github.com/HeavenAQ/nstc-linebot-2025/api/analysis/v1"
	"github.com/HeavenAQ/nstc-linebot-2025/commons"
	"github.com/stretchr/testify/require"
	"google.golang.org/grpc"
	"google.golang.org/grpc/test/bufconn"
)

type analysisContractServer struct {
	analysisv1.UnimplementedBadmintonAnalysisServer
	testing *testing.T
}

func (server *analysisContractServer) AnalyzeVideo(
	stream grpc.ClientStreamingServer[analysisv1.AnalyzeVideoChunk, analysisv1.AnalyzeVideoResponse],
) error {
	var header *analysisv1.AnalyzeVideoHeader
	var video []byte
	for {
		chunk, err := stream.Recv()
		if err == io.EOF {
			break
		}
		require.NoError(server.testing, err)
		if chunk.GetHeader() != nil {
			header = chunk.GetHeader()
		} else {
			video = append(video, chunk.GetData()...)
		}
	}
	require.NotNil(server.testing, header)
	require.Equal(server.testing, "serve.mp4", header.Filename)
	require.Equal(server.testing, []byte("video-bytes"), video)
	feedback := &analysisv1.StoredVideo{
		ObjectPath: "analyses/test/student_corrected.mp4",
		SignedUrl:  "https://media.test/feedback",
	}
	return stream.SendAndClose(&analysisv1.AnalyzeVideoResponse{
		AnalysisId: "analysis-1",
		Skill:      analysisv1.Skill_SKILL_SERVE,
		Handedness: analysisv1.Handedness_HANDEDNESS_RIGHT,
		Grade: &analysisv1.GradingOutcome{
			TotalGrade:  88,
			ScoreStatus: "expert_only_generated_distribution",
		},
		StudentVideo:  feedback,
		FeedbackVideo: feedback,
		SkeletonOverlayVideo: &analysisv1.StoredVideo{
			ObjectPath: "analyses/test/student_skeleton_overlay.mp4",
			SignedUrl:  "https://media.test/overlay",
		},
		Expert: &analysisv1.ExpertMatch{
			ExpertId:    "generated-prior-1",
			DisplayName: "Generated expert prior",
		},
	})
}

func TestAnalyzeVideoRejectsEmptyInputBeforeStartingStream(t *testing.T) {
	client := &Client{}
	result, err := client.AnalyzeVideo(
		context.Background(),
		"request-id",
		"user-id",
		"video.mp4",
		"serve",
		"right",
		nil,
	)

	require.Nil(t, result)
	require.EqualError(t, err, "video is empty")
}

func TestAnalyzeVideoStreamsAndMapsBothRenderedVideos(t *testing.T) {
	listener := bufconn.Listen(1024 * 1024)
	grpcServer := grpc.NewServer()
	analysisv1.RegisterBadmintonAnalysisServer(
		grpcServer,
		&analysisContractServer{testing: t},
	)
	go func() {
		require.NoError(t, grpcServer.Serve(listener))
	}()
	t.Cleanup(func() {
		grpcServer.Stop()
		require.NoError(t, listener.Close())
	})

	connection, err := grpc.NewClient(
		"passthrough:///bufnet",
		grpc.WithContextDialer(func(context.Context, string) (net.Conn, error) {
			return listener.Dial()
		}),
		grpc.WithInsecure(),
	)
	require.NoError(t, err)
	t.Cleanup(func() { require.NoError(t, connection.Close()) })
	client := &Client{
		connection: connection,
		service:    analysisv1.NewBadmintonAnalysisClient(connection),
		apiKey:     "test-key",
	}

	result, err := client.AnalyzeVideo(
		context.Background(),
		"request-1",
		"user-1",
		"serve.mp4",
		"serve",
		"right",
		[]byte("video-bytes"),
	)

	require.NoError(t, err)
	require.Equal(t, "analyses/test/student_corrected.mp4", result.StudentVideo.ObjectPath)
	require.Equal(t, result.StudentVideo, result.FeedbackVideo)
	require.Equal(t, "analyses/test/student_skeleton_overlay.mp4", result.SkeletonOverlayVideo.ObjectPath)
	require.Equal(t, "https://media.test/overlay", result.SkeletonOverlayVideo.SignedURL)
	require.Equal(t, "generated-prior-1", result.Expert.ExpertID)
}

// Playback needs the alignment in the same shape it will be written to
// Firestore in, so the mapping is worth pinning on its own.
func TestOutcomeCarriesTheExpertAlignment(t *testing.T) {
	response := &analysisv1.AnalyzeVideoResponse{
		Grade: &analysisv1.GradingOutcome{TotalGrade: 88},
		Expert: &analysisv1.ExpertMatch{
			ExpertId: "expert-3-1",
			Alignment: []*analysisv1.AlignmentSample{
				{NormalizedPosition: 0, ExpertSeconds: 0.767},
				{NormalizedPosition: 0.5, ExpertSeconds: 1.767},
				{NormalizedPosition: 1, ExpertSeconds: 2.233},
			},
		},
	}

	result := outcome(response)

	require.Equal(t, []commons.AlignmentSample{
		{NormalizedPosition: 0, ExpertSeconds: 0.767},
		{NormalizedPosition: 0.5, ExpertSeconds: 1.767},
		{NormalizedPosition: 1, ExpertSeconds: 2.233},
	}, result.Expert.Alignment)
}

// An analysis whose expert could not be warped still has to reach the player.
func TestOutcomeWithoutAnAlignmentIsEmptyRatherThanNil(t *testing.T) {
	result := outcome(&analysisv1.AnalyzeVideoResponse{
		Grade:  &analysisv1.GradingOutcome{TotalGrade: 88},
		Expert: &analysisv1.ExpertMatch{ExpertId: "expert-3-1"},
	})

	require.NotNil(t, result.Expert.Alignment)
	require.Empty(t, result.Expert.Alignment)
}
