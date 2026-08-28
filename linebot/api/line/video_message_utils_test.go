package line

import (
	"bytes"
	"context"
	"fmt"
	"io"
	"os"
	"testing"
	"time"

	"github.com/HeavenAQ/nstc-linebot-2025/api/analysis"
	"github.com/joho/godotenv"
	linebotsdk "github.com/line/line-bot-sdk-go/v7/linebot"
	"github.com/stretchr/testify/require"
)

func contentResponse(contentType string, body []byte) *linebotsdk.MessageContentResponse {
	return &linebotsdk.MessageContentResponse{
		Content:       io.NopCloser(bytes.NewReader(body)),
		ContentLength: int64(len(body)),
		ContentType:   contentType,
	}
}

func TestReadVideoContentRetriesWhileLINEIsTranscoding(t *testing.T) {
	attempt := 0
	waits := 0
	blob, err := readVideoContent(
		func() (*linebotsdk.MessageContentResponse, error) {
			attempt++
			if attempt < 3 {
				return contentResponse("application/json", nil), nil
			}
			return contentResponse("video/mp4", []byte("video-data")), nil
		},
		4,
		func() { waits++ },
	)

	require.NoError(t, err)
	require.Equal(t, []byte("video-data"), blob)
	require.Equal(t, 3, attempt)
	require.Equal(t, 2, waits)
}

func TestReadVideoContentRejectsPersistentEmptyResponse(t *testing.T) {
	waits := 0
	_, err := readVideoContent(
		func() (*linebotsdk.MessageContentResponse, error) {
			return contentResponse("application/json", nil), nil
		},
		3,
		func() { waits++ },
	)

	require.ErrorContains(t, err, "still processing after 3 attempts")
	require.Equal(t, 2, waits)
}

func TestLiveGetVideoContent(t *testing.T) {
	if os.Getenv("RUN_LIVE_LINE_CONTENT") != "1" {
		t.Skip("set RUN_LIVE_LINE_CONTENT=1 to download a real LINE video")
	}
	require.NoError(t, godotenv.Load("../../.env"))
	messageID := os.Getenv("LIVE_LINE_MESSAGE_ID")
	require.NotEmpty(t, messageID)
	client, err := NewBotClient(
		os.Getenv("LINE_CHANNEL_SECRET"),
		os.Getenv("LINE_CHANNEL_TOKEN"),
		os.Getenv("GCS_BUCKET_NAME"),
		os.Getenv("LIFF_REVIEW_URL"),
	)
	require.NoError(t, err)

	blob, err := client.GetVideoContent(messageID)
	require.NoError(t, err)
	require.Greater(t, len(blob), 1024)
	require.Equal(t, "ftyp", string(blob[4:8]))
}

func TestLiveLINEVideoAnalysis(t *testing.T) {
	if os.Getenv("RUN_LIVE_LINE_ANALYSIS") != "1" {
		t.Skip("set RUN_LIVE_LINE_ANALYSIS=1 to analyze a real LINE video")
	}
	_ = godotenv.Load("../../.env")
	messageID := os.Getenv("LIVE_LINE_MESSAGE_ID")
	target := os.Getenv("ANALYSIS_GRPC_TARGET")
	apiKey := os.Getenv("ANALYSIS_GRPC_API_KEY")
	skill := os.Getenv("LIVE_ANALYSIS_SKILL")
	handedness := os.Getenv("LIVE_ANALYSIS_HANDEDNESS")
	require.NotEmpty(t, messageID)
	require.NotEmpty(t, target)
	require.NotEmpty(t, apiKey)
	require.NotEmpty(t, skill)
	require.NotEmpty(t, handedness)

	lineClient, err := NewBotClient(
		os.Getenv("LINE_CHANNEL_SECRET"),
		os.Getenv("LINE_CHANNEL_TOKEN"),
		os.Getenv("GCS_BUCKET_NAME"),
		os.Getenv("LIFF_REVIEW_URL"),
	)
	require.NoError(t, err)
	video, err := lineClient.GetVideoContent(messageID)
	require.NoError(t, err)
	require.Greater(t, len(video), 1024)

	analysisClient, err := analysis.NewClient(target, apiKey, false, false)
	require.NoError(t, err)
	t.Cleanup(func() { require.NoError(t, analysisClient.Close()) })
	result, err := analysisClient.AnalyzeVideo(
		context.Background(),
		fmt.Sprintf("line-live-%d", time.Now().UnixNano()),
		"line-live-integration",
		messageID+".mp4",
		skill,
		handedness,
		video,
	)
	require.NoError(t, err)
	require.Equal(t, handedness, result.Handedness)
	require.NotEmpty(t, result.AnalysisID)
	require.NotEmpty(t, result.Expert.ExpertID)
	require.NotEmpty(t, result.StudentVideo.SignedURL)
	require.NotEmpty(t, result.Expert.Video.SignedURL)
	require.NotEmpty(t, result.SkeletonOverlayVideo.SignedURL)
}
