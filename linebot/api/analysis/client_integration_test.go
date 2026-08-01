package analysis_test

import (
	"context"
	"errors"
	"fmt"
	"net/http"
	"os"
	"path/filepath"
	"testing"
	"time"

	"github.com/HeavenAQ/nstc-linebot-2025/api/analysis"
	"github.com/joho/godotenv"
	"github.com/stretchr/testify/require"
)

func TestLiveAnalysisRejectsOppositeHandFallback(t *testing.T) {
	if os.Getenv("RUN_LIVE_ANALYSIS_NO_MATCH") != "1" {
		t.Skip("set RUN_LIVE_ANALYSIS_NO_MATCH=1 to verify strict expert matching")
	}
	_ = godotenv.Load("../../.env")
	target := os.Getenv("ANALYSIS_GRPC_TARGET")
	apiKey := os.Getenv("ANALYSIS_GRPC_API_KEY")
	videoPath := os.Getenv("LIVE_ANALYSIS_VIDEO")
	skill := os.Getenv("LIVE_ANALYSIS_SKILL")
	require.NotEmpty(t, target)
	require.NotEmpty(t, apiKey)
	require.NotEmpty(t, videoPath)
	require.NotEmpty(t, skill)

	video, err := os.ReadFile(videoPath)
	require.NoError(t, err)
	client, err := analysis.NewClient(target, apiKey, false)
	require.NoError(t, err)
	t.Cleanup(func() { require.NoError(t, client.Close()) })

	_, err = client.AnalyzeVideo(
		context.Background(),
		fmt.Sprintf("go-live-no-match-%d", time.Now().UnixNano()),
		"integration-test",
		filepath.Base(videoPath),
		skill,
		"left",
		video,
	)
	require.Error(t, err)
	require.True(t, errors.Is(err, analysis.ErrNoMatchingExpert), err)
}

func TestLiveAnalysisService(t *testing.T) {
	if os.Getenv("RUN_LIVE_ANALYSIS") != "1" {
		t.Skip("set RUN_LIVE_ANALYSIS=1 to exercise the deployed GPU service")
	}
	_ = godotenv.Load("../../.env")
	target := os.Getenv("ANALYSIS_GRPC_TARGET")
	apiKey := os.Getenv("ANALYSIS_GRPC_API_KEY")
	videoPath := os.Getenv("LIVE_ANALYSIS_VIDEO")
	require.NotEmpty(t, target)
	require.NotEmpty(t, apiKey)
	require.NotEmpty(t, videoPath)

	video, err := os.ReadFile(videoPath)
	require.NoError(t, err)
	client, err := analysis.NewClient(target, apiKey, os.Getenv("ANALYSIS_GRPC_INSECURE") == "true")
	require.NoError(t, err)
	t.Cleanup(func() { require.NoError(t, client.Close()) })

	require.NoError(t, client.Health(context.Background()))
	requestID := fmt.Sprintf("go-live-%d", time.Now().Unix())
	analysisStarted := time.Now()
	result, err := client.AnalyzeVideo(
		context.Background(), requestID, "integration-test", filepath.Base(videoPath),
		"clear", "right", video,
	)
	require.NoError(t, err)
	require.InDelta(t, 50, result.Grade.TotalGrade, 50)
	require.NotEmpty(t, result.Grade.GradingDetails)
	require.NotEmpty(t, result.StudentVideo.ObjectPath)
	require.NotEmpty(t, result.Expert.ExpertID)
	require.NotEmpty(t, result.Expert.Video.ObjectPath)
	require.GreaterOrEqual(t, result.Expert.MotionStartSeconds, 0.0)
	require.Greater(t, result.Expert.MotionEndSeconds, result.Expert.MotionStartSeconds)
	require.NotEmpty(t, result.Timeline)
	require.NotEmpty(t, result.OverallFeedback)
	require.NotEmpty(t, result.CoachingCues)
	require.Positive(t, result.Diagnostics["latency_pose_seconds"])
	require.Positive(t, result.Diagnostics["latency_pipeline_seconds"])
	require.Positive(t, result.Diagnostics["latency_service_seconds"])
	require.Equal(t, 1.0, result.Diagnostics["pose_tensorrt_active"])
	require.Equal(t, 1.0, result.Diagnostics["skeleton_tensorrt_active"])
	t.Logf("analysis latency: client=%s service=%.3fs stages=%v", time.Since(analysisStarted), result.Diagnostics["latency_service_seconds"], result.Diagnostics)

	refreshed, err := client.RefreshPlaybackURLs(
		context.Background(), result.StudentVideo.ObjectPath, result.Expert.Video.ObjectPath,
	)
	require.NoError(t, err)
	require.Len(t, refreshed, 2)
	for _, media := range refreshed {
		require.NotEmpty(t, media.SignedURL)
		request, requestErr := http.NewRequest(http.MethodGet, media.SignedURL, nil)
		require.NoError(t, requestErr)
		request.Header.Set("Range", "bytes=0-1023")
		response, requestErr := http.DefaultClient.Do(request)
		require.NoError(t, requestErr)
		response.Body.Close()
		require.Contains(t, []int{http.StatusOK, http.StatusPartialContent}, response.StatusCode)
	}
}
