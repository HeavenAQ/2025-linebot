package analysis_test

import (
	"context"
	"errors"
	"fmt"
	"net/http"
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"testing"
	"time"

	"github.com/HeavenAQ/nstc-linebot-2025/api/analysis"
	"github.com/HeavenAQ/nstc-linebot-2025/commons"
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
	client, err := analysis.NewClient(target, apiKey, false, false, "")
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
	skill := os.Getenv("LIVE_ANALYSIS_SKILL")
	if skill == "" {
		skill = "clear"
	}
	handedness := os.Getenv("LIVE_ANALYSIS_HANDEDNESS")
	if handedness == "" {
		handedness = "right"
	}
	require.NotEmpty(t, target)
	require.NotEmpty(t, apiKey)
	require.NotEmpty(t, videoPath)

	video, err := os.ReadFile(videoPath)
	require.NoError(t, err)
	client, err := analysis.NewClient(target, apiKey, os.Getenv("ANALYSIS_GRPC_INSECURE") == "true", false, "")
	require.NoError(t, err)
	t.Cleanup(func() { require.NoError(t, client.Close()) })

	require.NoError(t, client.Health(context.Background()))
	requestID := fmt.Sprintf("go-live-%d", time.Now().Unix())
	analysisStarted := time.Now()
	result, err := client.AnalyzeVideo(
		context.Background(), requestID, "integration-test", filepath.Base(videoPath),
		skill, handedness, video,
	)
	require.NoError(t, err)
	require.Equal(t, skill, result.Skill)
	require.Equal(t, handedness, result.Handedness)
	require.InDelta(t, 50, result.Grade.TotalGrade, 50)
	if rawMinimum := os.Getenv("LIVE_ANALYSIS_MIN_GRADE"); rawMinimum != "" {
		minimum, parseErr := strconv.ParseFloat(rawMinimum, 64)
		require.NoError(t, parseErr)
		require.GreaterOrEqual(t, result.Grade.TotalGrade, minimum)
	}
	require.NotEmpty(t, result.Grade.GradingDetails)
	require.NotEmpty(t, result.StudentVideo.ObjectPath)
	require.Equal(t, result.StudentVideo.ObjectPath, result.FeedbackVideo.ObjectPath)
	require.NotEmpty(t, result.SkeletonOverlayVideo.ObjectPath)
	require.NotEmpty(t, result.Expert.ExpertID)
	if prefix := os.Getenv("LIVE_ANALYSIS_EXPERT_PREFIX"); prefix != "" {
		require.True(t, len(result.Expert.ExpertID) >= len(prefix))
		require.Equal(t, prefix, result.Expert.ExpertID[:len(prefix)])
	}
	// The correction is generated, so the clip shown beside it is the nearest
	// real demonstration out of the expert reference bank. A skill with no bank
	// still analyses, it just has no video to play alongside -- which of the two
	// a fixture expects is the fixture's to declare.
	expectExpertVideo := os.Getenv("LIVE_ANALYSIS_EXPECT_EXPERT_VIDEO") == "1"
	if expectExpertVideo {
		require.NotEmpty(t, result.Expert.Video.ObjectPath)
		require.GreaterOrEqual(t, result.Expert.MotionStartSeconds, 0.0)
		require.Greater(t, result.Expert.MotionEndSeconds, result.Expert.MotionStartSeconds)
	} else {
		require.Empty(t, result.Expert.Video.ObjectPath)
		require.Equal(t, "Generated expert prior", result.Expert.DisplayName)
	}
	require.NotEmpty(t, result.Timeline)
	// Playback aligns marker for marker, so the expert must report the same
	// checkpoints in the same order, timed inside its own motion window.
	// Criteria are listed in scoring order, not stroke order — serve grades the
	// hip rotation (keyframe 4) before the wrist flick (keyframe 3) — so the
	// timestamps only run forwards once sorted by position, which is the order
	// playback interpolates through.
	if expectExpertVideo {
		require.Len(t, result.Expert.Timeline, len(result.Timeline))
		byPosition := append([]commons.PhaseMarker(nil), result.Expert.Timeline...)
		for index, marker := range result.Expert.Timeline {
			// Only the identity is shared: the frame is the expert's own, since the
			// expert reaches each checkpoint at its own point in the stroke.
			require.Equal(t, result.Timeline[index].ID, marker.ID)
			require.GreaterOrEqual(t, marker.TimestampSeconds, result.Expert.MotionStartSeconds)
			require.LessOrEqual(t, marker.TimestampSeconds, result.Expert.MotionEndSeconds)
		}
		sort.SliceStable(byPosition, func(i, j int) bool {
			return byPosition[i].NormalizedPosition < byPosition[j].NormalizedPosition
		})
		for index := 1; index < len(byPosition); index++ {
			require.GreaterOrEqual(t, byPosition[index].TimestampSeconds, byPosition[index-1].TimestampSeconds)
		}
	}
	require.NotEmpty(t, result.OverallFeedback)
	if os.Getenv("LIVE_ANALYSIS_EXPECT_NO_CUES") == "1" {
		require.Empty(t, result.CoachingCues)
		require.Zero(t, result.Diagnostics["latency_llm_inference_seconds"])
	} else {
		require.NotEmpty(t, result.CoachingCues)
	}
	require.Positive(t, result.Diagnostics["latency_pose_seconds"])
	require.Positive(t, result.Diagnostics["latency_pipeline_seconds"])
	require.Positive(t, result.Diagnostics["latency_service_seconds"])
	require.Equal(t, 1.0, result.Diagnostics["pose_tensorrt_active"])
	// Pose detection runs on a TensorRT engine; the diffusion prior that
	// replaced the skeleton corrector runs in torch and reports no engine.
	require.Equal(t, 0.0, result.Diagnostics["skeleton_tensorrt_active"])
	t.Logf("analysis latency: client=%s service=%.3fs stages=%v", time.Since(analysisStarted), result.Diagnostics["latency_service_seconds"], result.Diagnostics)
	t.Logf("grade=%.2f expert=%q distance=%.4f cues=%d",
		result.Grade.TotalGrade, result.Expert.ExpertID, result.Expert.CorrectionDistance, len(result.CoachingCues))

	playbackPaths := []string{
		result.FeedbackVideo.ObjectPath,
		result.SkeletonOverlayVideo.ObjectPath,
	}
	if result.Expert.Video.ObjectPath != "" {
		playbackPaths = append(playbackPaths, result.Expert.Video.ObjectPath)
	}
	refreshed, err := client.RefreshPlaybackURLs(context.Background(), playbackPaths...)
	require.NoError(t, err)
	require.Len(t, refreshed, len(playbackPaths))
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
