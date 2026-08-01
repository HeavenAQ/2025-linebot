package main

import (
	"context"
	"encoding/csv"
	"flag"
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"time"

	"github.com/HeavenAQ/nstc-linebot-2025/api/analysis"
)

var diagnosticColumns = []string{
	"pose_tensorrt_active",
	"skeleton_tensorrt_active",
	"latency_pose_seconds",
	"latency_preprocessing_seconds",
	"latency_scoring_seconds",
	"latency_preview_render_seconds",
	"latency_coaching_preparation_seconds",
	"latency_llm_inference_seconds",
	"latency_coaching_total_seconds",
	"latency_final_render_seconds",
	"latency_pipeline_seconds",
	"latency_catalog_seconds",
	"latency_storage_seconds",
	"latency_service_seconds",
	"input_video_bytes",
}

type benchmarkCase struct {
	skill string
	path  string
}

func main() {
	target := flag.String("target", os.Getenv("ANALYSIS_GRPC_TARGET"), "analysis gRPC host")
	apiKey := flag.String("api-key", os.Getenv("ANALYSIS_GRPC_API_KEY"), "analysis API key")
	casesValue := flag.String("cases", "", "comma-separated skill=/path/video.mp4 cases")
	runs := flag.Int("runs", 3, "number of runs per case")
	handedness := flag.String(
		"handedness", "right", "known handedness for latency fixtures (left or right)",
	)
	outputPath := flag.String("output", "analysis-latency.csv", "CSV result path")
	insecure := flag.Bool("insecure", false, "use plaintext gRPC")
	flag.Parse()

	if *target == "" || *apiKey == "" || *casesValue == "" || *runs < 1 {
		fatalf("target, api-key, at least one case, and a positive run count are required")
	}
	if *handedness != "left" && *handedness != "right" {
		fatalf("handedness must be left or right")
	}
	cases, err := parseCases(*casesValue)
	if err != nil {
		fatalf("parse cases: %v", err)
	}
	client, err := analysis.NewClient(*target, *apiKey, *insecure)
	if err != nil {
		fatalf("create client: %v", err)
	}
	defer client.Close()

	healthStarted := time.Now()
	if err := client.Health(context.Background()); err != nil {
		fatalf("health check: %v", err)
	}
	healthSeconds := time.Since(healthStarted).Seconds()

	output, err := os.Create(*outputPath)
	if err != nil {
		fatalf("create output: %v", err)
	}
	defer output.Close()
	writer := csv.NewWriter(output)
	header := []string{
		"recorded_at", "skill", "video", "run", "health_seconds",
		"client_analyze_seconds", "refresh_urls_seconds", "student_range_get_seconds",
		"expert_range_get_seconds", "score", "analysis_id", "expert_id", "handedness",
		"coaching_cue_count", "student_object_path", "expert_object_path",
	}
	header = append(header, diagnosticColumns...)
	if err := writer.Write(header); err != nil {
		fatalf("write header: %v", err)
	}

	for _, benchmark := range cases {
		video, err := os.ReadFile(benchmark.path)
		if err != nil {
			fatalf("read %s: %v", benchmark.path, err)
		}
		for run := 1; run <= *runs; run++ {
			requestID := fmt.Sprintf("benchmark-%s-%d-%d", benchmark.skill, time.Now().UnixNano(), run)
			started := time.Now()
			result, err := client.AnalyzeVideo(
				context.Background(), requestID, "latency-benchmark", filepath.Base(benchmark.path),
				benchmark.skill, *handedness, video,
			)
			if err != nil {
				fatalf("analyze %s run %d: %v", benchmark.skill, run, err)
			}
			analyzeSeconds := time.Since(started).Seconds()

			refreshStarted := time.Now()
			media, err := client.RefreshPlaybackURLs(
				context.Background(), result.StudentVideo.ObjectPath, result.Expert.Video.ObjectPath,
			)
			if err != nil || len(media) != 2 {
				fatalf("refresh %s run %d: count=%d error=%v", benchmark.skill, run, len(media), err)
			}
			refreshSeconds := time.Since(refreshStarted).Seconds()
			studentGetSeconds, err := rangeGet(media[0].SignedURL)
			if err != nil {
				fatalf("student range GET %s run %d: %v", benchmark.skill, run, err)
			}
			expertGetSeconds, err := rangeGet(media[1].SignedURL)
			if err != nil {
				fatalf("expert range GET %s run %d: %v", benchmark.skill, run, err)
			}

			row := []string{
				time.Now().UTC().Format(time.RFC3339), benchmark.skill, benchmark.path, strconv.Itoa(run),
				decimal(healthSeconds), decimal(analyzeSeconds), decimal(refreshSeconds),
				decimal(studentGetSeconds), decimal(expertGetSeconds), decimal(result.Grade.TotalGrade),
				result.AnalysisID, result.Expert.ExpertID, result.Handedness,
				strconv.Itoa(len(result.CoachingCues)), result.StudentVideo.ObjectPath,
				result.Expert.Video.ObjectPath,
			}
			for _, name := range diagnosticColumns {
				row = append(row, decimal(result.Diagnostics[name]))
			}
			if err := writer.Write(row); err != nil {
				fatalf("write result: %v", err)
			}
			writer.Flush()
			if err := writer.Error(); err != nil {
				fatalf("flush result: %v", err)
			}
			fmt.Fprintf(os.Stderr, "%s run %d/%d: %.2fs (service %.2fs)\n", benchmark.skill, run, *runs, analyzeSeconds, result.Diagnostics["latency_service_seconds"])
		}
	}
}

func parseCases(value string) ([]benchmarkCase, error) {
	var cases []benchmarkCase
	for _, raw := range strings.Split(value, ",") {
		parts := strings.SplitN(strings.TrimSpace(raw), "=", 2)
		if len(parts) != 2 || parts[0] == "" || parts[1] == "" {
			return nil, fmt.Errorf("invalid case %q; expected skill=/path", raw)
		}
		cases = append(cases, benchmarkCase{skill: parts[0], path: parts[1]})
	}
	return cases, nil
}

func rangeGet(url string) (float64, error) {
	request, err := http.NewRequest(http.MethodGet, url, nil)
	if err != nil {
		return 0, err
	}
	request.Header.Set("Range", "bytes=0-65535")
	started := time.Now()
	response, err := http.DefaultClient.Do(request)
	if err != nil {
		return 0, err
	}
	defer response.Body.Close()
	if response.StatusCode != http.StatusOK && response.StatusCode != http.StatusPartialContent {
		return 0, fmt.Errorf("unexpected status %s", response.Status)
	}
	if _, err := io.Copy(io.Discard, response.Body); err != nil {
		return 0, err
	}
	return time.Since(started).Seconds(), nil
}

func decimal(value float64) string { return strconv.FormatFloat(value, 'f', 6, 64) }

func fatalf(format string, values ...any) {
	fmt.Fprintf(os.Stderr, format+"\n", values...)
	os.Exit(1)
}
