# Badminton Motion Analysis Latency Benchmark

## Purpose

This benchmark measures the deployed production path, not an isolated model
microbenchmark. Each run streams a real student video from the Go client to the
Python gRPC service, performs pose extraction, skeleton correction and grading,
requests constrained OpenAI coaching, renders and uploads the annotated H.264
video, refreshes both signed playback URLs, and range-reads both videos from GCS.

## Environment

- Date: 2026-08-01
- GCP project: `nstc-linebot-2025`
- Cloud Run region: `asia-southeast1`
- Accelerator: one NVIDIA L4
- Container: PyTorch 2.5.1, CUDA 12.4, cuDNN 9, ONNX Runtime GPU 1.24.4,
  TensorRT 10.14, and `rtmlib` 0.0.15
- Cloud Run shape: 4 vCPU, 16 GiB, concurrency 1, maximum 1 instance
- Protocol: public HTTP/2 gRPC with API-key application authentication
- Run count: three sequential requests per skill, 12 analyses total

The one-time `Health` call is recorded separately. The deployment gate builds
the exact L4 RTMW3D TensorRT partitions before traffic is accepted and the
service keeps one warm instance. All 12 measured requests therefore reuse the
loaded process and sessions. The first request for a skill may still deserialize
that skill's prebuilt correction engine, but it does not compile an engine.

## Inputs

The four fixtures are deliberately low-scoring beginner examples from the
evaluation corpus. All are 1080 x 1920 at 30 FPS.

| Skill | Fixture | Duration | Frames | Bytes |
|---|---|---:|---:|---:|
| Serve | CG46 | 5.933 s | 178 | 5,365,026 |
| Lift | L3 | 5.500 s | 165 | 3,237,705 |
| Clear | EG3 | 5.033 s | 151 | 4,164,767 |
| Smash | EG12 | 5.333 s | 160 | 3,006,067 |

## Procedure

Run the committed benchmark client from `linebot/`:

```bash
go run ./cmd/analysis-benchmark \
  -target "$ANALYSIS_GRPC_TARGET" \
  -api-key "$ANALYSIS_GRPC_API_KEY" \
  -handedness right \
  -cases "serve=/path/serve-CG46.mp4,lift=/path/lift-L3.mp4,clear=/path/clear-EG3.mp4,smash=/path/smash-EG12.mp4" \
  -runs 3 \
  -output ../docs/benchmarks/badminton-analysis-latency-2026-08-01.csv
```

The CSV stores the analysis ID and GCS object paths, but never signed URLs or
credentials. `client_analyze_seconds` is the user-visible analysis latency.
Internal stage timings come from the Python response. In particular,
`latency_llm_inference_seconds` surrounds only the constrained OpenAI
`responses.parse` request and parsed response. Frame sampling and prompt-context
assembly are reported as `latency_coaching_preparation_seconds`, and their sum
as `latency_coaching_total_seconds`. Both are independent of the preview and
final rendering stages. URL refresh and 64 KiB range reads are measured from Go
after analysis completes.

Every production row must report both `pose_tensorrt_active=1` and
`skeleton_tensorrt_active=1`. A zero value makes the run a fallback measurement
and invalidates a TensorRT comparison.

## Results

All 12 rows reported `pose_tensorrt_active=1` and
`skeleton_tensorrt_active=1`. Times are seconds.

| Scope | Analysis mean | Analysis p50 | Analysis p95 | LLM mean | Pose mean | Score |
|---|---:|---:|---:|---:|---:|---:|
| Overall (12) | 35.679 | 35.274 | 44.146 | 16.350 | 2.139 | - |
| Serve: CG46 | 38.662 | 35.988 | 44.444 | 16.701 | 2.196 | 4.871 |
| Lift: L3 | 35.307 | 31.897 | 42.010 | 17.720 | 2.586 | 15.491 |
| Clear: EG3 | 35.665 | 35.833 | 36.385 | 15.833 | 1.841 | 26.718 |
| Smash: EG12 | 33.084 | 30.752 | 37.083 | 15.144 | 1.934 | 0.574 |

The deterministic grade and matched expert were stable across all three runs
of each fixture. These examples were chosen from the weak tail of each cohort;
their scores are not estimates of the 45.0 beginner-cohort mean.

| Stage | Mean | p50 | p95 |
|---|---:|---:|---:|
| RTMW pose extraction | 2.139 | 1.936 | 2.999 |
| Normalization and phase parsing | 0.026 | 0.024 | 0.032 |
| Correction and grading | 0.418 | 0.411 | 0.436 |
| Preview rendering | 3.733 | 3.854 | 4.010 |
| Coaching frame/prompt preparation | 1.981 | 1.965 | 2.116 |
| OpenAI inference | 16.350 | 15.568 | 22.921 |
| Final annotated rendering | 5.856 | 5.838 | 6.882 |
| Expert catalog lookup | 0.162 | 0.137 | 0.319 |
| GCS upload and signing | 0.755 | 0.722 | 0.973 |
| Signed URL refresh | 0.263 | 0.267 | 0.302 |

The raw observations, scores, expert IDs, object paths, provider flags, and all
stage timings are committed in
`docs/benchmarks/badminton-analysis-latency-2026-08-01.csv`.

The initial deployment gate also measured the unoptimized cold path: exact L4
detector and pose compilation plus one complete clear analysis took about 432
seconds. Its client connection reset at 405.93 seconds even though the server
finished and uploaded the video. Production now keeps one warm instance and the
CD gate retries once after compilation; cold-build time is not mixed into the
steady-state table.

## Interpretation

The optimization result must be evaluated from the measured stage breakdown.
TensorRT accelerates YOLOX, RTMW3D, and the selected skill's correction
Transformer; rendering, OpenAI, GCS, and network time remain independent. OpenAI
accounts for 45.8% of mean client latency, so model or reasoning changes have a
larger end-to-end effect than further optimizing the 2.14-second pose stage.

## Limitations

- Three observations per skill show operational scale but are not a load test.
- Requests are sequential because production concurrency is intentionally one.
- OpenAI and GCS latency vary independently of GPU inference.
- One preliminary smash request returned truncated structured JSON. It is not a
  benchmark row. The deployed service now retries schema generation once and
  falls back to deterministic Traditional Chinese rule advice; a retry would
  make LLM tail latency longer than the successful single-attempt observations.
- The fixtures have similar resolution and duration; longer uploads should be
  benchmarked separately before changing the current 150 MiB request limit.
