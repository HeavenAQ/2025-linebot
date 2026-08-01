# Badminton Motion Analysis Latency Benchmark

## Purpose

This benchmark measures the deployed production path, not an isolated model
microbenchmark. Each run streams a real student video from the Go client to the
Python gRPC service, performs pose extraction, skeleton correction and grading,
requests constrained OpenAI coaching, renders and uploads the annotated H.264
video, refreshes both signed playback URLs, and range-reads both videos from GCS.

## Environment

- Date: 2026-08-02 JST (observations recorded 2026-08-01 UTC)
- GCP project: `nstc-linebot-2025`
- Cloud Run region: `asia-southeast1`
- Accelerator: one NVIDIA L4
- Container: PyTorch 2.5.1, CUDA 12.4, cuDNN 9, ONNX Runtime GPU 1.24.4,
  TensorRT 10.14, and `rtmlib` 0.0.15
- Cloud Run shape: 4 vCPU, 16 GiB, concurrency 1, maximum 1 instance
- Protocol: public HTTP/2 gRPC with API-key application authentication
- Run count: three sequential requests per skill, 12 analyses total
- Production revision: `badminton-analysis-ai-00036-vah`
- Image digest: `sha256:8d3748a47ccbcd25f46fa8e4c125a5bd446c689a22ddb3babe2d6e63da5071b8`

The one-time `Health` call is recorded separately. The deployment gate builds
an image containing a checksum-verified 17-file exact-SM89 TensorRT cache before
traffic is accepted, then performs one real analysis and keeps one warm
instance. Production has no GCSFuse engine-cache mount. All 12 measured requests
therefore reuse the loaded process and sessions without compiling an engine.

## Inputs

The four fixtures are deliberately low-scoring examples. The serve fixture is
the supplied negative weight-transfer case; the other three come from the
frozen beginner evaluation corpus. All run at 30 FPS.

| Skill | Fixture | Duration | Frames | Bytes |
|---|---|---:|---:|---:|
| Serve | no-weight-transfer.mp4 | 3.100 s | 93 | 874,892 |
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
  -cases "serve=/path/no-weight-transfer.mp4,lift=/path/lift-L3.mp4,clear=/path/clear-EG3.mp4,smash=/path/smash-EG12.mp4" \
  -runs 3 \
  -output ../docs/benchmarks/badminton-analysis-latency-2026-08-01-fullbody.csv
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
| Overall (12) | 34.942 | 35.695 | 44.598 | 14.842 | 1.675 | - |
| Serve: no weight transfer | 23.768 | 24.424 | 24.449 | 13.120 | 1.028 | 39.290 |
| Lift: L3 | 39.047 | 41.696 | 44.497 | 16.031 | 1.983 | 11.053 |
| Clear: EG3 | 42.423 | 41.426 | 44.127 | 17.712 | 1.815 | 23.120 |
| Smash: EG12 | 34.532 | 32.662 | 38.121 | 12.506 | 1.876 | 15.348 |

The deterministic grade and matched expert were stable across all three runs of
each fixture. Serve matched `nstc_right_IMG_6623`, lift `張宸愷3`, clear
`李翊安6`, and smash `林國欽1`. The coaching cue count can vary because OpenAI
selects one to three image-supported problems; the grade cannot vary. These
examples are not estimates of the 45.0 beginner-cohort mean.

| Stage | Mean | p50 | p95 |
|---|---:|---:|---:|
| RTMW pose extraction | 1.675 | 1.870 | 2.004 |
| Normalization and phase parsing | 0.025 | 0.025 | 0.026 |
| Correction and grading | 0.488 | 0.570 | 0.667 |
| Preview rendering | 3.133 | 3.389 | 3.782 |
| Coaching frame/prompt preparation | 1.644 | 1.900 | 2.035 |
| OpenAI inference | 14.842 | 14.418 | 22.066 |
| Final annotated rendering | 5.169 | 5.008 | 6.847 |
| Expert catalog lookup | 0.174 | 0.140 | 0.341 |
| GCS upload and signing | 0.696 | 0.647 | 0.888 |
| Signed URL refresh | 0.296 | 0.277 | 0.391 |
| Student 64 KiB range read | 0.225 | 0.166 | 0.470 |
| Expert 64 KiB range read | 0.286 | 0.277 | 0.483 |

The raw observations, scores, expert IDs, object paths, provider flags, and all
stage timings are committed in
`docs/benchmarks/badminton-analysis-latency-2026-08-01-fullbody.csv`.

The revision-36 zero-traffic deployment gate processed the 151-frame clear
fixture in 49.82 seconds client-side and 49.55 seconds service-side. Its first
request includes one-time ONNX session deserialization: pose took 7.68 seconds,
while OpenAI took 23.21 seconds. Both TensorRT flags were active, signed playback
checks passed, and only then was the revision promoted. The workflow subsequently
removed every older revision and image.

## Interpretation

The optimization result must be evaluated from the measured stage breakdown.
TensorRT accelerates YOLOX, RTMW3D, and the selected skill's correction
Transformer; rendering, OpenAI, GCS, and network time remain independent. OpenAI
accounts for 42.5% of mean client latency, while warm pose extraction accounts
for 4.8%. Model/reasoning selection and rendering therefore have more remaining
latency leverage than further pose-engine work.

## Limitations

- Three observations per skill show operational scale but are not a load test.
- Requests are sequential because production concurrency is intentionally one.
- OpenAI and GCS latency vary independently of GPU inference.
- Invalid OpenAI rule references are retried once, then replaced by deterministic
  Traditional Chinese rule advice. Exact predefined criterion titles are
  canonicalized to their rule IDs. A retry can make LLM tail latency longer than
  these observations.
- The fixtures have similar resolution and duration; longer uploads should be
  benchmarked separately before changing the current 150 MiB request limit.
