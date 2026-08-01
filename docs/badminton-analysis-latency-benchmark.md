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

The one-time `Health` call is recorded separately. The first analysis includes
lazy RTMW3D session construction and prebuilt TensorRT engine deserialization,
so it is reported as a cold-model observation. Engine compilation is excluded
from request latency. Subsequent requests reuse the loaded process and sessions.

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

Results are populated after the production deployment completes.

## Interpretation

The optimization result must be evaluated from the measured stage breakdown.
TensorRT accelerates YOLOX, RTMW3D, and the selected skill's correction
Transformer; rendering, OpenAI, GCS, and network time remain independent.

## Limitations

- Three observations per skill show operational scale but are not a load test.
- Requests are sequential because production concurrency is intentionally one.
- OpenAI and GCS latency vary independently of GPU inference.
- The fixtures have similar resolution and duration; longer uploads should be
  benchmarked separately before changing the current 150 MiB request limit.
