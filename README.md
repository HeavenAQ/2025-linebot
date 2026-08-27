# Badminton Motion Coaching (no-AI variant)

This monorepo contains the Python motion-analysis service, Go LINE/API backend,
and LIFF review interface for badminton coaching.

> **This is the `variant/no-ai` branch.** It is the same product with every
> large-language-model layer removed, so it can be run beside the original and
> compared. The analysis itself was never GPT: pose estimation, the EIMD
> diffusion correction, grading, checkpoints and expert matching are all local
> models and are all unchanged. What is gone is the natural-language layer on
> top -- coaching cues, the chat coach, the AI summary and the weekly 課前預習
> push. See [VARIANT.md](VARIANT.md) for what was removed, how to stand the
> variant up, and the secrets it needs.
>
> **Never merge `variant/no-ai` into `main`.** It is a parallel product line,
> not a feature branch.

> Current production support: **serve and smash only**. Lift and clear remain in
> the shared enums and some research utilities for compatibility, but the
> analysis service does not load models for them and rejects those requests with
> `INVALID_ARGUMENT`. Do not present lift or clear as working analysis features.

## Current architecture

```text
LINE / LIFF client
       |
       v
Go backend (public playback and application API)
       |
       | client-streamed gRPC request: header + MP4 bytes
       v
Python analysis service
  - RTMW3D pose extraction
  - serve/smash phase alignment
  - expert-only diffusion inference
  - grading
  - one H.264 video render
  - upload to GCS
       |
       | protobuf response: grades, feedback, object metadata, signed URLs
       v
Go backend -> Firestore persistence and user-facing playback response
```

The legacy FastAPI video-serving responsibility is being retired. Python still
generates and uploads the videos, but it does not return video bytes or directly
serve media files to the browser. It returns GCS object metadata and signed URLs.
The Go backend owns the public playback/video-serving endpoint. During migration,
Go may use the internal `RefreshPlaybackUrls` gRPC method to refresh expiring
signed URLs; clients should call Go, not the Python service, for playback.

## Repository layout

- `badminton_analysis_ai/`: Python gRPC analysis, models, rendering, and GCS
  upload/signing.
- `linebot/`: Go gRPC client, public application/playback API, LINE workflow, and
  Firestore persistence.
- `liff/`: review interface for feedback and generated-expert overlay videos.
- `proto/`: language-neutral gRPC contract and generated Python/Go bindings.
- `.github/workflows/`: CI and Cloud Run deployment.

## Latest models

The current release is the expert-motion generator v3. These are the only motion
weights required at runtime:

| Skill | Artifact | Purpose | SHA-256 |
|---|---|---|---|
| Serve | `models/expert_motion/serve/expert_motion_model.pt` | Expert-only diffusion prior | `9fdbde74084aa444dd9db60793669047816b02fb4b427bee6996ccad8ce899b3` |
| Serve | `models/expert_motion/serve/expert_score_model.npz` | Expert-distribution criterion calibration | `aaad4f9afe81cba6aed076aee6cdc268fe192055034130487afbefcab6016254` |
| Smash | `models/expert_motion/smash/expert_motion_model.pt` | Expert-only diffusion prior | `643d2d426ce0f098ae27c89700671aa3313f14ea32c09bd4f50f2fb6e4d63936` |
| Smash | `models/expert_motion/smash/expert_score_model.npz` | Expert-distribution criterion calibration | `34a051fd15909e755363ede0a387c447f90fcda41d3aa4038a478a44c27aaf5a` |

Common inference settings:

- method: conditional diffusion;
- normalized output: 64 frames, 17 COCO joints, 2D pose plus root trajectory;
- diffusion steps: 30;
- candidates per request: 16;
- deterministic inference seed: 19;
- conditioning: stable student morphology, lower-body preparation stance,
  handedness, source coordinate system, and phase timing;
- student data is inference-only and was not used for training or score fitting.

Checkpoint provenance:

| Skill | Expert sequences | Expert identities | Training manifest SHA-256 | Wrist velocity ceiling |
|---|---:|---:|---|---:|
| Serve | 14 | 7 | `4f5d23a86682e48c155f045e249292424b6cd2cf55d9d0545bd16973fed7b751` | `0.8502273201942444` |
| Smash | 12 | 12 | `15bb5c0e1b20150390e1796bd0965e48693e4774f58f167083467c871e9dbec8` | not applied |

Serve rate-limits a generated correction only when its root-invariant dominant
wrist velocity exceeds the maximum derived from expert demonstrations. It keeps
the exact beginning and ending poses and advances the swing earlier through
arc-length interpolation instead of deleting intermediate frames.

The removed `models/skeleton_correction` Transformer/ONNX/calibration bundles and
TensorRT corrector caches are obsolete and must not be restored to deployment.
TensorRT remains in use for the detector and RTMW3D pose estimator only.

## Phase extraction and correction

1. RTMW3D extracts the athlete's 2D/3D body landmarks.
2. Handedness is taken from the request or estimated, then left-handed motion is
   canonicalized for inference.
3. The dominant wrist trajectory is measured relative to the dominant elbow so
   camera or body translation does not create a false acceleration peak.
4. Serve ends at the maximum shoulder angle occurring after maximum wrist
   acceleration. This rule is serve-specific.
5. Smash retains its original overhead ending-range mechanism, including delayed
   contact/follow-through refinement.
6. Motion is normalized to the student's coordinate system and phase-aligned to
   64 frames.
7. The diffusion prior generates a complete expert movement. It does not preserve
   a beginner's incorrect initial hand position, so missing hand-raise steps can
   be corrected.
8. Full-body forward-kinematic retargeting preserves the student's stable anatomy
   and generated root motion.

## Grading

Serve and smash are graded against the generated expert motion, not against one
catalog video. Each criterion combines confidence-masked Euclidean distance and
target-angle distance. Score tolerances are calibrated from held-out expert
identity distributions only.

Serve criteria and maxima:

| Criterion ID | Maximum |
|---|---:|
| `arms_raised` | 5 |
| `racket_foot_weight` | 5 |
| `weight_transfer` | 30 |
| `hip_rotation` | 10 |
| `wrist_flick` | 30 |
| `shoulder_rotation` | 20 |

Smash criteria and maxima:

| Criterion ID | Maximum |
|---|---:|
| `preparation` | 10 |
| `body_rotation` | 10 |
| `arm_balance` | 20 |
| `elbow_forward` | 20 |
| `wrist_flick` | 20 |
| `follow_through` | 20 |

The API reports `score_status=expert_only_generated_distribution`.

## Generated video

Every successful analysis generates and uploads one H.264/yuv420p video: the
detected and generated-expert skeletons drawn over the source footage, running
straight through with no inserted pauses. The annotated second render existed
only to burn coaching captions and pauses over this one; with nothing to draw it
would come out frame for frame identical, and an L4 is billed by the second.

`student_video`, `feedback_video` and `skeleton_overlay_video` therefore all name
that single upload, so every existing client keeps finding a video under the name
it knows. The protobuf response contains only URLs and metadata:

- GCS object path and `gs://` URI;
- expiring signed HTTPS URL;
- signed URL expiration time;
- duration, FPS, width, and height.

No generated video is embedded in protobuf, JSON, or base64.

## API contract

`proto/badminton/analysis/v1/analysis.proto` defines:

- `AnalyzeVideo`: client-streamed request containing one header followed by MP4
  chunks; returns analysis results and the generated-video URL. `coaching_cues`
  and `overall_feedback` remain in the contract and are always empty: nothing in
  this variant writes natural language.
- `Health`: reports service readiness and the loaded skills. It should currently
  return only `SKILL_SERVE` and `SKILL_SMASH`.
- `RefreshPlaybackUrls`: internal transition helper used by Go to refresh signed
  URLs. Browser and LIFF clients must use the Go playback endpoint.

Public playback is owned by Go:

```text
GET /api/db/playback?user_id=<id>&skill=<serve|smash>&work_date=<timestamp>
```

The Go client streams video input in 1 MiB chunks and persists the returned media
records. Serve/smash return `Generated expert prior` rather than a separate
catalog expert video; the clean overlay is the generated-expert review view.

## Configuration

Required Python analysis-service variables:

| Variable | Purpose |
|---|---|
| `ANALYSIS_GRPC_API_KEY` | Internal gRPC API-key authentication |
| `GCP_PROJECT_ID` | GCS project |
| `GCS_BUCKET_NAME` | Destination for generated videos |
| `GCP_SERVICE_ACCOUNT_EMAIL` | IAM signed-URL fallback identity |

Important optional variables:

| Variable | Default | Purpose |
|---|---|---|
| `EXPERT_MOTION_MODEL_ROOT` | `badminton_analysis_ai/models/expert_motion` | Directory containing `serve/` and `smash/` model pairs |
| `EXPERT_MOTION_DEVICE` | `auto` | `auto`, `cpu`, `mps`, `cuda`, or a concrete PyTorch device |
| `MAX_VIDEO_BYTES` | 150 MiB | Maximum streamed request size |
| `SIGNED_URL_MINUTES` | `60` | Generated-video URL lifetime |
| `POSE_EXECUTION_PROVIDER` | environment-dependent | RTMW3D provider; production uses `tensorrt` |
| `POSE_TENSORRT_CACHE_DIR` | unset | Detector/pose TensorRT cache directory |

`auto` chooses CUDA when available, otherwise Apple MPS, otherwise CPU. Production
Cloud Run uses an NVIDIA L4. Local Apple Silicon inference can use MPS.

Removed variables such as `SKELETON_MODEL_ROOT`, `SKELETON_EXECUTION_PROVIDER`,
and `EXPERT_VIDEOS_COLLECTION` are not part of the current serve/smash runtime.
`SKELETON_DEVICE` is accepted only as a temporary compatibility fallback for
`EXPERT_MOTION_DEVICE`.

## Running locally

Install the pinned dependencies, configure GCP ADC and environment variables,
then start the analysis service:

```bash
cd badminton_analysis_ai
export PYTHONPATH="$PWD:$PWD/generated"
python -m service.server
```

The service requires FFmpeg for final H.264 rendering. The container also
includes RTMW3D/YOLOX pose models and their production TensorRT dependencies.

Run checks:

```bash
cd badminton_analysis_ai
PYTHONPATH=.:generated pytest -q tests

cd ../linebot
go test ./...

cd ../liff
npm test
npm run build
```

## Operational checks

Before deployment, verify:

1. all four expert-motion artifacts exist and match the hashes above;
2. `Health` lists only serve and smash;
3. a streamed serve and smash request each return a non-empty `student_video`
   signed URL, and `coaching_cues` and `overall_feedback` come back empty;
4. that URL decodes as H.264/yuv420p;
5. the video duration matches the checkpoint timeline, which is measured in
   render time with no pause offset;
6. no `models/skeleton_correction` or TensorRT `correctors/` directory is present;
7. the Go playback endpoint refreshes and returns the media records.

## Main inference trace

1. `proto/badminton/analysis/v1/analysis.proto`
2. `badminton_analysis_ai/service/server.py`
3. `badminton_analysis_ai/service/pipeline.py`
4. `badminton_analysis_ai/badminton_analysis/services/pose_detector.py`
5. `badminton_analysis_ai/badminton_analysis/services/video_analyzer.py`
6. `badminton_analysis_ai/badminton_analysis/ml/expert_motion_preprocessing.py`
7. `badminton_analysis_ai/badminton_analysis/ml/expert_motion_backend.py`
8. `badminton_analysis_ai/badminton_analysis/ml/expert_motion_generator.py`
9. `badminton_analysis_ai/badminton_analysis/ml/kinematic_retargeting.py`
10. `badminton_analysis_ai/service/renderer.py`
11. `linebot/api/analysis/client.go`
12. `linebot/main.go`
13. `liff/src/components/VideoComparison.tsx`
