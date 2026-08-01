# Badminton Motion Coaching

This monorepo implements a LINE-based badminton coaching system for serve, lift,
clear, and smash. A Python GPU service extracts and corrects skeleton motion, a Go
service owns the LINE workflow and persistence, and a Next.js LIFF application
plays the student and matched-expert videos on one synchronized timeline.

The current analyzer is `badminton_analysis_ai`. It replaces the retired
angle-only HTTP analyzer, fixed expert video, base64 video response, and Go-side
video suggestion logic.

## Repository Layout

- `badminton_analysis_ai/`: Python gRPC motion-analysis service, four correction
  checkpoints, OpenAI coaching, video renderer, and expert catalog seeder.
- `linebot/`: Go LINE webhook/API service, Firestore persistence, gRPC client,
  playback URL refresh, and conversation-aware GPT chat.
- `liff/`: Next.js mobile portfolio and synchronized video comparison UI.
- `proto/`: language-neutral gRPC contract.
- `docs/`: architecture, operations, and measured benchmark results.
- `.github/workflows/`: CI plus LIFF, Go backend, and GPU analyzer CD.

## Knowledge Prerequisites

To follow the complete implementation, be comfortable with:

1. Python, NumPy array shapes, PyTorch inference/training, and confidence masks.
2. COCO's 17 body keypoints and basic pose concepts: position, joint angle,
   velocity, acceleration, body-relative coordinates, and bone length.
3. Temporal resampling and phase alignment of variable-length motion sequences.
4. gRPC client streaming and protobuf-generated Python/Go bindings.
5. Firestore documents, Cloud Storage object paths, V4 signed URLs, and Cloud Run.
6. Go HTTP handlers and LINE webhook state machines.
7. React media elements and synchronization through normalized playback progress.

Pose inference uses RTMW3D-X through `rtmlib` and ONNX Runtime. Production uses
TensorRT 10.14 FP16 engines with CUDA fallback for unsupported operators for
the YOLOX detector, RTMW3D pose network, and all four skeleton-correction
Transformers. The versioned exact-SM89 cache is checksum verified and baked into
the container image. The deployment gate loads these L4-specific engines during
a real analysis without a runtime GCS cache mount and keeps one warm instance
after verification.

## End-to-End Architecture

```text
LINE video
    |
    v
Go LINE webhook ---- client-streamed bytes ----> Python gRPC service on L4
    |                                              |
    |                                              +-- RTMW3D 2D/3D skeleton
    |                                              +-- phase/handedness parser
    |                                              +-- expert matching
    |                                              +-- correction + grading
    |                                              +-- constrained GPT coaching
    |                                              +-- annotated H.264 renderer
    |                                              +-- GCS upload/signing
    |<---- grades, cues, metadata, signed URLs ----+
    |
    +-- thumbnail -> GCS
    +-- complete analysis record -> Firestore
    +-- LINE completion reply
    |
    v
LIFF /personal -> Go playback API -> refreshed student/expert signed URLs
    |
    +-- two video elements, one normalized timeline
    +-- expert freezes during student coaching pauses
```

Videos never cross the Python-to-Go boundary as base64. The request is streamed
in 1 MiB gRPC chunks. The rendered result is uploaded by Python, and Go receives
only structured analysis data, GCS object paths, and expiring signed URLs.
LINE may return HTTP 202 with no bytes while an uploaded video is still being
transcoded. The Go adapter retries that pending response and refuses to open a
gRPC stream until it has non-empty `video/*` content.

## Motion Pipeline

`SkeletonAnalysisPipeline.analyze` executes these stages:

1. **Pose extraction**: YOLOX detects the athlete periodically and RTMW3D-X
   directly estimates 133 whole-body 2D/3D keypoints. The first 17 COCO body
   joints enter motion analysis; all 133 2D joints remain available to the
   renderer. There is no separate Human3.6M pose-lifting stage.
2. **Handedness**: explicit left/right input is honored. `auto` compares the
   motion and peak acceleration evidence of both wrists; left-handed motion is
   mirrored into the canonical dominant-right representation. This prevents a
   dominant left shoulder from being reported as the physical right shoulder.
3. **Skill window**: wrist/elbow trajectories locate the start, peak/contact,
   and end for the selected skill. Serve, lift, clear, and smash have separate
   parsing rules.
4. **Normalization**: missing observations are interpolated, the pelvis is the
   origin, body orientation is canonicalized, coordinates are divided by the
   median observed anatomical-segment length, and the window is resampled to 64
   frames. Using all major body segments prevents one noisy shoulder-depth
   estimate from amplifying the entire skeleton.
5. **Phase alignment**: five skill checkpoints map preparation through completion
   to a common timeline, so motion speed does not dominate comparison.
6. **Expert selection**: the checkpoint bank is filtered to the student's
   handedness before distance is calculated. Every remaining training expert
   is first adapted to the student's bone lengths, then ranked with the same
   confidence-masked, skill-weighted position, angle, velocity, and bone
   distance used for grading. This is not cosine similarity or an unrelated
   embedding score, and there is no cross-handed fallback.
7. **Correction**: a temporal/spatial Transformer consumes seven features per
   joint: student XYZ, selected expert XYZ, and observation confidence. Its
   output is optionally blended toward the selected expert according to the
   checkpoint, projected back onto the student's bone lengths, and restored to
   the student's original phase timing.
8. **Grading**: the magnitude of the required correction is scored globally and
   within qualitative skill-rule windows.
9. **Expert verification**: training accepts a checkpoint only when corrected
   held-out students move closer to experts, fall within held-out expert
   variability, preserve bone lengths, and satisfy correction smoothness bounds.
10. **Coaching**: OpenAI sees sampled frames, deterministic grade components,
    handedness, allowed rule IDs, allowed phase anchors, and permitted joints.
    It writes Traditional Chinese feedback but does not invent the grade. Rule,
    frame, and joint outputs are validated and snapped to the deterministic spec.
11. **Rendering**: yellow is the detected student skeleton, green is the
    corrected skeleton, red circles mark validated problematic joints, and the
    video pauses at coaching frames. Output is H.264/yuv420p.
12. **Storage**: the annotated student video is uploaded to GCS. The expert
    catalog resolves the selected expert's GCS video, and both objects receive
    short-lived V4 signed URLs.

## Grading Algorithm

Let `S` be the normalized student skeleton, `C` the corrected skeleton, and `M`
the joint-confidence mask. Each skill supplies a 17-element importance vector.
The correction distance is:

```text
D = position(S,C,M)
  + 0.5 * angle(S,C,M)
  + 0.5 * velocity(S,C,M)
  + 0.25 * bone_length(S,C,M)
  + serve_only(0.65 * support_transition(S,C,M)
             + 0.35 * torso_lean_transition(S,C,M))
```

Position and velocity are confidence- and skill-joint-weighted means. Angle is
the normalized error over eight limb triplets. Bone length covers twelve body
segments. For serve, the transition terms compare the complete first-to-last
lower-body support trajectory and signed shoulder-midpoint-to-hip-midpoint
forward lean.
They affect expert selection, training, the total score, the 20-point weight
transfer criterion, and the evidence sent to GPT. The final score is calibrated
per skill:

```text
score = 100 * exp(-alpha * max(D - distance_offset, 0))
```

`alpha` and `distance_offset` are fit from held-out expert and beginner
correction-distance distributions, targeting approximately 99.8 for experts and
45 for beginners. The current evaluation means are:

| Skill | Beginner mean | Expert mean | Separation AUC |
|---|---:|---:|---:|
| Serve | 45.00 | 99.82 | 1.00 |
| Lift | 45.00 | 99.80 | 1.00 |
| Clear | 45.00 | 99.79 | 1.00 |
| Smash | 45.00 | 99.81 | 0.98 |

Criterion scores use the same calibrated correction distance on rule-specific
frames and joints, capped by each rule's maximum and reconciled to the total.
The skill definitions in `skill_specs.py` are therefore the source of truth for
both quantitative grading and the feedback an LLM is allowed to provide.

## Expert Catalog

The legacy corpus contributes 50 right-handed experts for clear, lift, and
smash. Serve excludes `scoring_videos/發球/羽球隊同學`; its expert bank is NSTC
only. NSTC experts are included only from each skill's direct `left/` and
`right/` directories; person-named NSTC directories are excluded. The deployed
inventory is:

| Skill | Right | Left | Total |
|---|---:|---:|---:|
| Clear | 80 | 20 | 100 |
| Serve | 16 | 10 | 26 |
| Lift | 72 | 10 | 82 |
| Smash | 66 | 9 | 75 |

Firestore collection `badminton_experts_v2` stores one deterministic RTMW3D
record per video and vector with:

- skill, expert ID, display name, and handedness;
- GCS video and vector object paths;
- duration, FPS, width, height, and the expert action-window timestamps.

All legacy experts are authoritative right-handed samples. NSTC sequence IDs
use `nstc_left_` and `nstc_right_` prefixes so same-named videos cannot collide.
Training is stratified by handedness, and pseudo-target selection, checkpoint
reference selection, and catalog validation all reject cross-handed matches.

The checkpoint contains the training expert reference bank and filenames. The
selected filename becomes the catalog lookup key, so the video displayed by LIFF
is the same expert motion used to condition correction. LIFF maps its shared
timeline to that expert action window rather than to the entire raw clip. The
mobile personal page exposes this under the `影片比較` tab; legacy records without
synchronized media show an explicit re-upload message.

Seed or verify the catalog with:

```bash
cd badminton_analysis_ai
python seed_expert_catalog.py \
  --project-id nstc-linebot-2025 \
  --bucket nstc-2025-storage \
  --collection badminton_experts_v2 \
  --source-root /path/to/badminton-analysis \
  --prune \
  --workers 8
```

The seeder is deterministic. It skips existing videos, refreshes vectors,
validates exact handedness/source counts, and with `--prune` removes obsolete
managed catalog documents, videos, and vectors.

## API Contract

`proto/badminton/analysis/v1/analysis.proto` defines:

- `AnalyzeVideo`: client-streamed header/video bytes to one structured response.
- `RefreshPlaybackUrls`: validates stored analysis/expert paths and issues new
  signed URLs without rerunning inference.
- `Health`: reports service readiness and loaded skills.

The Go playback endpoint is:

```text
GET /api/db/playback?user_id=<id>&skill=<skill>&work_date=<timestamp>
```

It loads the persisted analysis, refreshes both URLs through gRPC, and returns
student/expert media metadata, phase markers, coaching cues, grade, and feedback.

## Code Trace Order

For online inference, read in this order:

1. `proto/badminton/analysis/v1/analysis.proto`
2. `badminton_analysis_ai/service/server.py`
3. `badminton_analysis_ai/service/pipeline.py`
4. `badminton_analysis_ai/badminton_analysis/services/pose_detector.py`
5. `badminton_analysis_ai/badminton_analysis/services/video_analyzer.py`
6. `badminton_analysis_ai/badminton_analysis/ml/handedness.py`
7. `badminton_analysis_ai/badminton_analysis/ml/skeleton_normalization.py`
8. `badminton_analysis_ai/badminton_analysis/ml/skill_specs.py`
9. `badminton_analysis_ai/badminton_analysis/ml/skeleton_backend.py`
10. `badminton_analysis_ai/badminton_analysis/ml/infer_skeleton_corrector.py`
11. `badminton_analysis_ai/badminton_analysis/ml/models/skeleton_denoiser.py`
12. `badminton_analysis_ai/badminton_analysis/ml/skeleton_scoring.py`
13. `badminton_analysis_ai/service/coaching.py`
14. `badminton_analysis_ai/service/renderer.py`
15. `badminton_analysis_ai/service/expert_catalog.py` and `storage.py`
16. `linebot/api/analysis/client.go`
17. `linebot/app/postback_handlers.go` and `video_utils.go`
18. `linebot/api/db/users.go` and `linebot/main.go`
19. `liff/src/lib/api/fetchPlayback.ts`
20. `liff/src/components/VideoComparison.tsx`

For training, start with `train_skeleton_corrector.py`, then read
`skeleton_dataset.py`, `skeleton_denoiser.py`, `skeleton_scoring.py`, and finally
the checkpoint acceptance block in the training script.

## Adding Another Skill

1. Add its enum to Python/Go/protobuf and regenerate both language bindings.
2. Define checkpoint anchors, joint weights, rule maxima, measured joints,
   coaching joints, and allowed feedback anchors in `skill_specs.py`.
3. Add its motion-window parser to `video_analyzer.py` and tests for edge cases.
4. Extract expert/beginner normalized sequences and verify handedness labels.
5. Train with disjoint train/validation/test splits and pass all expert-distance,
   bone-preservation, and smoothness quality gates.
6. Fit score calibration only from evaluation distributions; do not hardcode a
   cosmetic score transformation in the API.
7. Seed the expert videos/vectors into GCS and Firestore.
8. Add the checkpoint to the service loader and expose the skill in Go/LIFF.
9. Run contract tests, real GPU video analysis, qualitative rendering review,
   signed playback checks, and latency benchmarks.

## Local Development

Python contract tests do not load RTMW3D or require a GPU:

```bash
cd badminton_analysis_ai
pytest -q tests
```

The complete analyzer requires Linux, CUDA 12, cuDNN 9, TensorRT 10.14,
ONNX Runtime GPU 1.24.4, `rtmlib` 0.0.15, PyTorch 2.5.1, FFmpeg, GCP ADC, and
the environment variables used in `service/config.py`. Run it with:

```bash
PYTHONPATH="$PWD:$PWD/generated" python -m service.server
```

Run Go and LIFF checks:

```bash
cd linebot && go test ./...
cd ../liff && npm ci && npm run build
```

Live integrations are intentionally explicit:

```bash
RUN_LIVE_ANALYSIS=1 LIVE_ANALYSIS_VIDEO=/path/video.mp4 \
  go test -count=1 -v ./linebot/api/analysis -run TestLiveAnalysisService

RUN_LIVE_OPENAI=1 go test -count=1 -v ./linebot/api/gpt
RUN_LIVE_FIRESTORE=1 go test -count=1 -v ./linebot/api/db
RUN_LIVE_SECRET=1 go test -count=1 -v ./linebot/api/secret
```

Required production variables include `ANALYSIS_GRPC_TARGET`,
`ANALYSIS_GRPC_API_KEY`, `OPENAI_API_KEY`, `GCP_PROJECT_ID`, `GCS_BUCKET_NAME`,
`GCP_SERVICE_ACCOUNT_EMAIL`, and the existing LINE/Firestore settings. Secrets
belong in Secret Manager and local ignored `.env` files, never Git.

The analyzer deployment sets `POSE_EXECUTION_PROVIDER=tensorrt`,
`SKELETON_EXECUTION_PROVIDER=tensorrt`, and
`POSE_TENSORRT_CACHE_DIR=/app/models/tensorrt-cache`. Set either provider to
`cuda` only for a controlled compatibility or accuracy comparison.

## Deployment

- `ci.yml` runs Go tests, Python correction/spec tests, and the LIFF build.
- `cd-motion-analysis.yml` builds the CUDA image, deploys the
  `badminton-analysis-ai` Cloud Run service with one L4 GPU and HTTP/2 at zero
  traffic, then streams a real beginner clear video through the candidate URL
  and verifies signed playback. Only a passing candidate is promoted to 100%.
  Superseded GPU revisions and container images are removed after promotion.
- `cd-linebot.yml` builds/deploys the Go service and checks its live health
  endpoint.
- `cd-liff.yml` builds the static LIFF with the production Cloud Run backend,
  rejects bundles containing the localhost development URL, and publishes
  `liff/out` to the existing Netlify production site.

Generated videos, logs, local datasets, credentials, `.env` files, analysis
working directories, and TensorRT caches are ignored. PyTorch checkpoints,
calibration files, and portable ONNX graphs are tracked because they are
required runtime artifacts. The ignored 17-file exact-SM89 TensorRT cache is
downloaded from GCS, SHA-256 verified, and baked into the image by the GPU
deployment workflow.
