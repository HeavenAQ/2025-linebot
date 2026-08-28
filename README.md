# Badminton Motion Coaching

This monorepo contains the Python motion-analysis service, Go LINE/API backend,
and LIFF review interface for badminton coaching.

> Current production support: **serve and smash only**. Lift and clear remain in
> the shared protobuf enum for wire compatibility, but the analysis service does
> not load models for them and rejects those requests with `INVALID_ARGUMENT`.
> Do not present lift or clear as working analysis features.

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
  - RF-DETR Keypoint Preview COCO-17 pose extraction
  - serve/smash phase alignment
  - expert-only diffusion inference
  - grading and GPT coaching
  - two H.264 video renders
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

- `badminton_analysis_ai/`: Python gRPC analysis, models, rendering, GPT feedback,
  and GCS upload/signing.
- `linebot/`: Go gRPC client, public application/playback API, LINE workflow, and
  Firestore persistence.
- `liff/`: review interface for feedback and generated-expert overlay videos.
- `proto/`: language-neutral gRPC contract and generated Python/Go bindings.
- `.github/workflows/`: CI and Cloud Run deployment.

## Latest models

The current release is the expert-only Error-Isolated Motion Diffusion (EIMD)
v3 generator. These are the only motion weights required at runtime:

| Skill | Artifact | Purpose | SHA-256 |
|---|---|---|---|
| Serve | `models/error_isolated_motion/serve/error_isolated_motion.pt` | Expert-only EIMD prior | `bec47df5341429b0c6fd6c3bf470db83d80ce32645fc6931e00c4a050c7b8051` |
| Serve | `models/error_isolated_motion/serve/expert_score_model.npz` | 53-take, 7-subject RF-DETR expert-distribution and residual scorer | `1a0e3c7e5dc32ee019d071e35255d9337a25c38f5c7210d1ec62104058c9095c` |
| Smash | `models/error_isolated_motion/smash/error_isolated_motion.pt` | Expert-only EIMD prior | `956b567e407d88eff23ebc936f264e03d16543a550c08bf879268fbb85353977` |
| Smash | `models/error_isolated_motion/smash/expert_score_model.npz` | Expert-only qualitative scorer | `1cf4c958cbe360a1c739260e4c077cd286cdec900712c25bcde0a1e72a228f20` |
| Smash | `models/error_isolated_motion/smash/expert_semantic_score_model.npz` | Active expert-only semantic distribution scorer | `be6b704bd580ff362bb6718eefa4596b51c3edc8368b08d40626140ebc1b16a5` |
| Smash | `models/error_isolated_motion/smash/expert_trajectory_score_model.npz` | Phase-aligned Euclidean residual and expert-manifold gate | `1ed6ee9aa4218f05fdd60e8e0ccdd833ce7163ac2cb75666532ac83f653af027` |

Common inference settings:

- method: conditional diffusion;
- normalized output: 64 frames, 17 COCO joints, 2D pose plus root trajectory;
- diffusion steps: 30;
- candidates per request: 16;
- deterministic inference seed: 19;
- conditioning: stable student morphology, lower-body preparation stance,
  handedness, source coordinate system, and phase timing;
- student data is inference-only and was not used for training or score fitting.

Generator checkpoint provenance (the serve scorer separately uses the 53-take,
seven-identity RF-DETR bank listed above):

| Skill | Expert sequences | Expert identities | Training manifest SHA-256 | Wrist velocity ceiling |
|---|---:|---:|---|---:|
| Serve | 10 train + 4 held out | 5 train + 2 held out | `b70f4c82077a5f34d70cf01fd3a69aa6aab2139bd9950b5deecd778c3fa2720d` | `0.8502273201942444` |
| Smash | 8 train + 4 held out | 8 train + 4 held out | `284b7bcee32acba784ef06843f26170fa0c9953db96edd42b75c55ea32e9df21` | no output limiter |

Serve rate-limits a generated correction only when its root-invariant dominant
wrist velocity exceeds the maximum derived from expert demonstrations. It keeps
the exact beginning and ending poses and advances the swing earlier through
arc-length interpolation instead of deleting intermediate frames.

The removed `models/expert_motion` and `models/skeleton_correction`
Transformer/ONNX/calibration bundles and TensorRT corrector caches are obsolete
and must not be restored to deployment. TensorRT remains in use only for the
batched RF-DETR pose model in production.

## Phase extraction and correction

1. RF-DETR Keypoint Preview extracts one athlete's 17 COCO 2D joints. Production
   runs its fixed-batch FP16 TensorRT engine; local Apple Silicon validation uses
   the same RF-DETR weights through MPS.
2. Handedness is taken from the request or estimated, then left-handed motion is
   canonicalized for inference.
3. The dominant wrist trajectory is measured relative to the dominant elbow so
   camera or body translation does not create a false acceleration peak.
4. Serve ends at the maximum shoulder angle occurring after maximum wrist
   acceleration in the coherent forward swing. The acceleration search is
   restricted to the first 70% of onset-to-lowest-hand progress only when
   motion-onset recovery proves that the legacy pre-impact window truncated a
   substantial preparation segment. Otherwise the established contact anchor
   is retained. This rule is serve-specific.
5. Smash retains its original overhead ending-range mechanism, including delayed
   contact/follow-through refinement.
6. Motion is normalized to the student's coordinate system and phase-aligned to
   64 frames. Serve finds the stable interval immediately before the dominant
   wrist-motion episode, then selects the minimum smoothed canonical pelvis x
   only inside that preparation interval. This replaces the legacy fixed
   30-frame pre-impact offset.
7. The diffusion prior generates a complete expert movement. It does not preserve
   a beginner's incorrect initial hand position, so missing hand-raise steps can
   be corrected.
8. Full-body forward-kinematic retargeting preserves the student's stable anatomy
   and generated root motion.
9. Serve and smash both rotate the generated full body into the student's
   preparation ankle–spine camera frame before scoring. A preparation-derived
   ankle, knee-chain, then hip-chain placement is applied once to the whole
   motion; it does not follow the student frame by frame. Stable student bone
   lengths are enforced after placement. For serve, full-credit detected body
   chains remain unchanged and only deficient chains blend smoothly toward the
   generated expert motion.

The serve truncation gate requires two independent signals before replacing the
legacy acceleration anchor: at least four raw detected frames before the fixed
window, plus a substantial preparation extension in the confidence-interpolated
trajectory. Raw joints still determine acceleration and maximum-shoulder
completion. This preserves complete short clips while recovering preparation
hidden by brief pose-detector gaps.

## Grading

Serve generation and grading are separate. The correction video uses generated
expert motion, while the serve grade uses a subject-balanced expert-only
checkpoint distribution rather than one catalog video or one stochastic
diffusion sample. RF-DETR confidence remains continuous. Identity-held-out
expert folds set each tolerance; no learner recording is used for fitting.

All six serve checkpoints use continuous expert-only distribution distances.
Weight transfer compares preparation-to-completion balance; hip
rotation compares the joint pelvis/shoulder contraction and axial/twist proxy;
wrist action compares burst magnitude and temporal coherence; shoulder
follow-through compares contraction, cross-body reach, and terminal elbow/wrist
placement. Dynamic tolerances cover every valid identity-held-out expert
residual; stance and wrist use robust central expert tolerances. No checkpoint
is hard-zeroed by a completion rule. Each checkpoint also measures the
phase-aligned distance between the detected learner skeleton and the generated
correction. An expert-only angle manifold protects valid expert style, while
motions outside that manifold use the correction residual to suppress false
semantic positives. Learner videos are evaluation-only and do not set any
tolerance, gate, or score mapping.

The API product grade retains the qualitative rubric's original point weights:
`5/5/30/10/30/20`. A separate equal six-item checklist score exists only for
validation against the two-expert workbook, whose raters awarded one point per
completed checkpoint. Product grading and ICC therefore measure their intended
constructs without silently replacing the rubric weights.

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

Smash uses the same scoring family as serve: phase-aligned Euclidean distance
between the detected and corrected skeletons, combined with an expert-only
motion-manifold gate. The active trajectory artifact declares
`distance_method=euclidean` and `fusion=manifold_gate`. Learner recordings were
used only for held-out evaluation and did not fit the artifact. The workbook
release uploaded on 2026-08-28 has learner ICC(2,1) `0.7073` and pooled
100-video ICC(2,1) `0.9083`; these values describe the frozen validation cohort,
not external-dataset generalization.

The API reports `score_status=expert_only_generated_distribution`.

### Validation policy

Scorer artifacts are fitted from expert pose data only. Student/team/novice
recordings and the two-rater workbooks are held-out validation inputs. They may
be used to report ICC and inspect failure modes, but never to fit tolerances,
gates, score mappings, or filename/cohort-specific branches. Offline experiment
scripts and generated validation reports are intentionally excluded from the
deployment tree.

## Generated videos

Every successful analysis generates and uploads two H.264/yuv420p videos:

1. `feedback_video`: detected and generated-expert skeletons, GPT-selected problem
   circles, feedback panels, and inserted coaching pauses.
2. `skeleton_overlay_video`: the same detected/generated skeleton overlay without
   GPT annotations or pauses.

`student_video` is retained as a backward-compatible alias of `feedback_video`.
The protobuf response contains only URLs and metadata:

- GCS object path and `gs://` URI;
- expiring signed HTTPS URL;
- signed URL expiration time;
- duration, FPS, width, and height.

No generated video is embedded in protobuf, JSON, or base64.

## API contract

`proto/badminton/analysis/v1/analysis.proto` defines:

- `AnalyzeVideo`: client-streamed request containing one header followed by MP4
  chunks; returns analysis results and both generated-video URLs.
- `Health`: reports service readiness and the loaded skills. It should currently
  return only `SKILL_SERVE` and `SKILL_SMASH`.
- `RefreshPlaybackUrls`: internal transition helper used by Go to refresh signed
  URLs. Browser and LIFF clients must use the Go playback endpoint.

Public playback is owned by Go:

```text
GET /api/db/playback?user_id=<id>&skill=<serve|smash>&work_date=<timestamp>
```

The Go client streams video input in 1 MiB chunks and persists both returned media
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
| `OPENAI_API_KEY` | GPT coaching generation |

Important optional variables:

| Variable | Default | Purpose |
|---|---|---|
| `EXPERT_MOTION_MODEL_ROOT` | `badminton_analysis_ai/models/error_isolated_motion` | Directory containing the active `serve/` and `smash/` EIMD/scorer pairs |
| `EXPERT_MOTION_DEVICE` | `auto` | `auto`, `cpu`, `mps`, `cuda`, or a concrete PyTorch device |
| `OPENAI_COACHING_MODEL` | `gpt-5.6-terra` | Coaching model |
| `COACHING_PAUSE_SECONDS` | `2` | Pause inserted at each feedback frame |
| `MAX_VIDEO_BYTES` | 150 MiB | Maximum streamed request size |
| `SIGNED_URL_MINUTES` | `60` | Generated-video URL lifetime |
| `POSE_EXECUTION_PROVIDER` | environment-dependent | RF-DETR provider; production uses `tensorrt` |
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
includes RF-DETR Keypoint Preview and its production TensorRT dependencies.

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

1. all six active EIMD/scorer artifacts exist and match the hashes above;
2. `Health` lists only serve and smash;
3. a streamed serve and smash request each return non-empty `feedback_video` and
   `skeleton_overlay_video` signed URLs;
4. both URLs decode as H.264/yuv420p;
5. feedback duration is equal to or longer than the clean overlay because of
   coaching pauses;
6. no `models/expert_motion`, `models/skeleton_correction`, or TensorRT
   `correctors/` directory is present;
7. the Go playback endpoint refreshes and returns both media records.

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
10. `badminton_analysis_ai/service/coaching.py`
11. `badminton_analysis_ai/service/renderer.py`
12. `linebot/api/analysis/client.go`
13. `linebot/main.go`
14. `liff/src/components/VideoComparison.tsx`
