# Badminton Motion Coaching

This monorepo contains the Python motion-analysis service, Go LINE/API backend,
and LIFF review interface for badminton coaching.

> Current production support: **serve and smash only**. Lift and clear remain in
> the shared protobuf enum for wire compatibility, but the analysis service does
> not load models for them and rejects those requests with `INVALID_ARGUMENT`.
> Do not present lift or clear as working analysis features.

## Current architecture

```text
LINE app ──video──▶ Go backend (Cloud Run, asia-east1)
                      │ store input + thumbnail in GCS
                      │ Firestore transaction: pending attempt + analysis job
                      │ named Cloud Task ──▶ Go worker /internal/analysis/task
                      │
                      │ client-streamed gRPC: header + MP4 chunks
                      ▼
            Python analysis service (Cloud Run + NVIDIA L4, asia-southeast1)
              - RF-DETR Keypoint Preview COCO-17 pose
                (fixed-batch-16 FP16 TensorRT, cross-request microbatching)
              - serve/smash phase alignment and requested-skill guard
              - expert-only EIMD v3 diffusion and grading
              - GPT coaching
              - two H.264 video renders, uploaded to GCS
                      │
                      │ protobuf: grades, feedback, object metadata
                      ▼
            Go worker ──▶ Firestore (attempt completed)

LIFF app ──LINE ID token──▶ Go learner API (signs playback URLs itself)
```

Uploads are processed as durable jobs; see
`badminton_analysis_ai/api/ASYNC_ARCHITECTURE.md` for the queue, retry and
GPU-capacity schedule. Python generates and uploads the videos and returns GCS
object metadata; it never returns video bytes. The Go backend signs playback
URLs locally (`linebot/api/storage/playback.go`), so browsers and LIFF never
call the Python service.

## Repository layout

- `badminton_analysis_ai/`: Python gRPC analysis, models, rendering, GPT feedback,
  and GCS upload/signing.
- `linebot/`: Go LINE webhook, learner API, analysis job queue/worker, gRPC
  client, and Firestore persistence.
- `liff/`: review interface for feedback videos and the matched expert clip.
- `proto/`: language-neutral gRPC contract and generated Python/Go bindings.
- `scripts/`: Cloud Tasks / Cloud Scheduler provisioning and queue verification.
- `.github/workflows/`: CI and Cloud Run / Netlify deployment.

## Latest models

The current release is the expert-only Error-Isolated Motion Diffusion (EIMD)
v3 generator. These are the only motion weights required at runtime:

| Skill | Artifact | Purpose | SHA-256 |
|---|---|---|---|
| Serve | `models/error_isolated_motion/serve/error_isolated_motion.pt` | Expert-only EIMD prior | `bec47df5341429b0c6fd6c3bf470db83d80ce32645fc6931e00c4a050c7b8051` |
| Serve | `models/error_isolated_motion/serve/expert_score_model.npz` | 53-take, 7-subject RF-DETR expert-distribution and residual scorer | `1a0e3c7e5dc32ee019d071e35255d9337a25c38f5c7210d1ec62104058c9095c` |
| Smash | `models/error_isolated_motion/smash/error_isolated_motion.pt` | Expert-only EIMD prior | `956b567e407d88eff23ebc936f264e03d16543a550c08bf879268fbb85353977` |
| Smash | `models/error_isolated_motion/smash/expert_score_model.npz` | Expert-only qualitative scorer | `1cf4c958cbe360a1c739260e4c077cd286cdec900712c25bcde0a1e72a228f20` |
| Smash | `models/error_isolated_motion/smash/expert_trajectory_score_model.npz` | Phase-aligned Euclidean residual and expert-manifold gate | `1ed6ee9aa4218f05fdd60e8e0ccdd833ce7163ac2cb75666532ac83f653af027` |
| Smash | `models/error_isolated_motion/smash/checkpoint_scorer_v1/metric_graph.pt` | Frozen checkpoint graph heads | `c4cef078b3082584130e10d5bc189b88270b2f30ffeb5fc410ce305ce23b90f2` |
| Smash | `models/error_isolated_motion/smash/checkpoint_scorer_v1/expert_semantic_score_model.npz` | Active expert-only semantic distribution scorer | `e7006472527ff2e253535afd80ef93c749114972edf914cc51cae16d621ab212` |
| Smash | `models/error_isolated_motion/smash/checkpoint_scorer_v1/checkpoint_reference.npz` | Annotated expert checkpoint template | `685510316cc3aa498e0621e1650d04973e7ae9834dfa1cb5e8278080a1252280` |
| Smash | `models/error_isolated_motion/smash/checkpoint_scorer_v1/calibration.json` | Checkpoint scorer calibration | `078acd914ea59e81e1ba0942bfdef6c2d9482e208fbef169d7d1a5ca7fb413dc` |
| Both | `models/expert_reference_bank.npz` | Expert reference clips and requested-skill guard | `ed38bbb8873782a5cd5075522e66feca3abec3697a70117e7d3f5495741eb898` |

Common inference settings:

- method: conditional diffusion;
- normalized output: 64 frames, 17 COCO joints, 2D pose plus root trajectory;
- diffusion steps: 30;
- candidates per request: 8;
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

TensorRT is used only for the batched RF-DETR pose model; EIMD diffusion runs in
PyTorch on the same GPU. The engine is compiled once on an L4 by
`badminton_analysis_ai/build_rfdetr_engine.py`, published to Artifact Registry as
the generic artifact named in `badminton_analysis_ai/models/trt-engine.env`, and
downloaded, checksum-verified (`models/trt-engines.sha256`) and baked into the
image by the GPU deploy workflow. Cloud Storage holds learner and expert data
only; container images and build artifacts live in Artifact Registry.

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
| `preparation` | 5 |
| `body_rotation` | 20 |
| `arm_balance` | 5 |
| `elbow_forward` | 20 |
| `wrist_flick` | 30 |
| `follow_through` | 20 |

Smash is graded by the September checkpoint scorer: frozen graph heads for
preparation and arm balance, the semantic/trajectory heads (Euclidean residual
with the expert-manifold gate) for elbow and wrist evidence, and
contact-anchored ShapeDTW for checkpoint transfer and follow-through. The rules
per criterion, artifacts and validation limits are documented in
`badminton_analysis_ai/models/error_isolated_motion/README.md`. Learner
recordings are used only for held-out evaluation and never fit the scorer.

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
The protobuf response contains only object metadata:

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
- `RefreshPlaybackUrls`: re-signs stored objects; used by the latency benchmark
  and integration tests. Production playback URLs are signed by Go.

Public playback is owned by Go:

```text
GET /api/db/playback?user_id=<id>&skill=<serve|smash>&work_date=<timestamp>
```

The Go client streams video input in 1 MiB chunks and persists both returned media
records. When the reference bank has a matching expert, the response also pairs
the nearest real expert clip for LIFF comparison.

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
| `BADMINTON_TRT_CACHE_DIR` | unset | Prebuilt RF-DETR TensorRT engine directory; production `/app/models/trt-engines` |
| `COACHING_NO_SUGGESTION_MIN_SCORE` | `90` | Skip GPT coaching at or above this total… |
| `COACHING_NO_SUGGESTION_MIN_CRITERION_RATIO` | `0.8` | …when every criterion also reaches this share of its maximum |
| `OPENAI_COACHING_ATTEMPTS` | `2` | GPT attempts before deterministic rule-based advice |
| `ANALYSIS_POSE_DUMP_PREFIX` | unset | Optional GCS prefix for pose dumps used in offline debugging |

`auto` chooses CUDA when available, otherwise Apple MPS, otherwise CPU. Production
Cloud Run uses one NVIDIA L4 (4 vCPU, 16 GiB, concurrency 4, max 1 instance).
It scales to zero, except that Cloud Scheduler keeps one instance warm on
Mondays 13:45–18:10 Asia/Taipei. Local Apple Silicon inference can use MPS.

The Go backend's queue is configured with `ANALYSIS_TASKS_QUEUE`,
`ANALYSIS_ASYNC_ACCEPT`, `ANALYSIS_WORKER_URL` and
`ANALYSIS_TASK_SERVICE_ACCOUNT` (see `scripts/configure_async_analysis.sh`).
Uploads are only ever analyzed through the queue; `ANALYSIS_ASYNC_ACCEPT=false`
pauses new uploads while queued jobs drain.

## Observability and API protection

- Both services write one JSON object per log line (severity, message, source
  location, request ID and Cloud Trace fields), which Cloud Run forwards to
  Cloud Logging as structured logs. The Go backend forwards `x-request-id` (the
  analysis job ID for queued work) and `x-cloud-trace-context` to the GPU
  service over gRPC metadata, so both services' logs and request spans join one
  trace.
- `scripts/configure_observability.sh` creates log-based metrics for queued job
  outcomes and duration, GPU latency stages, analysis failures, and learner API
  rejections.
- Learner API credentials: an ES256 LIFF ID token is verified locally against
  LINE's published keys; once it expires the LIFF app sends its access token
  (`X-Line-Access-Token`), which LINE verifies. Rejected credentials are cached
  briefly so replays do not reach LINE.
- Per-instance rate limits cover each client IP, failed authentications per IP,
  each learner, and GPT summaries per learner; refusals return 429 with
  `Retry-After`. A global limit would belong at the edge (Cloud Armor).
- OpenAI calls have explicit timeouts: 2 minutes per attempt with one retry in
  Go; `OPENAI_TIMEOUT_SECONDS` (120) and `OPENAI_MAX_RETRIES` (1) in Python.

## Running locally

Install the pinned dependencies, configure GCP ADC and environment variables,
then start the analysis service:

```bash
cd badminton_analysis_ai
export PYTHONPATH="$PWD:$PWD/generated"
python -m api.server
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

1. all EIMD, scorer, `checkpoint_scorer_v1` and reference-bank artifacts exist
   and match the hashes above;
2. `Health` lists only serve and smash;
3. a streamed serve and smash request each return non-empty `feedback_video` and
   `skeleton_overlay_video` signed URLs;
4. both URLs decode as H.264/yuv420p;
5. feedback duration is equal to or longer than the clean overlay because of
   coaching pauses;
6. the Go playback endpoint signs and returns both media records.

## Main inference trace

1. `proto/badminton/analysis/v1/analysis.proto`
2. `badminton_analysis_ai/api/server.py`
3. `badminton_analysis_ai/api/pipeline.py`
4. `badminton_analysis_ai/badminton_analysis/services/pose_detector.py`
5. `badminton_analysis_ai/badminton_analysis/services/video_analyzer.py`
6. `badminton_analysis_ai/badminton_analysis/ml/expert_motion_preprocessing.py`
7. `badminton_analysis_ai/badminton_analysis/ml/expert_motion_backend.py`
8. `badminton_analysis_ai/badminton_analysis/ml/expert_motion_generator.py`
9. `badminton_analysis_ai/badminton_analysis/ml/kinematic_retargeting.py`
10. `badminton_analysis_ai/api/coaching.py`
11. `badminton_analysis_ai/api/renderer.py`
12. `linebot/api/analysis/client.go`
13. `linebot/main.go`
14. `liff/src/components/VideoComparison.tsx`
