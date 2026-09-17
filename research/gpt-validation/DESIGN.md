# GPT feedback validation (expert review)

Research tooling for validating the GPT coaching feedback the LLM product gives
beginners. It is **not** part of the learner product and is never deployed to
the no-LLM variant.

## Question and design

- **Reliability:** do two badminton experts agree? (Cohen's/Fleiss' κ per
  checkpoint, weighted κ and ICC for the overall score.)
- **Validity:** is GPT's feedback correct according to the experts?
  (Detection precision/recall/F1 against expert consensus; share of cues and
  responses judged correct, with 95% confidence intervals.)

The items are the **100 beginner videos the two experts already scored in the
rubric workbooks** (50 smash, 50 serve), listed in `processing/manifest.csv`
(written by `processing/build_manifest.py`). A mirrored `_left` copy is used only
when it is the student's sole recording. Each video is processed once by the
production pipeline, including GPT coaching, **on a local Mac** (RF-DETR and the
diffusion model on Apple MPS); no cloud GPU is used. Two experts rate every
eligible item independently and blind.

### Per-item form (about one minute)

1. **Step 1 — before seeing GPT.** The expert watches the RF-DETR pose-only
   overlay of the full original video (detected skeleton only; no generated
   expert skeleton, no score, no GPT text) and ticks which of the skill's six
   rubric checkpoints **need improvement**. Submitting locks step 1.
2. **Step 2 — GPT revealed.** The expert sees GPT's overall feedback, each GPT
   cue (checkpoint + advice), and the feedback video with the problem markers.
   They rate each cue `correct` / `partial` / `incorrect`, give one overall
   score, and may leave a comment.
   - `0` incorrect or potentially harmful advice
   - `1` partly correct
   - `2` correct but incomplete
   - `3` correct and useful

Experts never see: source filenames, student codes, the other expert's
ratings, or which items were excluded.

## Storage

Google Cloud Storage, bucket `nstc-2025-storage`, prefix
`gpt-validation/<batch_id>/` (temporary; delete after the study). Source videos
stay on the Mac; only renders are uploaded:

- `renders/<item_id>/detected_overlay.mp4` — step 1 video
- `renders/<item_id>/feedback.mp4` — step 2 video (GPT markers + pauses)
- `renders/<item_id>/skeleton_overlay.mp4` — detected + generated skeletons

Firestore, `(default)` database, project `nstc-linebot-2025`.

### `gpt_validation_items/{item_id}`

Written by `processing/process_videos.py`.

| Field | Type | Notes |
|---|---|---|
| `item_id` | string | `<skill>-<sha1(source_file)[:10]>`; stable across reruns |
| `batch_id` | string | e.g. `beginners-2026-09` |
| `skill` | string | `serve` or `smash` |
| `source_file` | string | original filename; **never sent to experts** |
| `workbook_id` | string | beginner ID in the rubric workbook (e.g. `EG01`); **never sent to experts** |
| `display_code` | string | neutral label shown to experts, e.g. `SV-017`, `SM-042` |
| `status` | string | `pending` · `processing` · `ready` · `failed` |
| `error` | string | failure reason when `failed` |
| `attempts` | int | processing attempts |
| `eligible` | bool | `status == ready` and `coaching_source == "openai"` |
| `coaching_source` | string | `openai` · `deterministic_fallback` · `score_gate` |
| `coaching_model` | string | e.g. `gpt-5.6-terra` |
| `handedness` | string | `left` · `right` |
| `total_grade` | number | system grade (not shown in step 1) |
| `criteria` | array | rubric order: `{id, name_zh, grade, maximum}` |
| `gpt_overall_feedback` | string | GPT overall feedback text |
| `gpt_cues` | array | `{index (1-based), criterion_id, title, feedback}` |
| `gpt_flagged_criteria` | array of string | criterion ids GPT raised a cue for |
| `media` | map | `detected_overlay`, `feedback_video`, `skeleton_overlay`: GCS object paths |
| `created_at`, `updated_at` | timestamp | |

### `gpt_validation_ratings/{item_id}__{expert_id}`

Written by the review service.

| Field | Type | Notes |
|---|---|---|
| `item_id`, `expert_id`, `skill`, `batch_id` | string | |
| `needs_improvement` | map criterion_id → bool | step 1; one key per rubric checkpoint |
| `step1_submitted_at` | timestamp | set once; step 1 is immutable afterwards |
| `step1_seconds` | number | client-measured time on step 1 |
| `cue_ratings` | map `"<cue index>"` → string | `correct` · `partial` · `incorrect` |
| `overall_score` | int | 0–3 |
| `comment` | string | optional |
| `step2_submitted_at` | timestamp | latest step 2 save |
| `step2_seconds` | number | client-measured time on step 2 |
| `completed` | bool | step 2 submitted |
| `updated_at` | timestamp | |

## Review service API

Cloud Run service `gpt-validation` (asia-east1, `min-instances 0`), code in
`research/gpt-validation/service`. Access codes come from Secret Manager secret
`gpt-validation-access-codes`, JSON `{"<expert_id>": "<code>", ...}`; the id
`admin` has the admin role. Every API call sends `X-Review-Code`.

| Method | Path | Role | Purpose |
|---|---|---|---|
| POST | `/api/login` | any | `{code}` → `{expert_id, role}` |
| GET | `/api/items` | expert | eligible items in this expert's fixed shuffled order: `{item_id, display_code, skill, step1_done, completed}` |
| GET | `/api/items/{item_id}` | expert | step 1 view: `skill`, `display_code`, `criteria [{id,name_zh}]`, signed `detected_overlay_url`; once step 1 is done also `gpt_overall_feedback`, `gpt_cues`, signed `feedback_video_url`, and the expert's own rating |
| PUT | `/api/items/{item_id}/step1` | expert | `{needs_improvement, seconds}`; 409 if already submitted |
| PUT | `/api/items/{item_id}/step2` | expert | `{cue_ratings, overall_score, comment, seconds}`; 409 before step 1 |
| GET | `/api/admin/progress` | admin | per-expert counts |
| GET | `/api/admin/export/items.csv` | admin | one row per item |
| GET | `/api/admin/export/criteria.csv` | admin | one row per item × expert × criterion |
| GET | `/api/admin/export/cues.csv` | admin | one row per item × expert × cue |
| GET | `/api/admin/export/overall.csv` | admin | one row per item × expert |

### CSV columns

- `items.csv`: `item_id, display_code, skill, source_file, workbook_id, status, eligible, coaching_source, coaching_model, handedness, total_grade, n_cues, gpt_flagged_criteria` (`;`-joined)
- `criteria.csv`: `item_id, skill, expert_id, criterion_id, criterion_name, expert_needs_improvement (0/1), gpt_flagged (0/1)`
- `cues.csv`: `item_id, skill, expert_id, cue_index, criterion_id, cue_rating (correct/partial/incorrect), cue_score (2/1/0)`
- `overall.csv`: `item_id, skill, expert_id, overall_score, comment, step1_seconds, step2_seconds, completed`

## Analysis

`research/gpt-validation/analysis/compute_validation_stats.py` reads the four
CSVs and reports reliability and validity (see its `--help`).
