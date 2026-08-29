# Frozen Error-Isolated Motion Diffusion models

This directory is the complete runtime checkpoint release. Production supports
`serve` and `smash` only. Clear/lift checkpoints, legacy expert-guided
correctors, validation evaluators, training reports, and exploratory artifacts
are intentionally excluded.

Required files:

- `serve/error_isolated_motion.pt`: expert-only EIMD correction prior;
- `serve/expert_score_model.npz`: serve expert-distribution scorer;
- `smash/error_isolated_motion.pt`: expert-only EIMD correction prior;
- `smash/expert_score_model.npz`: smash qualitative baseline;
- `smash/expert_semantic_score_model.npz`: smash semantic checkpoint scorer;
- `smash/expert_trajectory_score_model.npz`: phase-aligned Euclidean residual
  scorer with an expert-manifold gate; and
- `../expert_reference_bank.npz`: the serve/smash comparison-video bank and
  frozen expert-only temporal skill-support guard.

All fitted artifacts use expert data only. Student, team, and novice recordings
are allowed only for validation, testing, and inference. The diffusion output is
normalized to 64 COCO-17 frames and projected into the student's camera frame
using preparation ankle–spine normalization followed by one clip-level ankle,
knee-chain, and hip-chain placement. It never follows the student pose frame by
frame.

The frozen serving contract is `phase_contract=eimd_v3`, `candidates=8`, and
`seed=19`. The same values must be used when generating correction caches for
fitting and when serving. Calibration for the L4 deployment must be run with
`--device cuda`; a scorer fitted from MPS samples is not interchangeable because
candidate ranking can change across PyTorch devices. The deployed service uses
`device=auto`, which resolves to CUDA on the L4.

Scorer rebuild code intentionally remains outside the pruned deployment tree.
Its canonical location is the `fix/rfdetr-expert-score-calibration` worktree at
`badminton_analysis_ai/scripts/`. Run the projection and trajectory builders
there with explicit `--device cuda --candidates 8 --seed 19`, then copy only the
runtime files listed above into this directory and verify the complete learner
and expert cohorts before changing this manifest. The checked-in audit scripts
are runtime-only: they consume frozen RF-DETR caches and cannot train EIMD.

## Requested-skill consistency

The request skill is treated as untrusted. After the single RF-DETR tracking
pass and handedness normalization, the runtime forms both the serve and smash
EIMD-v3 phase hypotheses. It compares translation/scale/in-plane-view invariant
local joint-angle and torso-relative direction trajectories with constrained
shape-DTW against frozen expert support. A decisive alternative-skill win is
returned as gRPC `INVALID_ARGUMENT` before diffusion generation, grading,
rendering, GPT coaching, upload, or reference matching; the service never
silently relabels a request.

The support set contains 53 serve and 50 smash experts. The rejection boundary
is the minimum leave-one-identity-out expert separation minus fixed numerical
headroom (`0.0791043085379329 - 1e-6 = 0.0791033085379329`). No learner clip,
filename, cohort ID, or rating fitted this boundary. The frozen audit accepts
all 103 experts under their correct labels, rejects all 103 opposite-label
submissions, and accepts all 152 correctly labelled learner validation caches
(one ambiguous clip has no valid alternative phase hypothesis and is accepted
by the conservative rejection-only policy).

## Promotion hold

Local MPS parity is complete, but the no-traffic L4 candidate still must prove
the same committed contract on CUDA before promotion. If CUDA candidate ranking
changes scores beyond the reviewed tolerances, refit the score artifacts from
CUDA expert-only caches and rerun both learner distributions and expert safety.

The frozen local review contains 203 records (50 serve learners, 53 serve
experts, 50 smash learners, 50 smash experts). A clean recomputation matches
all visible totals, per-checkpoint results, trajectory hashes, and both media
streams' 30-fps frame counts and durations. Its learner ICC(2,1) values are
`0.9036520764` for serve and `0.7034850511` for smash; pooled learner+expert
values are `0.9892576824` and `0.9022686235`. Expert-only ICC is undefined
because both expert raters are constant at maximum. Human scores are validation
only and fit no runtime parameter.

Do not use the older serve benchmark's hard-coded `0:16` rigid-fit slice. The
model rubric defines the 64-frame preparation interval as `0:24`, and that same
`preparation.bounds(...)` interval must drive the serve dual-window alignment.
For smash, a cache whose analysis end exceeds a subsequently trimmed source
video is stale; the current v17 cache is authoritative (this affects EG40 in
the preserved validation cohort).

The clean overlay emits exactly one frame per input frame at the input video's
exact rational frame rate. The GPT-feedback render may be longer only because it
inserts explicit coaching pauses. Smash additionally repairs isolated generated
pose discontinuities before a small fixed-bone EMA and aligns the correction's
contact event to the detected dominant-wrist acceleration event.

Active SHA-256 manifest:

| Skill | File | SHA-256 |
|---|---|---|
| Serve | `error_isolated_motion.pt` | `bec47df5341429b0c6fd6c3bf470db83d80ce32645fc6931e00c4a050c7b8051` |
| Serve | `expert_score_model.npz` | `1a0e3c7e5dc32ee019d071e35255d9337a25c38f5c7210d1ec62104058c9095c` |
| Smash | `error_isolated_motion.pt` | `956b567e407d88eff23ebc936f264e03d16543a550c08bf879268fbb85353977` |
| Smash | `expert_score_model.npz` | `1cf4c958cbe360a1c739260e4c077cd286cdec900712c25bcde0a1e72a228f20` |
| Smash | `expert_semantic_score_model.npz` | `be6b704bd580ff362bb6718eefa4596b51c3edc8368b08d40626140ebc1b16a5` |
| Smash | `expert_trajectory_score_model.npz` | `1ed6ee9aa4218f05fdd60e8e0ccdd833ce7163ac2cb75666532ac83f653af027` |
| Both | `../expert_reference_bank.npz` | `ed38bbb8873782a5cd5075522e66feca3abec3697a70117e7d3f5495741eb898` |
