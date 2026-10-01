# Frozen motion models and current scoring

This tree contains the frozen inference artifacts for serve and smash.
Serve's generator, scoring, prompt, and display contract are unchanged by this port.
The September smash scorer is selected explicitly by `api/pipeline.py`.

## Architecture and contracts

- EIMD: expert-only conditional diffusion, 64 COCO-17 poses, eight candidates,
  seed 19. Both priors were retrained on ViTPose++-L expert poses in September
  2026; the smash training archives are cut with the EIMD-v3 window, which now
  ends where the shoulders finish turning forward.
- Serve retains its EIMD-v3 generation and dual-window grading.
- Smash serving uses the v6 wrist-velocity ending window and delayed-contact
  refinement (v7). Grading and generation use that same window.
- The requested-skill guard still compares EIMD-v3 hypotheses against the
  RF-DETR-era expert features in `../expert_reference_bank.npz`; with the
  shoulder ending, some real smashes now read as serves until that bank is
  rebuilt from ViTPose experts.
- Smash source video is normalized to 30fps **before** pose extraction. Its
  checkpoint frame numbers, overlay frames, and coaching evidence share this clock.
- Frozen checkpoint graph heads grade preparation and arm balance. Their input
  is detected and generated skeletons, not RGB, human grades, or clip identity.
  Legacy semantic/trajectory heads provide elbow and wrist evidence. The
  trajectory residual remains **Euclidean**, with its original manifold gate.
- Checkpoint transfer uses contact-anchored ShapeDTW on detected poses against
  the annotated expert template. This locates semantic intervals; it is not a
  whole-clip score.
- Smash display uses kinematic phase interpolation, one first-frame ankle–spine
  similarity and standing/pelvis offsets, bbox-position EMA (0.65), then player
  displacement transport. The renderer receives the exact scored pixel poses
  and does not perform another fit, temporal warp, or EMA.
- Display includes the upload's lead-in and tail. Before generated coverage,
  only the detected skeleton is visible. After it, the final local generated
  pose is held and transported; the motion is not stretched.
  Those held display frames do not create additional generated scoring evidence.
- Left-handed poses use the existing anatomical label swap, not an inferred
  camera reflection. The frozen endpoint convention is image-right-forward.
  Opposite-view/left-handed score parity is not established by the right-handed
  validation set; never choose a view by maximizing the resulting score.

## Smash rubric

| Criterion | Maximum | Current rule |
|---|---:|---|
| Preparation | 5 | Frozen graph; cap if dominant wrist rises above expert waist-height support |
| Body rotation | 20 | Pre-balance lower-body, torso and shoulder-axis angle changes; explicit legacy directional/span fallback when unassessed |
| Arm balance | 5 | Frozen graph; cap sustained five-frame low dominant hand while support hand is raised |
| Elbow forward | 20 | Frozen historical semantic/trajectory head |
| Wrist flick | 30 | Frozen historical semantic/trajectory head at the refined contact checkpoint |
| Follow-through | 20 | Shoulder turn: interval ShapeDTW and best post-contact endpoint on spine direction and the shoulder line (where the wrist ends is not counted); initial/final shoulder-span change can reduce even the original score to zero |

No endpoint pooling or shoulder-retreat penalty is applied. Experts and learners
follow identical inference rules. Cohort names, human ratings, and filename
exceptions never enter the scorer. Full-fit expert calibration is not evidence
of unseen-person generalization.

## Code and reproducibility

All paths below are relative to `badminton_analysis_ai/`.

- Code layout under `badminton_analysis/ml/`:
  - `skill.py`: `SkillDefinition`, the pipeline every skill shares (find the
    stroke, normalize it onto 64 frames, generate, score, coach), and the
    registry. A skill subclass names three modules that answer its hooks.
  - `serve/` and `smash/`, the same seven files each: `skill.py` (the
    subclass), `phases.py` (`eimd_v3_phases`, `current_phases`, …),
    `scorer.py` (its `SkillScorer` and `create_scorer`), `checkpoints.py`
    (per-checkpoint measurements), `experts.py` (what experts are fitted to),
    `coaching.py` (what the coach is told).
  - `eimd/`: the diffusion model; `motion/`: archives and camera-view placement.
  - `backend.py` is skill-agnostic: generate, `scorer.score()`, report.
- Adding a skill: a `Skill` value and rubric in `skill_specs.py`, its EIMD model
  under `models/error_isolated_motion/<skill>/`, a `ml/<skill>/` package with the
  seven files above, its prompts under `prompts/<skill>/`, and one line in
  `ml/skill.py`'s registry. The window detector in
  `services/video_analyzer.py` still needs a case for it.
- GPT prompts live in `prompts/` as text files (see `prompts/README.md`).
- Packaging: `scripts/export_current_smash_artifacts.py` exports the frozen
  research artifacts to a fresh output directory; it does not fit or train.
- Verification: `scripts/verify_current_smash_port.py` replays all 100 reviewed
  results; `scripts/verify_current_smash_runtime.py` additionally recomputes
  graph, semantic heads, alignment, placement, EMA and caps for all eight test clips.
  Add `--regenerate --device mps` or `--regenerate --device cuda` to regenerate
  diffusion from cached detected poses. `--cache-directory` accepts a portable
  private validation directory containing results.json and the referenced NPZs.

Example from this directory:

```sh
PYTHONPATH=.:generated python -m pytest -q
PYTHONPATH=.:generated python scripts/verify_current_smash_port.py \
  --research-root /Users/heavenchen/dev/badminton-analysis
PYTHONPATH=.:generated python scripts/verify_current_smash_runtime.py \
  --research-root /Users/heavenchen/dev/badminton-analysis --device mps --regenerate
```

## Verification and promotion boundary

On 2026-09-13, 100 saved-result numeric replays agree to 3.56e-15 points.
For all eight new test clips, fresh MPS diffusion from saved detections reproduces
native skeletons, projected overlays, and all six scores exactly. CPU graph
inference on their saved generations also agrees exactly.

On the lab Quadro RTX 6000 with PyTorch 2.5.1+cu124, fresh CUDA diffusion from
the same eight detected-pose caches differs by at most 0.003772 points per
criterion and 0.0094 overlay pixels. It fails the deliberately strict 1e-4
replay threshold; it is not bit-identical and is not a live L4 measurement.

This is **not** fresh pose extraction or live L4/CUDA parity. MPS and CUDA draw
different seeded diffusion noise. Before production promotion, verify real-video
expert **and learner** scores and overlays on the candidate; the two expert CD
fixtures alone cannot establish beginner parity. Do not relax the expert gate
or refit thresholds merely to conceal a failed parity check.

The 92-clip historical cohort retains cached generation provenance limitations.
Replaying its downstream scores does not establish fresh-generation equivalence
or authorize reporting its historical ICC as a new live-production measurement.

## SHA-256 manifest

The active smash semantic model is inside `checkpoint_scorer_v1/`. The root
semantic model is loaded by the smash backend but not used for its grade, and
is scored only by `scripts/verify_eimd_v3_review_parity.py`.

| Skill | File | SHA-256 |
|---|---|---|
| Serve | `error_isolated_motion.pt` | `20e8631582e6b9afd5d0331c2c7eb1973485527a7a10da2da33f7d614bc417b7` |
| Serve | `expert_score_model.npz` | `7b32797b2e3ade2548f8b80dbc15f680636f1e851a99e42f31a4e3e18b3c9c41` |
| Smash | `error_isolated_motion.pt` | `86aa170aabfb36c595cf7f390500c966e2ebb4c4d21af8c9767a9d675ca7c6eb` |
| Smash | `expert_score_model.npz` | `1cf4c958cbe360a1c739260e4c077cd286cdec900712c25bcde0a1e72a228f20` |
| Smash legacy | `expert_semantic_score_model.npz` | `be6b704bd580ff362bb6718eefa4596b51c3edc8368b08d40626140ebc1b16a5` |
| Smash | `expert_trajectory_score_model.npz` | `1ed6ee9aa4218f05fdd60e8e0ccdd833ce7163ac2cb75666532ac83f653af027` |
| Smash | `checkpoint_scorer_v1/metric_graph.pt` | `c4cef078b3082584130e10d5bc189b88270b2f30ffeb5fc410ce305ce23b90f2` |
| Smash | `checkpoint_scorer_v1/expert_semantic_score_model.npz` | `806278496fb1d585d3de03e0a332d07bd73de19b3072c15f6223eaf740bf472b` |
| Smash | `checkpoint_scorer_v1/checkpoint_reference.npz` | `685510316cc3aa498e0621e1650d04973e7ae9834dfa1cb5e8278080a1252280` |
| Smash | `checkpoint_scorer_v1/calibration.json` | `abfaf848908bd28d7b382d722581b74d68936abce2cb457225c4e68f24d9bd8a` |
| Both | `../expert_reference_bank.npz` | `ed38bbb8873782a5cd5075522e66feca3abec3697a70117e7d3f5495741eb898` |
