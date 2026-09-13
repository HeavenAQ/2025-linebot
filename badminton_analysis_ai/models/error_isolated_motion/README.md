# Frozen motion models and current scoring

This tree contains inference-only code and frozen artifacts for serve and smash.
Serve's generator, scoring, prompt, and display contract are unchanged by this port.
The September smash scorer is selected explicitly by `service/pipeline.py`.

## Architecture and contracts

- EIMD: expert-only conditional diffusion, 64 COCO-17 poses, eight candidates,
  seed 19. No retraining is performed by this release.
- Serve retains its existing EIMD-v3 generation and dual-window grading.
- Smash uses the v6 wrist-velocity ending window and existing delayed-contact
  refinement (v7) with the EIMD-v3 weights. Grading and generation use that same
  window. The independently calibrated requested-skill guard deliberately keeps
  its original EIMD-v3 hypotheses and unchanged reference bank.
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
| Follow-through | 20 | Interval ShapeDTW and best post-contact endpoint; initial/final shoulder-span change can reduce even the original score to zero |

No endpoint pooling or shoulder-retreat penalty is applied. Experts and learners
follow identical inference rules. Cohort names, human ratings, and filename
exceptions never enter the scorer. Full-fit expert calibration is not evidence
of unseen-person generalization.

## Code and reproducibility

All paths below are relative to `badminton_analysis_ai/`.

- Preprocessing: `service/smash_source.py`,
  `badminton_analysis/ml/expert_motion_preprocessing.py`.
- Generation/dispatch: `badminton_analysis/ml/expert_motion_backend.py`.
- Complete smash orchestration: `badminton_analysis/ml/smash_current_runtime.py`.
- Numeric rules: `smash_current_scoring.py`, `smash_current_geometry.py`,
  `smash_current_endpoint.py` in that same module directory.
- Frozen graph, checkpoint alignment and placement: `smash_current_graph.py`,
  `smash_current_alignment.py`, `smash_current_placement.py`.
- Rendering/GPT: `service/renderer.py`, `service/coaching.py`,
  `badminton_analysis/ml/smash_coaching_evidence.py`.
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

This is **not** fresh RF-DETR extraction or live L4/CUDA parity. MPS and CUDA draw
different seeded diffusion noise. Before production promotion, verify real-video
expert **and learner** scores and overlays on the candidate; the two expert CD
fixtures alone cannot establish beginner parity. Do not relax the expert gate
or refit thresholds merely to conceal a failed parity check.

The 92-clip historical cohort retains cached generation provenance limitations.
Replaying its downstream scores does not establish fresh-generation equivalence
or authorize reporting its historical ICC as a new live-production measurement.

## SHA-256 manifest

The new active smash semantic model is inside `checkpoint_scorer_v1/`.
The root semantic model is retained for the explicit legacy API path only.

| Skill | File | SHA-256 |
|---|---|---|
| Serve | `error_isolated_motion.pt` | `bec47df5341429b0c6fd6c3bf470db83d80ce32645fc6931e00c4a050c7b8051` |
| Serve | `expert_score_model.npz` | `1a0e3c7e5dc32ee019d071e35255d9337a25c38f5c7210d1ec62104058c9095c` |
| Smash | `error_isolated_motion.pt` | `956b567e407d88eff23ebc936f264e03d16543a550c08bf879268fbb85353977` |
| Smash | `expert_score_model.npz` | `1cf4c958cbe360a1c739260e4c077cd286cdec900712c25bcde0a1e72a228f20` |
| Smash legacy | `expert_semantic_score_model.npz` | `be6b704bd580ff362bb6718eefa4596b51c3edc8368b08d40626140ebc1b16a5` |
| Smash | `expert_trajectory_score_model.npz` | `1ed6ee9aa4218f05fdd60e8e0ccdd833ce7163ac2cb75666532ac83f653af027` |
| Smash | `checkpoint_scorer_v1/metric_graph.pt` | `c4cef078b3082584130e10d5bc189b88270b2f30ffeb5fc410ce305ce23b90f2` |
| Smash | `checkpoint_scorer_v1/expert_semantic_score_model.npz` | `e7006472527ff2e253535afd80ef93c749114972edf914cc51cae16d621ab212` |
| Smash | `checkpoint_scorer_v1/checkpoint_reference.npz` | `685510316cc3aa498e0621e1650d04973e7ae9834dfa1cb5e8278080a1252280` |
| Smash | `checkpoint_scorer_v1/calibration.json` | `078acd914ea59e81e1ba0942bfdef6c2d9482e208fbef169d7d1a5ca7fb413dc` |
| Both | `../expert_reference_bank.npz` | `ed38bbb8873782a5cd5075522e66feca3abec3697a70117e7d3f5495741eb898` |
