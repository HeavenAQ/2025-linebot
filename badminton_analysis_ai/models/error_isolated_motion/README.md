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
  scorer with an expert-manifold gate.

All fitted artifacts use expert data only. Student, team, and novice recordings
are allowed only for validation, testing, and inference. The diffusion output is
normalized to 64 COCO-17 frames and projected into the student's camera frame
using preparation ankle–spine normalization followed by one clip-level ankle,
knee-chain, and hip-chain placement. It never follows the student pose frame by
frame.

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
