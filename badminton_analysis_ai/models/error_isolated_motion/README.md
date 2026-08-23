# Frozen Error-Isolated Motion Diffusion models

This directory is the only tracked checkpoint release in the repository. It
contains the final expert-only EIMD v3 models for `serve` and `smash`; clear,
lift, legacy expert-guided correctors, exploratory variants, and learner
archives are intentionally excluded.

Each skill directory contains:

- `error_isolated_motion.pt`: the diffusion network, expert motion prior,
  morphology statistics, phase-duration envelope, and velocity limits;
- `expert_score_model.npz`: the expert-calibrated inference-time score model;
- `held_out_expert_evaluator.npz`: the subject-disjoint evaluator used for
  research validation; and
- `training_report.json`: training provenance and hyperparameters.

Both checkpoints declare `student_data_used = false`. They were trained only
from expert archives using synthetic expert corruption. Student recordings
are permitted only during validation, testing, and inference.

The copied artifacts are byte-identical to the final outputs under
`.artifacts/error-isolated-motion-v3/full/{serve,smash}/`. Their SHA-256
digests are:

| Skill | File | SHA-256 |
|---|---|---|
| Serve | `error_isolated_motion.pt` | `bec47df5341429b0c6fd6c3bf470db83d80ce32645fc6931e00c4a050c7b8051` |
| Serve | `expert_score_model.npz` | `1c744aa0f92c41fb6d72e4ea722a80b88e8e6369893c22fb1e8c449905b263e8` |
| Serve | `held_out_expert_evaluator.npz` | `839c46d8701306163bed45670778cf3dd8236747db9e9a54e5265fb6f9d73132` |
| Serve | `training_report.json` | `37618316ff758fcb9f229077c6346547914ea7cc8fb072768899425bbe9dd0e8` |
| Smash | `error_isolated_motion.pt` | `956b567e407d88eff23ebc936f264e03d16543a550c08bf879268fbb85353977` |
| Smash | `expert_score_model.npz` | `1cf4c958cbe360a1c739260e4c077cd286cdec900712c25bcde0a1e72a228f20` |
| Smash | `held_out_expert_evaluator.npz` | `4ea65d86a572efc20eb24b7045d593f1aa8d11c283eba3d263d71aca074c8e75` |
| Smash | `training_report.json` | `77f80bbf0a632b6fed4bf8fc853b0b83bfb0f72866ff793b5ca74a50a7c4643f` |
