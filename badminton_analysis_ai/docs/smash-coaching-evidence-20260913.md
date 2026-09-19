# Current smash scoring and coaching evidence

The numeric scorer, frozen artifacts, overlay placement and matching GPT prompt
are ported together. Serve is unchanged. See
[the model contract](../models/error_isolated_motion/README.md) for architecture,
artifact hashes, reproducibility commands and the production-promotion boundary.

## One scoring clock

`smash_current_runtime.CurrentSmashScorer` produces scores with maxima
5/20/5/20/30/20 and six source-indexed checkpoint intervals. Its same decisions
feed `smash_coaching_evidence.build_checkpoint_evidence`, the renderer and
frontend checkpoint timestamps. The full-source 30fps output avoids cropping
away early balancing evidence. Before generated coverage, only the detected
skeleton is visible; after it, continuation is explicitly display-only.

GPT receives actual scoring measurements and named criterion scores, not human
rater grades. It sees the exact five consecutive frames for a sustained
low-dominant-hand balance event, rather than only a later corrected pose.
Follow-through evidence includes the initial frame and the selected best
endpoint used for shoulder-span comparison. No pooling or retreat penalty
is described or scored.

## Response validation

Private evidence IDs identify additional input frames. A reported problem must
belong to its named criterion's evidence; otherwise the existing validator asks
the model to answer again. Missing evidence is explicit, not silently clamped.
This validates frame ownership and score consistency, not the truth of every
visual interpretation.

Public `frame_index` remains normalized to 0..63. Exact `video_frame_index`
and timestamp are retained. Rendering and gRPC cue construction both use
`api.coaching_timeline.coaching_video_frame`, so pauses follow video time,
not evidence-ID order. Frontend checkpoints separately use the scorer's exact
source anchor, including contact for wrist flick and the selected endpoint.

## Verification

Run `PYTHONPATH=.:generated python -m pytest -q` in `badminton_analysis_ai`.
The service suite has 239 passing tests, including scored-pixel passthrough,
source checkpoint timing, five-frame evidence extraction, and response validation.

The 100-result numeric replay and eight-clip fresh-MPS/cached-pose replay pass.
The lab CUDA run differs by at most 0.003772 criterion points and 0.0094 overlay
pixels; it fails the deliberately strict 1e-4 replay threshold and is not
bit-identical. That run does not test Cloud Run's L4 pose extraction.

Do not claim deployment or live learner/expert parity from these offline checks.
