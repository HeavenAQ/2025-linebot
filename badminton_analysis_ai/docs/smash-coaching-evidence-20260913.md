# Smash scoring-aligned coaching candidate

This change is **not deployed**. It updates coaching, not the numeric scorer or
model artifacts. Do not promote it as complete local/production scoring parity.
The accepted numeric smash scorer and its calibration must be ported together
before releasing these rubric changes on main. Serve is not part of this change.

## Contract

- Six rubric maxima: preparation 5, rotation 20, arm balance 5, elbow 20,
  wrist 30, follow-through 20.
- Instructions describe the accepted preparation-height guard, angular rotation
  evidence, sustained early low-dominant-hand guard, and maximum endpoint plus
  initial/final shoulder-span rule. No endpoint pooling or retreat penalty.
- The numeric scorer supplies `correction_grade.checkpoint_evidence`, with all
  six criterion IDs. Each contains exact `output_frame_indices` in the cropped,
  pause-free analysis video. `smash_coaching_evidence.build_checkpoint_evidence`
  converts source-indexed scoring decisions to that clock once.
- The builder uses existing scoring intervals/decisions. It does not infer new
  checkpoint locations or read human ratings. Original intervals and missing
  cropped frames remain in the evidence metadata.
- Arm balance includes **all five consecutive frames of the scored event**,
  rather than a later, already-corrected pose. Missing intervals are never
  silently clamped onto an unrelated visible frame.
- Extra input evidence IDs are private to the smash JSON request. Validation
  rejects a problem attributed to a frame outside its criterion's evidence,
  triggering the existing re-answer mechanism. It does not snap it to an old
  phase anchor. This is structural validation, not a guarantee that every visual
  interpretation by the model is correct.
- After validation, public `frame_index` remains normalized to 0..63. The exact
  `video_frame_index` and timestamp are retained. Both the renderer and gRPC cue
  builder use `service.coaching_timeline.coaching_video_frame`; pause ordering
  follows actual video time, not evidence-ID order.
- Requests without scorer-owned evidence retain legacy sampling. Therefore
  changing the prompt alone is **not** the finished production port: the new
  numeric scoring path must attach its actual evidence before calling coaching.
- Serve rules, system prompt, response JSON schema, and default frame manifest
  were compared byte-for-byte with deployed source 03239c5c and match.

## Verification

Run from `badminton_analysis_ai`:

```sh
PYTHONPATH=.:generated python -m pytest \
  tests/test_coaching.py tests/test_clear_feedback.py tests/test_skill_specs.py \
  tests/test_smash_coaching_evidence.py tests/test_renderer.py \
  tests/test_server_grpc.py tests/test_pipeline.py -q
```

91 tests passed on 2026-09-13. The new synthetic video test proves the images
come from the requested frames and the resulting cue retains the exact timestamp.

EG27 regression: source window starts at 60; the sustained low-hand event is
103–107. Exact output-local frames are therefore 43–47 (1.433–1.567 s at 30 fps).
Terra medium, Gemini medium, and Gemini high/8192 all returned model-generated
feedback identifying the dominant-hand-low issue and selected output frame 43.
Scores remain frozen: arm balance 1.526823207/5; total 94.375320405/100.

Research comparison runner and publisher:

```
/Users/heavenchen/dev/badminton-analysis/scripts/compare_smash_coaching_providers.py
/Users/heavenchen/dev/badminton-analysis/scripts/publish_smash_feedback_comparison.py
```

Use `--code-root` pointing to this `badminton_analysis_ai` tree and a fresh
`--output-root`. The runner snapshots and hashes code; it rejects resuming with
different source. Human rater values are only in the review page, never the model
request. Old deployed-prompt trials remain separately preserved.

Local page: http://localhost:8765/smash-feedback-model-comparison-20260913/

## Remaining release work

Port the accepted numeric scorer and its exact evidence producer, validate
calibration/inference parity, and then deploy through CD. This patch alone must
not be presented as a production update. Some existing review overlays omit the
beginning of the preparation scoring interval (35/49 learner clips have at least
one partially cropped interval); the comparison marks this limitation explicitly.
Complete interval review requires source frames before the overlay crop, not
fabricated images or silently reindexed checkpoints.
