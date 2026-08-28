# Historical Qualitative Criteria Verification (Superseded Pose Runtime)

> This report was produced against the former RTMW3D/ONNX Runtime path. The
> current service uses RF-DETR Keypoint Preview and the expert-only EIMD
> serve/smash pipeline. It is preserved as historical evidence, not as a
> description of the active production runtime.

Date: 2026-08-02 (Asia/Tokyo)

## Release

- Git commit: `aaf19b45c9dbaa9fcf2fb33fedf58e664f77dfd5`
- Cloud Run revision: `badminton-analysis-ai-00039-daz`
- Container digest: `sha256:9ad6213830d0b3218e8db195ea627d86f81ccf0c01444714325888adb421f632`
- Traffic: 100 percent
- GPU: NVIDIA L4
- Pose inference: RTMW3D-X through ONNX Runtime TensorRT
- Corrector inference: four independently trained skill correctors through ONNX Runtime TensorRT

The deployment workflow downloaded and checksum-verified the versioned TensorRT
cache, deployed a zero-traffic candidate, analyzed one right-handed beginner and
one held-out left-handed NSTC expert, checked both returned videos with signed
range requests, promoted the candidate, and removed all superseded revisions
and container images.

## Live Results

| Fixture | Skill | Hand | Grade | Matched expert | Same hand | Client | Service | LLM |
| --- | --- | --- | ---: | --- | ---: | ---: | ---: | ---: |
| `EG3.mp4` | clear | right | 19.610 | `李翊安6` | 1 | 28.939 s | 28.778 s | 10.914 s |
| `nstc_left_IMG_5568.mov` | serve | left | 100.000 | `nstc_left_IMG_5569` | 1 | 33.974 s | 33.845 s | 20.954 s |

Both responses reported `pose_tensorrt_active=1` and
`skeleton_tensorrt_active=1`. The left-handed request was restricted to the
left-handed reference bank and matched an `nstc_left_` expert.

## Qualitative Output Check

The low-scoring clear response used the exact first criterion
`球拍舉至腰部預備`, returned the Traditional Chinese instruction
`先把持拍手抬到腰部高度，再轉身。`, and circled COCO joints 6, 8, and 10 on
the physical right shoulder, elbow, and wrist. No degree threshold appeared in
the grading details or coaching cue.

The left-handed serve response used the canonical ordered descriptions:

1. `雙手平舉`
2. `將重心放至持拍腳`
3. `重心轉移至非持拍腳`
4. `髖關節前旋`
5. `持拍手手腕發力`
6. `肩膀旋轉朝前`

The response assigned 10, 10, 20, 20, 20, and 20 points respectively.

## Local Evidence

Generated media remains outside Git through
`badminton_analysis_ai/output/`:

- `badminton_analysis_ai/output/production-qualitative-verification/annotated-student.mp4`
- `badminton_analysis_ai/output/production-qualitative-verification/matched-expert.mp4`
- `badminton_analysis_ai/output/production-qualitative-verification/response.json`
- `badminton_analysis_ai/output/production-qualitative-verification/left-serve-expert/annotated-student.mp4`
- `badminton_analysis_ai/output/production-qualitative-verification/left-serve-expert/matched-expert.mp4`
- `badminton_analysis_ai/output/production-qualitative-verification/left-serve-expert/response.json`

The videos are H.264 portrait MP4 files. The beginner output is 1080 by 1920,
30 fps, and 4.13 seconds after the coaching pause was inserted. Its top overlay
shows `EG3 總分 19.6`; the lower overlay shows the criterion grade and coaching
instruction while the selected joints are circled.

## Automated Gates

- Production Python contract tests: 83 passed.
- Go tests: all packages passed.
- LIFF production build: passed.
- Source corrector and pipeline tests: 115 passed.
- Source mypy check: 27 runtime modules passed.
- Motion Analysis Service CD run: `30722349237`, passed.
- Source CI run: `30722560976`, passed.
