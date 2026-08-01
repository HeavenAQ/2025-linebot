# Pose Framework Selection

## Decision

The production service uses RTMW3D-X through RTMLib, ONNX Runtime 1.24.4, and
TensorRT 10.14 FP16. VideoPose3D-243 was the best candidate at imitating the
pseudo-label frame/window decisions, but RTMW3D was materially better at the
downstream grading task and was much faster in this repository's direct 3D
pipeline. The selection therefore follows product behavior rather than a
generic Human3.6M or COCO leaderboard.

## Benchmark Design

The frozen corpus contains 81 clips from serve, lift, clear, and smash: 33
beginners and 48 experts. Clear includes the known left-handed students EG28
and EG29. Each candidate owns its 2D pose, handedness decision, action window,
and five phase checkpoints.

The benchmark has two stages:

1. Compare handedness, action-window IoU, checkpoint timing, normalized 2D
   pose, PCK, and joint angles against a high-capacity pseudo labeler.
2. Run every candidate through one fixed leave-one-out, bone-adapted expert
   scorer and compare total/criterion score error, rank correlation, and
   expert-versus-beginner separation on held-out clips.

Pseudo labels are not human ground truth. They provide a repeatable reference,
while cohort labels and qualitative video overlays provide independent checks.

## Pseudo Labeler

[Sapiens2](https://arxiv.org/abs/2604.21681) was the preferred frontier model.
Its [repository](https://github.com/facebookresearch/sapiens2),
[pose instructions](https://github.com/facebookresearch/sapiens2/blob/main/docs/POSE.md),
and [license](https://github.com/facebookresearch/sapiens2/blob/main/LICENSE.md)
are public, but the pose checkpoint requires accepted Hugging Face access. The
Nislab environment had no authorized token and received HTTP 401, so Sapiens2
was recorded as unavailable rather than assigned fabricated results.

The approved fallback was `YOLO26x-pose` at 1280 pixels. References are the
[YOLO26 paper](https://arxiv.org/abs/2606.03748),
[Ultralytics repository](https://github.com/ultralytics/ultralytics), and
[pose documentation](https://docs.ultralytics.com/tasks/pose/). The largest
detected person supplies the 17 COCO joints.

## Candidates

| Candidate | Paper | Repository | Status |
| --- | --- | --- | --- |
| RTMW3D-X | [RTMW](https://arxiv.org/abs/2407.08634) | [MMPose project](https://github.com/open-mmlab/mmpose/tree/main/projects/rtmpose3d) | Ranked |
| MotionBERT | [Paper](https://arxiv.org/abs/2210.06551) | [Official code](https://github.com/Walter0807/MotionBERT) | Ranked |
| VideoPose3D-243 | [Paper](https://arxiv.org/abs/1811.11742) | [Official code](https://github.com/facebookresearch/VideoPose3D) | Ranked |
| MediaPipe Heavy | [BlazePose](https://arxiv.org/abs/2006.10204) | [MediaPipe](https://github.com/google-ai-edge/mediapipe) | Ranked |
| PoseMamba-L | [Paper](https://arxiv.org/abs/2408.03540) | [Official code](https://github.com/nankingjing/PoseMamba) | Not reproducible |
| MotionAGFormer | [Paper](https://arxiv.org/abs/2310.16288) | [Official code](https://github.com/TaatiTeam/MotionAGFormer) | Screened |
| PersPose | [ICCV 2025](https://openaccess.thecvf.com/content/ICCV2025/html/Hao_PersPose_3D_Human_Pose_Estimation_with_Perspective_Encoding_and_Perspective_ICCV_2025_paper.html) | [Official code](https://github.com/KenAdamsJoseph/PersPose) | Not a drop-in path |

PoseMamba's official custom CUDA extension could not be built in the server
environment, which lacks the required compiler/toolchain combination. PersPose
requires SMPL assets and focal-length inputs and does not expose the required
real-time video service path. Neither receives synthetic performance numbers.

[AthletePose3D](https://openaccess.thecvf.com/content/CVPR2025W/CVSPORTS/html/Yeung_AthletePose3D_A_Benchmark_Dataset_for_3D_Human_Pose_Estimation_and_CVPRW_2025_paper.html)
is an interpretation constraint: standard monocular benchmarks can hide
failures on fast athletic motion, which is why local checkpoint and scoring
behavior are the actual selection criteria.

## Stage 1 Results

All candidates completed all 81 clips with zero extraction failures.

| Candidate | Checkpoint MAE (ms) | Recall at 100 ms | Window IoU | Handedness | PCK@0.2 | Angle MAE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| RTMW3D-X | 80.74 | 0.8420 | 0.8973 | **0.9877** | 0.8068 | 6.38 deg |
| MotionBERT | 64.69 | 0.8815 | 0.9196 | 0.9630 | 0.8660 | 5.63 deg |
| **VideoPose3D-243** | **58.68** | **0.8938** | **0.9245** | 0.9753 | **0.8729** | **5.23 deg** |
| MediaPipe Heavy | 113.00 | 0.7975 | 0.8716 | 0.9506 | 0.7034 | 8.10 deg |

## Stage 2 Results

The held-out evaluation contains 48 leave-one-out experts and 17 beginners.

| Candidate | Total MAE | Criterion MAE | Spearman rho | Cohort AUC | Expert mean | Beginner mean |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| YOLO26x oracle | 0.00 | 0.00 | 1.0000 | 0.9167 | 97.17 | 44.83 |
| **RTMW3D-X** | **9.77** | **2.11** | **0.7811** | **0.9498** | **94.99** | 30.13 |
| MotionBERT | 29.13 | 5.51 | 0.2384 | 0.7181 | 72.95 | 42.89 |
| VideoPose3D-243 | 24.10 | 4.54 | 0.3709 | 0.7770 | 77.66 | 38.97 |
| MediaPipe Heavy | 14.74 | 2.91 | 0.6395 | 0.8983 | 89.13 | 25.31 |

The Stage 2 values come from a transparent shadow scorer and are not the final
production calibration. RTMW3D has 60% lower total-score MAE and 53% lower
criterion-score MAE than the Stage 1 winner, with substantially higher rank
correlation and cohort AUC.

## Production Audit

After selection, all four correction models were retrained and evaluated on the
complete 50-expert/50-beginner skill datasets:

| Skill | Beginner mean | Expert mean | AUC | Corrected/expert distance | Acceleration ratio | Improved students |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Serve | 45.000 | 99.801 | 0.99 | 0.275 | 0.997 | 100% |
| Lift | 45.000 | 99.803 | 1.00 | 0.356 | 0.995 | 100% |
| Clear | 45.000 | 99.795 | 1.00 | 0.312 | 0.988 | 100% |
| Smash | 45.000 | 99.805 | 0.98 | 0.319 | 0.995 | 100% |

The distance is measured against the nearest training expert after adapting the
expert to the student's bone lengths. The acceleration ratio compares the
corrected skeleton with that same adapted expert target, so a low distance
cannot pass by collapsing motion into a static pose.

## Environment and Artifacts

The offline benchmark ran on Nislab host `p920`, Quadro RTX 6000 24 GiB, driver
595.71.05, Python 3.12.13, PyTorch 2.5.1+cu124, MMPose 1.3.2, Ultralytics
8.4.75, and OpenCV 4.10. CUDA device ordering differs from `nvidia-smi` on this
host; the recorded benchmark command used CUDA device 1, which is the Quadro.

Production engines were built separately on the RTX A6000 (SM86) with ONNX
Runtime 1.24.4 and TensorRT 10.14.1.48. The four FP16 correction engines are
hardware-compatible SM80+ plans and load directly on the production L4. RTMW's
ONNX Runtime graph partition hashes differ in the production image, so the CD
gate builds exact SM89 detector and pose plans on a zero-traffic L4 candidate.
The workflow retries the same candidate after the cold request drains, promotes
it only after real video analysis and signed playback pass, and keeps one warm
instance. Failed candidates never receive production traffic.

The portable ignored cache is stored at:

```text
gs://nstc-2025-storage/models/rtmw3d-ort1.24.4-trt10.14-sm80plus
```

`badminton_analysis_ai/models/tensorrt-cache.sha256` verifies every portable
detector, pose, corrector, profile, and timing-cache object before the Docker
build. The production image retains only TensorRT's SM89 builder resource so a
missing or changed RTMW partition cannot silently build for a different GPU
architecture. The source benchmark's generated CSV, NPZ, log, and MP4 artifacts
remain in the ignored local folders `stats/pose-framework-benchmark-20260801`
and `stats/production-correction-20260801`.
