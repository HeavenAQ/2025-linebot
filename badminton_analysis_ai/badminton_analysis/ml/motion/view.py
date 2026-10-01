"""Put a generated correction in the learner's camera view."""

from __future__ import annotations

from dataclasses import replace

from numpy.typing import NDArray
import numpy as np

from badminton_analysis.ml.motion.samples import ExpertCorrection, MotionSample
from badminton_analysis.ml.skeleton_normalization import phase_align_sequence


_EPS = 1e-8


def _ankle_spine_frame(
    pose: NDArray[np.floating], *, start: int, end: int
) -> NDArray[np.float64]:
    values = np.asarray(pose, dtype=np.float64)
    if values.ndim != 3 or values.shape[1:] != (17, 2):
        raise ValueError("ankle-spine alignment requires shape (T, 17, 2)")
    if not 0 <= start < end <= len(values):
        raise ValueError("invalid ankle-spine preparation window")
    hip_center = 0.5 * (values[:, 11] + values[:, 12])
    shoulder_center = 0.5 * (values[:, 5] + values[:, 6])
    ankle_axis = np.median(values[start:end, 16] - values[start:end, 15], axis=0)
    spine_axis = np.median(shoulder_center[start:end] - hip_center[start:end], axis=0)
    frame = np.column_stack((ankle_axis, spine_axis))
    if not np.all(np.isfinite(frame)) or abs(float(np.linalg.det(frame))) <= 1e-6:
        return np.eye(2, dtype=np.float64)
    return frame


def ankle_spine_view_rotation(
    student_pose: NDArray[np.floating],
    corrected_pose: NDArray[np.floating],
    *,
    start: int,
    end: int,
) -> NDArray[np.float32]:
    """Estimate a rigid 2D camera-frame rotation from ankle and spine axes."""
    student_frame = _ankle_spine_frame(student_pose, start=start, end=end)
    corrected_frame = _ankle_spine_frame(corrected_pose, start=start, end=end)
    left, _, right_t = np.linalg.svd(student_frame @ corrected_frame.T)
    rotation = left @ right_t
    if float(np.linalg.det(rotation)) < 0.0:
        left[:, -1] *= -1.0
        rotation = left @ right_t
    return np.asarray(rotation, dtype=np.float32)


def project_pose_to_student_view(
    student_pose: NDArray[np.floating],
    corrected_pose: NDArray[np.floating],
    rotation: NDArray[np.floating],
) -> NDArray[np.float32]:
    """Rotate a complete corrected pose around its pelvis into student view."""
    student = np.asarray(student_pose, dtype=np.float64)
    corrected = np.asarray(corrected_pose, dtype=np.float64)
    transform = np.asarray(rotation, dtype=np.float64)
    if student.shape != corrected.shape or student.ndim != 3:
        raise ValueError("student and corrected poses must share shape (T, J, 2)")
    if transform.shape != (2, 2):
        raise ValueError("view rotation must have shape (2, 2)")
    student_pelvis = 0.5 * (student[:, 11] + student[:, 12])
    corrected_pelvis = 0.5 * (corrected[:, 11] + corrected[:, 12])
    centered = corrected - corrected_pelvis[:, None]
    return np.asarray(
        student_pelvis[:, None] + centered @ transform.T, dtype=np.float32
    )


def shift_expert_body_chain_to_student_hip(
    student_pose: NDArray[np.floating],
    corrected_pose: NDArray[np.floating],
    *,
    start: int,
    end: int,
) -> NDArray[np.float32]:
    """Align the hip centre after knee-chain placement."""
    student = np.asarray(student_pose, dtype=np.float64)
    corrected = np.asarray(corrected_pose, dtype=np.float64)
    if student.shape != corrected.shape or student.ndim != 3:
        raise ValueError("student and corrected poses must share shape (T, J, 2)")
    if not 0 <= start < end <= len(student):
        raise ValueError("invalid hip-shift preparation window")
    shifted = corrected.copy()
    student_pelvis = 0.5 * (student[:, 11] + student[:, 12])
    corrected_pelvis = 0.5 * (corrected[:, 11] + corrected[:, 12])
    # One robust placement translation avoids copying the student's dynamic pelvis trajectory into the expert motion.
    hip_translation = np.median(
        student_pelvis[start:end] - corrected_pelvis[start:end], axis=0
    )
    shifted[:, :13] += hip_translation
    return np.asarray(shifted, dtype=np.float32)


def shift_expert_body_chain_to_student_knee(
    student_pose: NDArray[np.floating],
    corrected_pose: NDArray[np.floating],
    *,
    start: int,
    end: int,
) -> NDArray[np.float32]:
    """Align knee centres while carrying hips and the upper body with them."""
    student = np.asarray(student_pose, dtype=np.float64)
    corrected = np.asarray(corrected_pose, dtype=np.float64)
    if student.shape != corrected.shape or student.ndim != 3:
        raise ValueError("student and corrected poses must share shape (T, J, 2)")
    if not 0 <= start < end <= len(student):
        raise ValueError("invalid knee-shift preparation window")
    shifted = corrected.copy()
    student_knees = 0.5 * (student[:, 13] + student[:, 14])
    corrected_knees = 0.5 * (corrected[:, 13] + corrected[:, 14])
    knee_translation = np.median(
        student_knees[start:end] - corrected_knees[start:end], axis=0
    )
    shifted[:, :15] += knee_translation
    return np.asarray(shifted, dtype=np.float32)


def apply_fixed_hierarchical_pose_placement(
    student_pose: NDArray[np.floating],
    corrected_pose: NDArray[np.floating],
    *,
    start: int,
    end: int,
) -> NDArray[np.float32]:
    """Normalize fixed placement without copying the student's trajectory."""
    student = np.asarray(student_pose, dtype=np.float64)
    corrected = np.asarray(corrected_pose, dtype=np.float64)
    if student.shape != corrected.shape or student.ndim != 3:
        raise ValueError("student and corrected poses must share shape (T, J, 2)")
    if not 0 <= start < end <= len(student):
        raise ValueError("invalid hierarchical-placement preparation window")
    ankle_y = [float(np.median(student[start:end, joint, 1])) for joint in (15, 16)]
    support_ankle = (15, 16)[int(np.argmax(ankle_y))]
    ankle_delta = np.median(
        student[start:end, support_ankle] - corrected[start:end, support_ankle],
        axis=0,
    )
    placed = np.asarray(corrected + ankle_delta, dtype=np.float32)
    placed = shift_expert_body_chain_to_student_knee(
        student, placed, start=start, end=end
    )
    return shift_expert_body_chain_to_student_hip(student, placed, start=start, end=end)


def align_expert_correction_to_ankle_spine_view(
    correction: ExpertCorrection,
    *,
    start: int,
    end: int,
) -> tuple[ExpertCorrection, NDArray[np.float32]]:
    """Rotate, then apply one clip-level hierarchical placement."""
    rotation = ankle_spine_view_rotation(
        correction.aligned_student_pose,
        correction.aligned_corrected_pose,
        start=start,
        end=end,
    )
    aligned_corrected = project_pose_to_student_view(
        correction.aligned_student_pose,
        correction.aligned_corrected_pose,
        rotation,
    )
    corrected = project_pose_to_student_view(
        correction.student.pose,
        correction.corrected_pose,
        rotation,
    )
    aligned_corrected = apply_fixed_hierarchical_pose_placement(
        correction.aligned_student_pose,
        aligned_corrected,
        start=start,
        end=end,
    )
    corrected = apply_fixed_hierarchical_pose_placement(
        correction.student.pose,
        corrected,
        start=start,
        end=end,
    )
    aligned_root_delta = (
        correction.aligned_corrected_root - correction.aligned_corrected_root[:1]
    ) @ rotation.T
    root_delta = (
        correction.corrected_root - correction.corrected_root[:1]
    ) @ rotation.T
    return (
        replace(
            correction,
            aligned_corrected_pose=aligned_corrected,
            corrected_pose=corrected,
            aligned_corrected_root=np.asarray(
                correction.aligned_student_root[:1] + aligned_root_delta,
                dtype=np.float32,
            ),
            corrected_root=np.asarray(
                correction.student.root[:1] + root_delta, dtype=np.float32
            ),
        ),
        rotation,
    )


def _aligned(sample: MotionSample) -> tuple[NDArray[np.float32], ...]:
    return (
        phase_align_sequence(sample.pose, sample.phase_indices),
        np.clip(
            phase_align_sequence(sample.confidence, sample.phase_indices),
            0.0,
            1.0,
        ),
        phase_align_sequence(sample.root, sample.phase_indices),
    )


def _retarget_root_with_contacts(
    pose: NDArray[np.floating],
    root: NDArray[np.floating],
    contacts: NDArray[np.floating],
    reference_pose: NDArray[np.floating],
) -> NDArray[np.float32]:
    """Preserve the expert's world-space support-foot path after retargeting."""
    local = np.asarray(pose, dtype=np.float64)
    reference = np.asarray(reference_pose, dtype=np.float64)
    prior_root = np.asarray(root, dtype=np.float64)
    weights = np.clip(np.asarray(contacts, dtype=np.float64), 0.0, 1.0)
    if reference.shape != local.shape:
        raise ValueError("reference_pose must match pose")
    if prior_root.shape != (len(local), local.shape[-1]):
        raise ValueError("root must have shape (T, D)")
    if weights.shape != (len(local), 2):
        raise ValueError("contacts must have shape (T, 2)")
    output = prior_root.copy()
    ankle_indices = (15, 16)
    for frame_index in range(len(local)):
        candidates = []
        active_weights = []
        for foot_index, ankle_index in enumerate(ankle_indices):
            weight = weights[frame_index, foot_index]
            if weight <= _EPS:
                continue
            candidates.append(
                reference[frame_index, ankle_index]
                + prior_root[frame_index]
                - local[frame_index, ankle_index]
            )
            active_weights.append(weight)
        if candidates:
            constrained = np.average(
                np.asarray(candidates), axis=0, weights=np.asarray(active_weights)
            )
            influence = min(float(np.sum(active_weights)), 1.0)
            output[frame_index] = (1.0 - influence) * prior_root[
                frame_index
            ] + influence * constrained
    return output.astype(np.float32)


def transport_corrected_by_student_displacement(
    corrected_pixels: NDArray[np.float32],
    detected_pixels: NDArray[np.float32],
    confidence: NDArray[np.floating],
) -> NDArray[np.float32]:
    """Add only the student's global displacement to an anchored correction."""
    corrected = np.asarray(corrected_pixels, dtype=np.float32)
    detected = np.asarray(detected_pixels, dtype=np.float32)
    observed = np.asarray(confidence, dtype=np.float32)
    if (
        corrected.shape != detected.shape
        or corrected.ndim != 3
        or corrected.shape[1:] != (17, 2)
    ):
        raise ValueError("student displacement poses must share shape (T, 17, 2)")
    if observed.shape != corrected.shape[:2]:
        raise ValueError("student displacement confidence must have shape (T, 17)")

    pelvis = 0.5 * (detected[:, 11] + detected[:, 12])
    torso = 0.25 * (detected[:, 5] + detected[:, 6] + detected[:, 11] + detected[:, 12])
    ankles = 0.5 * (detected[:, 15] + detected[:, 16])
    pelvis_ok = np.minimum(observed[:, 11], observed[:, 12]) > 0.05
    torso_ok = np.minimum.reduce(observed[:, (5, 6, 11, 12)], axis=1) > 0.05
    ankles_ok = np.minimum(observed[:, 15], observed[:, 16]) > 0.05
    position = np.full((len(corrected), 2), np.nan, dtype=np.float64)
    position[pelvis_ok] = pelvis[pelvis_ok]
    fallback = ~pelvis_ok & torso_ok
    position[fallback] = torso[fallback]
    fallback = ~pelvis_ok & ~torso_ok & ankles_ok
    position[fallback] = ankles[fallback]
    valid = np.isfinite(position).all(axis=1)
    if not np.any(valid):
        return corrected.copy()
    timeline = np.arange(len(position))
    for axis in range(2):
        position[:, axis] = np.interp(timeline, timeline[valid], position[valid, axis])
    # Reject isolated detector jitter without suppressing real player travel.
    smoothed = position.copy()
    padded = np.pad(position, ((2, 2), (0, 0)), mode="edge")
    for frame in range(len(position)):
        smoothed[frame] = np.median(padded[frame : frame + 5], axis=0)
    displacement = smoothed - smoothed[0]
    displacement[0] = 0.0
    return np.asarray(corrected + displacement[:, None], dtype=np.float32)


def smooth_corrected_bbox_placement(
    corrected_pixels: NDArray[np.float32],
    *,
    alpha_current: float = 0.65,
) -> NDArray[np.float32]:
    """Stabilize correction placement with one rigid per-frame translation."""
    corrected = np.asarray(corrected_pixels, dtype=np.float32)
    if corrected.ndim != 3 or corrected.shape[1:] != (17, 2):
        raise ValueError("bbox placement smoothing requires shape (T, 17, 2)")
    if not 0.0 < alpha_current <= 1.0:
        raise ValueError("alpha_current must be in (0, 1]")
    if len(corrected) <= 2 or alpha_current >= 1.0:
        return corrected.copy()

    core = corrected[:, 5:17].astype(np.float64)
    anchor = 0.5 * (np.min(core, axis=1) + np.max(core, axis=1))
    forward = anchor.copy()
    for frame in range(1, len(anchor)):
        forward[frame] = (
            alpha_current * anchor[frame] + (1.0 - alpha_current) * forward[frame - 1]
        )
    backward = anchor.copy()
    for frame in range(len(anchor) - 2, -1, -1):
        backward[frame] = (
            alpha_current * anchor[frame] + (1.0 - alpha_current) * backward[frame + 1]
        )
    stable_anchor = 0.5 * (forward + backward)
    stable_anchor[0] = anchor[0]
    stable_anchor[-1] = anchor[-1]
    translation = stable_anchor - anchor
    return np.asarray(corrected + translation[:, None], dtype=np.float32)


MIN_CONFIDENCE = 0.05


def first_frame_ankle_spine_map(corrected_first, detected_first, confidence_first):
    """One proper similarity transform, fitted on the first frame only."""
    corrected = np.asarray(corrected_first, dtype=float)
    detected = np.asarray(detected_first, dtype=float)
    confidence = np.asarray(confidence_first, dtype=float)
    if (
        corrected.shape != (17, 2)
        or detected.shape != (17, 2)
        or confidence.shape != (17,)
    ):
        raise ValueError("Expected two COCO17 poses and 17 confidences")
    torso = [5, 6, 11, 12]
    if (
        not np.isfinite(corrected[torso]).all()
        or not np.isfinite(detected[torso]).all()
        or not np.isfinite(confidence[torso]).all()
        or np.any(confidence[torso] <= MIN_CONFIDENCE)
    ):
        raise ValueError(
            "Reliable first-frame torso required; do not silently refit later"
        )
    spine = lambda p: p[[5, 6]].mean(0) - p[[11, 12]].mean(0)
    source, target = spine(corrected), spine(detected)
    lengths = np.linalg.norm(source), np.linalg.norm(target)
    if min(lengths) <= 1e-6:
        raise ValueError("Nondegenerate first-frame spines required")
    a, b = source / lengths[0], target / lengths[1]
    cosine = np.dot(a, b)
    sine = a[0] * b[1] - a[1] * b[0]
    matrix = np.array([[cosine, sine], [-sine, cosine]]) * (lengths[1] / lengths[0])
    ankles = [
        j
        for j in (15, 16)
        if np.isfinite(corrected[j]).all()
        and np.isfinite(detected[j]).all()
        and np.isfinite(confidence[j])
        and confidence[j] > MIN_CONFIDENCE
    ]
    if not ankles:
        raise ValueError("Reliable first-frame ankle required")
    support = max(ankles, key=lambda j: detected[j, 1])
    translation = detected[support] - corrected[support] @ matrix
    return matrix, translation, support


def first_frame_standing_offsets(corrected_first, detected_first, confidence_first):
    """Retarget initial standing placement once, after camera projection."""
    q = np.asarray(corrected_first, float)
    p = np.asarray(detected_first, float)
    c = np.asarray(confidence_first, float)
    if q.shape != (17, 2) or p.shape != q.shape or c.shape != (17,):
        raise ValueError("Expected two COCO17 poses and 17 confidences")
    joints = [11, 12, 13, 14, 15, 16]
    if (
        not np.isfinite(q[joints]).all()
        or not np.isfinite(p[joints]).all()
        or not np.isfinite(c[joints]).all()
        or np.any(c[joints] <= MIN_CONFIDENCE)
    ):
        raise ValueError("Reliable first-frame pelvis, knees and ankles required")
    offsets = np.zeros((17, 2), dtype=float)
    offsets[:13] = p[[11, 12]].mean(0) - q[[11, 12]].mean(0)
    offsets[13:] = p[13:] - q[13:]
    return offsets
