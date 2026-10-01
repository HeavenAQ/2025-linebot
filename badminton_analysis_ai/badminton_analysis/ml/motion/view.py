"""Put a generated correction in the learner's camera view.

Projects an EIMD-generated correction into the learner's ankle--spine view
with one clip-level hierarchical placement, shared by serve and smash.
"""

from __future__ import annotations

from dataclasses import replace
import numpy as np
from numpy.typing import NDArray
from badminton_analysis.ml.skeleton_normalization import phase_align_sequence
from badminton_analysis.ml.motion.samples import (
    ExpertCorrection,
    MotionSample,
)


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
    """Estimate a rigid 2D camera-frame rotation from ankle and spine axes.

    The 2D cross product is the determinant used to reject a degenerate or
    reflected basis. Orthogonal Procrustes then returns a proper rotation;
    scale, stance width, and torso lean are deliberately not normalized away.
    """
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
    """Align the hip centre after knee-chain placement.

    The generated expert articulation is retained: joints 0..12 receive one
    shared preparation-window translation from the generated pelvis centre to
    the student's pelvis centre. Knees and ankles (13..16) remain fixed.

    Both arrays must already be expressed in the same absolute coordinate
    system. In rendering this means calling the transform after support-ankle
    grounding, because pelvis-centred model-local poses have no placement
    residual to correct.
    """
    student = np.asarray(student_pose, dtype=np.float64)
    corrected = np.asarray(corrected_pose, dtype=np.float64)
    if student.shape != corrected.shape or student.ndim != 3:
        raise ValueError("student and corrected poses must share shape (T, J, 2)")
    if not 0 <= start < end <= len(student):
        raise ValueError("invalid hip-shift preparation window")
    shifted = corrected.copy()
    student_pelvis = 0.5 * (student[:, 11] + student[:, 12])
    corrected_pelvis = 0.5 * (corrected[:, 11] + corrected[:, 12])
    # One robust placement translation avoids copying the student's dynamic
    # pelvis trajectory into the expert motion.
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
    """Preserve the expert's world-space support-foot path after retargeting.

    Retargeting expert limbs to the student's lengths moves the local ankle.
    During a labelled contact, compensate through the global root so that the
    resulting world ankle follows the same path as the selected expert. Soft
    contact confidence blends this constraint into the unmodified expert root;
    simultaneous contacts use their least-squares weighted root translation.
    """
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
