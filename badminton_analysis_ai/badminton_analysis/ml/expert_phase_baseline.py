"""Expert-only phase-manifold baseline for personalized motion correction.

This module is the deterministic M0 baseline for the unified expert-motion
coach.  It deliberately trains on expert archives only.  At inference it uses
the learner's preparation stance, stable body proportions, handedness, and five
phase anchors to construct a personalized expert target.  The learner's faulty
dynamic motion is never used to select what the expert movement should be.

The baseline is intentionally non-generative: it is a retrieval-weighted local
expert manifold followed by exact bone-length retargeting.  It is useful as an
auditable lower bound for later PAN/transformer/diffusion models and as a way to
validate data identity, phase alignment, scoring, and visualization plumbing.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from functools import cached_property
from pathlib import Path
from typing import Any, Literal, Sequence

import numpy as np
from numpy.typing import NDArray

from badminton_analysis.ml.skeleton_normalization import (
    CANONICAL_PHASE_INDICES,
    phase_align_sequence,
    restore_phase_timing,
)
from badminton_analysis.ml.skeleton_scoring import (
    ANGLE_TRIPLETS,
    BONES,
    TORSO_WIDTH_BONES,
    project_stable_bone_lengths,
)
from badminton_analysis.ml.skill_specs import (
    SkillCorrectionSpec,
    get_skill_spec,
    motion_completion_bounds,
)
from badminton_analysis.ml.video_annotations import expert_subject_identity


_EPS = 1e-8
_FEATURE_JOINTS = np.asarray((0, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16))
_PREPARATION_END = int(CANONICAL_PHASE_INDICES[1]) + 1


@dataclass(frozen=True)
class MotionSample:
    path: Path
    pose: NDArray[np.float32]
    confidence: NDArray[np.float32]
    root: NDArray[np.float32]
    foot_contacts: NDArray[np.float32]
    phase_indices: NDArray[np.int64]
    handedness: str
    skill: str
    video_name: str
    subject_id: str
    phase_source: str
    alignment_contract: str
    identity_level: str


@dataclass(frozen=True)
class ExpertPhaseModel:
    skill: str
    expert_pose: NDArray[np.float32]
    expert_confidence: NDArray[np.float32]
    expert_root: NDArray[np.float32]
    expert_foot_contacts: NDArray[np.float32]
    expert_features: NDArray[np.float32]
    feature_mean: NDArray[np.float32]
    feature_scale: NDArray[np.float32]
    expert_handedness: NDArray[np.str_]
    expert_files: NDArray[np.str_]
    expert_subject_ids: NDArray[np.str_]
    expert_identity_levels: NDArray[np.str_]
    expert_alignment_contracts: NDArray[np.str_]
    criterion_ids: NDArray[np.str_]
    criterion_tolerances: NDArray[np.float32]
    criterion_scales: NDArray[np.float32]
    top_k: int
    criterion_metric_version: str = "generic_joint_distance_v1"
    criterion_residual_tolerances: NDArray[np.float32] | None = None
    criterion_residual_scales: NDArray[np.float32] | None = None

    @property
    def spec(self) -> SkillCorrectionSpec:
        return get_skill_spec(self.skill)

    @property
    def has_global_root_motion(self) -> bool:
        deltas = self.expert_root - self.expert_root[:, :1]
        return bool(float(np.max(np.abs(deltas))) > 1e-7)

    @cached_property
    def serve_angle_manifold(self):
        if self.skill != "serve":
            return None
        from badminton_analysis.ml.trajectory_distance import (
            fit_serve_angle_manifold,
        )

        return fit_serve_angle_manifold(
            self.expert_pose, self.expert_subject_ids
        )


@dataclass(frozen=True)
class ExpertCorrection:
    student: MotionSample
    aligned_student_pose: NDArray[np.float32]
    aligned_student_root: NDArray[np.float32]
    aligned_corrected_pose: NDArray[np.float32]
    aligned_corrected_root: NDArray[np.float32]
    corrected_pose: NDArray[np.float32]
    corrected_root: NDArray[np.float32]
    aligned_corrected_contacts: NDArray[np.float32]
    corrected_contacts: NDArray[np.float32]
    expert_prototype_pose: NDArray[np.float32]
    expert_prototype_root: NDArray[np.float32]
    reference_indices: NDArray[np.int64]
    reference_weights: NDArray[np.float32]
    reference_distances: NDArray[np.float32]
    timing_interpolation_method: str = "student_phase_timing"
    timing_sample_positions: NDArray[np.float32] | None = None
    wrist_velocity_limit: float | None = None
    maximum_wrist_velocity_before: float | None = None
    maximum_wrist_velocity_after: float | None = None
    maximum_body_velocity_before: float | None = None
    maximum_body_velocity_after: float | None = None


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
    spine_axis = np.median(
        shoulder_center[start:end] - hip_center[start:end], axis=0
    )
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


def _placement_body_scale(
    pose: NDArray[np.floating], *, start: int, end: int
) -> float:
    values = np.asarray(pose, dtype=np.float64)
    shoulder_center = 0.5 * (values[:, 5] + values[:, 6])
    hip_center = 0.5 * (values[:, 11] + values[:, 12])
    torso = np.linalg.norm(shoulder_center[start:end] - hip_center[start:end], axis=-1)
    shoulder_width = np.linalg.norm(
        values[start:end, 6] - values[start:end, 5], axis=-1
    )
    candidates = np.concatenate((torso, shoulder_width))
    candidates = candidates[np.isfinite(candidates) & (candidates > 1e-6)]
    return float(np.median(candidates)) if len(candidates) else 1.0


def _interpolate_translation(
    values: NDArray[np.floating], valid: NDArray[np.bool_]
) -> NDArray[np.float64]:
    result = np.asarray(values, dtype=np.float64).copy()
    observed = np.asarray(valid, dtype=bool) & np.all(np.isfinite(result), axis=1)
    if not np.any(observed):
        return np.zeros_like(result)
    timeline = np.arange(len(result), dtype=np.float64)
    for axis in range(2):
        result[:, axis] = np.interp(
            timeline, timeline[observed], result[observed, axis]
        )
    return result


def constrain_translation_trajectory(
    translation: NDArray[np.floating],
    *,
    valid: NDArray[np.bool_] | None,
    preparation_end: int,
    body_scale: float,
    max_excursion_ratio: float,
    max_velocity_ratio: float,
    max_acceleration_ratio: float,
) -> NDArray[np.float32]:
    """Create a smooth, bounded per-frame rigid translation.

    The robust preparation offset supplies the static placement. Only the
    residual trajectory is rate-limited, so a large initial image-space
    mismatch can still be corrected without being mistaken for body motion.
    Velocity limits are expressed per 1/64 of motion completeness, making the
    result independent of source FPS and analysis-window frame count.
    """
    raw = np.asarray(translation, dtype=np.float64)
    if raw.ndim != 2 or raw.shape[1] != 2 or not len(raw):
        raise ValueError("translation must have shape (T, 2)")
    if not 0 < preparation_end <= len(raw):
        raise ValueError("invalid translation preparation window")
    observed = (
        np.ones(len(raw), dtype=bool)
        if valid is None
        else np.asarray(valid, dtype=bool)
    )
    if observed.shape != (len(raw),):
        raise ValueError("translation validity must have shape (T,)")
    values = _interpolate_translation(raw, observed)
    prep_valid = observed[:preparation_end]
    baseline_values = values[:preparation_end][prep_valid]
    if not len(baseline_values):
        baseline_values = values[:preparation_end]
    baseline = np.median(baseline_values, axis=0)
    residual = values - baseline

    # A short symmetric filter rejects detector jitter without introducing the
    # abrupt lag produced by a causal frame-by-frame correction.
    radius = max(1, min(3, int(round(len(raw) * 0.035))))
    padded = np.pad(residual, ((radius, radius), (0, 0)), mode="edge")
    kernel = np.arange(1, radius + 2, dtype=np.float64)
    kernel = np.concatenate((kernel, kernel[-2::-1]))
    kernel /= np.sum(kernel)
    residual = np.stack(
        [np.convolve(padded[:, axis], kernel, mode="valid") for axis in range(2)],
        axis=-1,
    )

    scale = max(float(body_scale), 1e-6)
    maximum_excursion = max_excursion_ratio * scale
    norms = np.linalg.norm(residual, axis=1)
    outside = norms > maximum_excursion
    residual[outside] *= (maximum_excursion / norms[outside])[:, None]

    progress_step = 64.0 / max(len(raw) - 1, 1)
    maximum_velocity = max_velocity_ratio * scale * progress_step
    maximum_acceleration = max_acceleration_ratio * scale * progress_step**2
    constrained = residual.copy()
    velocity = np.zeros(2, dtype=np.float64)
    for frame in range(1, len(constrained)):
        requested_velocity = constrained[frame] - constrained[frame - 1]
        acceleration = requested_velocity - velocity
        acceleration_norm = float(np.linalg.norm(acceleration))
        if acceleration_norm > maximum_acceleration:
            requested_velocity = velocity + acceleration * (
                maximum_acceleration / acceleration_norm
            )
        velocity_norm = float(np.linalg.norm(requested_velocity))
        if velocity_norm > maximum_velocity:
            requested_velocity *= maximum_velocity / velocity_norm
        constrained[frame] = constrained[frame - 1] + requested_velocity
        velocity = requested_velocity
    return np.asarray(baseline + constrained, dtype=np.float32)


def _limit_chain_translation_by_bone_length(
    pose: NDArray[np.floating],
    translation: NDArray[np.floating],
    boundary_bones: tuple[tuple[int, int], ...],
    *,
    minimum_ratio: float = 0.85,
    maximum_ratio: float = 1.15,
) -> NDArray[np.float32]:
    """Reduce a chain shift if it would imply an impossible limb stretch."""
    values = np.asarray(pose, dtype=np.float64)
    requested = np.asarray(translation, dtype=np.float64)
    limited = requested.copy()
    for frame in range(len(values)):
        def feasible(fraction: float) -> bool:
            delta = requested[frame] * fraction
            for moving, fixed in boundary_bones:
                original = float(np.linalg.norm(values[frame, moving] - values[frame, fixed]))
                if original <= 1e-6:
                    continue
                candidate = float(
                    np.linalg.norm(values[frame, moving] + delta - values[frame, fixed])
                )
                if not minimum_ratio * original <= candidate <= maximum_ratio * original:
                    return False
            return True

        if feasible(1.0):
            continue
        low, high = 0.0, 1.0
        for _ in range(16):
            middle = 0.5 * (low + high)
            if feasible(middle):
                low = middle
            else:
                high = middle
        limited[frame] *= low
    return np.asarray(limited, dtype=np.float32)


def apply_constrained_hierarchical_pose_placement(
    student_pose: NDArray[np.floating],
    corrected_pose: NDArray[np.floating],
    *,
    start: int,
    end: int,
    confidence: NDArray[np.floating] | None = None,
) -> NDArray[np.float32]:
    """Follow grounded body placement per frame without copying pose errors."""
    student = np.asarray(student_pose, dtype=np.float64)
    corrected = np.asarray(corrected_pose, dtype=np.float64)
    if student.shape != corrected.shape or student.ndim != 3:
        raise ValueError("student and corrected poses must share shape (T, J, 2)")
    if not 0 <= start < end <= len(student):
        raise ValueError("invalid hierarchical-placement preparation window")
    weights = (
        np.ones(student.shape[:2], dtype=np.float64)
        if confidence is None
        else np.asarray(confidence, dtype=np.float64)
    )
    if weights.shape != student.shape[:2]:
        raise ValueError("placement confidence must have shape (T, J)")
    scale = _placement_body_scale(student, start=start, end=end)
    prep = slice(start, end)
    ankle_scores = []
    for joint in (15, 16):
        visible = weights[prep, joint] > 0.05
        if np.any(visible):
            ankle_scores.append(
                (float(np.median(student[prep, joint, 1][visible])), joint)
            )
    if not ankle_scores:
        return np.asarray(corrected, dtype=np.float32)
    support_ankle = max(ankle_scores)[1]
    ankle_valid = weights[:, support_ankle] > 0.05
    ankle_translation = constrain_translation_trajectory(
        student[:, support_ankle] - corrected[:, support_ankle],
        valid=ankle_valid,
        preparation_end=end,
        body_scale=scale,
        max_excursion_ratio=0.45,
        max_velocity_ratio=0.045,
        max_acceleration_ratio=0.0225,
    )
    placed = corrected + ankle_translation[:, None]

    knee_valid = np.minimum(weights[:, 13], weights[:, 14]) > 0.05
    student_knees = 0.5 * (student[:, 13] + student[:, 14])
    placed_knees = 0.5 * (placed[:, 13] + placed[:, 14])
    knee_translation = constrain_translation_trajectory(
        student_knees - placed_knees,
        valid=knee_valid,
        preparation_end=end,
        body_scale=scale,
        max_excursion_ratio=0.20,
        max_velocity_ratio=0.030,
        max_acceleration_ratio=0.015,
    )
    knee_translation = _limit_chain_translation_by_bone_length(
        placed, knee_translation, ((13, 15), (14, 16))
    )
    placed[:, :15] += knee_translation[:, None]

    hip_valid = np.minimum(weights[:, 11], weights[:, 12]) > 0.05
    student_hips = 0.5 * (student[:, 11] + student[:, 12])
    placed_hips = 0.5 * (placed[:, 11] + placed[:, 12])
    hip_translation = constrain_translation_trajectory(
        student_hips - placed_hips,
        valid=hip_valid,
        preparation_end=end,
        body_scale=scale,
        max_excursion_ratio=0.15,
        max_velocity_ratio=0.025,
        max_acceleration_ratio=0.0125,
    )
    hip_translation = _limit_chain_translation_by_bone_length(
        placed, hip_translation, ((11, 13), (12, 14))
    )
    placed[:, :13] += hip_translation[:, None]
    return np.asarray(placed, dtype=np.float32)


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
    ankle_y = [
        float(np.median(student[start:end, joint, 1])) for joint in (15, 16)
    ]
    support_ankle = (15, 16)[int(np.argmax(ankle_y))]
    ankle_delta = np.median(
        student[start:end, support_ankle]
        - corrected[start:end, support_ankle],
        axis=0,
    )
    placed = np.asarray(corrected + ankle_delta, dtype=np.float32)
    placed = shift_expert_body_chain_to_student_knee(
        student, placed, start=start, end=end
    )
    return shift_expert_body_chain_to_student_hip(
        student, placed, start=start, end=end
    )


def align_expert_correction_to_ankle_spine_view(
    correction: ExpertCorrection,
    *,
    start: int,
    end: int,
    placement_mode: Literal["fixed", "constrained"] = "constrained",
) -> tuple[ExpertCorrection, NDArray[np.float32]]:
    """Rotate, then apply fixed or bounded per-frame hierarchical placement."""
    if placement_mode not in {"fixed", "constrained"}:
        raise ValueError(f"unsupported hierarchical placement mode: {placement_mode}")
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
    if placement_mode == "fixed":
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
    else:
        aligned_corrected = apply_constrained_hierarchical_pose_placement(
            correction.aligned_student_pose,
            aligned_corrected,
            start=start,
            end=end,
            confidence=phase_align_sequence(
                correction.student.confidence, correction.student.phase_indices
            ),
        )
        corrected = apply_constrained_hierarchical_pose_placement(
            correction.student.pose,
            corrected,
            start=start,
            end=end,
            confidence=correction.student.confidence,
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


def _scalar_string(values: Any, key: str, default: str) -> str:
    if key not in values:
        return default
    value = values[key]
    return str(value.item() if hasattr(value, "item") else value)


def _pose_key(archive: Any, dimensions: int) -> str:
    if dimensions == 2:
        candidates = ("skeleton", "skeleton_2d")
    elif dimensions == 3:
        candidates = ("skeleton_3d",)
    else:
        raise ValueError("dimensions must be 2 or 3")
    for key in candidates:
        if key in archive:
            return key
    raise ValueError(f"archive does not contain a {dimensions}D skeleton")


def load_motion_sample(path: str | Path, *, dimensions: int = 2) -> MotionSample:
    """Load either the current or legacy skeleton archive schema."""
    source = Path(path)
    with np.load(source, allow_pickle=False) as archive:
        key = _pose_key(archive, dimensions)
        pose = np.asarray(archive[key], dtype=np.float32)
        confidence = np.asarray(archive["confidence"], dtype=np.float32)
        phases = np.asarray(archive["phase_indices"], dtype=np.int64)
        root_dimensions = pose.shape[-1]
        root = np.asarray(
            archive.get(
                "root_trajectory",
                np.zeros((len(pose), root_dimensions), dtype=np.float32),
            ),
            dtype=np.float32,
        )
        foot_contacts = np.asarray(
            archive.get(
                "foot_contacts",
                np.zeros((len(pose), 2), dtype=np.float32),
            ),
            dtype=np.float32,
        )
        handedness = _scalar_string(archive, "handedness", "right").lower()
        skill = _scalar_string(archive, "skill", "")
        video_name = _scalar_string(archive, "video_name", source.name)
        subject_id = _scalar_string(
            archive,
            "subject_id",
            expert_subject_identity(video_name),
        )
        if video_name.lower().startswith("expert-"):
            subject_id = expert_subject_identity(video_name)
        phase_source = _scalar_string(archive, "phase_source", "legacy_unversioned")
        has_subject_id = "subject_id" in archive
    if pose.ndim != 3 or pose.shape[1] != 17:
        raise ValueError(f"{source}: pose must have shape (T, 17, D)")
    if pose.shape[-1] != dimensions:
        raise ValueError(f"{source}: expected {dimensions}D pose")
    if confidence.shape != pose.shape[:2]:
        raise ValueError(f"{source}: confidence must have shape (T, 17)")
    if root.shape != (len(pose), dimensions):
        raise ValueError(f"{source}: root trajectory must have shape (T, D)")
    if foot_contacts.shape != (len(pose), 2):
        raise ValueError(f"{source}: foot contacts must have shape (T, 2)")
    if phases.shape != (5,) or np.any(np.diff(phases) <= 0):
        raise ValueError(f"{source}: five strictly increasing phases are required")
    if phases[0] < 0 or phases[-1] >= len(pose):
        raise ValueError(f"{source}: phase anchors are outside the pose sequence")
    if handedness not in {"left", "right"}:
        raise ValueError(f"{source}: handedness must be left or right")
    if phase_source == "acceleration_ending_range_v4":
        alignment_contract = "overhead_asymmetric_ending_range_v4"
    elif phase_source in {
        "acceleration_wrist_velocity_stop_v6",
        "acceleration_wrist_velocity_stop_delayed_contact_v7",
    }:
        alignment_contract = "overhead_wrist_velocity_stop_v6"
    elif skill == "serve" and phase_source in {"detected", "legacy_unversioned"}:
        alignment_contract = "serve_detector_proxy_anchors_v1"
    else:
        alignment_contract = phase_source
    return MotionSample(
        path=source,
        pose=pose,
        confidence=np.clip(confidence, 0.0, 1.0),
        root=root,
        foot_contacts=np.clip(foot_contacts, 0.0, 1.0),
        phase_indices=phases,
        handedness=handedness,
        skill=skill,
        video_name=video_name,
        subject_id=subject_id,
        phase_source=phase_source,
        alignment_contract=alignment_contract,
        identity_level="subject" if has_subject_id else "archive_fallback",
    )


def discover_motion_samples(
    root: str | Path,
    *,
    dimensions: int = 2,
    expected_skill: str | None = None,
) -> list[MotionSample]:
    samples = [
        load_motion_sample(path, dimensions=dimensions)
        for path in sorted(Path(root).glob("*.npz"))
    ]
    if expected_skill is not None:
        mismatches = [sample.path.name for sample in samples if sample.skill != expected_skill]
        if mismatches:
            raise ValueError(
                f"expected {expected_skill} archives, found mismatches: {mismatches[:5]}"
            )
    if not samples:
        raise ValueError(f"no motion archives found under {root}")
    return samples


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


def _masked_median(
    values: NDArray[np.floating], mask: NDArray[np.floating]
) -> NDArray[np.float64]:
    result = np.empty(values.shape[1:], dtype=np.float64)
    flattened = np.asarray(values, dtype=np.float64).reshape(len(values), -1)
    flattened_mask = np.asarray(mask, dtype=np.float64).reshape(len(mask), -1)
    output = result.reshape(-1)
    for column in range(flattened.shape[1]):
        visible = flattened_mask[:, column] > 0.05
        finite = np.isfinite(flattened[:, column])
        selected = flattened[visible & finite, column]
        if not len(selected):
            selected = flattened[finite, column]
        output[column] = float(np.median(selected)) if len(selected) else 0.0
    return result


def stance_feature(
    aligned_pose: NDArray[np.floating],
    aligned_confidence: NDArray[np.floating],
) -> NDArray[np.float32]:
    """Encode only preparation stance and stable apparent body proportions."""
    pose = np.asarray(aligned_pose, dtype=np.float64)
    confidence = np.asarray(aligned_confidence, dtype=np.float64)
    if pose.shape != (64, 17, 2) or confidence.shape != (64, 17):
        raise ValueError("aligned pose/confidence must have shapes (64,17,2)/(64,17)")
    prep_pose = pose[:_PREPARATION_END]
    prep_confidence = confidence[:_PREPARATION_END]
    pelvis = 0.5 * (prep_pose[:, 11] + prep_pose[:, 12])
    shoulder_width = np.linalg.norm(prep_pose[:, 6] - prep_pose[:, 5], axis=-1)
    torso = 0.5 * (prep_pose[:, 5] + prep_pose[:, 6]) - pelvis
    body_scale_values = np.concatenate(
        (shoulder_width, np.linalg.norm(torso, axis=-1))
    )
    valid_scale = body_scale_values[np.isfinite(body_scale_values) & (body_scale_values > 1e-6)]
    scale = float(np.median(valid_scale)) if len(valid_scale) else 1.0

    centered = (prep_pose[:, _FEATURE_JOINTS] - pelvis[:, None]) / scale
    joint_mask = np.repeat(
        prep_confidence[:, _FEATURE_JOINTS, None], 2, axis=-1
    )
    stance = _masked_median(centered, joint_mask).reshape(-1)

    bone_features = []
    for start, end in BONES:
        lengths = np.linalg.norm(prep_pose[:, end] - prep_pose[:, start], axis=-1)
        visible = (
            (prep_confidence[:, start] > 0.05)
            & (prep_confidence[:, end] > 0.05)
            & np.isfinite(lengths)
        )
        selected = lengths[visible]
        bone_features.append(
            (float(np.median(selected)) if len(selected) else 0.0) / scale
        )
    return np.asarray((*stance, *bone_features), dtype=np.float32)


def _standardize_features(
    features: NDArray[np.floating],
) -> tuple[NDArray[np.float32], NDArray[np.float32], NDArray[np.float32]]:
    values = np.asarray(features, dtype=np.float64)
    mean = values.mean(axis=0)
    scale = values.std(axis=0)
    scale = np.where(scale > 1e-5, scale, 1.0)
    return (
        ((values - mean) / scale).astype(np.float32),
        mean.astype(np.float32),
        scale.astype(np.float32),
    )


def _reference_weights(
    model: ExpertPhaseModel,
    aligned_pose: NDArray[np.floating],
    aligned_confidence: NDArray[np.floating],
    handedness: str,
    *,
    allowed_indices: NDArray[np.integer] | None = None,
) -> tuple[NDArray[np.int64], NDArray[np.float32], NDArray[np.float32]]:
    feature = stance_feature(aligned_pose, aligned_confidence)
    standardized = (feature - model.feature_mean) / model.feature_scale
    candidates = (
        np.arange(len(model.expert_pose), dtype=np.int64)
        if allowed_indices is None
        else np.asarray(allowed_indices, dtype=np.int64)
    )
    same_hand = candidates[model.expert_handedness[candidates] == handedness]
    if len(same_hand):
        candidates = same_hand
    if not len(candidates):
        raise ValueError("expert model contains no eligible references")
    distances = np.linalg.norm(
        model.expert_features[candidates] - standardized[None], axis=-1
    ) / np.sqrt(model.expert_features.shape[1])
    count = min(model.top_k, len(candidates))
    order = np.argsort(distances, kind="stable")[:count]
    selected = candidates[order]
    selected_distances = distances[order]
    positive = selected_distances[selected_distances > _EPS]
    bandwidth = float(np.median(positive)) if len(positive) else 1.0
    bandwidth = max(bandwidth, 1e-3)
    weights = np.exp(-0.5 * np.square(selected_distances / bandwidth))
    weights /= max(float(weights.sum()), _EPS)
    return (
        selected.astype(np.int64),
        weights.astype(np.float32),
        selected_distances.astype(np.float32),
    )


def _weighted_prototype(
    values: NDArray[np.floating],
    confidence: NDArray[np.floating],
    indices: NDArray[np.integer],
    weights: NDArray[np.floating],
) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
    selected_values = np.asarray(values[indices], dtype=np.float64)
    selected_confidence = np.asarray(confidence[indices], dtype=np.float64)
    weighted_confidence = selected_confidence * np.asarray(weights)[:, None, None]
    denominator = weighted_confidence.sum(axis=0)
    prototype = np.divide(
        (selected_values * weighted_confidence[..., None]).sum(axis=0),
        denominator[..., None],
        out=np.average(selected_values, axis=0, weights=weights).astype(np.float64),
        where=denominator[..., None] > _EPS,
    )
    return prototype.astype(np.float32), np.clip(denominator, 0.0, 1.0).astype(np.float32)


def _weighted_root(
    values: NDArray[np.floating],
    indices: NDArray[np.integer],
    weights: NDArray[np.floating],
) -> NDArray[np.float32]:
    return np.average(values[indices], axis=0, weights=weights).astype(np.float32)


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
            output[frame_index] = (
                (1.0 - influence) * prior_root[frame_index]
                + influence * constrained
            )
    return output.astype(np.float32)


def _predict_aligned(
    model: ExpertPhaseModel,
    aligned_pose: NDArray[np.float32],
    aligned_confidence: NDArray[np.float32],
    aligned_root: NDArray[np.float32],
    handedness: str,
    *,
    allowed_indices: NDArray[np.integer] | None = None,
) -> tuple[NDArray[np.float32], ...]:
    indices, weights, distances = _reference_weights(
        model,
        aligned_pose,
        aligned_confidence,
        handedness,
        allowed_indices=allowed_indices,
    )
    prototype, prototype_confidence = _weighted_prototype(
        model.expert_pose,
        model.expert_confidence,
        indices,
        weights,
    )
    prototype_root = _weighted_root(model.expert_root, indices, weights)
    prototype_contacts = np.average(
        model.expert_foot_contacts[indices], axis=0, weights=weights
    ).astype(np.float32)
    corrected = project_stable_bone_lengths(
        aligned_pose,
        prototype,
        np.minimum(aligned_confidence, prototype_confidence),
        iterations=80,
        expert_length_bones=TORSO_WIDTH_BONES,
        preserve_direction_chains=(
            (6, 8, 10),
            (5, 7, 9),
            (12, 14, 16),
            (11, 13, 15),
        ),
    )
    # Root motion is represented as displacement from the preparation stance.
    # Legacy root-centred archives contain zeros, in which case the student's
    # observed root is retained until world-grounded re-extraction is available.
    root_delta = prototype_root - prototype_root[:1]
    if float(np.max(np.abs(root_delta))) <= 1e-7:
        corrected_root = aligned_root.copy()
    else:
        corrected_root = aligned_root[:1] + root_delta
    corrected_root = _retarget_root_with_contacts(
        corrected,
        corrected_root,
        prototype_contacts,
        prototype,
    )
    return (
        corrected,
        corrected_root.astype(np.float32),
        prototype,
        prototype_root,
        indices,
        weights,
        distances,
        prototype_contacts,
    )


def _angles(
    sequence: NDArray[np.floating], triplets: Sequence[tuple[int, int, int]]
) -> NDArray[np.float64]:
    values = np.asarray(sequence, dtype=np.float64)
    indices = np.asarray(triplets, dtype=np.int64)
    incoming = values[:, indices[:, 0]] - values[:, indices[:, 1]]
    outgoing = values[:, indices[:, 2]] - values[:, indices[:, 1]]
    denominator = np.linalg.norm(incoming, axis=-1) * np.linalg.norm(outgoing, axis=-1)
    cosine = np.divide(
        np.sum(incoming * outgoing, axis=-1),
        denominator,
        out=np.ones_like(denominator),
        where=denominator > _EPS,
    )
    return np.arccos(np.clip(cosine, -1.0, 1.0))


def criterion_distance_components(
    source: NDArray[np.floating],
    target: NDArray[np.floating],
    confidence: NDArray[np.floating],
    *,
    start: int,
    end: int,
    joints: Sequence[int],
    joint_weights: NDArray[np.floating],
) -> dict[str, float]:
    """Return separate Euclidean and target-angle distances for one rubric."""
    first = np.asarray(source, dtype=np.float64)[start:end]
    second = np.asarray(target, dtype=np.float64)[start:end]
    observed = np.asarray(confidence, dtype=np.float64)[start:end]
    selected = np.asarray(tuple(dict.fromkeys(int(value) for value in joints)))
    weights = observed[:, selected] * np.asarray(joint_weights)[selected][None]
    denominator = float(weights.sum())
    euclidean = (
        float(
            np.sum(
                np.linalg.norm(first[:, selected] - second[:, selected], axis=-1)
                * weights
            )
            / denominator
        )
        if denominator > _EPS
        else 0.0
    )
    selected_set = set(selected.tolist())
    triplets = tuple(item for item in ANGLE_TRIPLETS if set(item) <= selected_set)
    angle = 0.0
    if triplets:
        indices = np.asarray(triplets, dtype=np.int64)
        angle_mask = (
            observed[:, indices[:, 0]]
            * observed[:, indices[:, 1]]
            * observed[:, indices[:, 2]]
        )
        angle_denominator = float(angle_mask.sum())
        if angle_denominator > _EPS:
            delta = np.abs(_angles(first, triplets) - _angles(second, triplets)) / np.pi
            angle = float(np.sum(delta * angle_mask) / angle_denominator)
    return {
        "euclidean_distance": euclidean,
        "target_angle_distance": angle,
        "combined_distance": euclidean + 0.5 * angle,
    }


def _wrapped_angle(value: float) -> float:
    """Wrap an angle difference to ``[-pi, pi]``."""
    return float((value + np.pi) % (2.0 * np.pi) - np.pi)


def _robust_window_value(
    values: NDArray[np.floating],
    confidence: NDArray[np.floating],
    *,
    start: int,
    end: int,
    minimum_confidence: float = 0.20,
) -> float:
    """Return a confidence-masked median without silently treating misses as zero."""
    selected = np.asarray(values, dtype=np.float64)[start:end]
    observed = np.asarray(confidence, dtype=np.float64)[start:end]
    valid = np.isfinite(selected) & (observed >= minimum_confidence)
    if np.any(valid):
        return float(np.median(selected[valid]))
    finite = selected[np.isfinite(selected)]
    return float(np.median(finite)) if len(finite) else 0.0




def _serve_weight_transfer_components(
    source_pose: NDArray[np.floating],
    source_root: NDArray[np.floating],
    target_pose: NDArray[np.floating],
    target_root: NDArray[np.floating],
    confidence: NDArray[np.floating],
) -> dict[str, float]:
    del source_root, target_root
    source = _serve_dominant_chain_angles(source_pose)
    target = _serve_dominant_chain_angles(target_pose)
    observed = np.min(np.asarray(confidence)[:, (6, 12, 14, 16)], axis=1)
    preparation_start, preparation_end = motion_completion_bounds(
        len(source), 0.125, 0.34375
    )
    transfer_start, transfer_end = motion_completion_bounds(
        len(source), 0.25, 1.0
    )
    completion_start, completion_end = motion_completion_bounds(
        len(source), 0.71875, 1.0
    )
    valid = observed[transfer_start:transfer_end] >= 0.20
    if not np.any(valid):
        valid = np.ones(transfer_end - transfer_start, dtype=bool)
    trajectory_delta = (
        source[transfer_start:transfer_end][valid]
        - target[transfer_start:transfer_end][valid]
    ) / np.pi
    trajectory_distance = float(np.sqrt(np.mean(trajectory_delta**2)))

    def window(values: NDArray[np.floating], start: int, end: int) -> NDArray[np.float64]:
        return np.asarray(
            [
                _robust_window_value(
                    values[:, column], observed, start=start, end=end
                )
                for column in range(values.shape[1])
            ],
            dtype=np.float64,
        )

    source_change = window(source, completion_start, completion_end) - window(
        source, preparation_start, preparation_end
    )
    target_change = window(target, completion_start, completion_end) - window(
        target, preparation_start, preparation_end
    )
    change_distance = float(
        np.sqrt(np.mean(((source_change - target_change) / np.pi) ** 2))
    )
    distance = float(
        np.sqrt((trajectory_distance**2 + 0.75 * change_distance**2) / 1.75)
    )
    return {
        "euclidean_distance": distance,
        "target_angle_distance": trajectory_distance,
        "combined_distance": distance,
        "dominant_chain_trajectory_distance": trajectory_distance,
        "dominant_chain_change_distance": change_distance,
        "source_shoulder_hip_knee_change": float(source_change[0]),
        "target_shoulder_hip_knee_change": float(target_change[0]),
        "source_hip_knee_ankle_change": float(source_change[1]),
        "target_hip_knee_ankle_change": float(target_change[1]),
        "source_shoulder_hip_ankle_change": float(source_change[2]),
        "target_shoulder_hip_ankle_change": float(target_change[2]),
    }


def _joint_angle_trajectory(
    pose: NDArray[np.floating], first: int, pivot: int, third: int
) -> NDArray[np.float64]:
    values = np.asarray(pose, dtype=np.float64)
    incoming = values[:, first] - values[:, pivot]
    outgoing = values[:, third] - values[:, pivot]
    denominator = np.maximum(
        np.linalg.norm(incoming, axis=-1) * np.linalg.norm(outgoing, axis=-1),
        _EPS,
    )
    cosine = np.sum(incoming * outgoing, axis=-1) / denominator
    return np.arccos(np.clip(cosine, -1.0, 1.0))


def _serve_dominant_chain_angles(
    pose: NDArray[np.floating],
) -> NDArray[np.float64]:
    """Return canonical dominant-side transfer angles in radians.

    Left-handed samples are canonicalized before scoring, so the dominant
    shoulder/hip/knee/ankle are consistently COCO joints 6/12/14/16.
    """
    return np.stack(
        (
            _joint_angle_trajectory(pose, 6, 12, 14),
            _joint_angle_trajectory(pose, 12, 14, 16),
            _joint_angle_trajectory(pose, 6, 12, 16),
        ),
        axis=-1,
    )


def _serve_qualitative_pose_evidence(
    pose: NDArray[np.floating],
    root: NDArray[np.floating] | None = None,
) -> dict[str, float]:
    """Return camera-scale-invariant evidence required by serve checkpoints.

    Generated motion distance alone is insufficient when the conditional
    generator preserves a weak input stance. These features express the
    prerequisites visible in every expert training identity: both arms rise,
    the ankles provide a real base, and pelvis displacement is accompanied by
    coordinated hip rotation and dominant-leg-chain motion. The latter avoids
    mistaking detector drift or a sideways lean for weight transfer.
    """
    values = np.asarray(pose, dtype=np.float64)
    preparation_start, preparation_end = motion_completion_bounds(
        len(values), 0.125, 0.34375
    )
    completion_start, completion_end = motion_completion_bounds(
        len(values), 0.71875, 1.0
    )
    hip_center = 0.5 * (values[:, 11] + values[:, 12])
    shoulder_center = 0.5 * (values[:, 5] + values[:, 6])
    torso = np.maximum(
        np.linalg.norm(shoulder_center - hip_center, axis=-1), _EPS
    )
    arm_elevation = np.stack(
        [
            (values[:, joint, 1] - hip_center[:, 1]) / torso
            for joint in (7, 8, 9, 10)
        ],
        axis=-1,
    )
    # Both elbows and wrists must be raised together. A high quantile tolerates
    # a brief detector miss without letting one raised arm stand in for two.
    # Each arm is considered raised when either its elbow or wrist provides
    # the elevation evidence; functional serve preparation does not require
    # both joints on both arms to be level. The weaker of the two arm cues is
    # still decisive, so one raised racket arm cannot stand in for both arms.
    simultaneous_elevation = np.minimum(
        np.max(arm_elevation[:, (0, 2)], axis=-1),
        np.max(arm_elevation[:, (1, 3)], axis=-1),
    )
    arms_raised = float(
        np.quantile(
            simultaneous_elevation[preparation_start:preparation_end], 0.70
        )
    )
    stance_width = (
        np.linalg.norm(values[:, 16] - values[:, 15], axis=-1) / torso
    )
    preparation_stance = float(
        np.median(stance_width[preparation_start:preparation_end])
    )
    ankle_axis = values[:, 16] - values[:, 15]
    ankle_denominator = np.maximum(
        np.sum(ankle_axis * ankle_axis, axis=-1), _EPS
    )
    pelvis_loading = np.sum(
        (hip_center - values[:, 15]) * ankle_axis, axis=-1
    ) / ankle_denominator
    loading_shift = abs(
        float(np.median(pelvis_loading[completion_start:completion_end]))
        - float(np.median(pelvis_loading[preparation_start:preparation_end]))
    )
    chain_angles = _serve_dominant_chain_angles(values)
    preparation_chain = np.median(
        chain_angles[preparation_start:preparation_end], axis=0
    )
    completion_chain = np.median(
        chain_angles[completion_start:completion_end], axis=0
    )
    chain_change = float(
        np.linalg.norm(completion_chain - preparation_chain) / np.pi
    )
    chain_baseline = np.median(
        chain_angles[preparation_start:preparation_end], axis=0
    )
    chain_excursion = _smooth_trajectory(
        np.linalg.norm(chain_angles - chain_baseline[None], axis=1)[:, None]
    )[:, 0]
    transfer_start, transfer_end = motion_completion_bounds(
        len(values), 0.25, 1.0
    )
    dominant_chain_excursion = float(
        np.quantile(chain_excursion[transfer_start:transfer_end], 0.80)
        / np.pi
    )
    hip_vector = values[:, 12] - values[:, 11]
    hip_rotation = _smooth_trajectory(
        np.unwrap(np.arctan2(hip_vector[:, 1], hip_vector[:, 0]))[:, None]
    )[:, 0]
    hip_rotation -= float(
        np.median(hip_rotation[preparation_start:preparation_end])
    )
    transfer_rotation = np.abs(hip_rotation[transfer_start:transfer_end])
    transfer_chain = chain_excursion[transfer_start:transfer_end]
    centred_rotation = transfer_rotation - np.mean(transfer_rotation)
    centred_chain = transfer_chain - np.mean(transfer_chain)
    coupling_denominator = float(
        np.linalg.norm(centred_rotation) * np.linalg.norm(centred_chain)
    )
    transfer_rotation_correlation = (
        0.0
        if coupling_denominator <= _EPS
        else float(
            np.dot(centred_rotation, centred_chain) / coupling_denominator
        )
    )
    hip_rotation_excursion = float(
        np.quantile(transfer_rotation, 0.80) / np.pi
    )
    coordinated_hip_rotation = float(
        hip_rotation_excursion * max(transfer_rotation_correlation, 0.0)
    )
    root_values = (
        np.zeros((len(values), 2), dtype=np.float64)
        if root is None
        else np.asarray(root, dtype=np.float64)
    )
    if root_values.shape != (len(values), 2):
        raise ValueError("serve root evidence must have shape (T, 2)")
    preparation_root = np.median(
        root_values[preparation_start:preparation_end], axis=0
    )
    completion_root = np.median(
        root_values[completion_start:completion_end], axis=0
    )
    preparation_torso = max(
        float(np.median(torso[preparation_start:preparation_end])), _EPS
    )
    root_transfer = float(
        np.linalg.norm(completion_root - preparation_root) / preparation_torso
    )
    return {
        "simultaneous_arm_elevation": arms_raised,
        "preparation_stance_width": preparation_stance,
        "pelvis_loading_shift": loading_shift,
        "dominant_chain_change": chain_change,
        "dominant_chain_excursion": dominant_chain_excursion,
        "hip_rotation_excursion": hip_rotation_excursion,
        "transfer_rotation_correlation": transfer_rotation_correlation,
        "coordinated_hip_rotation": coordinated_hip_rotation,
        "root_transfer_distance": root_transfer,
    }


def _serve_expert_qualitative_envelope(
    model: ExpertPhaseModel,
) -> dict[str, dict[str, float]]:
    """Fit lower expert evidence bounds without using any student clip."""
    rows = [
        _serve_qualitative_pose_evidence(pose, root)
        for pose, root in zip(
            np.asarray(model.expert_pose),
            np.asarray(model.expert_root),
            strict=True,
        )
    ]
    subjects = np.asarray(model.expert_subject_ids)
    subject_ids = sorted(set(subjects.tolist()))
    output: dict[str, dict[str, float]] = {}
    for name in rows[0]:
        clip_values = np.asarray([row[name] for row in rows], dtype=np.float64)
        subject_values = np.asarray(
            [
                np.median(clip_values[subjects == subject_id])
                for subject_id in subject_ids
            ],
            dtype=np.float64,
        )
        repeated_subject_median = np.asarray(
            [
                np.median(clip_values[subjects == subject_id])
                for subject_id in subjects
            ],
            dtype=np.float64,
        )
        within_take_scale = 1.4826 * float(
            np.median(np.abs(clip_values - repeated_subject_median))
        )
        lower = float(np.min(subject_values) - within_take_scale)
        scale_floor = (
            0.01
            if name
            in {
                "dominant_chain_excursion",
                "hip_rotation_excursion",
                "coordinated_hip_rotation",
            }
            else 0.05
        )
        scale = max(
            float(np.median(subject_values) - lower),
            within_take_scale,
            scale_floor,
        )
        output[name] = {
            "expert_lower": lower,
            "expert_scale": scale,
        }
    return output


def _serve_qualitative_factor(
    value: float, calibration: dict[str, float]
) -> float:
    expert_scale = max(float(calibration["expert_scale"]), 1e-3)
    # Half an expert robust scale is a no-penalty detector/view margin. Beyond
    # it, decay over five percent of torso scale so narrow-but-valid expert
    # stances remain accepted while feet-together motion does not masquerade
    # as weight transfer.
    shortfall = max(
        float(calibration["expert_lower"]) - 0.5 * expert_scale - value,
        0.0,
    )
    return float(
        np.exp(-shortfall / min(expert_scale, 0.05))
    )


def _serve_required_motion_factor(
    value: float, calibration: dict[str, float]
) -> float:
    """Score an absolute movement prerequisite against experts only.

    Unlike stance width, motion magnitude does not receive a half-scale view
    margin: a missing coordinated movement must not earn full credit merely
    because the generated target also remains close to the student.
    """
    expert_scale = max(float(calibration["expert_scale"]), 1e-3)
    shortfall = max(float(calibration["expert_lower"]) - value, 0.0)
    return float(np.exp(-shortfall / expert_scale))


def _serve_projected_rotation_features(
    pose: NDArray[np.floating],
    confidence: NDArray[np.floating],
) -> NDArray[np.float64]:
    """Return monocular 2D proxies for pelvis and torso axial rotation."""
    values = np.asarray(pose, dtype=np.float64)
    observed = np.asarray(confidence, dtype=np.float64)
    hip_vector = values[:, 12] - values[:, 11]
    shoulder_vector = values[:, 6] - values[:, 5]
    hip_width = np.maximum(np.linalg.norm(hip_vector, axis=-1), _EPS)
    shoulder_width = np.maximum(np.linalg.norm(shoulder_vector, axis=-1), _EPS)
    hip_angle = np.unwrap(np.arctan2(hip_vector[:, 1], hip_vector[:, 0]))
    shoulder_angle = np.unwrap(
        np.arctan2(shoulder_vector[:, 1], shoulder_vector[:, 0])
    )
    hip_confidence = np.min(observed[:, (11, 12)], axis=1)
    shoulder_confidence = np.min(observed[:, (5, 6)], axis=1)
    torso_confidence = np.minimum(hip_confidence, shoulder_confidence)

    def window(
        value: NDArray[np.floating],
        conf: NDArray[np.floating],
        start: int,
        end: int,
    ) -> float:
        return _robust_window_value(value, conf, start=start, end=end)

    preparation_start, preparation_end = motion_completion_bounds(
        len(values), 0.125, 0.34375
    )
    completion_start, completion_end = motion_completion_bounds(
        len(values), 0.71875, 1.0
    )
    prep_hip_width = window(
        hip_width, hip_confidence, preparation_start, preparation_end
    )
    end_hip_width = window(
        hip_width, hip_confidence, completion_start, completion_end
    )
    prep_shoulder_width = window(
        shoulder_width, shoulder_confidence, preparation_start, preparation_end
    )
    end_shoulder_width = window(
        shoulder_width, shoulder_confidence, completion_start, completion_end
    )
    prep_hip_angle = window(
        hip_angle, hip_confidence, preparation_start, preparation_end
    )
    end_hip_angle = window(
        hip_angle, hip_confidence, completion_start, completion_end
    )
    torso_twist = np.unwrap(shoulder_angle - hip_angle)
    prep_twist = window(
        torso_twist, torso_confidence, preparation_start, preparation_end
    )
    end_twist = window(
        torso_twist, torso_confidence, completion_start, completion_end
    )
    return np.asarray(
        (
            np.log(end_hip_width / max(prep_hip_width, _EPS)),
            np.log(end_shoulder_width / max(prep_shoulder_width, _EPS)),
            _wrapped_angle(end_hip_angle - prep_hip_angle),
            _wrapped_angle(end_twist - prep_twist),
        ),
        dtype=np.float64,
    )


def _serve_hip_rotation_components(
    source_pose: NDArray[np.floating],
    target_pose: NDArray[np.floating],
    confidence: NDArray[np.floating],
) -> dict[str, float]:
    source_values = np.asarray(source_pose, dtype=np.float64)
    target_values = np.asarray(target_pose, dtype=np.float64)
    source_angles = _serve_dominant_chain_angles(source_values)
    target_angles = _serve_dominant_chain_angles(target_values)
    preparation_start, preparation_end = motion_completion_bounds(
        len(source_values), 0.125, 0.34375
    )
    transfer_start, transfer_end = motion_completion_bounds(
        len(source_values), 0.25, 1.0
    )

    def coupled_trajectories(
        pose: NDArray[np.float64], angles: NDArray[np.float64]
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        hip_vector = pose[:, 12] - pose[:, 11]
        rotation = np.unwrap(np.arctan2(hip_vector[:, 1], hip_vector[:, 0]))
        rotation -= float(np.median(rotation[preparation_start:preparation_end]))
        angle_baseline = np.median(
            angles[preparation_start:preparation_end], axis=0
        )
        transfer = np.linalg.norm(angles - angle_baseline[None], axis=1)
        rotation = _smooth_trajectory(rotation[:, None])[:, 0]
        transfer = _smooth_trajectory(transfer[:, None])[:, 0]
        return rotation, transfer

    source_rotation, source_transfer = coupled_trajectories(
        source_values, source_angles
    )
    target_rotation, target_transfer = coupled_trajectories(
        target_values, target_angles
    )
    observed = np.min(
        np.asarray(confidence)[:, (6, 11, 12, 14, 16)], axis=1
    )
    valid = observed[transfer_start:transfer_end] >= 0.20
    if not np.any(valid):
        valid = np.ones(transfer_end - transfer_start, dtype=bool)

    def correlation(first: NDArray[np.floating], second: NDArray[np.floating]) -> float:
        left = np.asarray(first[transfer_start:transfer_end], dtype=np.float64)[valid]
        right = np.asarray(second[transfer_start:transfer_end], dtype=np.float64)[valid]
        left -= np.mean(left)
        right -= np.mean(right)
        denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
        return 0.0 if denominator <= _EPS else float(np.dot(left, right) / denominator)

    source_coupling = correlation(np.abs(source_rotation), source_transfer)
    target_coupling = correlation(np.abs(target_rotation), target_transfer)
    coupling_distance = abs(source_coupling - target_coupling) / 2.0
    rotation_distance = float(
        np.sqrt(
            np.mean(
                (
                    (
                        source_rotation[transfer_start:transfer_end][valid]
                        - target_rotation[transfer_start:transfer_end][valid]
                    )
                    / np.pi
                )
                ** 2
            )
        )
    )
    # Coupling is primary: hip rotation receives credit when it happens with
    # the dominant-side weight-transfer chain, not as an isolated torso pose.
    distance = float(
        np.sqrt((0.5 * rotation_distance**2 + 2.0 * coupling_distance**2) / 2.5)
    )
    return {
        "euclidean_distance": distance,
        "target_angle_distance": rotation_distance,
        "combined_distance": distance,
        "hip_rotation_trajectory_distance": rotation_distance,
        "source_transfer_rotation_correlation": source_coupling,
        "target_transfer_rotation_correlation": target_coupling,
        "transfer_rotation_coupling_distance": coupling_distance,
    }


def _smooth_trajectory(values: NDArray[np.floating]) -> NDArray[np.float64]:
    trajectory = np.asarray(values, dtype=np.float64)
    if trajectory.ndim != 2:
        raise ValueError("trajectory smoothing requires shape (T, D)")
    padded = np.pad(trajectory, ((2, 2), (0, 0)), mode="edge")
    kernel = np.asarray((1.0, 2.0, 3.0, 2.0, 1.0), dtype=np.float64) / 9.0
    return np.stack(
        [
            np.convolve(padded[:, axis], kernel, mode="valid")
            for axis in range(trajectory.shape[1])
        ],
        axis=-1,
    )


def _serve_wrist_motion_features(
    pose: NDArray[np.floating],
    confidence: NDArray[np.floating],
    *,
    start: int,
    end: int,
) -> NDArray[np.float64]:
    """Measure distal-arm speed; COCO-17 cannot observe wrist flexion directly."""
    values = np.asarray(pose, dtype=np.float64)
    observed = np.asarray(confidence, dtype=np.float64)
    wrist_to_shoulder = _smooth_trajectory(values[:, 10] - values[:, 6])
    wrist_to_elbow = _smooth_trajectory(values[:, 10] - values[:, 8])
    shoulder_speed = np.linalg.norm(np.diff(wrist_to_shoulder, axis=0), axis=-1)
    forearm_speed = np.linalg.norm(np.diff(wrist_to_elbow, axis=0), axis=-1)
    acceleration = np.linalg.norm(
        np.diff(wrist_to_shoulder, n=2, axis=0), axis=-1
    )
    joint_confidence = np.min(observed[:, (6, 8, 10)], axis=1)
    speed_confidence = np.minimum(joint_confidence[:-1], joint_confidence[1:])
    acceleration_confidence = np.minimum(
        np.minimum(joint_confidence[:-2], joint_confidence[1:-1]),
        joint_confidence[2:],
    )

    def summarize(
        signal: NDArray[np.floating],
        signal_confidence: NDArray[np.floating],
        left: int,
        right: int,
    ) -> tuple[float, float]:
        selected = np.asarray(signal, dtype=np.float64)[left:right]
        selected_confidence = np.asarray(signal_confidence, dtype=np.float64)[
            left:right
        ]
        valid = np.isfinite(selected) & (selected_confidence >= 0.20)
        if not np.any(valid):
            valid = np.isfinite(selected)
        available = selected[valid]
        if not len(available):
            return 0.0, 0.0
        return float(np.mean(available)), float(np.quantile(available, 0.90))

    shoulder_mean, shoulder_peak = summarize(
        shoulder_speed, speed_confidence, start, end - 1
    )
    forearm_mean, forearm_peak = summarize(
        forearm_speed, speed_confidence, start, end - 1
    )
    acceleration_mean, acceleration_peak = summarize(
        acceleration, acceleration_confidence, start, end - 2
    )
    return np.asarray(
        (
            shoulder_mean,
            shoulder_peak,
            forearm_mean,
            forearm_peak,
            acceleration_mean,
            acceleration_peak,
        ),
        dtype=np.float64,
    )


def _serve_wrist_action_components(
    source_pose: NDArray[np.floating],
    target_pose: NDArray[np.floating],
    confidence: NDArray[np.floating],
    *,
    start: int,
    end: int,
) -> dict[str, float]:
    source = _serve_wrist_motion_features(
        source_pose, confidence, start=start, end=end
    )
    target = _serve_wrist_motion_features(
        target_pose, confidence, start=start, end=end
    )
    delta = source - target
    # Sustained wrist-to-shoulder speed is weighted above isolated peaks,
    # which are easily inflated by detector jitter. The remaining distal-arm
    # and acceleration summaries verify that the movement is coordinated.
    weights = np.asarray((3.0, 0.25, 0.25, 0.25, 0.25, 0.25))
    distance = float(np.sqrt(np.sum(weights * delta**2) / np.sum(weights)))
    return {
        "euclidean_distance": distance,
        "target_angle_distance": 0.0,
        "combined_distance": distance,
        "source_wrist_speed_mean": float(source[0]),
        "target_wrist_speed_mean": float(target[0]),
        "source_wrist_speed_p90": float(source[1]),
        "target_wrist_speed_p90": float(target[1]),
        "source_forearm_speed_p90": float(source[3]),
        "target_forearm_speed_p90": float(target[3]),
        "source_wrist_acceleration_p90": float(source[5]),
        "target_wrist_acceleration_p90": float(target[5]),
    }


def _serve_semantic_evidence(
    rule_id: str,
    pose: NDArray[np.floating],
    root: NDArray[np.floating],
    confidence: NDArray[np.floating],
) -> tuple[NDArray[np.float64], tuple[str, ...], NDArray[np.float64]]:
    """Return higher-is-better evidence supported by every training expert."""
    if rule_id == "arms_raised":
        evidence = _serve_qualitative_pose_evidence(pose, root)
        return (
            np.asarray(
                (evidence["simultaneous_arm_elevation"],), dtype=np.float64
            ),
            ("simultaneous_arm_elevation",),
            np.asarray((1.0,), dtype=np.float64),
        )
    if rule_id == "racket_foot_weight":
        values = np.asarray(pose, dtype=np.float64)
        observed = np.asarray(confidence, dtype=np.float64)
        start, end = motion_completion_bounds(len(values), 0.125, 0.34375)
        hip_center = 0.5 * (values[:, 11] + values[:, 12])
        shoulder_center = 0.5 * (values[:, 5] + values[:, 6])
        torso = np.maximum(
            np.linalg.norm(shoulder_center - hip_center, axis=-1), _EPS
        )
        ankle_axis = values[:, 16] - values[:, 15]
        ankle_squared = np.maximum(
            np.sum(ankle_axis * ankle_axis, axis=-1), _EPS
        )
        # Canonical joint 16 is the racket-side ankle. A value near one means
        # the pelvis begins above that side; the stance term prevents a
        # feet-together pose from earning this checkpoint through an unstable
        # near-zero projection denominator.
        racket_side_loading = np.sum(
            (hip_center - values[:, 15]) * ankle_axis, axis=-1
        ) / ankle_squared
        stance_width = np.linalg.norm(ankle_axis, axis=-1) / torso
        loading_confidence = np.min(
            observed[:, (11, 12, 15, 16)], axis=1
        )
        return (
            np.asarray(
                (
                    _robust_window_value(
                        racket_side_loading,
                        loading_confidence,
                        start=start,
                        end=end,
                    ),
                    _robust_window_value(
                        stance_width,
                        loading_confidence,
                        start=start,
                        end=end,
                    ),
                ),
                dtype=np.float64,
            ),
            ("preparation_racket_side_loading", "preparation_stance_width"),
            np.asarray((1.5, 0.75), dtype=np.float64),
        )
    if rule_id == "weight_transfer":
        motion = _serve_qualitative_pose_evidence(pose, root)
        return (
            np.asarray(
                (
                    motion["dominant_chain_excursion"],
                    motion["dominant_chain_change"],
                    motion["pelvis_loading_shift"],
                    motion["root_transfer_distance"],
                    motion["coordinated_hip_rotation"],
                ),
                dtype=np.float64,
            ),
            (
                "dominant_chain_excursion",
                "dominant_chain_completion_change",
                "pelvis_loading_shift",
                "root_transfer_distance",
                "coordinated_hip_rotation",
            ),
            # The two dominant-side joint-angle summaries are invariant to
            # translation, scale, and the ankle-spine in-plane rotation and
            # therefore carry most of the decision. Pelvis loading remains a
            # required supporting cue: limb articulation without a change in
            # loading is not body-weight transfer. Root translation stays
            # outside the weighted distance because monocular perspective and
            # camera motion can suppress or exaggerate it; the camera-robust
            # aggregation below uses it only as an alternative supporting cue
            # when the dominant-side joint chain independently agrees.
            np.asarray((2.5, 1.5, 0.5, 0.0, 0.0), dtype=np.float64),
        )
    if rule_id == "hip_rotation":
        rotation = _serve_projected_rotation_features(pose, confidence)
        return (
            np.asarray(
                (
                    -rotation[0],
                    -rotation[1],
                    abs(rotation[2]),
                    abs(rotation[3]),
                ),
                dtype=np.float64,
            ),
            (
                "projected_hip_contraction",
                "projected_shoulder_contraction",
                "projected_hip_axis_rotation",
                "projected_torso_twist",
            ),
            np.ones(4, dtype=np.float64),
        )
    if rule_id == "wrist_flick":
        # Measure a tempo-invariant contact impulse. Absolute per-normalized-
        # frame speed made a slower complete serve look incorrect and a short
        # detector twitch look powerful. The displacement/acceleration
        # geometric mean accepts either a compact sharp wrist action or a
        # broader forearm-led action only when it creates a coherent contact
        # event relative to the performer's own torso and motion baseline.
        start, end = motion_completion_bounds(len(pose), 0.375, 0.625)
        values = np.asarray(pose, dtype=np.float64)
        shoulder_center = 0.5 * (values[:, 5] + values[:, 6])
        hip_center = 0.5 * (values[:, 11] + values[:, 12])
        torso_scale = max(
            float(
                np.median(
                    np.linalg.norm(shoulder_center - hip_center, axis=-1)
                )
            ),
            _EPS,
        )
        relative_wrist = _smooth_trajectory(values[:, 10] - values[:, 6])
        acceleration = np.linalg.norm(
            np.diff(relative_wrist, n=2, axis=0), axis=-1
        )
        event_displacement = float(
            np.linalg.norm(relative_wrist[end - 1] - relative_wrist[start])
            / torso_scale
        )
        event_acceleration = float(
            np.quantile(acceleration[start : max(start + 1, end - 2)], 0.90)
        )
        baseline_acceleration = max(
            float(np.quantile(acceleration, 0.50)), 1e-4
        )
        acceleration_prominence = event_acceleration / baseline_acceleration
        contact_impulse = float(
            np.sqrt(max(event_displacement * acceleration_prominence, 0.0))
        )
        forward_axis = np.median(
            values[start:end, 5] - values[start:end, 6], axis=0
        )
        forward_axis /= max(float(np.linalg.norm(forward_axis)), _EPS)
        projected_acceleration = (
            np.diff(relative_wrist, n=2, axis=0) @ forward_axis
        )[start : max(start + 1, end - 2)]
        directional_acceleration_ratio = float(
            np.sum(np.maximum(projected_acceleration, 0.0))
            / max(float(np.sum(np.abs(projected_acceleration))), _EPS)
        )
        return (
            np.asarray(
                (contact_impulse, directional_acceleration_ratio),
                dtype=np.float64,
            ),
            (
                "tempo_invariant_contact_impulse",
                "forward_acceleration_coherence",
            ),
            np.asarray((1.0, 1.0), dtype=np.float64),
        )
    if rule_id == "shoulder_rotation":
        values = np.asarray(pose, dtype=np.float64)
        observed = np.asarray(confidence, dtype=np.float64)
        shoulder_center = 0.5 * (values[:, 5] + values[:, 6])
        hip_center = 0.5 * (values[:, 11] + values[:, 12])
        preparation_start, preparation_end = motion_completion_bounds(
            len(values), 0.125, 0.34375
        )
        torso = np.linalg.norm(
            shoulder_center[preparation_start:preparation_end]
            - hip_center[preparation_start:preparation_end],
            axis=-1,
        )
        torso_scale = max(float(np.median(torso[np.isfinite(torso)])), _EPS)
        completion_start, completion_end = motion_completion_bounds(
            len(values), 0.875, 1.0
        )
        confidence_mask = np.min(observed[:, (5, 6, 8, 10)], axis=1)
        forearm_offset = (
            0.75 * values[:, 8, 0]
            + 0.25 * values[:, 10, 0]
            - shoulder_center[:, 0]
        ) / torso_scale
        elbow_drop = (values[:, 8, 1] - shoulder_center[:, 1]) / torso_scale
        wrist_drop = (values[:, 10, 1] - shoulder_center[:, 1]) / torso_scale
        rotation = _serve_projected_rotation_features(values, observed)
        return (
            np.asarray(
                (
                    -rotation[1],
                    -_robust_window_value(
                        forearm_offset,
                        confidence_mask,
                        start=completion_start,
                        end=completion_end,
                    ),
                    _robust_window_value(
                        elbow_drop,
                        confidence_mask,
                        start=completion_start,
                        end=completion_end,
                    ),
                    _robust_window_value(
                        wrist_drop,
                        confidence_mask,
                        start=completion_start,
                        end=completion_end,
                    ),
                ),
                dtype=np.float64,
            ),
            (
                "terminal_shoulder_contraction",
                "terminal_cross_body_reach",
                "terminal_elbow_drop",
                "terminal_wrist_drop",
            ),
            # The checkpoint is shoulder-forward rotation. Elbow/wrist height
            # remains diagnostic but must not turn a valid high or low
            # follow-through style into a shoulder failure.
            np.asarray((1.5, 0.0, 0.0, 0.0), dtype=np.float64),
        )
    raise KeyError(f"serve rule has no semantic expert evidence: {rule_id}")


def _serve_expert_envelope(
    model: ExpertPhaseModel,
    *,
    allowed_indices: NDArray[np.integer] | None = None,
) -> dict[str, dict[str, Any]]:
    """Fit subject-balanced checkpoint patterns from expert trajectories."""
    selected = (
        np.arange(len(model.expert_pose), dtype=np.int64)
        if allowed_indices is None
        else np.asarray(allowed_indices, dtype=np.int64)
    )
    if not len(selected):
        raise ValueError("serve expert patterns require at least one clip")
    output: dict[str, dict[str, Any]] = {}
    for rule_id in (
        "arms_raised",
        "racket_foot_weight",
        "weight_transfer",
        "hip_rotation",
        "wrist_flick",
        "shoulder_rotation",
    ):
        evidence = []
        names: tuple[str, ...] = ()
        weights = np.empty(0, dtype=np.float64)
        for index in selected:
            pose = model.expert_pose[index]
            root = model.expert_root[index]
            confidence = model.expert_confidence[index]
            values, names, weights = _serve_semantic_evidence(
                rule_id, pose, root, confidence
            )
            evidence.append(values)
        matrix = np.stack(evidence)
        selected_subjects = model.expert_subject_ids[selected]
        subject_ids = sorted(set(selected_subjects.tolist()))
        subject_values = np.stack(
            [
                np.median(
                    matrix[selected_subjects == subject_id], axis=0
                )
                for subject_id in subject_ids
            ]
        )
        clip_median = np.median(matrix, axis=0)
        within_take_scale = 1.4826 * np.median(
            np.abs(matrix - clip_median[None]), axis=0
        )
        # The deployed expert distribution must retain every demonstrated
        # expert identity, including legitimate camera/style extremes. A 10th
        # percentile with only seven identities put the lowest valid identity
        # outside its own acceptance boundary. Extend the minimum
        # subject-median by detector within-take noise; student clips remain
        # completely absent from this boundary.
        if rule_id == "wrist_flick":
            lower = np.quantile(subject_values, 0.10, axis=0)
        elif rule_id == "racket_foot_weight":
            # Preparation pose varies substantially between repeated takes
            # from the same expert.  Every observed expert take is a valid
            # preparation example, so retain the full expert-only support.
            lower = np.min(matrix, axis=0) - within_take_scale
        else:
            # Dynamic checkpoints use identity medians so a single occluded
            # or truncated take cannot erase the required motion pattern.
            lower = np.min(subject_values, axis=0) - within_take_scale
        median = np.median(subject_values, axis=0)
        scale = np.maximum.reduce(
            (
                median - lower,
                within_take_scale,
                0.10 * np.maximum(lower, 0.0),
                np.full_like(lower, 1e-3),
            )
        )
        output[rule_id] = {
            "feature_names": names,
            "lower_envelope": lower,
            "feature_scale": scale,
            "feature_weights": weights,
            "subject_values": subject_values,
            "subject_ids": np.asarray(subject_ids),
        }
    return output


def _serve_expert_envelope_components(
    rule_id: str,
    pose: NDArray[np.floating],
    root: NDArray[np.floating],
    confidence: NDArray[np.floating],
    envelope: dict[str, dict[str, Any]],
) -> dict[str, float]:
    evidence, names, _ = _serve_semantic_evidence(
        rule_id, pose, root, confidence
    )
    calibration = envelope[rule_id]
    lower = np.asarray(calibration["lower_envelope"], dtype=np.float64)
    scale = np.asarray(calibration["feature_scale"], dtype=np.float64)
    weights = np.asarray(calibration["feature_weights"], dtype=np.float64)
    deficiency = np.maximum(lower - evidence, 0.0) / scale
    positive_weight = float(np.sum(weights))
    if positive_weight <= _EPS:
        raise ValueError(f"{rule_id} expert evidence has no positive weight")
    strict_required_cue_distance = float(
        np.sqrt(np.sum(weights * deficiency**2) / positive_weight)
    )
    aggregation = "weighted_required_cues"
    if rule_id == "wrist_flick":
        # Compact impulse and forward-coherent acceleration are alternative
        # expert wrist-action styles.
        distance = float(np.min(deficiency[weights > 0.0]))
        aggregation = "either_impulse_or_directional_acceleration"
    elif rule_id == "weight_transfer":
        # The dominant shoulder-hip-knee and hip-knee-ankle chain is the
        # camera-robust primary evidence. Pelvis, root, and coordinated hip
        # motion are alternative supporting views because any one can become
        # unreliable under occlusion or a different camera azimuth. Keep the
        # stricter all-cues distance in the diagnostics; the runtime rubric
        # attribution uses it to avoid assigning the full 30 points when the
        # aggregate model passes on root translation alone.
        chain_weights = weights[:2]
        chain_distance = float(
            np.sqrt(
                np.sum(chain_weights * deficiency[:2] ** 2)
                / max(float(np.sum(chain_weights)), _EPS)
            )
        )
        support_distance = float(np.min(deficiency[2:5]))
        distance = float(max(chain_distance, support_distance))
        aggregation = "dominant_chain_with_pelvis_or_root_support"
    elif rule_id == "hip_rotation":
        # Axial rotation changes apparent hip width, shoulder width, hip-axis
        # direction, and torso twist differently with camera azimuth. Require
        # one contraction cue and one orientation cue, while allowing either
        # member of each group to carry the evidence. The score-level motion
        # completeness gate below prevents an isolated 2D cue from passing an
        # otherwise incomplete serve.
        contraction = float(np.min(deficiency[:2]))
        orientation = float(np.min(deficiency[2:]))
        distance = float(max(contraction, orientation))
        aggregation = "contraction_and_orientation_camera_robust"
    elif rule_id == "shoulder_rotation":
        # Forward shoulder rotation is a depth movement and shoulder-width
        # contraction is weak from frontal views. A coordinated cross-body
        # forearm/elbow/wrist completion is the observable alternative, but all
        # three arm cues must agree so a single noisy distal joint cannot pass.
        shoulder_depth_proxy = float(deficiency[0])
        arm_completion = float(np.sqrt(np.mean(deficiency[1:4] ** 2)))
        distance = float(min(shoulder_depth_proxy, arm_completion))
        aggregation = "shoulder_contraction_or_cross_body_completion"
    else:
        # Every positively weighted feature is a necessary higher-is-better
        # cue for the remaining checkpoints.
        distance = strict_required_cue_distance
    components: dict[str, float] = {
        "euclidean_distance": distance,
        "target_angle_distance": 0.0,
        "combined_distance": distance,
        "expert_pattern_distance": distance,
        "matched_expert_subject": "subject_balanced_lower_envelope",
        "semantic_cue_aggregation": aggregation,
        "strict_required_cue_distance": strict_required_cue_distance,
    }
    for name, value, target, feature_scale, shortfall in zip(
        names,
        evidence,
        lower,
        scale,
        deficiency,
        strict=True,
    ):
        components[f"source_{name}"] = float(value)
        components[f"expert_lower_{name}"] = float(target)
        components[f"expert_scale_{name}"] = float(feature_scale)
        components[f"standardized_shortfall_{name}"] = float(shortfall)
    return components


def _criterion_components_for_spec(
    spec: SkillCorrectionSpec,
    source_pose: NDArray[np.float32],
    source_root: NDArray[np.float32],
    target_pose: NDArray[np.float32],
    target_root: NDArray[np.float32],
    confidence: NDArray[np.float32],
    *,
    serve_expert_envelope: dict[str, dict[str, Any]] | None = None,
) -> list[dict[str, float]]:
    frame_count = len(source_pose)
    if not (
        len(source_root)
        == len(target_pose)
        == len(target_root)
        == len(confidence)
        == frame_count
    ):
        raise ValueError("criterion inputs must use the same motion length")
    output = []
    for detail, rule in zip(spec.details, spec.rules, strict=True):
        joints = detail.joints or rule.measured_joints
        source = source_pose
        target = target_pose
        start, end = detail.bounds(frame_count)
        if (
            spec.slug == "serve"
            and serve_expert_envelope is not None
            and rule.id in serve_expert_envelope
        ):
            components = _serve_expert_envelope_components(
                rule.id,
                source_pose,
                source_root,
                confidence,
                serve_expert_envelope,
            )
        elif spec.slug == "serve" and rule.id == "weight_transfer":
            components = _serve_weight_transfer_components(
                source_pose,
                source_root,
                target_pose,
                target_root,
                confidence,
            )
        elif spec.slug == "serve" and rule.id == "hip_rotation":
            components = _serve_hip_rotation_components(
                source_pose, target_pose, confidence
            )
        elif spec.slug == "serve" and rule.id == "wrist_flick":
            components = _serve_wrist_action_components(
                source_pose,
                target_pose,
                confidence,
                start=start,
                end=end,
            )
        else:
            components = criterion_distance_components(
                source,
                target,
                confidence,
                start=start,
                end=end,
                joints=joints,
                joint_weights=spec.joint_weights_array,
            )
        if (
            detail.metric == "serve_follow_through_cross_body"
            and not (
                spec.slug == "serve"
                and serve_expert_envelope is not None
                and rule.id in serve_expert_envelope
            )
        ):
            # Endpoint detection is noisy, especially across pose backends.
            # Evaluate the last 12.5% of motion instead of making one selected
            # frame responsible for the entire 20-point checkpoint.
            completion_start, completion_end = motion_completion_bounds(
                frame_count, 0.875, 1.0
            )
            terminal = criterion_distance_components(
                source,
                target,
                confidence,
                start=completion_start,
                end=completion_end,
                joints=joints,
                joint_weights=spec.joint_weights_array,
            )
            required = (5, 6, 8, 10)
            terminal_confidence = confidence[completion_start:completion_end]
            valid = np.prod(terminal_confidence[:, list(required)], axis=1) > 0.2
            if np.any(valid):
                source_frames = source[completion_start:completion_end]
                target_frames = target[completion_start:completion_end]
                source_shoulder_center = 0.5 * (
                    source_frames[:, 5] + source_frames[:, 6]
                )
                target_shoulder_center = 0.5 * (
                    target_frames[:, 5] + target_frames[:, 6]
                )
                source_forearm_offset = (
                    0.75 * source_frames[:, 8, 0]
                    + 0.25 * source_frames[:, 10, 0]
                    - source_shoulder_center[:, 0]
                )
                target_forearm_offset = (
                    0.75 * target_frames[:, 8, 0]
                    + 0.25 * target_frames[:, 10, 0]
                    - target_shoulder_center[:, 0]
                )
                cross_body_deficiency = float(
                    np.median(
                        np.maximum(
                            source_forearm_offset[valid]
                            - target_forearm_offset[valid],
                            0.0,
                        )
                    )
                )
            else:
                cross_body_deficiency = 0.0
            terminal_angle = float(terminal["target_angle_distance"])
            terminal_euclidean = float(terminal["euclidean_distance"])
            components = {
                "euclidean_distance": terminal_euclidean,
                "target_angle_distance": terminal_angle,
                "combined_distance": max(
                    cross_body_deficiency,
                    terminal_euclidean + 0.5 * terminal_angle,
                ),
                "window_euclidean_distance": float(
                    components["euclidean_distance"]
                ),
                "window_target_angle_distance": float(
                    components["target_angle_distance"]
                ),
                "completion_start_fraction": 0.875,
                "completion_end_fraction": 1.0,
                "cross_body_deficiency": cross_body_deficiency,
            }
        output.append(components)
    return output


def train_expert_phase_model(
    samples: Sequence[MotionSample],
    *,
    skill: str,
    top_k: int = 5,
) -> tuple[ExpertPhaseModel, dict[str, Any]]:
    """Fit an expert-only local manifold and identity-held-out tolerances.

    Current archives carry a real ``subject_id``.  Legacy archives fall back
    to one identity per archive, which is recorded in the report rather than
    being overstated as a subject-disjoint evaluation.
    """
    if top_k < 1:
        raise ValueError("top_k must be positive")
    if skill == "serve" and top_k != 1:
        raise ValueError(
            "serve full-body correction requires one coherent expert trajectory"
        )
    if not samples:
        raise ValueError("expert training requires at least one sample")
    if any(sample.skill != skill for sample in samples):
        raise ValueError("all training samples must match the requested skill")
    spec = get_skill_spec(skill)
    aligned = [_aligned(sample) for sample in samples]
    poses = np.stack([item[0] for item in aligned]).astype(np.float32)
    confidence = np.stack([item[1] for item in aligned]).astype(np.float32)
    roots = np.stack([item[2] for item in aligned]).astype(np.float32)
    contacts = np.stack(
        [
            phase_align_sequence(sample.foot_contacts, sample.phase_indices)
            for sample in samples
        ]
    ).astype(np.float32)
    raw_features = np.stack(
        [stance_feature(pose, conf) for pose, conf, _ in aligned]
    )
    features, feature_mean, feature_scale = _standardize_features(raw_features)
    empty_tolerances = np.zeros(len(spec.rules), dtype=np.float32)
    empty_scales = np.ones(len(spec.rules), dtype=np.float32)
    provisional = ExpertPhaseModel(
        skill=skill,
        expert_pose=poses,
        expert_confidence=confidence,
        expert_root=roots,
        expert_foot_contacts=contacts,
        expert_features=features,
        feature_mean=feature_mean,
        feature_scale=feature_scale,
        expert_handedness=np.asarray([sample.handedness for sample in samples]),
        expert_files=np.asarray([sample.path.name for sample in samples]),
        expert_subject_ids=np.asarray([sample.subject_id for sample in samples]),
        expert_identity_levels=np.asarray(
            [sample.identity_level for sample in samples]
        ),
        expert_alignment_contracts=np.asarray(
            [sample.alignment_contract for sample in samples]
        ),
        criterion_ids=np.asarray([rule.id for rule in spec.rules]),
        criterion_tolerances=empty_tolerances,
        criterion_scales=empty_scales,
        top_k=top_k,
        criterion_metric_version=(
            "serve_expert_distribution_v6"
            if skill == "serve"
            else "generic_joint_distance_v1"
        ),
    )
    if skill == "serve" and not provisional.has_global_root_motion:
        raise ValueError(
            "serve expert training requires current-schema global root motion; "
            "legacy pelvis-centred archives cannot supervise full-body correction"
        )
    serve_envelope = (
        _serve_expert_envelope(provisional) if skill == "serve" else None
    )

    fold_rows: list[dict[str, Any]] = []
    fold_distances: list[list[float]] = []
    subject_ids = provisional.expert_subject_ids
    for index, sample in enumerate(samples):
        allowed = np.flatnonzero(subject_ids != sample.subject_id)
        if not len(allowed):
            continue
        prediction = _predict_aligned(
            provisional,
            poses[index],
            confidence[index],
            roots[index],
            sample.handedness,
            allowed_indices=allowed,
        )
        corrected, corrected_root = prediction[:2]
        # Serve scoring is a separate one-class expert distribution model.
        # The generator remains responsible for producing the visualization,
        # but a valid student is not penalized merely because a diffusion
        # sample chose a different expert style.  Each calibration row is
        # evaluated against an envelope that excludes its entire identity.
        fold_envelope = (
            _serve_expert_envelope(provisional, allowed_indices=allowed)
            if skill == "serve"
            else None
        )
        components = _criterion_components_for_spec(
            spec,
            poses[index],
            roots[index],
            poses[index] if skill == "serve" else corrected,
            roots[index] if skill == "serve" else corrected_root,
            confidence[index],
            serve_expert_envelope=fold_envelope,
        )
        distances = [item["combined_distance"] for item in components]
        fold_distances.append(distances)
        fold_rows.append(
            {
                "file": sample.path.name,
                "subject_id": sample.subject_id,
                "identity_level": sample.identity_level,
                "references": [
                    provisional.expert_files[value] for value in prediction[4]
                ],
                "criteria": {
                    rule.id: component
                    for rule, component in zip(spec.rules, components, strict=True)
                },
            }
        )
    if fold_distances:
        matrix = np.asarray(fold_distances, dtype=np.float64)
        tolerances = np.quantile(matrix, 0.90, axis=0)
        median = np.median(matrix, axis=0)
        mad = np.median(np.abs(matrix - median[None]), axis=0)
        scales = np.maximum(1.4826 * mad, np.maximum(0.10 * tolerances, 1e-3))
        if skill == "serve":
            # Preparation stance and wrist impulse retain the central expert
            # tolerance because their extreme held-out residuals can be
            # inflated by monocular foot/wrist localization. The remaining
            # qualitative motion checkpoints must cover every valid
            # identity-held-out expert residual. This avoids falsely grading
            # a demonstrated camera/style variant as an error while keeping
            # the boundary entirely expert-only.
            pattern_ids = set(str(value) for value in provisional.criterion_ids)
            p75 = np.quantile(matrix, 0.75, axis=0)
            for criterion_index, rule in enumerate(spec.rules):
                if rule.id not in pattern_ids:
                    continue
                # Distances are already expressed in robust expert feature
                # scales. Use the central identity-held-out expert range as a
                # no-penalty uncertainty margin, then decay over a fixed
                # fraction of one standardized unit. Using the p90-p75 spread
                # as the decay scale made heterogeneous camera identities
                # produce enormous scales and let motions missing an entire
                # checkpoint retain nearly full credit.
                tolerance_quantile = (
                    p75[criterion_index]
                    if rule.id in {"racket_foot_weight", "wrist_flick"}
                    else np.max(matrix[:, criterion_index])
                )
                tolerances[criterion_index] = tolerance_quantile
                scales[criterion_index] = max(
                    float(0.25 * tolerance_quantile),
                    0.10,
                )
    else:
        tolerances = np.full(len(spec.rules), 0.05, dtype=np.float64)
        scales = np.full(len(spec.rules), 0.01, dtype=np.float64)
    model = ExpertPhaseModel(
        **{
            **provisional.__dict__,
            "criterion_tolerances": tolerances.astype(np.float32),
            "criterion_scales": scales.astype(np.float32),
        }
    )
    report = {
        "method": "expert_only_stance_conditioned_phase_manifold_baseline_v1",
        "skill": skill,
        "expert_samples": len(samples),
        "expert_subjects": len(set(subject_ids.tolist())),
        "identity_levels": sorted(set(sample.identity_level for sample in samples)),
        "alignment_contracts": sorted(
            set(sample.alignment_contract for sample in samples)
        ),
        "handedness_counts": {
            value: int(np.sum(model.expert_handedness == value))
            for value in ("right", "left")
        },
        "top_k": top_k,
        "phase_indices": CANONICAL_PHASE_INDICES.tolist(),
        "criterion_metric_version": model.criterion_metric_version,
        "serve_expert_envelope": (
            {
                rule_id: {
                    "feature_names": list(values["feature_names"]),
                    "lower_envelope": np.asarray(
                        values["lower_envelope"]
                    ).tolist(),
                    "feature_scale": np.asarray(
                        values["feature_scale"]
                    ).tolist(),
                    "feature_weights": np.asarray(
                        values["feature_weights"]
                    ).tolist(),
                    "subject_values": np.asarray(
                        values["subject_values"]
                    ).tolist(),
                }
                for rule_id, values in serve_envelope.items()
            }
            if serve_envelope is not None
            else None
        ),
        "criterion_tolerances": {
            rule.id: {
                "combined_distance_p90": float(tolerance),
                "robust_scale": float(scale),
            }
            for rule, tolerance, scale in zip(
                spec.rules,
                model.criterion_tolerances,
                model.criterion_scales,
                strict=True,
            )
        },
        "held_out_expert_folds": fold_rows,
    }
    return model, report


def correct_student_motion(
    model: ExpertPhaseModel, sample: MotionSample
) -> ExpertCorrection:
    if sample.skill != model.skill:
        raise ValueError(
            f"student skill {sample.skill!r} does not match model {model.skill!r}"
        )
    aligned_pose, aligned_confidence, aligned_root = _aligned(sample)
    prediction = _predict_aligned(
        model,
        aligned_pose,
        aligned_confidence,
        aligned_root,
        sample.handedness,
    )
    (
        corrected,
        corrected_root,
        prototype,
        prototype_root,
        indices,
        weights,
        distances,
        contacts,
    ) = prediction
    return ExpertCorrection(
        student=sample,
        aligned_student_pose=aligned_pose,
        aligned_student_root=aligned_root,
        aligned_corrected_pose=corrected,
        aligned_corrected_root=corrected_root,
        corrected_pose=restore_phase_timing(corrected, sample.phase_indices),
        corrected_root=restore_phase_timing(corrected_root, sample.phase_indices),
        aligned_corrected_contacts=contacts,
        corrected_contacts=restore_phase_timing(contacts, sample.phase_indices),
        expert_prototype_pose=prototype,
        expert_prototype_root=prototype_root,
        reference_indices=indices,
        reference_weights=weights,
        reference_distances=distances,
    )


def _aggregate_qualitative_checkpoint_ratios(
    ratios: np.ndarray,
    *,
    power: float = 1.0 / 3.0,
    floor: float = 1e-3,
) -> float:
    """Aggregate equally important checkpoints as a fixed soft conjunction."""

    values = np.asarray(ratios, dtype=np.float64)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("checkpoint ratios must be a non-empty vector")
    if power <= 0.0:
        raise ValueError("generalized-mean power must be positive")
    bounded = np.maximum(np.clip(values, 0.0, 1.0), floor)
    return float(np.mean(bounded**power) ** (1.0 / power))


def _serve_checklist_aggregation(
    criteria: list[dict[str, Any]],
) -> tuple[float, str, float, float, bool]:
    """Aggregate serve checkpoints without validation-video exceptions.

    Incomplete movements use a fixed soft conjunction, so several isolated
    2D false positives cannot dominate the grade. A six-item additive
    checklist is used only when expert-derived evidence identifies one
    bounded preparation-pose deviation and every independent movement
    checkpoint is complete.
    """

    by_rule = {str(item["rule_reference"]): item for item in criteria}
    ordered_ids = (
        "arms_raised",
        "racket_foot_weight",
        "weight_transfer",
        "hip_rotation",
        "wrist_flick",
        "shoulder_rotation",
    )
    ratios = np.asarray(
        [
            float(by_rule[rule_id]["score"])
            / max(float(by_rule[rule_id]["maximum"]), _EPS)
            for rule_id in ordered_ids
        ],
        dtype=np.float64,
    )
    preparation = by_rule["arms_raised"]
    support_limit = float(preparation["expert_tolerance"]) + 2.5 * max(
        float(preparation["expert_robust_scale"]), 1e-3
    )
    isolated_preparation_deviation = bool(
        ratios[0] < 0.20
        and float(np.min(ratios[1:])) >= 0.90
        and float(preparation["generated_target_distance"])
        <= support_limit
    )
    if isolated_preparation_deviation:
        return (
            float(100.0 * np.mean(ratios)),
            "expert_supported_isolated_preparation_additive_v4",
            1.0,
            0.0,
            True,
        )
    power = 1.0 / 3.0
    floor = 1e-3
    return (
        float(
            100.0
            * _aggregate_qualitative_checkpoint_ratios(
                ratios,
                power=power,
                floor=floor,
            )
        ),
        "soft_conjunctive_qualitative_checkpoints_v4",
        power,
        floor,
        False,
    )


def _serve_corrected_residual_checklist(
    model: ExpertPhaseModel,
    correction: ExpertCorrection,
    criteria: list[dict[str, Any]],
    *,
    semantic_total: float,
    semantic_policy: str,
    semantic_power: float,
    semantic_floor: float,
    manifold_pose: NDArray[np.floating] | None = None,
) -> tuple[float, str, float, float, NDArray[np.float64], dict[str, Any]]:
    """Fuse semantic evidence with the learner-to-correction residual.

    Expert-manifold motion retains the semantic checklist so valid expert
    style differences are not penalized. Outside that support, each
    checkpoint must also agree with the generated corrected skeleton. All
    residual tolerances and the trajectory support boundary are expert-only.
    """

    tolerances = model.criterion_residual_tolerances
    scales = model.criterion_residual_scales
    semantic_ratios = np.asarray(
        [
            float(item["score"]) / max(float(item["maximum"]), _EPS)
            for item in criteria
        ],
        dtype=np.float64,
    )
    if (
        tolerances is None
        or scales is None
        or np.asarray(tolerances).shape != semantic_ratios.shape
        or np.asarray(scales).shape != semantic_ratios.shape
    ):
        return (
            semantic_total,
            semantic_policy,
            semantic_power,
            semantic_floor,
            semantic_ratios,
            {"corrected_residual_fusion_active": False},
        )

    from badminton_analysis.ml.trajectory_distance import (
        corrected_motion_distance,
        expert_residual_ratio,
        serve_angle_manifold_distance,
    )

    costs = []
    residual_ratios = []
    for criterion, rule, tolerance, scale in zip(
        model.spec.details,
        model.spec.rules,
        tolerances,
        scales,
        strict=True,
    ):
        cost = corrected_motion_distance(
            correction.aligned_student_pose,
            correction.aligned_corrected_pose,
            joints=tuple(criterion.joints or rule.measured_joints),
            start_fraction=criterion.start_fraction,
            end_fraction=criterion.end_fraction,
            method="euclidean",
        )
        costs.append(cost)
        residual_ratios.append(
            expert_residual_ratio(
                cost,
                tolerance=float(tolerance),
                scale=float(scale),
            )
        )
    residual = np.asarray(residual_ratios, dtype=np.float64)
    for item, cost, ratio in zip(criteria, costs, residual, strict=True):
        item["corrected_skeleton_euclidean_cost"] = float(cost)
        item["corrected_skeleton_residual_ratio"] = float(ratio)

    manifold = model.serve_angle_manifold
    assert manifold is not None
    manifold_distance = serve_angle_manifold_distance(
        (
            correction.aligned_student_pose
            if manifold_pose is None
            else manifold_pose
        ),
        manifold,
    )
    low = np.flatnonzero(semantic_ratios < 0.20)
    isolated_wrist_uncertainty = bool(
        len(low) == 1
        and low[0] == 4
        and float(np.min(semantic_ratios[[0, 1, 2, 3, 5]])) >= 0.40
    )
    isolated_preparation_style = bool(
        len(low) == 1
        and low[0] == 0
        and manifold_distance <= manifold.expert_q80
    )
    diagnostics: dict[str, Any] = {
        "corrected_residual_fusion_active": True,
        "trajectory_manifold_distance": manifold_distance,
        "trajectory_manifold_expert_q80": manifold.expert_q80,
        "trajectory_manifold_expert_scale": manifold.expert_scale,
        "isolated_wrist_observability_fallback": isolated_wrist_uncertainty,
        "isolated_preparation_style_fallback": isolated_preparation_style,
    }
    if isolated_wrist_uncertainty or isolated_preparation_style:
        return (
            float(100.0 * np.mean(semantic_ratios)),
            "additive_isolated_pose_observability_v1",
            1.0,
            0.0,
            semantic_ratios,
            diagnostics,
        )
    if manifold_distance <= manifold.expert_q80:
        return (
            semantic_total,
            f"expert_manifold_supported_{semantic_policy}",
            semantic_power,
            semantic_floor,
            semantic_ratios,
            diagnostics,
        )

    effective = np.minimum(semantic_ratios, residual)
    # COCO-17 cannot see true wrist flexion. Outside the expert manifold the
    # observable distal-arm residual is therefore the least-assumptive wrist
    # evidence, rather than requiring two correlated imperfect proxies.
    effective[4] = residual[4]
    power = 1.0 / 3.0
    floor = 1e-3
    checklist = float(
        100.0
        * _aggregate_qualitative_checkpoint_ratios(
            effective, power=power, floor=floor
        )
    )
    structural_indices = np.asarray((0, 1, 2, 3, 5), dtype=np.int64)
    novelty_factor = 1.0
    if float(np.min(effective[structural_indices])) < 0.20:
        standardized_novelty = max(
            0.0,
            (manifold_distance - manifold.expert_q80)
            / manifold.expert_scale,
        )
        novelty_factor = float(np.exp(-0.75 * standardized_novelty))
        checklist *= novelty_factor
    diagnostics["trajectory_novelty_factor"] = novelty_factor
    return (
        checklist,
        "corrected_euclidean_residual_outside_expert_manifold_v1",
        power,
        floor,
        effective,
        diagnostics,
    )


def score_expert_correction(
    model: ExpertPhaseModel,
    correction: ExpertCorrection,
    *,
    canonical_phase_indices: NDArray[np.integer] = CANONICAL_PHASE_INDICES,
) -> dict[str, Any]:
    spec = model.spec
    confidence = np.clip(
        phase_align_sequence(
            correction.student.confidence,
            correction.student.phase_indices,
            canonical_indices=canonical_phase_indices,
        ),
        0.0,
        1.0,
    )
    components = _criterion_components_for_spec(
        spec,
        correction.aligned_student_pose,
        correction.aligned_student_root,
        correction.aligned_corrected_pose,
        correction.aligned_corrected_root,
        confidence,
    )
    if model.criterion_metric_version in {
        "serve_subject_pattern_trajectory_v4",
        "serve_expert_distribution_v6",
    }:
        # The diffusion bundle may use skill-specific canonical anchors (the
        # current serve checkpoint uses 0/21/46/54/63), while this scorer's
        # frozen expert bank uses the grading contract 0/16/32/48/63. Absolute
        # expert evidence must be measured in the scorer's own time basis.
        semantic_pose, semantic_confidence, semantic_root = _aligned(
            correction.student
        )
        semantic_components = _criterion_components_for_spec(
            spec,
            semantic_pose,
            semantic_root,
            semantic_pose,
            semantic_root,
            semantic_confidence,
            serve_expert_envelope=_serve_expert_envelope(model),
        )
        for index, rule in enumerate(spec.rules):
            if rule.id in {
                "arms_raised",
                "racket_foot_weight",
                "weight_transfer",
                "hip_rotation",
                "wrist_flick",
                "shoulder_rotation",
            }:
                semantic = semantic_components[index]
                generated_distance = float(components[index]["combined_distance"])
                components[index] = {
                    **semantic,
                    "generated_target_distance": generated_distance,
                    "semantic_envelope_distance": float(
                        semantic["combined_distance"]
                    ),
                    "selected_expert_evidence": (
                        "expert_only_identity_distribution"
                        if model.criterion_metric_version
                        == "serve_expert_distribution_v6"
                        else "nearest_expert_subject_pattern"
                    ),
                }
    criteria = []
    qualitative_evidence: dict[str, float] | None = None
    qualitative_envelope: dict[str, dict[str, float]] | None = None
    if model.criterion_metric_version == "serve_dominant_chain_coupled_v5":
        qualitative_evidence = _serve_qualitative_pose_evidence(
            correction.aligned_student_pose,
            correction.aligned_student_root,
        )
        qualitative_envelope = _serve_expert_qualitative_envelope(model)
    for index, (rule, component) in enumerate(
        zip(spec.rules, components, strict=True)
    ):
        tolerance = float(model.criterion_tolerances[index])
        scale = float(model.criterion_scales[index])
        distance = component["combined_distance"]
        excess = max(0.0, distance - tolerance)
        ratio = float(np.exp(-excess / max(scale, 1e-3)))
        qualitative_factor = 1.0
        qualitative_diagnostics: dict[str, float | str] = {}
        if qualitative_evidence is not None and qualitative_envelope is not None:
            evidence_name = (
                "simultaneous_arm_elevation"
                if rule.id == "arms_raised"
                else (
                    "preparation_stance_width"
                    if rule.id
                    in {
                        "racket_foot_weight",
                        "shoulder_rotation",
                    }
                    else None
                )
            )
            if rule.id in {"weight_transfer", "hip_rotation"}:
                stance_factor = _serve_qualitative_factor(
                    float(qualitative_evidence["preparation_stance_width"]),
                    qualitative_envelope["preparation_stance_width"],
                )
                movement_factors = {
                    name: _serve_required_motion_factor(
                        float(qualitative_evidence[name]),
                        qualitative_envelope[name],
                    )
                    for name in (
                        "pelvis_loading_shift",
                        "dominant_chain_excursion",
                        "coordinated_hip_rotation",
                    )
                }
                # All three are prerequisites. Multiplication deliberately
                # prevents a large 2D pelvis translation from hiding absent
                # leg-chain motion or absent coordinated hip rotation.
                transfer_magnitude_factor = float(
                    np.prod(list(movement_factors.values()))
                )
                qualitative_factor = stance_factor * transfer_magnitude_factor
                qualitative_diagnostics = {
                    "qualitative_evidence": (
                        "coordinated_absolute_weight_transfer"
                    ),
                    "qualitative_evidence_factor": qualitative_factor,
                    "preparation_stance_factor": stance_factor,
                    "pelvis_loading_shift": float(
                        qualitative_evidence["pelvis_loading_shift"]
                    ),
                    "pelvis_loading_shift_factor": movement_factors[
                        "pelvis_loading_shift"
                    ],
                    "dominant_chain_change_magnitude": float(
                        qualitative_evidence["dominant_chain_change"]
                    ),
                    "dominant_chain_excursion": float(
                        qualitative_evidence["dominant_chain_excursion"]
                    ),
                    "dominant_chain_excursion_factor": movement_factors[
                        "dominant_chain_excursion"
                    ],
                    "hip_rotation_excursion": float(
                        qualitative_evidence["hip_rotation_excursion"]
                    ),
                    "transfer_rotation_correlation": float(
                        qualitative_evidence[
                            "transfer_rotation_correlation"
                        ]
                    ),
                    "coordinated_hip_rotation": float(
                        qualitative_evidence["coordinated_hip_rotation"]
                    ),
                    "coordinated_hip_rotation_factor": movement_factors[
                        "coordinated_hip_rotation"
                    ],
                    "root_transfer_distance": float(
                        qualitative_evidence["root_transfer_distance"]
                    ),
                    "transfer_magnitude_factor": transfer_magnitude_factor,
                    "qualitative_calibration_policy": (
                        "expert_identity_coordinated_transfer_envelope_only"
                    ),
                }
            elif evidence_name is not None:
                evidence_value = float(qualitative_evidence[evidence_name])
                evidence_calibration = qualitative_envelope[evidence_name]
                qualitative_factor = _serve_qualitative_factor(
                    evidence_value, evidence_calibration
                )
                qualitative_diagnostics = {
                    "qualitative_evidence": evidence_name,
                    "qualitative_evidence_value": evidence_value,
                    "qualitative_expert_lower": float(
                        evidence_calibration["expert_lower"]
                    ),
                    "qualitative_expert_scale": float(
                        evidence_calibration["expert_scale"]
                    ),
                    "qualitative_evidence_factor": qualitative_factor,
                    "qualitative_calibration_policy": (
                        "expert_identity_lower_envelope_only"
                    ),
                }
        criteria.append(
            {
                "name_zh_tw": rule.name_zh_tw,
                "rule_reference": rule.id,
                "score": rule.maximum * ratio * qualitative_factor,
                "maximum": rule.maximum,
                "expert_tolerance": tolerance,
                "expert_robust_scale": scale,
                "standardized_excess": excess / max(scale, 1e-3),
                **component,
                **qualitative_diagnostics,
            }
        )
    if model.criterion_metric_version == "serve_expert_distribution_v6":
        by_rule = {item["rule_reference"]: item for item in criteria}
        dynamic_completion_gate = min(
            float(by_rule[rule_id]["score"])
            / max(float(by_rule[rule_id]["maximum"]), _EPS)
            for rule_id in (
                "weight_transfer",
                "wrist_flick",
            )
        )
        for rule_id in ("hip_rotation", "shoulder_rotation"):
            item = by_rule[rule_id]
            if item.get("semantic_cue_aggregation") not in {
                "contraction_and_orientation_camera_robust",
                "shoulder_contraction_or_cross_body_completion",
            }:
                continue
            tolerance = float(item["expert_tolerance"])
            scale = max(float(item["expert_robust_scale"]), 1e-3)
            strict_distance = float(item["strict_required_cue_distance"])
            strict_ratio = float(
                np.exp(-max(0.0, strict_distance - tolerance) / scale)
            )
            alternative_ratio = float(item["score"]) / max(
                float(item["maximum"]), _EPS
            )
            generated_distance = float(item["generated_target_distance"])
            generated_tolerance = max(tolerance, 0.5 * scale)
            generated_ratio = float(
                np.exp(
                    -max(0.0, generated_distance - generated_tolerance)
                    / scale
                )
            )
            supported_alternative_ratio = max(
                alternative_ratio,
                generated_ratio,
            )
            selected_ratio = max(
                strict_ratio,
                supported_alternative_ratio * dynamic_completion_gate,
            )
            item["score"] = float(item["maximum"]) * selected_ratio
            item["camera_robust_alternative_ratio"] = alternative_ratio
            item["generated_agreement_ratio"] = generated_ratio
            item["generated_agreement_tolerance"] = generated_tolerance
            item["supported_camera_evidence_ratio"] = (
                supported_alternative_ratio
            )
            item["strict_required_cue_ratio"] = strict_ratio
            item["serve_motion_completeness_gate"] = (
                dynamic_completion_gate
            )
            item["motion_completeness_gate_policy"] = (
                "weight_transfer_and_wrist_dynamic_completion"
            )
            item["selected_camera_evidence_ratio"] = selected_ratio
    weighted_total = float(sum(item["score"] for item in criteria))
    criterion_ratios = np.asarray(
        [
            float(item["score"]) / max(float(item["maximum"]), _EPS)
            for item in criteria
        ],
        dtype=np.float64,
    )
    arithmetic_checklist_total = float(100.0 * np.mean(criterion_ratios))
    isolated_preparation_deviation = False
    residual_fusion_diagnostics: dict[str, Any] = {
        "corrected_residual_fusion_active": False
    }
    if model.criterion_metric_version == "serve_expert_distribution_v6":
        (
            checklist_total,
            total_aggregation,
            aggregation_power,
            aggregation_floor,
            isolated_preparation_deviation,
        ) = _serve_checklist_aggregation(criteria)
        (
            checklist_total,
            total_aggregation,
            aggregation_power,
            aggregation_floor,
            effective_checklist_ratios,
            residual_fusion_diagnostics,
        ) = _serve_corrected_residual_checklist(
            model,
            correction,
            criteria,
            semantic_total=checklist_total,
            semantic_policy=total_aggregation,
            semantic_power=aggregation_power,
            semantic_floor=aggregation_floor,
            manifold_pose=semantic_pose,
        )
        bounded = np.maximum(
            np.clip(effective_checklist_ratios, 0.0, 1.0),
            aggregation_floor,
        )
        if aggregation_power == 1.0:
            contribution_weights = effective_checklist_ratios / max(
                float(np.sum(effective_checklist_ratios)), _EPS
            )
        else:
            transformed = bounded**aggregation_power
            contribution_weights = transformed / max(
                float(np.sum(transformed)), _EPS
            )
        for index, (item, ratio) in enumerate(
            zip(criteria, criterion_ratios, strict=True)
        ):
            # Product grading retains the original qualitative rubric
            # (5/5/30/10/30/20).  The validation workbook is a different
            # construct: a six-item expert checklist scored 0--6.  Expose its
            # equal-item attribution separately instead of changing every
            # product checkpoint to 16.67 points.
            item["raw_checkpoint_ratio"] = float(ratio)
            item["effective_checklist_ratio"] = float(
                effective_checklist_ratios[index]
            )
            item["checklist_score_contribution"] = float(
                checklist_total * contribution_weights[index]
            )
            item["checklist_maximum"] = 100.0 / len(criteria)
    else:
        checklist_total = arithmetic_checklist_total
        aggregation_power = 1.0
        aggregation_floor = 0.0
        total_aggregation = "equal_qualitative_checkpoint_mean_v1"
    return {
        "filename": correction.student.video_name,
        "skill": model.skill,
        "handedness": correction.student.handedness,
        "score_method": (
            "expert_only_generated_projection_residual_v4"
            if model.criterion_metric_version
            == "expert_generated_projection_residual_v4"
            else (
                "expert_only_coordinated_transfer_coupled_v4"
                if model.criterion_metric_version
                == "serve_dominant_chain_coupled_v5"
                else (
                    (
                        "expert_only_identity_distribution_v6"
                        if model.criterion_metric_version
                        == "serve_expert_distribution_v6"
                        else "expert_only_subject_pattern_trajectory_v8"
                    )
                    if model.criterion_metric_version
                    in {
                        "serve_subject_pattern_trajectory_v4",
                        "serve_expert_distribution_v6",
                    }
                    else (
                        "expert_only_semantic_criterion_tolerance_v2"
                        if model.criterion_metric_version
                        == "serve_semantic_motion_features_v2"
                        else "expert_only_held_out_identity_tolerance_v1"
                    )
                )
            )
        ),
        "correction_policy": (
            "full_body_generated_expert_projection_energy"
            if model.criterion_metric_version
            == "expert_generated_projection_residual_v4"
            else "full_body_coherent_expert_phase_projection"
        ),
        "score_reference_policy": (
            "subject_balanced_generated_projection_residual"
            if model.criterion_metric_version
            == "expert_generated_projection_residual_v4"
            else (
                "generated_expert_dominant_chain_and_rotation_coupling"
                if model.criterion_metric_version
                == "serve_dominant_chain_coupled_v5"
                else (
                    (
                        "expert_identity_held_out_checkpoint_distribution"
                        if model.criterion_metric_version
                        == "serve_expert_distribution_v6"
                        else "nearest_subject_checkpoint_trajectory_pattern"
                    )
                    if model.criterion_metric_version
                    in {
                        "serve_subject_pattern_trajectory_v4",
                        "serve_expert_distribution_v6",
                    }
                    else "generated_correction_distance"
                )
            )
        ),
        "limitations": (
            ["wrist_action_uses_coco17_distal_arm_motion_proxy"]
            if model.criterion_metric_version in {
                "serve_subject_pattern_trajectory_v4",
                "serve_dominant_chain_coupled_v5",
                "serve_expert_distribution_v6",
            }
            else []
        ),
        # The product grade and expert-validation checklist are intentionally
        # both reported.  Criteria sum exactly to the product grade, while ICC
        # must use ``checklist_total_score`` because the human workbook counts
        # six binary qualitative items rather than weighted rubric points.
        "total_score": weighted_total,
        "weighted_total_score": weighted_total,
        "checklist_total_score": checklist_total,
        "checklist_score_0_6": checklist_total * 6.0 / 100.0,
        "arithmetic_checklist_total_score": arithmetic_checklist_total,
        "weighted_coaching_total_score": weighted_total,
        "total_aggregation": total_aggregation,
        "aggregation_power": aggregation_power,
        "aggregation_floor": aggregation_floor,
        "isolated_preparation_deviation": isolated_preparation_deviation,
        **residual_fusion_diagnostics,
        "criteria": criteria,
        "references": [
            {
                "file": str(model.expert_files[index]),
                "subject_id": str(model.expert_subject_ids[index]),
                "identity_level": str(model.expert_identity_levels[index]),
                "alignment_contract": str(
                    model.expert_alignment_contracts[index]
                ),
                "weight": float(weight),
                "stance_distance": float(distance),
            }
            for index, weight, distance in zip(
                correction.reference_indices,
                correction.reference_weights,
                correction.reference_distances,
                strict=True,
            )
        ],
    }


def save_expert_phase_model(model: ExpertPhaseModel, path: str | Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        destination,
        format_version=np.asarray(1, dtype=np.int64),
        method=np.asarray("expert_only_stance_conditioned_phase_manifold_baseline_v1"),
        skill=np.asarray(model.skill),
        expert_pose=model.expert_pose,
        expert_confidence=model.expert_confidence,
        expert_root=model.expert_root,
        expert_foot_contacts=model.expert_foot_contacts,
        expert_features=model.expert_features,
        feature_mean=model.feature_mean,
        feature_scale=model.feature_scale,
        expert_handedness=model.expert_handedness,
        expert_files=model.expert_files,
        expert_subject_ids=model.expert_subject_ids,
        expert_identity_levels=model.expert_identity_levels,
        expert_alignment_contracts=model.expert_alignment_contracts,
        criterion_ids=model.criterion_ids,
        criterion_tolerances=model.criterion_tolerances,
        criterion_scales=model.criterion_scales,
        top_k=np.asarray(model.top_k, dtype=np.int64),
        criterion_metric_version=np.asarray(model.criterion_metric_version),
        criterion_residual_tolerances=np.asarray(
            ()
            if model.criterion_residual_tolerances is None
            else model.criterion_residual_tolerances,
            dtype=np.float32,
        ),
        criterion_residual_scales=np.asarray(
            ()
            if model.criterion_residual_scales is None
            else model.criterion_residual_scales,
            dtype=np.float32,
        ),
        canonical_phase_indices=CANONICAL_PHASE_INDICES,
    )


def load_expert_phase_model(path: str | Path) -> ExpertPhaseModel:
    with np.load(path, allow_pickle=False) as archive:
        if int(archive["format_version"].item()) != 1:
            raise ValueError("unsupported expert phase model format")
        if not np.array_equal(
            archive["canonical_phase_indices"], CANONICAL_PHASE_INDICES
        ):
            raise ValueError("model canonical phases do not match this runtime")
        expert_pose = np.asarray(archive["expert_pose"], dtype=np.float32)
        expert_foot_contacts = (
            np.asarray(archive["expert_foot_contacts"], dtype=np.float32)
            if "expert_foot_contacts" in archive
            else np.zeros((*expert_pose.shape[:2], 2), dtype=np.float32)
        )
        return ExpertPhaseModel(
            skill=str(archive["skill"].item()),
            expert_pose=expert_pose,
            expert_confidence=np.asarray(
                archive["expert_confidence"], dtype=np.float32
            ),
            expert_root=np.asarray(archive["expert_root"], dtype=np.float32),
            expert_foot_contacts=expert_foot_contacts,
            expert_features=np.asarray(archive["expert_features"], dtype=np.float32),
            feature_mean=np.asarray(archive["feature_mean"], dtype=np.float32),
            feature_scale=np.asarray(archive["feature_scale"], dtype=np.float32),
            expert_handedness=np.asarray(archive["expert_handedness"]),
            expert_files=np.asarray(archive["expert_files"]),
            expert_subject_ids=np.asarray(archive["expert_subject_ids"]),
            expert_identity_levels=np.asarray(archive["expert_identity_levels"]),
            expert_alignment_contracts=np.asarray(
                archive["expert_alignment_contracts"]
            ),
            criterion_ids=np.asarray(archive["criterion_ids"]),
            criterion_tolerances=np.asarray(
                archive["criterion_tolerances"], dtype=np.float32
            ),
            criterion_scales=np.asarray(
                archive["criterion_scales"], dtype=np.float32
            ),
            top_k=int(archive["top_k"].item()),
            criterion_metric_version=_scalar_string(
                archive,
                "criterion_metric_version",
                "generic_joint_distance_v1",
            ),
            criterion_residual_tolerances=(
                np.asarray(
                    archive["criterion_residual_tolerances"], dtype=np.float32
                )
                if "criterion_residual_tolerances" in archive
                and archive["criterion_residual_tolerances"].size
                else None
            ),
            criterion_residual_scales=(
                np.asarray(
                    archive["criterion_residual_scales"], dtype=np.float32
                )
                if "criterion_residual_scales" in archive
                and archive["criterion_residual_scales"].size
                else None
            ),
        )
