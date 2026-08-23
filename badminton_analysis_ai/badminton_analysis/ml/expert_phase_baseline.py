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

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

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
from badminton_analysis.ml.skill_specs import SkillCorrectionSpec, get_skill_spec
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

    @property
    def spec(self) -> SkillCorrectionSpec:
        return get_skill_spec(self.skill)

    @property
    def has_global_root_motion(self) -> bool:
        deltas = self.expert_root - self.expert_root[:, :1]
        return bool(float(np.max(np.abs(deltas))) > 1e-7)


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


def _serve_balance_features(
    pose: NDArray[np.floating],
    root: NDArray[np.floating],
    confidence: NDArray[np.floating],
) -> NDArray[np.float64]:
    """Measure body-centre transfer relative to the stance, not camera motion."""
    world = np.asarray(pose, dtype=np.float64) + np.asarray(
        root, dtype=np.float64
    )[:, None]
    observed = np.asarray(confidence, dtype=np.float64)
    pelvis = 0.5 * (world[:, 11] + world[:, 12])
    shoulders = 0.5 * (world[:, 5] + world[:, 6])
    body_centre = 0.55 * pelvis + 0.45 * shoulders
    foot_centre = 0.5 * (world[:, 15] + world[:, 16])
    stance_width = np.maximum(
        np.linalg.norm(world[:, 16] - world[:, 15], axis=-1), 0.15
    )
    balance = (body_centre[:, 0] - foot_centre[:, 0]) / stance_width
    balance_confidence = np.min(observed[:, (5, 6, 11, 12, 15, 16)], axis=1)
    preparation = _robust_window_value(
        balance, balance_confidence, start=8, end=22
    )
    completion = _robust_window_value(
        balance, balance_confidence, start=46, end=64
    )
    return np.asarray(
        (preparation, completion, preparation - completion), dtype=np.float64
    )


def _serve_weight_transfer_components(
    source_pose: NDArray[np.floating],
    source_root: NDArray[np.floating],
    target_pose: NDArray[np.floating],
    target_root: NDArray[np.floating],
    confidence: NDArray[np.floating],
) -> dict[str, float]:
    source = _serve_balance_features(source_pose, source_root, confidence)
    target = _serve_balance_features(target_pose, target_root, confidence)
    transfer_scale = max(abs(float(target[2])), 0.15)
    deficiency = np.asarray(
        (
            abs(float(source[0] - target[0])) / transfer_scale,
            max(float(source[1] - target[1]), 0.0) / transfer_scale,
            max(float(target[2] - source[2]), 0.0) / transfer_scale,
        ),
        dtype=np.float64,
    )
    # Terminal balance and the total transfer are the actual coaching target;
    # preparation balance is retained at lower weight to distinguish a real
    # transfer from two equally displaced static poses.
    weights = np.asarray((0.25, 1.0, 1.5), dtype=np.float64)
    distance = float(
        np.sqrt(np.sum(weights * deficiency**2) / np.sum(weights))
    )
    return {
        "euclidean_distance": distance,
        "target_angle_distance": 0.0,
        "combined_distance": distance,
        "source_preparation_balance": float(source[0]),
        "target_preparation_balance": float(target[0]),
        "source_completion_balance": float(source[1]),
        "target_completion_balance": float(target[1]),
        "source_weight_transfer": float(source[2]),
        "target_weight_transfer": float(target[2]),
    }


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

    prep_hip_width = window(hip_width, hip_confidence, 8, 22)
    end_hip_width = window(hip_width, hip_confidence, 46, 64)
    prep_shoulder_width = window(shoulder_width, shoulder_confidence, 8, 22)
    end_shoulder_width = window(shoulder_width, shoulder_confidence, 46, 64)
    prep_hip_angle = window(hip_angle, hip_confidence, 8, 22)
    end_hip_angle = window(hip_angle, hip_confidence, 46, 64)
    torso_twist = np.unwrap(shoulder_angle - hip_angle)
    prep_twist = window(torso_twist, torso_confidence, 8, 22)
    end_twist = window(torso_twist, torso_confidence, 46, 64)
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
    source = _serve_projected_rotation_features(source_pose, confidence)
    target = _serve_projected_rotation_features(target_pose, confidence)
    # Projected width contracts as the torso rotates away from a front-facing
    # view. A learner is deficient only when the generated expert target has
    # more contraction/rotation evidence; experts with a stronger valid turn
    # must not be punished for differing from another expert's exact pose.
    deficiency = np.asarray(
        (
            max(float(source[0] - target[0]), 0.0),
            max(float(source[1] - target[1]), 0.0),
            max(float(abs(target[2]) - abs(source[2])), 0.0),
            max(float(abs(target[3]) - abs(source[3])), 0.0),
        ),
        dtype=np.float64,
    )
    # Shoulder foreshortening is the clearest 2D evidence in the deployed
    # generated targets. Pelvis contraction and line/twist changes provide
    # supporting evidence without claiming unobservable true 3D axial angle.
    weights = np.asarray((0.25, 1.5, 0.1, 0.25), dtype=np.float64)
    distance = float(
        np.sqrt(np.sum(weights * deficiency**2) / np.sum(weights))
    )
    return {
        "euclidean_distance": distance,
        "target_angle_distance": float(np.mean(deficiency[2:]) / np.pi),
        "combined_distance": distance,
        "source_projected_hip_contraction": float(source[0]),
        "target_projected_hip_contraction": float(target[0]),
        "source_projected_shoulder_contraction": float(source[1]),
        "target_projected_shoulder_contraction": float(target[1]),
        "source_projected_torso_twist": float(source[3]),
        "target_projected_torso_twist": float(target[3]),
        "projected_rotation_deficiency": distance,
    }


def _smooth_trajectory(values: NDArray[np.floating]) -> NDArray[np.float64]:
    trajectory = np.asarray(values, dtype=np.float64)
    padded = np.pad(trajectory, ((2, 2), (0, 0)), mode="edge")
    kernel = np.asarray((1.0, 2.0, 3.0, 2.0, 1.0), dtype=np.float64) / 9.0
    return np.stack(
        [np.convolve(padded[:, axis], kernel, mode="valid") for axis in range(2)],
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
    if rule_id == "weight_transfer":
        balance = _serve_balance_features(pose, root, confidence)
        return (
            np.asarray((balance[2], -balance[1]), dtype=np.float64),
            ("weight_transfer", "completion_on_non_dominant_side"),
            np.asarray((1.5, 1.0), dtype=np.float64),
        )
    if rule_id == "hip_rotation":
        rotation = _serve_projected_rotation_features(pose, confidence)
        return (
            np.asarray((-rotation[0], -rotation[1]), dtype=np.float64),
            ("projected_hip_contraction", "projected_shoulder_contraction"),
            np.asarray((1.0, 1.5), dtype=np.float64),
        )
    if rule_id == "wrist_flick":
        # Serve archives place maximum wrist acceleration at canonical anchor
        # 2. The empirical expert burst spans frames 24--40; the old 36--56
        # window mostly measured deceleration and follow-through.
        wrist = _serve_wrist_motion_features(
            pose, confidence, start=24, end=40
        )
        return (
            np.asarray((wrist[0], wrist[1], wrist[5]), dtype=np.float64),
            (
                "wrist_event_speed_mean",
                "wrist_event_speed_p90",
                "wrist_event_acceleration_p90",
            ),
            np.asarray((2.0, 1.0, 0.5), dtype=np.float64),
        )
    raise KeyError(f"serve rule has no semantic expert evidence: {rule_id}")


def _serve_expert_envelope(
    model: ExpertPhaseModel,
) -> dict[str, dict[str, Any]]:
    """Fit subject-balanced lower expert envelopes from the frozen bank."""
    output: dict[str, dict[str, Any]] = {}
    for rule_id in ("weight_transfer", "hip_rotation", "wrist_flick"):
        evidence = []
        names: tuple[str, ...] = ()
        weights = np.empty(0, dtype=np.float64)
        for pose, root, confidence in zip(
            model.expert_pose,
            model.expert_root,
            model.expert_confidence,
            strict=True,
        ):
            values, names, weights = _serve_semantic_evidence(
                rule_id, pose, root, confidence
            )
            evidence.append(values)
        matrix = np.stack(evidence)
        subject_values = np.stack(
            [
                np.median(
                    matrix[model.expert_subject_ids == subject_id], axis=0
                )
                for subject_id in sorted(set(model.expert_subject_ids.tolist()))
            ]
        )
        lower = np.quantile(subject_values, 0.10, axis=0)
        median = np.median(subject_values, axis=0)
        clip_median = np.median(matrix, axis=0)
        within_take_scale = 1.4826 * np.median(
            np.abs(matrix - clip_median[None]), axis=0
        )
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
    distance = float(
        np.sqrt(np.sum(weights * deficiency**2) / np.sum(weights))
    )
    components: dict[str, float] = {
        "euclidean_distance": distance,
        "target_angle_distance": 0.0,
        "combined_distance": distance,
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
    output = []
    for detail, rule in zip(spec.details, spec.rules, strict=True):
        joints = detail.joints or rule.measured_joints
        source = source_pose
        target = target_pose
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
                start=detail.start,
                end=detail.end,
            )
        else:
            components = criterion_distance_components(
                source,
                target,
                confidence,
                start=detail.start,
                end=detail.end,
                joints=joints,
                joint_weights=spec.joint_weights_array,
            )
        if detail.metric == "serve_follow_through_cross_body":
            # Serve extraction defines frame ``end - 1`` as the maximum
            # post-acceleration shoulder angle. Averaging that endpoint over
            # the whole follow-through window made a visibly wrong last pose
            # almost disappear from the score. Preserve the original
            # cross-body/target-angle semantics, but evaluate them at the
            # explicitly aligned completion frame.
            terminal = criterion_distance_components(
                source,
                target,
                confidence,
                start=detail.end - 1,
                end=detail.end,
                joints=joints,
                joint_weights=spec.joint_weights_array,
            )
            source_frame = source[detail.end - 1]
            target_frame = target[detail.end - 1]
            terminal_confidence = confidence[detail.end - 1]
            required = (5, 6, 8, 10)
            if float(np.prod(terminal_confidence[list(required)])) > 0.2:
                source_shoulder_center = 0.5 * (
                    source_frame[5] + source_frame[6]
                )
                target_shoulder_center = 0.5 * (
                    target_frame[5] + target_frame[6]
                )
                source_forearm_offset = (
                    0.75 * source_frame[8, 0]
                    + 0.25 * source_frame[10, 0]
                    - source_shoulder_center[0]
                )
                target_forearm_offset = (
                    0.75 * target_frame[8, 0]
                    + 0.25 * target_frame[10, 0]
                    - target_shoulder_center[0]
                )
                cross_body_deficiency = max(
                    float(source_forearm_offset - target_forearm_offset), 0.0
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
            "serve_subject_balanced_expert_envelope_v3"
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
        components = _criterion_components_for_spec(
            spec,
            poses[index],
            roots[index],
            corrected,
            corrected_root,
            confidence[index],
            serve_expert_envelope=serve_envelope,
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
    else:
        tolerances = np.full(len(spec.rules), 0.05, dtype=np.float64)
        scales = np.full(len(spec.rules), 0.01, dtype=np.float64)
    if serve_envelope is not None:
        for index, rule in enumerate(spec.rules):
            if rule.id in serve_envelope:
                # Envelope components are already standardized by the natural
                # between-subject expert range. One exponential unit is the
                # interpretable e-folding penalty below that lower envelope.
                tolerances[index] = 0.0
                scales[index] = 1.0
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
    if (
        model.criterion_metric_version
        == "serve_subject_balanced_expert_envelope_v3"
    ):
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
            if rule.id in {"weight_transfer", "hip_rotation", "wrist_flick"}:
                components[index] = semantic_components[index]
    criteria = []
    for index, (rule, component) in enumerate(
        zip(spec.rules, components, strict=True)
    ):
        tolerance = float(model.criterion_tolerances[index])
        scale = float(model.criterion_scales[index])
        distance = component["combined_distance"]
        excess = max(0.0, distance - tolerance)
        ratio = float(np.exp(-excess / max(scale, 1e-3)))
        criteria.append(
            {
                "name_zh_tw": rule.name_zh_tw,
                "rule_reference": rule.id,
                "score": rule.maximum * ratio,
                "maximum": rule.maximum,
                "expert_tolerance": tolerance,
                "expert_robust_scale": scale,
                "standardized_excess": excess / max(scale, 1e-3),
                **component,
            }
        )
    return {
        "filename": correction.student.video_name,
        "skill": model.skill,
        "handedness": correction.student.handedness,
        "score_method": (
            "expert_only_subject_balanced_semantic_envelope_v3"
            if model.criterion_metric_version
            == "serve_subject_balanced_expert_envelope_v3"
            else (
                "expert_only_semantic_criterion_tolerance_v2"
                if model.criterion_metric_version
                == "serve_semantic_motion_features_v2"
                else "expert_only_held_out_identity_tolerance_v1"
            )
        ),
        "correction_policy": "full_body_coherent_expert_phase_projection",
        "score_reference_policy": (
            "subject_balanced_absolute_expert_envelope"
            if model.criterion_metric_version
            == "serve_subject_balanced_expert_envelope_v3"
            else "generated_correction_distance"
        ),
        "limitations": (
            ["wrist_action_uses_coco17_distal_arm_motion_proxy"]
            if model.criterion_metric_version
            == "serve_subject_balanced_expert_envelope_v3"
            else []
        ),
        "total_score": float(sum(item["score"] for item in criteria)),
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
        )


def model_training_inputs(paths: Iterable[Path]) -> list[str]:
    """Expose auditable training inputs for reports and regression tests."""
    return [str(path) for path in sorted(paths)]
