"""What a serve is measured against: tolerances fitted on expert takes only."""

from __future__ import annotations

from typing import Any

from numpy.typing import NDArray
import numpy as np

from badminton_analysis.ml.motion.samples import ExpertPhaseModel
from badminton_analysis.ml.motion.view import _EPS
from badminton_analysis.ml.serve.checkpoints import (
    _serve_qualitative_pose_evidence,
    _serve_semantic_evidence,
    _SERVE_UNSEEN_EXPERT_MARGIN,
)


_SERVE_CHECKPOINT_MANIFOLDS: dict[tuple[int, str], Any] = {}


def _serve_checkpoint_manifold(
    model: "ExpertPhaseModel", rule_id: str, joints: tuple[int, ...], start: int, end: int
) -> Any:
    from badminton_analysis.ml.trajectory_distance import fit_serve_checkpoint_manifold

    key = (id(model), rule_id)
    if key not in _SERVE_CHECKPOINT_MANIFOLDS:
        _SERVE_CHECKPOINT_MANIFOLDS[key] = fit_serve_checkpoint_manifold(
            np.asarray(model.expert_pose), model.expert_subject_ids, joints, start, end
        )
    return _SERVE_CHECKPOINT_MANIFOLDS[key]


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
            [np.median(clip_values[subjects == subject_id]) for subject_id in subjects],
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
                np.median(matrix[selected_subjects == subject_id], axis=0)
                for subject_id in subject_ids
            ]
        )
        clip_median = np.median(matrix, axis=0)
        within_take_scale = 1.4826 * np.median(
            np.abs(matrix - clip_median[None]), axis=0
        )
        # The deployed expert distribution must retain every demonstrated expert identity.
        if rule_id == "wrist_flick":
            lower = np.quantile(subject_values, 0.10, axis=0)
        elif rule_id == "racket_foot_weight":
            # Preparation pose varies substantially between repeated takes from the same expert.
            lower = np.min(matrix, axis=0) - within_take_scale
        else:
            # Dynamic checkpoints use identity medians so a single occluded or truncated take cannot erase the required motion pattern.
            lower = np.min(subject_values, axis=0) - within_take_scale
        if rule_id == "weight_transfer":
            # One expert identity barely moves the pelvis across its stance.
            loading = names.index("pelvis_loading_shift")
            lower[loading] = (
                float(np.sort(subject_values[:, loading])[1]) - within_take_scale[loading]
            )
            # Stance varies from take to take within one expert.
            stance = names.index("stance_retention")
            lower[stance] = float(np.min(matrix[:, stance])) * (
                1.0 - _SERVE_UNSEEN_EXPERT_MARGIN
            )
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
    evidence, names, _ = _serve_semantic_evidence(rule_id, pose, root, confidence)
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
        # Compact impulse and forward-coherent acceleration are alternative expert wrist-action styles.
        distance = float(np.min(deficiency[weights > 0.0]))
        aggregation = "either_impulse_or_directional_acceleration"
    elif rule_id == "weight_transfer":
        # The dominant shoulder-hip-knee and hip-knee-ankle chain is the camera-robust primary evidence.
        chain_weights = weights[:2]
        chain_distance = float(
            np.sqrt(
                np.sum(chain_weights * deficiency[:2] ** 2)
                / max(float(np.sum(chain_weights)), _EPS)
            )
        )
        support_distance = float(deficiency[2])
        # A stance that closes leaves nothing to transfer across.
        stance_distance = float(deficiency[5])
        distance = float(max(chain_distance, support_distance, stance_distance))
        aggregation = "dominant_chain_with_pelvis_or_root_support"
        transfer_parts = {
            "transfer_chain_distance": chain_distance,
            "transfer_support_distance": support_distance,
        }
    elif rule_id == "hip_rotation":
        # Axial rotation changes apparent hip width, shoulder width, hip-axis direction.
        contraction = float(np.min(deficiency[:2]))
        orientation = float(np.min(deficiency[2:]))
        distance = float(max(contraction, orientation))
        aggregation = "contraction_and_orientation_camera_robust"
    elif rule_id == "shoulder_rotation":
        # Forward shoulder rotation is a depth movement and shoulder-width contraction is weak from frontal views.
        shoulder_depth_proxy = float(deficiency[0])
        arm_completion = float(np.sqrt(np.mean(deficiency[1:4] ** 2)))
        # An elbow left off to the side is a body that never rotated, whichever way the arm finished.
        elbow_between_shoulders = float(deficiency[4])
        shoulders_turned = float(deficiency[5])
        distance = float(
            max(
                min(shoulder_depth_proxy, arm_completion),
                elbow_between_shoulders,
                shoulders_turned,
            )
        )
        aggregation = "shoulder_contraction_or_cross_body_completion"
    else:
        # Every positively weighted feature is a necessary higher-is-better cue for the remaining checkpoints.
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
    if rule_id == "weight_transfer":
        components.update(transfer_parts)
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
