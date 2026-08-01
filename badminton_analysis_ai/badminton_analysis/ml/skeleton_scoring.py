from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Mapping

import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor

JOINT_WEIGHTS = np.asarray(
    [
        0.5, 0.25, 0.25, 0.25, 0.25,
        1.5, 2.0, 1.25, 3.0, 1.5, 4.0,
        1.5, 1.5, 1.25, 1.25, 1.25, 1.25,
    ],
    dtype=np.float64,
)

BONES = (
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 11), (6, 12), (11, 12), (11, 13), (13, 15),
    (12, 14), (14, 16),
)

ANGLE_TRIPLETS = (
    (5, 7, 9), (6, 8, 10),
    (7, 5, 11), (8, 6, 12),
    (5, 11, 13), (6, 12, 14),
    (11, 13, 15), (12, 14, 16),
)

DEFAULT_COMPONENT_WEIGHTS = {
    "position": 1.0,
    "angle": 0.5,
    "velocity": 0.5,
    "bone_length": 0.25,
}


def _masked_weighted_mean(values: Tensor, weights: Tensor, mask: Tensor) -> Tensor:
    combined = weights * mask
    return (values * combined).sum() / combined.sum().clamp_min(1e-8)


def _torch_angles(sequence: Tensor, triplets: Tensor) -> Tensor:
    first = sequence[..., triplets[:, 0], :]
    center = sequence[..., triplets[:, 1], :]
    last = sequence[..., triplets[:, 2], :]
    vector_a = first - center
    vector_b = last - center
    cosine = (vector_a * vector_b).sum(-1) / (
        vector_a.norm(dim=-1) * vector_b.norm(dim=-1)
    ).clamp_min(1e-8)
    return torch.acos(cosine.clamp(-1.0 + 1e-6, 1.0 - 1e-6))


def sequence_training_losses(
    prediction: Tensor,
    target: Tensor,
    confidence: Tensor,
    joint_weights: NDArray[np.floating] | Tensor = JOINT_WEIGHTS,
) -> dict[str, Tensor]:
    """Differentiable position, velocity, angle, and bone-length losses."""
    joint_weight_tensor = torch.as_tensor(
        joint_weights, dtype=prediction.dtype, device=prediction.device
    ).view(1, 1, -1)
    if joint_weight_tensor.shape[-1] != prediction.shape[-2]:
        raise ValueError("joint_weights must contain one value per keypoint")
    position_error = (prediction - target).norm(dim=-1)
    position = _masked_weighted_mean(
        position_error, joint_weight_tensor, confidence
    )

    prediction_velocity = prediction[:, 1:] - prediction[:, :-1]
    target_velocity = target[:, 1:] - target[:, :-1]
    velocity_mask = confidence[:, 1:] * confidence[:, :-1]
    velocity_error = (prediction_velocity - target_velocity).norm(dim=-1)
    velocity = _masked_weighted_mean(
        velocity_error, joint_weight_tensor, velocity_mask
    )

    bones = torch.as_tensor(BONES, dtype=torch.long, device=prediction.device)
    prediction_bones = (
        prediction[..., bones[:, 0], :] - prediction[..., bones[:, 1], :]
    ).norm(dim=-1)
    target_bones = (
        target[..., bones[:, 0], :] - target[..., bones[:, 1], :]
    ).norm(dim=-1)
    bone_mask = confidence[..., bones[:, 0]] * confidence[..., bones[:, 1]]
    bone_length = _masked_weighted_mean(
        (prediction_bones - target_bones).abs(),
        torch.ones_like(bone_mask),
        bone_mask,
    )

    triplets = torch.as_tensor(
        ANGLE_TRIPLETS, dtype=torch.long, device=prediction.device
    )
    prediction_angles = _torch_angles(prediction, triplets)
    target_angles = _torch_angles(target, triplets)
    angle_mask = (
        confidence[..., triplets[:, 0]]
        * confidence[..., triplets[:, 1]]
        * confidence[..., triplets[:, 2]]
    )
    angle = _masked_weighted_mean(
        (prediction_angles - target_angles).abs() / np.pi,
        torch.ones_like(angle_mask),
        angle_mask,
    )

    total = position + 0.5 * velocity + 0.25 * angle + bone_length
    return {
        "loss": total,
        "position": position,
        "velocity": velocity,
        "angle": angle,
        "bone_length": bone_length,
    }


def _numpy_masked_mean(
    values: NDArray[np.floating],
    mask: NDArray[np.floating],
    weights: NDArray[np.floating] | None = None,
) -> float:
    combined = np.asarray(mask, dtype=np.float64)
    if weights is not None:
        combined = combined * np.asarray(weights, dtype=np.float64)
    denominator = float(np.sum(combined))
    if denominator < 1e-8:
        return 0.0
    return float(np.sum(np.asarray(values, dtype=np.float64) * combined) / denominator)


def _numpy_angles(
    sequence: NDArray[np.floating], triplets: tuple[tuple[int, int, int], ...]
) -> NDArray[np.float64]:
    coordinates = np.asarray(sequence, dtype=np.float64)
    indices = np.asarray(triplets, dtype=np.int64)
    vector_a = coordinates[:, indices[:, 0]] - coordinates[:, indices[:, 1]]
    vector_b = coordinates[:, indices[:, 2]] - coordinates[:, indices[:, 1]]
    denominator = np.linalg.norm(vector_a, axis=-1) * np.linalg.norm(vector_b, axis=-1)
    cosine = np.divide(
        np.sum(vector_a * vector_b, axis=-1),
        denominator,
        out=np.ones_like(denominator),
        where=denominator > 1e-8,
    )
    return np.asarray(np.arccos(np.clip(cosine, -1.0, 1.0)), dtype=np.float64)


def correction_distance_components(
    original: NDArray[np.floating],
    corrected: NDArray[np.floating],
    confidence: NDArray[np.floating],
    joint_weights: NDArray[np.floating] = JOINT_WEIGHTS,
) -> dict[str, float]:
    """Return normalized correction components for one skeleton sequence."""
    source = np.asarray(original, dtype=np.float64)
    target = np.asarray(corrected, dtype=np.float64)
    mask = np.asarray(confidence, dtype=np.float64)
    if source.shape != target.shape or source.ndim != 3 or source.shape[-1] != 3:
        raise ValueError("original and corrected must have matching shape (T, J, 3)")
    if mask.shape != source.shape[:2]:
        raise ValueError("confidence must have shape (T, J)")
    weights = np.asarray(joint_weights, dtype=np.float64)
    if weights.shape != (source.shape[1],):
        raise ValueError("joint_weights must contain one value per keypoint")

    position_error = np.linalg.norm(source - target, axis=-1)
    position = _numpy_masked_mean(position_error, mask, weights[None, :])

    source_velocity = np.diff(source, axis=0)
    target_velocity = np.diff(target, axis=0)
    velocity_mask = mask[1:] * mask[:-1]
    velocity = _numpy_masked_mean(
        np.linalg.norm(source_velocity - target_velocity, axis=-1),
        velocity_mask,
        weights[None, :],
    )

    bone_indices = np.asarray(BONES, dtype=np.int64)
    source_bones = np.linalg.norm(
        source[:, bone_indices[:, 0]] - source[:, bone_indices[:, 1]], axis=-1
    )
    target_bones = np.linalg.norm(
        target[:, bone_indices[:, 0]] - target[:, bone_indices[:, 1]], axis=-1
    )
    bone_mask = mask[:, bone_indices[:, 0]] * mask[:, bone_indices[:, 1]]
    bone_length = _numpy_masked_mean(np.abs(source_bones - target_bones), bone_mask)

    angle_indices = np.asarray(ANGLE_TRIPLETS, dtype=np.int64)
    angle_mask = (
        mask[:, angle_indices[:, 0]]
        * mask[:, angle_indices[:, 1]]
        * mask[:, angle_indices[:, 2]]
    )
    source_angles = _numpy_angles(source, ANGLE_TRIPLETS)
    target_angles = _numpy_angles(target, ANGLE_TRIPLETS)
    angle = _numpy_masked_mean(np.abs(source_angles - target_angles) / np.pi, angle_mask)
    return {
        "position_distance": position,
        "angle_distance": angle,
        "velocity_distance": velocity,
        "bone_length_distance": bone_length,
    }


def correction_distance(
    original: NDArray[np.floating],
    corrected: NDArray[np.floating],
    confidence: NDArray[np.floating],
    component_weights: Mapping[str, float] = DEFAULT_COMPONENT_WEIGHTS,
    joint_weights: NDArray[np.floating] = JOINT_WEIGHTS,
) -> tuple[float, dict[str, float]]:
    components = correction_distance_components(
        original, corrected, confidence, joint_weights
    )
    total = (
        float(component_weights["position"]) * components["position_distance"]
        + float(component_weights["angle"]) * components["angle_distance"]
        + float(component_weights["velocity"]) * components["velocity_distance"]
        + float(component_weights["bone_length"]) * components["bone_length_distance"]
    )
    return float(total), components


def keypoint_correction_components(
    original: NDArray[np.floating],
    corrected: NDArray[np.floating],
    confidence: NDArray[np.floating],
) -> dict[str, NDArray[np.float64]]:
    """Attribute correction distance components to individual COCO keypoints."""
    source = np.asarray(original, dtype=np.float64)
    target = np.asarray(corrected, dtype=np.float64)
    mask = np.asarray(confidence, dtype=np.float64)
    if source.shape != target.shape or source.ndim != 3 or source.shape[-1] != 3:
        raise ValueError("original and corrected must have matching shape (T, J, 3)")
    if mask.shape != source.shape[:2]:
        raise ValueError("confidence must have shape (T, J)")

    def per_keypoint_mean(
        values: NDArray[np.float64], value_mask: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        denominator = np.sum(value_mask, axis=0)
        return np.divide(
            np.sum(values * value_mask, axis=0),
            denominator,
            out=np.zeros(source.shape[1], dtype=np.float64),
            where=denominator > 1e-8,
        )

    position_error = np.linalg.norm(source - target, axis=-1)
    position = per_keypoint_mean(position_error, mask)

    source_velocity = np.diff(source, axis=0)
    target_velocity = np.diff(target, axis=0)
    velocity_mask = mask[1:] * mask[:-1]
    velocity = per_keypoint_mean(
        np.linalg.norm(source_velocity - target_velocity, axis=-1),
        velocity_mask,
    )

    angle = np.zeros(source.shape[1], dtype=np.float64)
    angle_counts = np.zeros(source.shape[1], dtype=np.float64)
    source_angles = _numpy_angles(source, ANGLE_TRIPLETS)
    target_angles = _numpy_angles(target, ANGLE_TRIPLETS)
    angle_indices = np.asarray(ANGLE_TRIPLETS, dtype=np.int64)
    angle_mask = (
        mask[:, angle_indices[:, 0]]
        * mask[:, angle_indices[:, 1]]
        * mask[:, angle_indices[:, 2]]
    )
    angle_error = np.abs(source_angles - target_angles) / np.pi
    for triplet_index, (_, center, _) in enumerate(ANGLE_TRIPLETS):
        observed_count = float(np.sum(angle_mask[:, triplet_index]))
        if observed_count <= 1e-8:
            continue
        angle[center] += float(
            np.sum(
                angle_error[:, triplet_index]
                * angle_mask[:, triplet_index]
            )
            / observed_count
        )
        angle_counts[center] += 1.0
    angle = np.divide(
        angle,
        angle_counts,
        out=np.zeros_like(angle),
        where=angle_counts > 0.0,
    )

    bone_length = np.zeros(source.shape[1], dtype=np.float64)
    bone_counts = np.zeros(source.shape[1], dtype=np.float64)
    for start, end in BONES:
        source_length = np.linalg.norm(source[:, start] - source[:, end], axis=-1)
        target_length = np.linalg.norm(target[:, start] - target[:, end], axis=-1)
        bone_mask = mask[:, start] * mask[:, end]
        observed_count = float(np.sum(bone_mask))
        if observed_count <= 1e-8:
            continue
        error = float(
            np.sum(np.abs(source_length - target_length) * bone_mask)
            / observed_count
        )
        bone_length[start] += error
        bone_length[end] += error
        bone_counts[start] += 1.0
        bone_counts[end] += 1.0
    bone_length = np.divide(
        bone_length,
        bone_counts,
        out=np.zeros_like(bone_length),
        where=bone_counts > 0.0,
    )

    total = position + 0.5 * angle + 0.5 * velocity + 0.25 * bone_length
    return {
        "correction_distance": total,
        "position_distance": position,
        "angle_distance": angle,
        "velocity_distance": velocity,
        "bone_length_distance": bone_length,
    }


def correction_quality_metrics(
    original: NDArray[np.floating],
    corrected: NDArray[np.floating],
    confidence: NDArray[np.floating],
) -> dict[str, float]:
    """Diagnostics for implausible stretching and high-frequency corrections."""
    source = np.asarray(original, dtype=np.float64)
    target = np.asarray(corrected, dtype=np.float64)
    mask = np.asarray(confidence, dtype=np.float64)
    delta = target - source
    magnitudes = np.linalg.norm(delta, axis=-1)
    observed_magnitudes = magnitudes[mask > 0]

    bone_indices = np.asarray(BONES, dtype=np.int64)
    source_lengths = np.linalg.norm(
        source[:, bone_indices[:, 0]] - source[:, bone_indices[:, 1]], axis=-1
    )
    target_lengths = np.linalg.norm(
        target[:, bone_indices[:, 0]] - target[:, bone_indices[:, 1]], axis=-1
    )
    bone_mask = mask[:, bone_indices[:, 0]] * mask[:, bone_indices[:, 1]]
    relative_bone_change = np.abs(target_lengths - source_lengths) / np.maximum(
        source_lengths, 0.1
    )
    observed_bone_change = relative_bone_change[bone_mask > 0]

    correction_velocity = np.diff(delta, axis=0)
    velocity_mask = mask[1:] * mask[:-1]
    correction_acceleration = np.diff(delta, n=2, axis=0)
    acceleration_mask = mask[2:] * mask[1:-1] * mask[:-2]
    return {
        "mean_joint_correction": (
            float(np.mean(observed_magnitudes)) if observed_magnitudes.size else 0.0
        ),
        "max_joint_correction": (
            float(np.max(observed_magnitudes)) if observed_magnitudes.size else 0.0
        ),
        "mean_relative_bone_change": (
            float(np.mean(observed_bone_change)) if observed_bone_change.size else 0.0
        ),
        "p95_relative_bone_change": (
            float(np.quantile(observed_bone_change, 0.95))
            if observed_bone_change.size
            else 0.0
        ),
        "mean_correction_velocity": _numpy_masked_mean(
            np.linalg.norm(correction_velocity, axis=-1), velocity_mask
        ),
        "mean_correction_acceleration": _numpy_masked_mean(
            np.linalg.norm(correction_acceleration, axis=-1), acceleration_mask
        ),
    }


def project_bone_lengths(
    original: NDArray[np.floating],
    corrected: NDArray[np.floating],
    *,
    iterations: int = 20,
) -> NDArray[np.float32]:
    """Project corrected joints onto the source skeleton's bone-length constraints."""
    source = np.asarray(original, dtype=np.float64)
    projected = np.asarray(corrected, dtype=np.float64).copy()
    if source.shape != projected.shape or source.ndim != 3 or source.shape[-1] != 3:
        raise ValueError("original and corrected must have matching shape (T, J, 3)")
    bone_indices = np.asarray(BONES, dtype=np.int64)
    desired_lengths = np.linalg.norm(
        source[:, bone_indices[:, 1]] - source[:, bone_indices[:, 0]], axis=-1
    )
    pelvis_anchor = (source[:, 11] + source[:, 12]) / 2.0
    for _ in range(iterations):
        for bone_index, (start, end) in enumerate(BONES):
            vector = projected[:, end] - projected[:, start]
            length = np.linalg.norm(vector, axis=-1)
            valid = length > 1e-8
            adjustment = np.zeros_like(vector)
            adjustment[valid] = (
                0.5
                * ((length[valid] - desired_lengths[valid, bone_index]) / length[valid])[:, None]
                * vector[valid]
            )
            projected[:, start] += adjustment
            projected[:, end] -= adjustment
        pelvis = (projected[:, 11] + projected[:, 12]) / 2.0
        projected += (pelvis_anchor - pelvis)[:, None, :]
    return projected.astype(np.float32)


def select_bone_adapted_expert(
    sequence: NDArray[np.floating],
    expert_sequences: NDArray[np.floating],
    confidence: NDArray[np.floating],
    expert_confidence: NDArray[np.floating],
    joint_weights: NDArray[np.floating] = JOINT_WEIGHTS,
) -> tuple[int, NDArray[np.float32], NDArray[np.float64], float]:
    """Select the adapted expert with the lowest grading distance."""
    source = np.asarray(sequence, dtype=np.float64)
    experts = np.asarray(expert_sequences, dtype=np.float64)
    source_confidence = np.asarray(confidence, dtype=np.float64)
    reference_confidence = np.asarray(expert_confidence, dtype=np.float64)
    if experts.ndim != 4 or experts.shape[1:] != source.shape:
        raise ValueError("expert sequences must have shape (N, T, J, 3)")
    if reference_confidence.shape != experts.shape[:3]:
        raise ValueError("expert confidence must have shape (N, T, J)")

    matches: list[tuple[float, int, NDArray[np.float32], NDArray[np.float64]]] = []
    for index, expert in enumerate(experts):
        adapted = project_bone_lengths(source, expert)
        match_confidence = source_confidence * reference_confidence[index]
        distance, _ = correction_distance(
            source,
            adapted,
            match_confidence,
            joint_weights=joint_weights,
        )
        matches.append((distance, index, adapted, match_confidence))
    distance, index, adapted, match_confidence = min(
        matches, key=lambda match: match[0]
    )
    return index, adapted, match_confidence, float(distance)


def expert_euclidean_distances(
    sequence: NDArray[np.floating],
    expert_sequences: NDArray[np.floating],
    confidence: NDArray[np.floating] | None = None,
    expert_confidence: NDArray[np.floating] | None = None,
) -> NDArray[np.float64]:
    """Return mean per-joint Euclidean distance to every expert sequence."""
    source = np.asarray(sequence, dtype=np.float64)
    experts = np.asarray(expert_sequences, dtype=np.float64)
    if source.ndim != 3 or source.shape[-1] != 3:
        raise ValueError("sequence must have shape (T, J, 3)")
    if experts.ndim != 4 or experts.shape[1:] != source.shape:
        raise ValueError("expert sequences must have shape (N, T, J, 3)")
    source_mask = (
        np.ones(source.shape[:2], dtype=np.float64)
        if confidence is None
        else np.asarray(confidence, dtype=np.float64)
    )
    reference_mask = (
        np.ones(experts.shape[:3], dtype=np.float64)
        if expert_confidence is None
        else np.asarray(expert_confidence, dtype=np.float64)
    )
    if source_mask.shape != source.shape[:2]:
        raise ValueError("confidence must have shape (T, J)")
    if reference_mask.shape != experts.shape[:3]:
        raise ValueError("expert confidence must have shape (N, T, J)")
    mask = reference_mask * source_mask[None, ...]
    distances = np.linalg.norm(experts - source[None, ...], axis=-1)
    denominator = np.sum(mask, axis=(1, 2))
    return np.divide(
        np.sum(distances * mask, axis=(1, 2)),
        denominator,
        out=np.full(len(experts), np.inf, dtype=np.float64),
        where=denominator > 1e-8,
    )


@dataclass(frozen=True)
class ScoreCalibration:
    distance_offset: float
    alpha: float
    target_beginner_mean: float = 45.0
    target_expert_mean: float = 99.0
    target_reachable: bool = True

    def score(self, distance: float | NDArray[np.floating]) -> float | NDArray[np.float64]:
        values = np.asarray(distance, dtype=np.float64)
        scores = 100.0 * np.exp(-self.alpha * np.maximum(values - self.distance_offset, 0.0))
        if values.ndim == 0:
            return float(scores)
        return scores

    def to_dict(self) -> dict[str, float | bool]:
        return asdict(self)


def fit_score_calibration(
    expert_distances: NDArray[np.floating],
    beginner_distances: NDArray[np.floating],
    *,
    target_beginner_mean: float = 45.0,
    target_expert_mean: float = 99.0,
) -> ScoreCalibration:
    experts = np.asarray(expert_distances, dtype=np.float64)
    beginners = np.asarray(beginner_distances, dtype=np.float64)
    if experts.size == 0 or beginners.size == 0:
        raise ValueError("expert and beginner distances are required for calibration")

    def solve_alpha(offset: float) -> tuple[float, float, bool]:
        adjusted = np.maximum(beginners - offset, 0.0)
        minimum_mean = 100.0 * float(np.mean(adjusted == 0.0))
        reachable = minimum_mean <= target_beginner_mean
        low, high = 0.0, 1.0
        while (
            float(np.mean(100.0 * np.exp(-high * adjusted)))
            > target_beginner_mean
            and high < 1e6
        ):
            high *= 2.0
        for _ in range(80):
            midpoint = (low + high) / 2.0
            mean_score = float(np.mean(100.0 * np.exp(-midpoint * adjusted)))
            if mean_score > target_beginner_mean:
                low = midpoint
            else:
                high = midpoint
        alpha = (low + high) / 2.0
        achieved = float(np.mean(100.0 * np.exp(-alpha * adjusted)))
        return alpha, achieved, reachable

    candidates = np.unique(
        np.concatenate(
            (np.asarray((0.0,)), np.quantile(experts, np.linspace(0.0, 1.0, 201)))
        )
    )
    best: tuple[float, float, bool, float] | None = None
    best_objective = float("inf")
    for candidate in candidates:
        offset = float(candidate)
        alpha, beginner_mean, reachable = solve_alpha(offset)
        expert_mean = float(
            np.mean(100.0 * np.exp(-alpha * np.maximum(experts - offset, 0.0)))
        )
        objective = abs(expert_mean - target_expert_mean) + abs(
            beginner_mean - target_beginner_mean
        )
        if not reachable:
            objective += 100.0
        if objective < best_objective:
            best_objective = objective
            best = (offset, alpha, reachable, expert_mean)
    assert best is not None
    distance_offset, alpha, reachable, _ = best
    return ScoreCalibration(
        distance_offset=distance_offset,
        alpha=alpha,
        target_beginner_mean=target_beginner_mean,
        target_expert_mean=target_expert_mean,
        target_reachable=reachable,
    )
