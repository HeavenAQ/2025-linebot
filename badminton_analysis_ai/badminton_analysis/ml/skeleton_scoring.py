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

# Lateral torso spans are projection-dependent rather than rigid limb lengths.
# The expert-motion generator preserves their generated per-frame profile so
# shoulder/hip rotation remains visible after student-anatomy retargeting.
TORSO_WIDTH_BONES = ((5, 6), (11, 12))

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
    transition_weight: float = 0.0,
    transition_joints: tuple[int, ...] = (),
    transition_lean_joints: tuple[int, ...] = (),
    transition_direction_joint: int | None = None,
) -> dict[str, Tensor]:
    """Differentiable pose losses, including an optional full-body transition."""
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

    transition = prediction.new_zeros(())
    direction = prediction.new_zeros(())
    if transition_weight > 0.0:
        if not transition_joints:
            raise ValueError(
                "transition_joints are required when transition_weight is positive"
            )
        indices = torch.as_tensor(
            transition_joints, dtype=torch.long, device=prediction.device
        )
        source_trajectory = prediction[..., indices, :] - prediction[:, :1, indices, :]
        target_trajectory = target[..., indices, :] - target[:, :1, indices, :]
        trajectory_mask = confidence[..., indices] * confidence[:, :1, indices]
        trajectory = _masked_weighted_mean(
            (source_trajectory - target_trajectory).norm(dim=-1),
            torch.ones_like(trajectory_mask),
            trajectory_mask,
        )
        endpoint_mask = confidence[:, 0, indices] * confidence[:, -1, indices]
        source_endpoint = prediction[:, -1, indices] - prediction[:, 0, indices]
        target_endpoint = target[:, -1, indices] - target[:, 0, indices]
        endpoint = _masked_weighted_mean(
            (source_endpoint - target_endpoint).norm(dim=-1),
            torch.ones_like(endpoint_mask),
            endpoint_mask,
        )
        lower_transition = 0.5 * trajectory + 0.5 * endpoint
        if len(transition_lean_joints) != 4:
            raise ValueError(
                "transition_lean_joints must contain both shoulders and both hips"
            )
        left_shoulder, right_shoulder, left_hip, right_hip = (
            transition_lean_joints
        )
        source_torso = (
            (prediction[..., left_shoulder, :] + prediction[..., right_shoulder, :])
            - (prediction[..., left_hip, :] + prediction[..., right_hip, :])
        ) * 0.5
        target_torso = (
            (target[..., left_shoulder, :] + target[..., right_shoulder, :])
            - (target[..., left_hip, :] + target[..., right_hip, :])
        ) * 0.5
        source_lean = torch.atan2(source_torso[..., 2], source_torso[..., 1])
        target_lean = torch.atan2(target_torso[..., 2], target_torso[..., 1])
        lean_mask = (
            confidence[..., left_shoulder]
            * confidence[..., right_shoulder]
            * confidence[..., left_hip]
            * confidence[..., right_hip]
        )
        lean_trajectory_mask = lean_mask * lean_mask[:, :1]
        lean_trajectory = _masked_weighted_mean(
            (
                (source_lean - source_lean[:, :1])
                - (target_lean - target_lean[:, :1])
            ).abs(),
            torch.ones_like(lean_trajectory_mask),
            lean_trajectory_mask,
        )
        lean_endpoint_mask = lean_mask[:, 0] * lean_mask[:, -1]
        lean_endpoint = _masked_weighted_mean(
            (
                (source_lean[:, -1] - source_lean[:, 0])
                - (target_lean[:, -1] - target_lean[:, 0])
            ).abs(),
            torch.ones_like(lean_endpoint_mask),
            lean_endpoint_mask,
        )
        lean_transition = 0.5 * lean_trajectory + 0.5 * lean_endpoint
        if transition_direction_joint is None:
            transition = 0.65 * lower_transition + 0.35 * lean_transition
        else:
            pelvis_prediction = (
                prediction[..., 11, :] + prediction[..., 12, :]
            ) * 0.5
            pelvis_target = (target[..., 11, :] + target[..., 12, :]) * 0.5
            source_relative = (
                prediction[..., transition_direction_joint, :]
                - pelvis_prediction
            )
            target_relative = target[..., transition_direction_joint, :] - pelvis_target
            source_displacement = source_relative - source_relative[:, :1]
            target_displacement = target_relative - target_relative[:, :1]
            source_horizontal = source_displacement[..., (0, 2)]
            target_horizontal = target_displacement[..., (0, 2)]
            window_start = prediction.shape[1] // 2
            window_end = max(window_start + 1, prediction.shape[1] * 7 // 8)
            target_window = target_horizontal[:, window_start:window_end]
            peak_offset = target_window.norm(dim=-1).argmax(dim=1)
            peak_index = peak_offset + window_start
            gather_index = peak_index[:, None, None].expand(-1, 1, 2)
            source_vector = torch.gather(
                source_horizontal, 1, gather_index
            ).squeeze(1)
            target_vector = torch.gather(
                target_horizontal, 1, gather_index
            ).squeeze(1)
            source_norm = source_vector.norm(dim=-1)
            target_norm = target_vector.norm(dim=-1)
            cosine = (source_vector * target_vector).sum(dim=-1) / (
                source_norm * target_norm
            ).clamp_min(1e-8)
            direction_error = 0.5 * (1.0 - cosine.clamp(-1.0, 1.0))
            direction_error = torch.where(
                source_norm > 1e-6,
                direction_error,
                torch.ones_like(direction_error),
            )
            peak_confidence = torch.gather(
                confidence[..., transition_direction_joint],
                1,
                peak_index[:, None],
            ).squeeze(1)
            direction_mask = (
                peak_confidence
                * confidence[:, 0, transition_direction_joint]
                * confidence[:, 0, 11]
                * confidence[:, 0, 12]
                * (target_norm > 1e-6).to(confidence.dtype)
            )
            direction = _masked_weighted_mean(
                direction_error,
                torch.ones_like(direction_mask),
                direction_mask,
            )
            transition = (
                0.45 * lower_transition
                + 0.20 * lean_transition
                + 0.35 * direction
            )

    total = (
        position
        + 0.5 * velocity
        + 0.25 * angle
        + bone_length
        + transition_weight * transition
    )
    return {
        "loss": total,
        "position": position,
        "velocity": velocity,
        "angle": angle,
        "bone_length": bone_length,
        "transition": transition,
        "direction": direction,
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
        "support_transition_distance": 0.0,
        "torso_lean_transition_distance": 0.0,
        "lunge_direction_distance": 0.0,
        "transition_distance": 0.0,
    }


def full_transition_components(
    original: NDArray[np.floating],
    corrected: NDArray[np.floating],
    confidence: NDArray[np.floating],
    joints: tuple[int, ...],
    lean_joints: tuple[int, ...],
    direction_joint: int | None = None,
) -> dict[str, float]:
    """Compare full support-transfer and signed torso-lean trajectories."""
    if not joints:
        raise ValueError("joints must contain the lower-body support joints")
    source = np.asarray(original, dtype=np.float64)[:, joints]
    target = np.asarray(corrected, dtype=np.float64)[:, joints]
    mask = np.asarray(confidence, dtype=np.float64)[:, joints]
    if source.shape != target.shape or source.ndim != 3 or source.shape[-1] != 3:
        raise ValueError("original and corrected must have matching shape (T, J, 3)")
    if mask.shape != source.shape[:2]:
        raise ValueError("confidence must have shape (T, J)")
    if len(source) < 2:
        raise ValueError("transition distance requires at least two frames")

    source_trajectory = source - source[:1]
    target_trajectory = target - target[:1]
    trajectory_mask = mask * mask[:1]
    trajectory = _numpy_masked_mean(
        np.linalg.norm(source_trajectory - target_trajectory, axis=-1),
        trajectory_mask,
    )
    endpoint_mask = mask[0] * mask[-1]
    source_endpoint = source[-1] - source[0]
    target_endpoint = target[-1] - target[0]
    endpoint = _numpy_masked_mean(
        np.linalg.norm(source_endpoint - target_endpoint, axis=-1),
        endpoint_mask,
    )
    lower_transition = 0.5 * trajectory + 0.5 * endpoint

    if len(lean_joints) != 4:
        raise ValueError("lean_joints must contain both shoulders and both hips")
    left_shoulder, right_shoulder, left_hip, right_hip = lean_joints
    full_source = np.asarray(original, dtype=np.float64)
    full_target = np.asarray(corrected, dtype=np.float64)
    full_mask = np.asarray(confidence, dtype=np.float64)
    source_torso = (
        (full_source[:, left_shoulder] + full_source[:, right_shoulder])
        - (full_source[:, left_hip] + full_source[:, right_hip])
    ) * 0.5
    target_torso = (
        (full_target[:, left_shoulder] + full_target[:, right_shoulder])
        - (full_target[:, left_hip] + full_target[:, right_hip])
    ) * 0.5
    source_lean = np.arctan2(source_torso[:, 2], source_torso[:, 1])
    target_lean = np.arctan2(target_torso[:, 2], target_torso[:, 1])
    lean_mask = (
        full_mask[:, left_shoulder]
        * full_mask[:, right_shoulder]
        * full_mask[:, left_hip]
        * full_mask[:, right_hip]
    )
    lean_trajectory = _numpy_masked_mean(
        np.abs(
            (source_lean - source_lean[0])
            - (target_lean - target_lean[0])
        ),
        lean_mask * lean_mask[0],
    )
    lean_endpoint = _numpy_masked_mean(
        np.asarray(
            [
                abs(
                    (source_lean[-1] - source_lean[0])
                    - (target_lean[-1] - target_lean[0])
                )
            ]
        ),
        np.asarray([lean_mask[0] * lean_mask[-1]]),
    )
    lean_transition = 0.5 * lean_trajectory + 0.5 * lean_endpoint
    direction_distance = 0.0
    if direction_joint is not None:
        source_pelvis = (full_source[:, 11] + full_source[:, 12]) * 0.5
        target_pelvis = (full_target[:, 11] + full_target[:, 12]) * 0.5
        source_relative = full_source[:, direction_joint] - source_pelvis
        target_relative = full_target[:, direction_joint] - target_pelvis
        source_displacement = source_relative - source_relative[:1]
        target_displacement = target_relative - target_relative[:1]
        source_horizontal = source_displacement[:, (0, 2)]
        target_horizontal = target_displacement[:, (0, 2)]
        window_start = len(full_target) // 2
        window_end = max(window_start + 1, len(full_target) * 7 // 8)
        peak_index = window_start + int(
            np.argmax(
                np.linalg.norm(
                    target_horizontal[window_start:window_end], axis=-1
                )
            )
        )
        source_vector = source_horizontal[peak_index]
        target_vector = target_horizontal[peak_index]
        source_norm = float(np.linalg.norm(source_vector))
        target_norm = float(np.linalg.norm(target_vector))
        direction_mask = (
            full_mask[0, direction_joint]
            * full_mask[peak_index, direction_joint]
            * full_mask[0, 11]
            * full_mask[0, 12]
        )
        if direction_mask > 1e-8 and target_norm > 1e-6:
            if source_norm <= 1e-6:
                direction_distance = 1.0
            else:
                cosine = float(
                    np.dot(source_vector, target_vector)
                    / (source_norm * target_norm)
                )
                direction_distance = 0.5 * (1.0 - np.clip(cosine, -1.0, 1.0))
    transition_distance = (
        0.65 * lower_transition + 0.35 * lean_transition
        if direction_joint is None
        else 0.45 * lower_transition
        + 0.20 * lean_transition
        + 0.35 * direction_distance
    )
    return {
        "support_transition_distance": lower_transition,
        "torso_lean_transition_distance": lean_transition,
        "lunge_direction_distance": direction_distance,
        "transition_distance": transition_distance,
    }


def full_transition_distance(
    original: NDArray[np.floating],
    corrected: NDArray[np.floating],
    confidence: NDArray[np.floating],
    joints: tuple[int, ...],
    lean_joints: tuple[int, ...],
    direction_joint: int | None = None,
) -> float:
    return full_transition_components(
        original, corrected, confidence, joints, lean_joints, direction_joint
    )["transition_distance"]


def correction_distance(
    original: NDArray[np.floating],
    corrected: NDArray[np.floating],
    confidence: NDArray[np.floating],
    component_weights: Mapping[str, float] = DEFAULT_COMPONENT_WEIGHTS,
    joint_weights: NDArray[np.floating] = JOINT_WEIGHTS,
    transition_weight: float = 0.0,
    transition_joints: tuple[int, ...] = (),
    transition_lean_joints: tuple[int, ...] = (),
    transition_direction_joint: int | None = None,
) -> tuple[float, dict[str, float]]:
    components = correction_distance_components(
        original, corrected, confidence, joint_weights
    )
    transition_components = (
        full_transition_components(
            original,
            corrected,
            confidence,
            transition_joints,
            transition_lean_joints,
            transition_direction_joint,
        )
        if transition_weight > 0.0
        else {
            "support_transition_distance": 0.0,
            "torso_lean_transition_distance": 0.0,
            "lunge_direction_distance": 0.0,
            "transition_distance": 0.0,
        }
    )
    components.update(transition_components)
    total = (
        float(component_weights["position"]) * components["position_distance"]
        + float(component_weights["angle"]) * components["angle_distance"]
        + float(component_weights["velocity"]) * components["velocity_distance"]
        + float(component_weights["bone_length"]) * components["bone_length_distance"]
        + float(transition_weight) * components["transition_distance"]
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


def project_stable_bone_lengths(
    original: NDArray[np.floating],
    corrected: NDArray[np.floating],
    confidence: NDArray[np.floating],
    *,
    iterations: int = 20,
    expert_length_bones: tuple[tuple[int, int], ...] = (),
    preserve_target_pelvis: bool = False,
    preserve_direction_chains: tuple[tuple[int, ...], ...] = (),
) -> NDArray[np.float32]:
    """Retarget a motion using stable clip-level student anatomy.

    Rigid limbs use the student's median observed length. Projection-dependent
    torso widths may instead retain the generated expert profile. Optional
    distal chains are rebuilt after the global solve so their generated
    directions survive the length projection exactly.
    """
    source = np.asarray(original, dtype=np.float64)
    corrected_source = np.asarray(corrected, dtype=np.float64)
    projected = corrected_source.copy()
    observed = np.asarray(confidence, dtype=np.float64)
    if source.shape != projected.shape or source.ndim != 3 or source.shape[-1] != 2:
        raise ValueError("original and corrected must have matching shape (T, J, 2)")
    if observed.shape != source.shape[:2]:
        raise ValueError("confidence must have shape (T, J)")
    if iterations <= 0:
        raise ValueError("iterations must be positive")
    if not np.all(np.isfinite(projected)):
        raise ValueError("corrected skeleton must contain finite coordinates")

    free_bones = {frozenset(pair) for pair in expert_length_bones}
    bone_lookup = {
        frozenset((start, end)): index for index, (start, end) in enumerate(BONES)
    }
    direction_chains = tuple(tuple(chain) for chain in preserve_direction_chains)
    for chain in direction_chains:
        if len(chain) < 2:
            raise ValueError("preserved direction chains need at least two joints")
        for start, end in zip(chain[:-1], chain[1:], strict=True):
            if frozenset((start, end)) not in bone_lookup:
                raise ValueError(
                    f"preserved direction segment {start}-{end} is not a bone"
                )

    desired_lengths = np.empty((len(source), len(BONES)), dtype=np.float64)
    timeline = np.arange(len(source), dtype=np.float64)
    for bone_index, (start, end) in enumerate(BONES):
        is_free = frozenset((start, end)) in free_bones
        length_reference = corrected_source if is_free else source
        reference_vectors = length_reference[:, end] - length_reference[:, start]
        reference_lengths = np.linalg.norm(reference_vectors, axis=-1)
        visible = (
            (observed[:, start] > 0.05)
            & (observed[:, end] > 0.05)
            & np.isfinite(reference_lengths)
            & (reference_lengths > 1e-8)
        )
        fallback = np.isfinite(reference_lengths) & (reference_lengths > 1e-8)
        lengths = (
            reference_lengths[visible]
            if np.any(visible)
            else reference_lengths[fallback]
        )
        if not len(lengths):
            raise ValueError(f"reference bone {start}-{end} has no finite length")
        median_length = float(np.median(lengths))
        desired_lengths[:, bone_index] = (
            np.where(fallback, reference_lengths, median_length)
            if is_free
            else median_length
        )

        source_vectors = source[:, end] - source[:, start]
        source_lengths = np.linalg.norm(source_vectors, axis=-1)
        target_vectors = projected[:, end] - projected[:, start]
        target_lengths = np.linalg.norm(target_vectors, axis=-1)
        valid_direction = (
            np.isfinite(target_lengths)
            & (target_lengths > 1e-8)
            & (observed[:, start] > 0.05)
            & (observed[:, end] > 0.05)
        )
        if not np.any(valid_direction):
            valid_direction = fallback
            target_vectors = source_vectors.copy()
            target_lengths = source_lengths.copy()
        unit = np.zeros_like(target_vectors)
        unit[valid_direction] = (
            target_vectors[valid_direction] / target_lengths[valid_direction, None]
        )
        for dimension in range(unit.shape[-1]):
            unit[:, dimension] = np.interp(
                timeline, timeline[valid_direction], unit[valid_direction, dimension]
            )
        unit /= np.maximum(np.linalg.norm(unit, axis=-1, keepdims=True), 1e-8)
        unreliable = ~valid_direction
        projected[unreliable, end] = (
            projected[unreliable, start]
            + unit[unreliable] * desired_lengths[unreliable, bone_index][:, None]
        )

    bone_indices = np.asarray(BONES, dtype=np.int64)
    pelvis_reference = corrected_source if preserve_target_pelvis else source
    pelvis_anchor = (pelvis_reference[:, 11] + pelvis_reference[:, 12]) / 2.0
    for _ in range(iterations):
        for bone_index, (start, end) in enumerate(BONES):
            vector = projected[:, end] - projected[:, start]
            length = np.linalg.norm(vector, axis=-1)
            valid = length > 1e-8
            adjustment = np.zeros_like(vector)
            adjustment[valid] = (
                0.5
                * (
                    (length[valid] - desired_lengths[valid, bone_index])
                    / length[valid]
                )[:, None]
                * vector[valid]
            )
            projected[:, start] += adjustment
            projected[:, end] -= adjustment
        pelvis = (projected[:, 11] + projected[:, 12]) / 2.0
        projected += (pelvis_anchor - pelvis)[:, None, :]

    for chain in direction_chains:
        for start, end in zip(chain[:-1], chain[1:], strict=True):
            vectors = corrected_source[:, end] - corrected_source[:, start]
            lengths = np.linalg.norm(vectors, axis=-1)
            valid = np.isfinite(lengths) & (lengths > 1e-8)
            if not np.any(valid):
                vectors = projected[:, end] - projected[:, start]
                lengths = np.linalg.norm(vectors, axis=-1)
                valid = np.isfinite(lengths) & (lengths > 1e-8)
            units = np.zeros_like(vectors)
            units[valid] = vectors[valid] / lengths[valid, None]
            for dimension in range(2):
                units[:, dimension] = np.interp(
                    timeline, timeline[valid], units[valid, dimension]
                )
            units /= np.maximum(np.linalg.norm(units, axis=-1, keepdims=True), 1e-8)
            bone_index = bone_lookup[frozenset((start, end))]
            projected[:, end] = (
                projected[:, start] + units * desired_lengths[:, bone_index, None]
            )

    final_lengths = np.linalg.norm(
        projected[:, bone_indices[:, 1]] - projected[:, bone_indices[:, 0]], axis=-1
    )
    if np.any(final_lengths <= 1e-6):
        raise ValueError("stable bone projection produced a collapsed bone")
    return projected.astype(np.float32)


def select_bone_adapted_expert(
    sequence: NDArray[np.floating],
    expert_sequences: NDArray[np.floating],
    confidence: NDArray[np.floating],
    expert_confidence: NDArray[np.floating],
    joint_weights: NDArray[np.floating] = JOINT_WEIGHTS,
    transition_weight: float = 0.0,
    transition_joints: tuple[int, ...] = (),
    transition_lean_joints: tuple[int, ...] = (),
    transition_direction_joint: int | None = None,
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
            transition_weight=transition_weight,
            transition_joints=transition_joints,
            transition_lean_joints=transition_lean_joints,
            transition_direction_joint=transition_direction_joint,
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
