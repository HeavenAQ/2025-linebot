"""Expert-only semantic distribution scoring for badminton smash.

The smash generator and the smash grader intentionally have different jobs.
The generator supplies an articulated correction for visualization.  This
module grades observable checkpoint evidence in the detected motion itself so
that a valid performer is not penalized for choosing a different expert style.

All calibration statistics are fitted from expert RF-DETR skeletons.  Student
poses and ratings are accepted only by the evaluation script, never here.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from numpy.typing import NDArray

from badminton_analysis.ml.skeleton_normalization import phase_align_sequence


_EPS = 1e-8

FEATURE_NAMES = (
    "preparation_racket_wrist_height",
    "preparation_racket_elbow_height",
    "rotation_shoulder_axis_excursion",
    "rotation_hip_axis_excursion",
    "rotation_ankle_stagger",
    "rotation_dominant_elbow_elevation",
    "rotation_non_dominant_elbow_elevation",
    "rotation_elbow_span",
    "contact_dominant_upper_arm_excursion",
    "contact_dominant_elbow_elevation",
    "contact_dominant_upper_arm_length",
    "contact_downward_wrist_displacement",
    "contact_downward_velocity_fraction",
    "contact_downward_acceleration_fraction",
    "contact_forearm_angle_excursion",
    "follow_through_wrist_drop",
    "follow_through_elbow_drop",
    "follow_through_shoulder_axis_excursion",
    "follow_through_cross_body_reach",
    "contact_upper_arm_phase_change",
    "contact_forearm_phase_change",
    "contact_elbow_phase_displacement",
    "rotation_non_dominant_wrist_elevation",
    "rotation_wrist_span",
    "contact_signed_upper_arm_phase_change",
    "contact_signed_forearm_phase_change",
    "contact_downstroke_order_margin",
    "contact_elbow_lead_margin",
)

CRITERION_IDS = (
    "preparation",
    "body_rotation",
    "arm_balance",
    "elbow_forward",
    "wrist_flick",
    "follow_through",
)


def allocate_smash_total_to_weighted_criteria(
    ratios: NDArray[np.floating],
    maxima: NDArray[np.floating],
    total_score: float,
) -> NDArray[np.float64]:
    """Attribute an aggregate smash grade to the product rubric.

    Smash validation uses a nonlinear aggregate across the six qualitative
    checkpoints, while the product UI retains the original
    10/10/20/20/20/20 point layout.  Re-summing ``maximum * ratio`` silently
    replaces the validated aggregate with a different arithmetic scorer.  The
    attribution below preserves the relative checkpoint evidence, respects
    every checkpoint maximum, and sums exactly to the aggregate grade.
    """
    checkpoint_ratios = np.clip(np.asarray(ratios, dtype=np.float64), 0.0, 1.0)
    checkpoint_maxima = np.asarray(maxima, dtype=np.float64)
    if checkpoint_ratios.ndim != 1 or checkpoint_maxima.shape != checkpoint_ratios.shape:
        raise ValueError("smash criterion ratios and maxima must be matching vectors")
    if np.any(checkpoint_maxima < 0.0):
        raise ValueError("smash criterion maxima must be non-negative")
    maximum_total = float(np.sum(checkpoint_maxima))
    target = float(np.clip(total_score, 0.0, maximum_total))
    raw = checkpoint_ratios * checkpoint_maxima
    raw_total = float(np.sum(raw))
    if target <= raw_total and raw_total > _EPS:
        attributed = raw * (target / raw_total)
    elif target > raw_total:
        headroom = np.maximum(checkpoint_maxima - raw, 0.0)
        available = float(np.sum(headroom))
        attributed = (
            raw + (target - raw_total) * headroom / available
            if available > _EPS
            else raw
        )
    elif maximum_total > _EPS:
        attributed = target * checkpoint_maxima / maximum_total
    else:
        attributed = np.zeros_like(checkpoint_maxima)
    # Absorb floating-point residue without violating a criterion cap.
    residue = target - float(np.sum(attributed))
    if abs(residue) > 1e-10 and len(attributed):
        if residue > 0.0:
            index = int(np.argmax(checkpoint_maxima - attributed))
        else:
            index = int(np.argmax(attributed))
        attributed[index] += residue
    return np.clip(attributed, 0.0, checkpoint_maxima)


@dataclass(frozen=True)
class SmashDistribution:
    lower: NDArray[np.float64]
    upper: NDArray[np.float64]
    scale: NDArray[np.float64]
    subject_ids: NDArray[np.str_]
    subject_values: NDArray[np.float64]
    calibration_policy: str


@dataclass(frozen=True)
class SmashVariant:
    name: str
    envelope_policy: str
    decay: float
    aggregation: str
    checkpoint_profile: str


def _bounds(start: float, end: float, length: int) -> tuple[int, int]:
    left = min(length - 1, int(np.floor(start * length)))
    right = min(length, max(left + 1, int(np.ceil(end * length))))
    return left, right


def _wrapped(values: NDArray[np.floating]) -> NDArray[np.float64]:
    array = np.asarray(values, dtype=np.float64)
    return (array + np.pi) % (2.0 * np.pi) - np.pi


def _angle(vector: NDArray[np.floating]) -> NDArray[np.float64]:
    values = np.asarray(vector, dtype=np.float64)
    return np.unwrap(np.arctan2(values[:, 1], values[:, 0]))


def _smooth(values: NDArray[np.floating]) -> NDArray[np.float64]:
    trajectory = np.asarray(values, dtype=np.float64)
    padded = np.pad(trajectory, ((2, 2), (0, 0)), mode="edge")
    kernel = np.asarray((1.0, 2.0, 3.0, 2.0, 1.0)) / 9.0
    return np.stack(
        [np.convolve(padded[:, axis], kernel, mode="valid") for axis in range(2)],
        axis=-1,
    )


def extract_smash_evidence(
    pose: NDArray[np.floating],
    confidence: NDArray[np.floating],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Return camera/scale-invariant semantic evidence and cue reliability.

    ``pose`` must already use the canonical handedness convention.  The local
    coordinate frame is estimated once from the preparation interval, matching
    the fixed ankle-spine projection contract; it is not recomputed per frame.
    """
    values = np.asarray(pose, dtype=np.float64)
    observed = np.clip(np.asarray(confidence, dtype=np.float64), 0.0, 1.0)
    if values.ndim != 3 or values.shape[1:] != (17, 2):
        raise ValueError("smash evidence requires pose shape (T, 17, 2)")
    if observed.shape != values.shape[:2]:
        raise ValueError("smash evidence confidence must have shape (T, 17)")
    length = len(values)
    preparation = _bounds(0.0, 0.25, length)
    rotation = _bounds(0.125, 0.5, length)
    arm_window = _bounds(0.25, 0.625, length)
    contact = _bounds(0.421875, 0.59375, length)
    wrist_window = _bounds(0.375, 0.75, length)
    follow = _bounds(0.625, 1.0, length)

    hip = 0.5 * (values[:, 11] + values[:, 12])
    shoulder = 0.5 * (values[:, 5] + values[:, 6])
    spine = shoulder - hip
    torso = max(float(np.median(np.linalg.norm(spine, axis=-1))), _EPS)
    prep = slice(*preparation)
    vertical = np.median(spine[prep], axis=0)
    vertical /= max(float(np.linalg.norm(vertical)), _EPS)
    ankle_axis = np.median(values[prep, 16] - values[prep, 15], axis=0)
    ankle_axis -= vertical * float(np.dot(ankle_axis, vertical))
    horizontal = ankle_axis / max(float(np.linalg.norm(ankle_axis)), _EPS)

    def local(joint: int) -> NDArray[np.float64]:
        relative = (values[:, joint] - hip) / torso
        return np.stack((relative @ horizontal, relative @ vertical), axis=-1)

    local_pose = {joint: local(joint) for joint in range(5, 17)}

    def upper_quantile(signal: NDArray[np.floating], window: tuple[int, int]) -> float:
        return float(np.quantile(np.asarray(signal)[slice(*window)], 0.85))

    def excursion(signal: NDArray[np.floating], window: tuple[int, int]) -> float:
        baseline = float(np.median(np.asarray(signal)[prep]))
        delta = np.abs(_wrapped(np.asarray(signal)[slice(*window)] - baseline))
        return float(np.quantile(delta, 0.90))

    shoulder_angle = _angle(values[:, 6] - values[:, 5])
    hip_angle = _angle(values[:, 12] - values[:, 11])
    upper_arm_angle = _angle(values[:, 8] - values[:, 6])
    forearm_angle = _angle(values[:, 10] - values[:, 8])

    relative_wrist = (values[:, 10] - values[:, 6]) / torso
    smoothed_wrist = _smooth(relative_wrist)
    velocity = np.diff(smoothed_wrist, axis=0)
    acceleration = np.diff(smoothed_wrist, n=2, axis=0)
    downward_velocity = -(velocity @ vertical)
    downward_acceleration = -(acceleration @ vertical)
    wrist_start, wrist_end = wrist_window
    speed_slice = slice(wrist_start, max(wrist_start + 1, wrist_end - 1))
    acceleration_slice = slice(
        wrist_start, max(wrist_start + 1, wrist_end - 2)
    )
    peak = wrist_start + int(
        np.argmax(relative_wrist[wrist_start:wrist_end] @ vertical)
    )
    terminal = max(peak + 1, wrist_end - 1)
    downward_displacement = max(
        0.0,
        float((relative_wrist[peak] - relative_wrist[terminal]) @ vertical),
    )
    selected_velocity = downward_velocity[speed_slice]
    selected_acceleration = downward_acceleration[acceleration_slice]
    downward_velocity_fraction = float(
        np.sum(np.maximum(selected_velocity, 0.0))
        / max(float(np.sum(np.abs(selected_velocity))), _EPS)
    )
    downward_acceleration_fraction = float(
        np.sum(np.maximum(selected_acceleration, 0.0))
        / max(float(np.sum(np.abs(selected_acceleration))), _EPS)
    )
    rotation_phase = _bounds(0.25, 0.421875, length)
    contact_phase = _bounds(0.421875, 0.59375, length)

    def signed_phase_angle_change(signal: NDArray[np.floating]) -> float:
        first = float(np.median(np.asarray(signal)[slice(*rotation_phase)]))
        second = float(np.median(np.asarray(signal)[slice(*contact_phase)]))
        return float(_wrapped(np.asarray(second - first)))

    elbow_phase_displacement = float(
        np.linalg.norm(
            np.mean(local_pose[8][slice(*contact_phase)], axis=0)
            - np.mean(local_pose[8][slice(*rotation_phase)], axis=0)
        )
    )
    upper_arm_signed_change = signed_phase_angle_change(upper_arm_angle)
    forearm_signed_change = signed_phase_angle_change(forearm_angle)
    height_search = slice(*_bounds(0.375, 0.6875, length))
    downstroke_search = slice(*_bounds(0.50, 0.875, length - 1))
    wrist_height_index = height_search.start + int(
        np.argmax(relative_wrist[height_search] @ vertical)
    )
    downstroke_index = downstroke_search.start + int(
        np.argmax(downward_velocity[downstroke_search])
    )
    upper_arm_speed = np.abs(np.diff(_smooth(upper_arm_angle[:, None].repeat(2, axis=1))[:, 0]))
    elbow_search = slice(*_bounds(0.3125, 0.6875, length - 1))
    elbow_index = elbow_search.start + int(
        np.argmax(upper_arm_speed[elbow_search])
    )

    evidence = np.asarray(
        (
            upper_quantile(local_pose[10][:, 1], preparation),
            upper_quantile(local_pose[8][:, 1], preparation),
            excursion(shoulder_angle, rotation),
            excursion(hip_angle, rotation),
            upper_quantile(
                np.abs(local_pose[16][:, 1] - local_pose[15][:, 1]),
                rotation,
            ),
            upper_quantile(
                local_pose[8][:, 1] - local_pose[6][:, 1], arm_window
            ),
            upper_quantile(
                local_pose[7][:, 1] - local_pose[5][:, 1], arm_window
            ),
            upper_quantile(
                np.linalg.norm(local_pose[8] - local_pose[7], axis=-1),
                arm_window,
            ),
            excursion(upper_arm_angle, contact),
            upper_quantile(
                local_pose[8][:, 1] - local_pose[6][:, 1], contact
            ),
            upper_quantile(
                np.linalg.norm(local_pose[8] - local_pose[6], axis=-1),
                contact,
            ),
            downward_displacement,
            downward_velocity_fraction,
            downward_acceleration_fraction,
            excursion(forearm_angle, wrist_window),
            upper_quantile(-local_pose[10][:, 1], follow),
            upper_quantile(-local_pose[8][:, 1], follow),
            excursion(shoulder_angle, follow),
            upper_quantile(-local_pose[10][:, 0], follow),
            abs(upper_arm_signed_change),
            abs(forearm_signed_change),
            elbow_phase_displacement,
            upper_quantile(
                local_pose[9][:, 1] - local_pose[5][:, 1], arm_window
            ),
            upper_quantile(
                np.linalg.norm(local_pose[10] - local_pose[9], axis=-1),
                arm_window,
            ),
            upper_arm_signed_change,
            forearm_signed_change,
            float((downstroke_index - wrist_height_index) / max(length - 1, 1)),
            float((downstroke_index - elbow_index) / max(length - 1, 1)),
        ),
        dtype=np.float64,
    )

    def reliability(joints: Sequence[int], window: tuple[int, int]) -> float:
        joint_confidence = np.min(observed[:, joints], axis=1)
        return float(np.quantile(joint_confidence[slice(*window)], 0.50))

    reliability_values = np.asarray(
        (
            reliability((6, 10, 11, 12), preparation),
            reliability((6, 8, 11, 12), preparation),
            reliability((5, 6), rotation),
            reliability((11, 12), rotation),
            reliability((15, 16), rotation),
            reliability((6, 8), arm_window),
            reliability((5, 7), arm_window),
            reliability((5, 6, 7, 8), arm_window),
            reliability((6, 8), contact),
            reliability((6, 8), contact),
            reliability((6, 8), contact),
            reliability((6, 8, 10), wrist_window),
            reliability((6, 8, 10), wrist_window),
            reliability((6, 8, 10), wrist_window),
            reliability((6, 8, 10), wrist_window),
            reliability((6, 10), follow),
            reliability((6, 8), follow),
            reliability((5, 6), follow),
            reliability((6, 10), follow),
            reliability((6, 8), contact_phase),
            reliability((6, 8, 10), contact_phase),
            reliability((6, 8), contact_phase),
            reliability((5, 9), arm_window),
            reliability((5, 6, 9, 10), arm_window),
            reliability((6, 8), contact_phase),
            reliability((6, 8, 10), contact_phase),
            reliability((6, 8, 10), wrist_window),
            reliability((6, 8, 10), wrist_window),
        ),
        dtype=np.float64,
    )
    return evidence, reliability_values


def aligned_smash_evidence(
    pose: NDArray[np.floating],
    confidence: NDArray[np.floating],
    phase_indices: NDArray[np.integer],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    return extract_smash_evidence(
        phase_align_sequence(pose, phase_indices),
        phase_align_sequence(confidence, phase_indices),
    )


def fit_smash_distribution(
    evidence: NDArray[np.floating],
    subject_ids: Sequence[str],
    *,
    policy: str = "identity_p10",
) -> SmashDistribution:
    matrix = np.asarray(evidence, dtype=np.float64)
    subjects = np.asarray(subject_ids, dtype=np.str_)
    if matrix.ndim != 2 or matrix.shape[1] != len(FEATURE_NAMES):
        raise ValueError("smash evidence matrix has the wrong shape")
    if len(subjects) != len(matrix) or not len(matrix):
        raise ValueError("one expert subject id is required per evidence row")
    identities = np.asarray(sorted(set(subjects.tolist())), dtype=np.str_)
    subject_values = np.stack(
        [np.median(matrix[subjects == identity], axis=0) for identity in identities]
    )
    clip_median = np.median(matrix, axis=0)
    within_take_scale = 1.4826 * np.median(
        np.abs(matrix - clip_median[None]), axis=0
    )
    if policy == "identity_p10":
        lower = np.quantile(subject_values, 0.10, axis=0) - within_take_scale
        upper = np.quantile(subject_values, 0.90, axis=0) + within_take_scale
    elif policy == "identity_support":
        lower = np.min(subject_values, axis=0) - within_take_scale
        upper = np.max(subject_values, axis=0) + within_take_scale
    elif policy == "clip_support":
        lower = np.min(matrix, axis=0) - within_take_scale
        upper = np.max(matrix, axis=0) + within_take_scale
    else:
        raise ValueError(f"unknown smash expert-envelope policy: {policy}")
    identity_median = np.median(subject_values, axis=0)
    scale = np.maximum.reduce(
        (
            identity_median - lower,
            upper - identity_median,
            within_take_scale,
            0.10 * np.maximum(np.abs(identity_median), 0.30),
            np.full(len(FEATURE_NAMES), 1e-3),
        )
    )
    return SmashDistribution(
        lower=lower,
        upper=upper,
        scale=scale,
        subject_ids=identities,
        subject_values=subject_values,
        calibration_policy=policy,
    )


def _feature_deficiency(
    evidence: NDArray[np.floating], distribution: SmashDistribution
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    values = np.asarray(evidence, dtype=np.float64)
    lower = np.maximum(distribution.lower - values, 0.0) / distribution.scale
    upper = np.maximum(values - distribution.upper, 0.0) / distribution.scale
    bounded = np.maximum(lower, upper)
    return lower, upper, bounded


def score_smash_evidence(
    evidence: NDArray[np.floating],
    reliability: NDArray[np.floating],
    distribution: SmashDistribution,
    variant: SmashVariant,
) -> dict[str, Any]:
    """Score six qualitative checkpoints without generated-pose distance."""
    values = np.asarray(evidence, dtype=np.float64)
    cue_reliability = np.clip(np.asarray(reliability, dtype=np.float64), 0.0, 1.0)
    lower, _, bounded = _feature_deficiency(values, distribution)

    if variant.checkpoint_profile == "semantic_base":
        distances = np.asarray(
            (
                lower[0],
                min(lower[2], lower[3]),
                lower[6],
                np.sqrt(0.5 * (lower[19] ** 2 + lower[21] ** 2)),
                np.sqrt(0.5 * (lower[11] ** 2 + lower[20] ** 2)),
                np.sqrt(0.5 * (lower[17] ** 2 + lower[18] ** 2)),
            )
        )
        reliabilities = np.asarray(
            (
                cue_reliability[0],
                max(cue_reliability[2], cue_reliability[3]),
                cue_reliability[6],
                min(cue_reliability[19], cue_reliability[21]),
                min(cue_reliability[11], cue_reliability[20]),
                min(cue_reliability[17], cue_reliability[18]),
            )
        )
        aggregations = (
            "racket_wrist_height",
            "shoulder_or_hip_rotation",
            "non_dominant_elbow_support",
            "upper_arm_phase_change_and_elbow_displacement",
            "downward_displacement_and_forearm_phase_change",
            "shoulder_rotation_and_cross_body_completion",
        )
    elif variant.checkpoint_profile == "semantic_strict_rotation":
        distances = np.asarray(
            (
                lower[0],
                max(min(lower[2], lower[3]), lower[4]),
                np.sqrt(0.5 * (lower[5] ** 2 + lower[6] ** 2)),
                np.sqrt(0.5 * (lower[19] ** 2 + lower[21] ** 2)),
                np.sqrt(0.5 * (lower[11] ** 2 + lower[20] ** 2)),
                np.sqrt(0.5 * (lower[17] ** 2 + lower[18] ** 2)),
            )
        )
        reliabilities = np.asarray(
            (
                cue_reliability[0],
                min(max(cue_reliability[2], cue_reliability[3]), cue_reliability[4]),
                min(cue_reliability[5], cue_reliability[6]),
                min(cue_reliability[19], cue_reliability[21]),
                min(cue_reliability[11], cue_reliability[20]),
                min(cue_reliability[17], cue_reliability[18]),
            )
        )
        aggregations = (
            "racket_wrist_height",
            "body_rotation_and_ankle_stagger",
            "both_elbows_raised",
            "upper_arm_phase_change_and_elbow_displacement",
            "downward_displacement_and_forearm_phase_change",
            "shoulder_rotation_and_cross_body_completion",
        )
    elif variant.checkpoint_profile in {
        "semantic_occlusion_robust",
        "semantic_bounded_temporal",
    }:
        bounded_temporal = (
            variant.checkpoint_profile == "semantic_bounded_temporal"
        )
        elbow_distance = (
            np.sqrt(0.5 * (bounded[24] ** 2 + lower[21] ** 2))
            if bounded_temporal
            else min(lower[19], lower[21])
        )
        wrist_distance = (
            np.sqrt(
                np.mean(
                    (
                        lower[11] ** 2,
                        bounded[25] ** 2,
                        lower[26] ** 2,
                        lower[27] ** 2,
                    )
                )
            )
            if bounded_temporal
            else np.sqrt(0.5 * (lower[11] ** 2 + lower[20] ** 2))
        )
        follow_arm_completion = np.sqrt(
            0.5 * (lower[15] ** 2 + lower[18] ** 2)
        )
        distances = np.asarray(
            (
                min(lower[0], lower[1]),
                min(lower[2], lower[3]),
                min(lower[6], lower[22]),
                elbow_distance,
                wrist_distance,
                min(lower[17], follow_arm_completion),
            )
        )
        reliabilities = np.asarray(
            (
                max(cue_reliability[0], cue_reliability[1]),
                max(cue_reliability[2], cue_reliability[3]),
                max(cue_reliability[6], cue_reliability[22]),
                min(cue_reliability[24], cue_reliability[21])
                if bounded_temporal
                else max(cue_reliability[19], cue_reliability[21]),
                min(
                    cue_reliability[11],
                    cue_reliability[25] if bounded_temporal else cue_reliability[20],
                ),
                max(
                    cue_reliability[17],
                    min(cue_reliability[15], cue_reliability[18]),
                ),
            )
        )
        aggregations = (
            "racket_wrist_or_elbow_height",
            "shoulder_or_hip_rotation",
            "non_dominant_elbow_or_wrist_support",
            (
                "bounded_signed_upper_arm_change_and_elbow_displacement"
                if bounded_temporal
                else "upper_arm_change_or_elbow_displacement"
            ),
            (
                "bounded_signed_forearm_change_and_ordered_downstroke"
                if bounded_temporal
                else "downward_displacement_and_forearm_phase_change"
            ),
            "shoulder_rotation_or_wrist_drop_cross_body_completion",
        )
    else:
        raise ValueError(f"unknown smash checkpoint profile: {variant.checkpoint_profile}")

    raw_ratios = np.exp(-distances / max(float(variant.decay), 1e-3))
    # An unobserved elbow is not evidence of an incorrect elbow. Blend toward
    # an explicitly uncertain half-credit state instead of silently treating a
    # detector miss as zero motion or awarding full credit.
    ratios = reliabilities * raw_ratios + (1.0 - reliabilities) * 0.5
    if variant.aggregation == "arithmetic":
        total_ratio = float(np.mean(ratios))
    elif variant.aggregation == "geometric":
        total_ratio = float(np.exp(np.mean(np.log(np.maximum(ratios, 0.03)))))
    elif variant.aggregation == "harmonic":
        total_ratio = float(len(ratios) / np.sum(1.0 / np.maximum(ratios, 0.03)))
    elif variant.aggregation == "power_minus_half":
        total_ratio = float(
            np.mean(np.maximum(ratios, 0.03) ** -0.5) ** -2.0
        )
    elif variant.aggregation == "power_minus_two":
        total_ratio = float(
            np.mean(np.maximum(ratios, 0.03) ** -2.0) ** -0.5
        )
    elif variant.aggregation == "rubric_weighted_geometric":
        rubric_weights = np.asarray((1.0, 1.0, 2.0, 2.0, 2.0, 2.0))
        total_ratio = float(
            np.exp(
                np.sum(rubric_weights * np.log(np.maximum(ratios, 0.03)))
                / np.sum(rubric_weights)
            )
        )
    elif variant.aggregation == "rubric_weighted_arithmetic":
        rubric_weights = np.asarray((1.0, 1.0, 2.0, 2.0, 2.0, 2.0))
        total_ratio = float(
            np.sum(rubric_weights * ratios) / np.sum(rubric_weights)
        )
    else:
        raise ValueError(f"unknown smash checkpoint aggregation: {variant.aggregation}")

    criteria = []
    for index, criterion_id in enumerate(CRITERION_IDS):
        criteria.append(
            {
                "rule_reference": criterion_id,
                "score": float(100.0 / 6.0 * ratios[index]),
                "maximum": float(100.0 / 6.0),
                "ratio": float(ratios[index]),
                "semantic_distance": float(distances[index]),
                "cue_reliability": float(reliabilities[index]),
                "semantic_cue_aggregation": aggregations[index],
            }
        )
    return {
        "total_score": 100.0 * total_ratio,
        "score_method": "smash_expert_only_semantic_distribution_v1",
        "criterion_metric_version": "smash_expert_distribution_v1",
        "calibration_policy": distribution.calibration_policy,
        "variant": variant.name,
        "student_data_used_for_training_or_calibration": False,
        "criteria": criteria,
        "evidence": {
            name: float(value) for name, value in zip(FEATURE_NAMES, values, strict=True)
        },
    }


def save_smash_distribution(
    distribution: SmashDistribution, variant: SmashVariant, path: str | Path
) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        destination,
        format_version=np.asarray(1, dtype=np.int64),
        method=np.asarray("smash_expert_only_semantic_distribution_v1"),
        feature_names=np.asarray(FEATURE_NAMES),
        criterion_ids=np.asarray(CRITERION_IDS),
        lower=distribution.lower,
        upper=distribution.upper,
        scale=distribution.scale,
        subject_ids=distribution.subject_ids,
        subject_values=distribution.subject_values,
        calibration_policy=np.asarray(distribution.calibration_policy),
        variant_name=np.asarray(variant.name),
        decay=np.asarray(variant.decay, dtype=np.float64),
        aggregation=np.asarray(variant.aggregation),
        checkpoint_profile=np.asarray(variant.checkpoint_profile),
        student_data_used_for_training_or_calibration=np.asarray(False),
    )


def load_smash_distribution(
    path: str | Path,
) -> tuple[SmashDistribution, SmashVariant]:
    with np.load(path, allow_pickle=False) as archive:
        if str(archive["method"].item()) != "smash_expert_only_semantic_distribution_v1":
            raise ValueError("not a smash semantic distribution artifact")
        if tuple(archive["feature_names"].tolist()) != FEATURE_NAMES:
            raise ValueError("smash semantic feature contract mismatch")
        distribution = SmashDistribution(
            lower=np.asarray(archive["lower"], dtype=np.float64),
            upper=np.asarray(archive["upper"], dtype=np.float64),
            scale=np.asarray(archive["scale"], dtype=np.float64),
            subject_ids=np.asarray(archive["subject_ids"], dtype=np.str_),
            subject_values=np.asarray(archive["subject_values"], dtype=np.float64),
            calibration_policy=str(archive["calibration_policy"].item()),
        )
        variant = SmashVariant(
            name=str(archive["variant_name"].item()),
            envelope_policy=distribution.calibration_policy,
            decay=float(archive["decay"].item()),
            aggregation=str(archive["aggregation"].item()),
            checkpoint_profile=str(archive["checkpoint_profile"].item()),
        )
    return distribution, variant
