"""What a smash is measured against: models fitted on expert takes only."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence
import math

from numpy.typing import NDArray
from scipy.ndimage import median_filter
from torch import nn
import numpy as np
import torch
import torch.nn.functional as F

from badminton_analysis.ml.skeleton_normalization import phase_align_sequence


_EPS = 1e-8


# The keypoint confidence a clearly seen joint reaches.
SEEN_JOINT_CONFIDENCE = 0.853


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
    """Attribute an aggregate smash grade to the product rubric."""
    checkpoint_ratios = np.clip(np.asarray(ratios, dtype=np.float64), 0.0, 1.0)
    checkpoint_maxima = np.asarray(maxima, dtype=np.float64)
    if (
        checkpoint_ratios.ndim != 1
        or checkpoint_maxima.shape != checkpoint_ratios.shape
    ):
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
    """Return camera/scale-invariant semantic evidence and cue reliability."""
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
    acceleration_slice = slice(wrist_start, max(wrist_start + 1, wrist_end - 2))
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
    upper_arm_speed = np.abs(
        np.diff(_smooth(upper_arm_angle[:, None].repeat(2, axis=1))[:, 0])
    )
    elbow_search = slice(*_bounds(0.3125, 0.6875, length - 1))
    elbow_index = elbow_search.start + int(np.argmax(upper_arm_speed[elbow_search]))

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
            upper_quantile(local_pose[8][:, 1] - local_pose[6][:, 1], arm_window),
            upper_quantile(local_pose[7][:, 1] - local_pose[5][:, 1], arm_window),
            upper_quantile(
                np.linalg.norm(local_pose[8] - local_pose[7], axis=-1),
                arm_window,
            ),
            excursion(upper_arm_angle, contact),
            upper_quantile(local_pose[8][:, 1] - local_pose[6][:, 1], contact),
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
            upper_quantile(local_pose[9][:, 1] - local_pose[5][:, 1], arm_window),
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
        typical = float(np.quantile(joint_confidence[slice(*window)], 0.50))
        return float(np.clip(typical / SEEN_JOINT_CONFIDENCE, 0.0, 1.0))

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
                # Follow-through is the shoulder turn alone; cross-body wrist reach no longer counts.
                lower[17],
            )
        )
        reliabilities = np.asarray(
            (
                cue_reliability[0],
                max(cue_reliability[2], cue_reliability[3]),
                cue_reliability[6],
                min(cue_reliability[19], cue_reliability[21]),
                min(cue_reliability[11], cue_reliability[20]),
                cue_reliability[17],
            )
        )
        aggregations = (
            "racket_wrist_height",
            "shoulder_or_hip_rotation",
            "non_dominant_elbow_support",
            "upper_arm_phase_change_and_elbow_displacement",
            "downward_displacement_and_forearm_phase_change",
            "shoulder_rotation",
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
        bounded_temporal = variant.checkpoint_profile == "semantic_bounded_temporal"
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
        follow_arm_completion = np.sqrt(0.5 * (lower[15] ** 2 + lower[18] ** 2))
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
                (
                    min(cue_reliability[24], cue_reliability[21])
                    if bounded_temporal
                    else max(cue_reliability[19], cue_reliability[21])
                ),
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
        raise ValueError(
            f"unknown smash checkpoint profile: {variant.checkpoint_profile}"
        )

    raw_ratios = np.exp(-distances / max(float(variant.decay), 1e-3))
    # An unobserved elbow is not evidence of an incorrect elbow.
    ratios = reliabilities * raw_ratios + (1.0 - reliabilities) * 0.5
    if variant.aggregation == "arithmetic":
        total_ratio = float(np.mean(ratios))
    elif variant.aggregation == "geometric":
        total_ratio = float(np.exp(np.mean(np.log(np.maximum(ratios, 0.03)))))
    elif variant.aggregation == "harmonic":
        total_ratio = float(len(ratios) / np.sum(1.0 / np.maximum(ratios, 0.03)))
    elif variant.aggregation == "power_minus_half":
        total_ratio = float(np.mean(np.maximum(ratios, 0.03) ** -0.5) ** -2.0)
    elif variant.aggregation == "power_minus_two":
        total_ratio = float(np.mean(np.maximum(ratios, 0.03) ** -2.0) ** -0.5)
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
        total_ratio = float(np.sum(rubric_weights * ratios) / np.sum(rubric_weights))
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
            name: float(value)
            for name, value in zip(FEATURE_NAMES, values, strict=True)
        },
    }


def load_smash_distribution(
    path: str | Path,
) -> tuple[SmashDistribution, SmashVariant]:
    with np.load(path, allow_pickle=False) as archive:
        if (
            str(archive["method"].item())
            != "smash_expert_only_semantic_distribution_v1"
        ):
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


EDGES = (
    (0, 1),
    (0, 2),
    (1, 3),
    (2, 4),
    (5, 6),
    (5, 7),
    (7, 9),
    (6, 8),
    (8, 10),
    (5, 11),
    (6, 12),
    (11, 12),
    (11, 13),
    (13, 15),
    (12, 14),
    (14, 16),
)


def normalized_graph_pose(pose):
    p = np.asarray(pose, dtype=np.float32)
    if p.shape != (64, 17, 2) or not np.isfinite(p).all():
        raise ValueError("Expected finite COCO17 sequence")
    pelvis = p[:, [11, 12]].mean(1)
    scale = np.median(np.linalg.norm(p[:5, [5, 6]].mean(1) - pelvis[:5], axis=1))
    if scale < 1e-6:
        raise ValueError("Degenerate preparation torso")
    # Feature coordinates only; does not change either video overlay.
    return (p - pelvis[:, None]) / scale


def checkpoint_index_map(phase_indices, anchor_groups):
    phases = np.asarray(phase_indices)
    if (
        phases.shape != (5,)
        or not np.issubdtype(phases.dtype, np.integer)
        or phases[0] != 0
        or phases[-1] != 63
        or np.any(np.diff(phases) <= 0)
    ):
        raise ValueError("Expected five increasing cached anchors spanning 64 frames")
    windows, rule_indices, anchor_indices = [], [], []
    for rule, anchors in enumerate(anchor_groups):
        if not anchors:
            raise ValueError("Each checkpoint needs specified anchors")
        for anchor in anchors:
            if anchor not in range(5):
                raise ValueError("Invalid checkpoint anchor")
            windows.append(np.clip(phases[anchor] + np.arange(-2, 3), 0, 63))
            rule_indices.append(rule)
            anchor_indices.append(anchor)
    return np.asarray(windows), np.asarray(rule_indices), np.asarray(anchor_indices)


def checkpoint_windows(pose, confidence, indices):
    p = normalized_graph_pose(pose)
    c = np.asarray(confidence, np.float32)
    if c.shape != (64, 17) or not np.isfinite(c).all():
        raise ValueError("Expected finite confidence")
    indices = np.asarray(indices)
    if (
        indices.ndim != 2
        or indices.shape[1] != 5
        or not np.issubdtype(indices.dtype, np.integer)
        or np.any((indices < 0) | (indices > 63))
    ):
        raise ValueError("Explicit in-range five-frame checkpoint indices required")
    return p[indices], np.clip(c[indices], 0, 1)


def interpolate_vector_direction_length(left, right, amount):
    """Shortest-arc 2D direction interpolation with linearly varying length."""
    a, b = np.asarray(left, float), np.asarray(right, float)
    la, lb = np.linalg.norm(a, axis=-1), np.linalg.norm(b, axis=-1)
    ta, tb = np.arctan2(a[..., 1], a[..., 0]), np.arctan2(b[..., 1], b[..., 0])
    ta = np.where(la > 1e-8, ta, tb)
    tb = np.where(lb > 1e-8, tb, ta)
    delta = np.arctan2(np.sin(tb - ta), np.cos(tb - ta))
    theta = ta + amount * delta
    length = (1 - amount) * la + amount * lb
    return np.stack((np.cos(theta), np.sin(theta)), axis=-1) * length[..., None]


def kinematic_temporal_interpolation(left, right, amount):
    """Interpolate only a reference's own geometry, with no observed-pose fit."""
    a, b = np.asarray(left, float), np.asarray(right, float)
    if a.shape != b.shape or a.shape[-2:] != (17, 2) or amount.shape != a.shape[:-2]:
        raise ValueError("Expected matching COCO17 endpoint poses and fractions")
    pelvis_a, pelvis_b = a[..., [11, 12], :].mean(-2), b[..., [11, 12], :].mean(-2)
    shoulders_a, shoulders_b = a[..., [5, 6], :].mean(-2), b[..., [5, 6], :].mean(-2)
    pelvis = (1 - amount[..., None]) * pelvis_a + amount[..., None] * pelvis_b
    vector = lambda x, y: interpolate_vector_direction_length(x, y, amount)
    shoulders = pelvis + vector(shoulders_a - pelvis_a, shoulders_b - pelvis_b)
    hip_axis = vector(a[..., 12, :] - a[..., 11, :], b[..., 12, :] - b[..., 11, :])
    shoulder_axis = vector(a[..., 6, :] - a[..., 5, :], b[..., 6, :] - b[..., 5, :])
    out = np.zeros_like(a)
    out[..., 11, :], out[..., 12, :] = pelvis - hip_axis / 2, pelvis + hip_axis / 2
    out[..., 5, :], out[..., 6, :] = (
        shoulders - shoulder_axis / 2,
        shoulders + shoulder_axis / 2,
    )
    out[..., 0, :] = shoulders + vector(
        a[..., 0, :] - shoulders_a, b[..., 0, :] - shoulders_b
    )
    for parent, child in (
        (0, 1),
        (0, 2),
        (1, 3),
        (2, 4),
        (5, 7),
        (7, 9),
        (6, 8),
        (8, 10),
        (11, 13),
        (13, 15),
        (12, 14),
        (14, 16),
    ):
        out[..., child, :] = out[..., parent, :] + vector(
            a[..., child, :] - a[..., parent, :], b[..., child, :] - b[..., parent, :]
        )
    return out


def checkpoint_reference_at_source_frames(
    pose, native_phases, source_phases, target_source_frames, interpolation="cartesian"
):
    """Sample the corrected pose at exactly the observed checkpoint timestamps."""
    native = np.asarray(native_phases, float)
    source = np.asarray(source_phases, float)
    target = np.asarray(target_source_frames, float)
    if (
        native.shape != (5,)
        or source.shape != (5,)
        or not np.isfinite(native).all()
        or not np.isfinite(source).all()
        or native[0] != 0
        or native[-1] != 63
        or np.any(np.diff(native) <= 0)
        or np.any(np.diff(source) <= 0)
    ):
        raise ValueError("Expected five increasing native/source anchors")
    if target.ndim != 2 or target.shape[1] != 5 or not np.isfinite(target).all():
        raise ValueError("Expected explicit five-frame source checkpoint windows")
    if (
        np.any(target < source[0])
        or np.any(target > source[-1])
        or np.any(np.diff(target, axis=1) < 0)
    ):
        raise ValueError("Target times must be ordered and inside the source interval")
    query = np.interp(target, source, native)
    before = np.floor(query).astype(int)
    after = np.minimum(before + 1, 63)
    alpha = (query - before)[..., None, None]
    normalized = normalized_graph_pose(pose)
    if interpolation == "cartesian":
        values = (1 - alpha) * normalized[before] + alpha * normalized[after]
    elif interpolation == "kinematic":
        values = kinematic_temporal_interpolation(
            normalized[before], normalized[after], query - before
        )
    else:
        raise ValueError("Expected cartesian or kinematic temporal interpolation")
    return values.astype(np.float32), query


def graph_inputs(coordinates, confidence, visible=None):
    """No masked target leaks through velocity or confidence channels."""
    if coordinates.ndim != 4 or coordinates.shape[1:] != (5, 17, 2):
        raise ValueError("Expected B,5,17,2 checkpoint coordinates")
    if confidence.shape != coordinates.shape[:-1]:
        raise ValueError("Confidence shape mismatch")
    visible = torch.ones_like(confidence) if visible is None else visible
    if visible.shape != confidence.shape:
        raise ValueError("Visibility shape mismatch")
    positions = coordinates * visible[..., None]
    valid_pair = visible[:, 1:] * visible[:, :-1]
    velocity = torch.zeros_like(positions)
    velocity[:, 1:] = (positions[:, 1:] - positions[:, :-1]) * valid_pair[..., None]
    values = torch.cat(
        [positions, velocity, (confidence * visible)[..., None], visible[..., None]],
        dim=-1,
    )
    return values.contiguous()


class GraphTemporalBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        adjacency = torch.zeros(17, 17)
        for a, b in EDGES:
            adjacency[a, b] = adjacency[b, a] = 1
        adjacency /= adjacency.sum(1, keepdim=True).clamp_min(1)
        self.register_buffer("adjacency", adjacency)
        self.self_map = nn.Linear(channels, channels)
        self.neighbor_map = nn.Linear(channels, channels, bias=False)
        self.temporal = nn.Linear(3 * channels, channels)
        self.norm = nn.LayerNorm(channels)

    def forward(self, x):
        # Spatial aggregation with an explicit shared adjacency matrix.
        neighbors = torch.matmul(self.adjacency, x)
        spatial = F.silu(self.self_map(x) + self.neighbor_map(neighbors))
        # Explicit three-tap temporal convolution in channel-last layout avoids this environment's MPS Conv2d-backward noncontiguous-view failure.
        padded = F.pad(spatial, (0, 0, 0, 0, 1, 1))
        taps = torch.cat([padded[:, :-2], padded[:, 1:-1], padded[:, 2:]], dim=-1)
        return F.silu(x + self.norm(self.temporal(taps)))


class CheckpointGraphEncoder(nn.Module):
    def __init__(self, channels=32, layers=3, input_channels=6):
        super().__init__()
        self.config = {
            "channels": channels,
            "layers": layers,
            "input_channels": input_channels,
        }
        self.input = nn.Linear(input_channels, channels)
        self.blocks = nn.Sequential(
            *(GraphTemporalBlock(channels) for _ in range(layers))
        )
        self.decoder = nn.Linear(channels, 2)

    def encode(self, x):
        return self.blocks(F.silu(self.input(x)))

    def forward(self, x):
        return self.decoder(self.encode(x))


class CheckpointMetricGraph(nn.Module):
    def __init__(self, measured_joints, active_rules=(0, 2, 3, 4)):
        super().__init__()
        if (
            len(measured_joints) != 6
            or not active_rules
            or any(k not in range(6) for k in active_rules)
        ):
            raise ValueError("Invalid checkpoint contract")
        self.config = dict(
            measured_joints=measured_joints, active_rules=list(active_rules)
        )
        self.active_rules = tuple(active_rules)
        weights = torch.zeros(6, 17)
        for k, joints in enumerate(measured_joints):
            if not joints or any(j not in range(17) for j in joints):
                raise ValueError("Invalid measured joints")
            weights[k, joints] = 1 / len(joints)
        self.register_buffer("pool_weights", weights)
        active = torch.zeros(6, dtype=torch.bool)
        active[list(active_rules)] = True
        self.register_buffer("active", active)
        self.encoder = CheckpointGraphEncoder(input_channels=12)
        self.encoder.decoder = nn.Identity()
        self.projection = nn.Linear(32, 6 * 16)
        self.raw_intercept = nn.Parameter(torch.full((6,), math.log(math.expm1(2.0))))
        self.raw_scale = nn.Parameter(torch.full((6,), math.log(math.expm1(8.0))))

    def encode(self, pose, confidence, rule_ids):
        inputs = graph_inputs(pose, confidence)
        identity = (
            F.one_hot(rule_ids, 6).to(inputs.dtype)[:, None, None].expand(-1, 5, 17, -1)
        )
        features = self.encoder.encode(torch.cat((inputs, identity), dim=-1))
        pooled = (features * self.pool_weights[rule_ids, None, :, None]).sum(2).mean(1)
        projected = self.projection(pooled).reshape(-1, 6, 16)
        return F.normalize(
            projected[torch.arange(len(pose), device=pose.device), rule_ids],
            dim=-1,
            eps=1e-6,
        )

    def distance(self, observed, confidence, corrected, rule_ids):
        if (
            observed.shape != corrected.shape
            or rule_ids.shape != observed.shape[:1]
            or rule_ids.dtype != torch.long
            or torch.any((rule_ids < 0) | (rule_ids >= 6))
        ):
            raise ValueError("Matching checkpoint windows and IDs required")
        if not self.active[rule_ids].all():
            raise ValueError(
                "Unsupported head must use the explicitly declared baseline, not this model"
            )
        both = self.encode(
            torch.cat((observed, corrected)),
            torch.cat((confidence, confidence)),
            torch.cat((rule_ids, rule_ids)),
        )
        left, right = both.chunk(2)
        return (left - right).square().sum(-1)

    def forward(self, observed, confidence, corrected, rule_ids):
        return F.softplus(self.raw_intercept[rule_ids]) - F.softplus(
            self.raw_scale[rule_ids]
        ) * self.distance(observed, confidence, corrected, rule_ids)


def infer(model, p, c, q, ids):
    selected = np.isin(ids, model.active_rules)
    device = next(model.parameters()).device
    tensors = [
        torch.as_tensor(a[:, selected].reshape(-1, *a.shape[2:]), device=device)
        for a in (p, c, q)
    ]
    rid = torch.as_tensor(
        np.tile(ids[selected], len(p)), device=device, dtype=torch.long
    )
    with torch.no_grad():
        values = torch.sigmoid(model(*tensors, rid)).cpu().numpy().reshape(len(p), -1)
    output = np.full(p.shape[:2], np.nan)
    output[:, selected] = values
    return output


def aggregate(values, ids, active):
    output = np.full((len(values), 6), np.nan)
    for k in active:
        output[:, k] = values[:, ids == k].mean(1)
    return output


TRIPLES = (
    (11, 5, 7),
    (12, 6, 8),
    (5, 7, 9),
    (6, 8, 10),
    (5, 11, 13),
    (6, 12, 14),
    (11, 13, 15),
    (12, 14, 16),
)


def motion_features(pose, confidence):
    p, c = np.asarray(pose, float), np.asarray(confidence, float)
    if p.ndim != 3 or p.shape[1:] != (17, 2) or c.shape != p.shape[:-1]:
        raise ValueError("Matching interval COCO17 coordinates and confidence required")
    if (
        not np.isfinite(p).all()
        or not np.isfinite(c).all()
        or np.any((c < 0) | (c > 1))
    ):
        raise ValueError("Finite input and bounded confidence required")
    values, weights = [], []
    for a, b, d in TRIPLES:
        u, v = p[:, a] - p[:, b], p[:, d] - p[:, b]
        denominator = np.linalg.norm(u, axis=1) * np.linalg.norm(v, axis=1)
        valid = denominator > 1e-8
        cos = np.divide((u * v).sum(1), denominator, out=np.zeros(len(p)), where=valid)
        sin = np.divide(
            abs(u[:, 0] * v[:, 1] - u[:, 1] * v[:, 0]),
            denominator,
            out=np.zeros(len(p)),
            where=valid,
        )
        values.extend((np.clip(cos, -1, 1), np.clip(sin, 0, 1)))
        weight = np.min(c[:, [a, b, d]], axis=1) * valid
        weights.extend((weight, weight))
    vectors = (
        p[:, [5, 6]].mean(1) - p[:, [11, 12]].mean(1),
        p[:, 6] - p[:, 5],
        p[:, 12] - p[:, 11],
    )
    joints = ([5, 6, 11, 12], [5, 6], [11, 12])
    for vector, ids in zip(vectors, joints):
        norm = np.linalg.norm(vector, axis=1)
        unit = np.divide(
            vector, norm[:, None], out=np.zeros_like(vector), where=norm[:, None] > 1e-8
        )
        values.extend((unit[:, 0], unit[:, 1]))
        weight = np.min(c[:, ids], axis=1) * (norm > 1e-8)
        weights.extend((weight, weight))
    return np.array(values).T, np.array(weights).T


def descriptors(values, radius=2):
    x = np.asarray(values, float)
    if x.ndim != 2 or len(x) < 2 or not np.isfinite(x).all():
        raise ValueError("Finite interval feature matrix required")
    padded = np.pad(x, ((radius, radius), (0, 0)), mode="edge")
    return np.stack([padded[i : i + 2 * radius + 1].ravel() for i in range(len(x))])


def local_cost(a, b, ca, cb):
    weights = np.minimum(ca[:, None], cb[None])
    denominator = weights.sum(-1)
    return np.sqrt(
        np.divide(
            ((a[:, None] - b[None]) ** 2 * weights).sum(-1),
            denominator,
            out=np.full(denominator.shape, np.inf),
            where=denominator > 1e-8,
        )
    )


def observed_features(pixels, confidence):
    """Translation/scale invariant, without rotating away shoulder/hip motion."""
    points = np.asarray(pixels, dtype=float).copy()
    confidence = np.asarray(confidence, dtype=float)
    if (
        points.ndim != 3
        or points.shape[1:] != (17, 2)
        or confidence.shape != points.shape[:2]
    ):
        raise ValueError("Expected full source-clock COCO17 poses and confidence")
    clock = np.arange(len(points))
    for joint in range(17):
        if joint not in range(5, 13):
            # Alignment uses shoulders, elbows, wrists and hips only.
            points[:, joint] = np.nan_to_num(points[:, joint])
            continue
        valid = np.isfinite(points[:, joint]).all(1) & (confidence[:, joint] >= 0.25)
        if valid.sum() < 3:
            raise ValueError(f"Insufficient observed evidence for joint {joint}")
        for axis in range(2):
            points[:, joint, axis] = np.interp(
                clock, clock[valid], points[valid, joint, axis]
            )
    # Three-frame median rejects isolated detector spikes, not global movement.
    points = median_filter(points, size=(3, 1, 1), mode="nearest")
    root = points[:, [11, 12]].mean(1)
    torso = points[:, [5, 6]].mean(1) - root
    scale = float(np.median(np.linalg.norm(torso, axis=1)))
    if scale < 1:
        raise ValueError("Degenerate observed torso scale")
    angles, angle_confidence = motion_features(points, np.clip(confidence, 0, 1))
    # Both shoulders/elbows, spine direction, shoulder axis and hip axis.
    columns = [*range(8), *range(16, 22)]
    angles = angles[:, columns]
    angle_confidence = angle_confidence[:, columns]
    relative = ((points[:, 5:13] - root[:, None]) / scale).reshape(len(points), -1)
    relative_confidence = np.repeat(confidence[:, 5:13], 2, axis=1)
    return angles, angle_confidence, relative, relative_confidence


def observed_cost(reference, target):
    a, ca, pa, cpa = reference
    b, cb, pb, cpb = target
    angular = local_cost(
        descriptors(a), descriptors(b), descriptors(ca), descriptors(cb)
    )
    positional = local_cost(
        descriptors(pa), descriptors(pb), descriptors(cpa), descriptors(cpb)
    )
    return 0.6 * angular + 0.4 * positional


def outward_path(cost, expected_ratio=1.0, minimum_target_length=2):
    """Symmetric2 DTW, fixed contact origin and free outer endpoint."""
    raw = np.asarray(cost, float)
    n, m = raw.shape
    expected = np.arange(n)[:, None] * expected_ratio
    prior = 0.025 * np.abs(np.arange(m)[None] - expected) / 15.0
    local = raw + prior
    dp = np.full((n + 1, m + 1), np.inf)
    parent = np.full((n + 1, m + 1), -1, np.int8)
    dp[0, 0] = 0
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            options = (
                dp[i - 1, j - 1] + 2 * local[i - 1, j - 1],
                dp[i - 1, j] + local[i - 1, j - 1] + 0.04,
                dp[i, j - 1] + local[i - 1, j - 1] + 0.04,
            )
            step = int(np.argmin(options))
            dp[i, j], parent[i, j] = options[step], step
    eligible = np.arange(max(2, minimum_target_length), m + 1)
    if not len(eligible):
        raise ValueError("Insufficient target frames for an open-end alignment")
    endpoint = int(eligible[np.argmin(dp[n, eligible] / (n + eligible))])
    i, j = n, endpoint
    path = []
    while i or j:
        path.append((i - 1, j - 1))
        step = parent[i, j]
        if step in (0, 1):
            i -= 1
        if step in (0, 2):
            j -= 1
    return np.asarray(path[::-1], int), float(dp[n, endpoint] / (n + endpoint))


def align_contacts(cost, reference_contact, target_contact):
    n, m = cost.shape
    if not 0 < reference_contact < n - 2 or not 0 < target_contact < m - 2:
        raise ValueError("Both contact anchors must have pre/post-contact evidence")
    before = cost[: reference_contact + 1, : target_contact + 1][::-1, ::-1]
    ratio = min(1.0, target_contact / reference_contact)
    pre, pre_cost = outward_path(before, ratio, max(3, int(min(before.shape) * 0.7)))
    # The short reference ends during follow-through. Ignore long target rest tails.
    post_limit = min(m, target_contact + 3 * (n - reference_contact))
    after = cost[reference_contact:, target_contact:post_limit]
    post, post_cost = outward_path(after, 1.0, max(3, int(min(after.shape) * 0.5)))
    pre = np.array([reference_contact, target_contact]) - pre
    post = post + [reference_contact, target_contact]
    path = np.concatenate((pre[::-1], post[1:]))
    mapping = np.asarray(
        [int(np.rint(np.median(path[path[:, 0] == i, 1]))) for i in range(n)]
    )
    mapping[reference_contact] = target_contact
    if np.any(np.diff(mapping) < 0):
        raise ValueError("Nonmonotone contact mapping")
    return mapping, path, (pre_cost + post_cost) / 2


def transfer_intervals(annotations, mapping):
    output = {}
    for key, original in annotations.items():
        value = {
            field: int(mapping[original[field]]) for field in ("start", "end", "anchor")
        }
        value.update(
            reviewed=False,
            note="依張宸愷1.mp4人工區間，以偵測骨架時序對齊提出；待人工確認。",
        )
        output[key] = value
    return output
