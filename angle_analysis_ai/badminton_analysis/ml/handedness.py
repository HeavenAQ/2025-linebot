from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from badminton_analysis.models.types import COCOKeypoints, Handedness


@dataclass(frozen=True)
class HandednessEstimate:
    handedness: Handedness | None
    left_motion_score: float
    right_motion_score: float
    confidence_ratio: float


def _interpolate(
    values: NDArray[np.floating], observed: NDArray[np.floating]
) -> NDArray[np.float64]:
    positions = np.asarray(values, dtype=np.float64).copy()
    confidence = np.asarray(observed, dtype=np.float64)
    timeline = np.arange(len(positions), dtype=np.float64)
    for dimension in range(positions.shape[1]):
        valid = (confidence > 0.05) & np.isfinite(positions[:, dimension])
        if np.count_nonzero(valid) < 2:
            return np.empty((0, positions.shape[1]), dtype=np.float64)
        positions[:, dimension] = np.interp(
            timeline, timeline[valid], positions[valid, dimension]
        )
    return positions


def interpolated_keypoint(
    skeleton: NDArray[np.floating],
    confidence: NDArray[np.floating],
    keypoint: COCOKeypoints,
) -> NDArray[np.float64]:
    coordinates = np.asarray(skeleton, dtype=np.float64)
    observed = np.asarray(confidence, dtype=np.float64)
    return _interpolate(coordinates[:, int(keypoint)], observed[:, int(keypoint)])


def _motion_score(positions: NDArray[np.float64]) -> float:
    if len(positions) < 7:
        return 0.0
    window = 3
    padding = window // 2
    kernel = np.ones(window, dtype=np.float64) / window
    smoothed = np.column_stack(
        [
            np.convolve(
                np.pad(positions[:, dimension], (padding, padding), mode="edge"),
                kernel,
                mode="valid",
            )
            for dimension in range(positions.shape[1])
        ]
    )
    speed = np.linalg.norm(np.diff(smoothed, axis=0), axis=-1)
    positive_acceleration = np.diff(speed)
    positive_acceleration = positive_acceleration[positive_acceleration > 0.0]
    if not len(speed) or not len(positive_acceleration):
        return 0.0
    peak_speed = float(np.quantile(speed, 0.95))
    peak_acceleration = float(np.quantile(positive_acceleration, 0.95))
    return peak_speed * peak_acceleration


def estimate_handedness(
    skeleton: NDArray[np.floating],
    confidence: NDArray[np.floating],
    *,
    minimum_confidence_ratio: float = 2.0,
) -> HandednessEstimate:
    """Estimate racket hand from scale-normalized peak wrist motion."""
    coordinates = np.asarray(skeleton, dtype=np.float64)
    observed = np.asarray(confidence, dtype=np.float64)
    if coordinates.ndim != 3 or coordinates.shape[1] != 17:
        raise ValueError("skeleton must have shape (T, 17, D)")
    if observed.shape != coordinates.shape[:2]:
        raise ValueError("confidence must have shape (T, 17)")
    if minimum_confidence_ratio <= 1.0:
        raise ValueError("minimum confidence ratio must be greater than one")

    joints = {
        keypoint: interpolated_keypoint(coordinates, observed, keypoint)
        for keypoint in (
            COCOKeypoints.LEFT_WRIST,
            COCOKeypoints.RIGHT_WRIST,
            COCOKeypoints.LEFT_SHOULDER,
            COCOKeypoints.RIGHT_SHOULDER,
            COCOKeypoints.LEFT_HIP,
            COCOKeypoints.RIGHT_HIP,
        )
    }
    if any(len(values) != len(coordinates) for values in joints.values()):
        return HandednessEstimate(None, 0.0, 0.0, 1.0)

    torso_center = np.mean(
        np.stack(
            (
                joints[COCOKeypoints.LEFT_SHOULDER],
                joints[COCOKeypoints.RIGHT_SHOULDER],
                joints[COCOKeypoints.LEFT_HIP],
                joints[COCOKeypoints.RIGHT_HIP],
            )
        ),
        axis=0,
    )
    shoulder_width = np.linalg.norm(
        joints[COCOKeypoints.RIGHT_SHOULDER]
        - joints[COCOKeypoints.LEFT_SHOULDER],
        axis=-1,
    )
    valid_width = shoulder_width > 1e-8
    if np.count_nonzero(valid_width) < 2:
        return HandednessEstimate(None, 0.0, 0.0, 1.0)
    median_width = float(np.median(shoulder_width[valid_width]))
    shoulder_width = np.where(valid_width, shoulder_width, median_width)

    left_positions = (
        joints[COCOKeypoints.LEFT_WRIST] - torso_center
    ) / shoulder_width[:, None]
    right_positions = (
        joints[COCOKeypoints.RIGHT_WRIST] - torso_center
    ) / shoulder_width[:, None]
    left_score = _motion_score(left_positions)
    right_score = _motion_score(right_positions)
    smaller = min(left_score, right_score)
    larger = max(left_score, right_score)
    ratio = larger / max(smaller, 1e-12)
    if larger <= 1e-12 or ratio < minimum_confidence_ratio:
        handedness = None
    else:
        handedness = (
            Handedness.LEFT if left_score > right_score else Handedness.RIGHT
        )
    return HandednessEstimate(handedness, left_score, right_score, ratio)
