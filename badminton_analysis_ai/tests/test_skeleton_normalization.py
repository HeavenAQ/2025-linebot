from __future__ import annotations

import numpy as np

from badminton_analysis.ml.handedness import estimate_handedness
from badminton_analysis.ml.skeleton_normalization import (
    CANONICAL_PHASE_INDICES,
    interpolate_pose_sequence,
    normalize_skeleton_sequence,
    phase_align_sequence,
    resample_sequence,
    restore_phase_timing,
)
from badminton_analysis.models.types import Handedness

LEFT_RIGHT_PAIRS = (
    (1, 2),
    (3, 4),
    (5, 6),
    (7, 8),
    (9, 10),
    (11, 12),
    (13, 14),
    (15, 16),
)


def _pose_sequence(frames: int = 8) -> np.ndarray:
    pose = np.zeros((frames, 17, 3), dtype=np.float32)
    pose[:, 5] = (-1.0, 2.0, 0.1)
    pose[:, 6] = (1.0, 2.0, -0.1)
    pose[:, 7] = (-1.5, 1.0, 0.2)
    pose[:, 8] = (1.5, 1.0, -0.2)
    pose[:, 9] = (-1.8, 0.2, 0.3)
    pose[:, 10] = (2.0, 0.1, -0.3)
    pose[:, 11] = (-0.8, 0.0, 0.0)
    pose[:, 12] = (0.8, 0.0, 0.0)
    pose[:, 13] = (-0.8, -1.7, 0.1)
    pose[:, 14] = (0.8, -1.7, -0.1)
    pose[:, 15] = (-0.8, -3.2, 0.2)
    pose[:, 16] = (0.8, -3.2, -0.2)
    pose[:, 0] = (0.0, 3.2, 0.0)
    pose[:, 10, 1] += np.linspace(0.0, 0.7, frames)
    return pose


def _pose_sequence_2d(frames: int = 8) -> np.ndarray:
    """The x/y of the shared fixture: normalization now takes 2D poses."""
    return _pose_sequence(frames)[..., :2].copy()


def test_normalization_preserves_shape_and_mirrors_handedness() -> None:
    right = _pose_sequence_2d()
    left = right.copy()
    for left_index, right_index in LEFT_RIGHT_PAIRS:
        left[:, [left_index, right_index]] = right[:, [right_index, left_index]]
    left[..., 0] *= -1.0
    confidence = np.ones((len(right), 17), dtype=np.float32)

    normalized_right, right_confidence = normalize_skeleton_sequence(
        right, confidence, Handedness.RIGHT
    )
    normalized_left, left_confidence = normalize_skeleton_sequence(
        left, confidence, Handedness.LEFT
    )

    assert normalized_right.shape == right.shape
    np.testing.assert_allclose(normalized_left, normalized_right, atol=1e-5)
    np.testing.assert_array_equal(left_confidence, right_confidence)


def test_resampling_returns_fixed_sequence_length() -> None:
    sequence = _pose_sequence(frames=11)
    result = resample_sequence(sequence, 64)
    assert result.shape == (64, 17, 3)
    np.testing.assert_allclose(result[0], sequence[0])
    np.testing.assert_allclose(result[-1], sequence[-1])


def test_phase_alignment_maps_and_restores_phase_anchors() -> None:
    timeline = np.arange(64, dtype=np.float32)[:, None]
    phases = np.asarray((0, 23, 40, 53, 63), dtype=np.int64)
    aligned = phase_align_sequence(timeline, phases)
    np.testing.assert_allclose(
        aligned[CANONICAL_PHASE_INDICES, 0], phases.astype(np.float32)
    )
    restored = restore_phase_timing(aligned, phases)
    np.testing.assert_allclose(restored[phases, 0], phases.astype(np.float32))


def test_pose_outlier_rejection_interpolates_impossible_limb_length() -> None:
    sequence = _pose_sequence(frames=9)[:, :, :2]
    confidence = np.ones(sequence.shape[:2], dtype=np.float32)
    sequence[4, 10] = (40.0, -30.0)

    filtered, filtered_confidence = interpolate_pose_sequence(sequence, confidence)

    assert filtered_confidence[4, 10] == 0.0
    np.testing.assert_allclose(
        filtered[4, 10], (filtered[3, 10] + filtered[5, 10]) * 0.5, atol=1e-5
    )


def test_normalization_preserves_rotation_relative_to_preparation() -> None:
    sequence = _pose_sequence_2d(frames=2)
    angle = np.deg2rad(30.0)
    rotation = np.asarray(
        ((np.cos(angle), -np.sin(angle)), (np.sin(angle), np.cos(angle))),
        dtype=np.float32,
    )
    sequence[1, :, :2] = sequence[1, :, :2] @ rotation.T
    confidence = np.ones(sequence.shape[:2], dtype=np.float32)
    normalized, _ = normalize_skeleton_sequence(sequence, confidence, Handedness.RIGHT)
    shoulder_vector_0 = normalized[0, 6] - normalized[0, 5]
    shoulder_vector_1 = normalized[1, 6] - normalized[1, 5]
    assert abs(float(shoulder_vector_0[1])) < 1e-5
    assert abs(float(shoulder_vector_1[1])) > 0.2


def test_normalization_does_not_amplify_compressed_shoulders() -> None:
    sequence = _pose_sequence_2d()
    sequence[:, 5, 0] = -0.05
    sequence[:, 6, 0] = 0.05
    confidence = np.ones(sequence.shape[:2], dtype=np.float32)

    normalized, _ = normalize_skeleton_sequence(sequence, confidence, Handedness.RIGHT)

    assert float(np.ptp(normalized[..., 1])) < 10.0


def test_handedness_estimate_uses_decisive_wrist_motion() -> None:
    sequence = resample_sequence(_pose_sequence(), 64)
    confidence = np.ones(sequence.shape[:2], dtype=np.float32)
    timeline = np.linspace(0.0, 1.0, len(sequence), dtype=np.float32)
    sequence[:, 9, 0] += 4.0 * timeline**3
    sequence[:, 10, 0] += 0.1 * timeline

    estimate = estimate_handedness(sequence, confidence)

    assert estimate.handedness == Handedness.LEFT
    assert estimate.left_motion_score > estimate.right_motion_score
    assert estimate.confidence_ratio >= 2.0


def test_handedness_estimate_rejects_ambiguous_motion() -> None:
    sequence = resample_sequence(_pose_sequence(), 64)
    confidence = np.ones(sequence.shape[:2], dtype=np.float32)
    timeline = np.linspace(0.0, 1.0, len(sequence), dtype=np.float32)
    motion = timeline**3
    sequence[:, 9, 0] += motion
    sequence[:, 10, 0] += motion

    estimate = estimate_handedness(sequence, confidence)

    assert estimate.handedness is None
    assert estimate.confidence_ratio < 2.0
