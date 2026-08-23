from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from badminton_analysis.ml.handedness import estimate_handedness
from badminton_analysis.ml.skeleton_normalization import (
    CANONICAL_PHASE_INDICES,
    interpolate_pose_sequence,
    normalize_skeleton_sequence,
    phase_align_sequence,
    resample_sequence,
    restore_phase_timing,
    restore_phase_timing_dtw,
)
from badminton_analysis.ml.skeleton_scoring import (
    ScoreCalibration,
    correction_distance,
    correction_quality_metrics,
    expert_euclidean_distances,
    fit_score_calibration,
    full_transition_components,
    keypoint_correction_components,
    project_bone_lengths,
    select_bone_adapted_expert,
    sequence_training_losses,
)
from badminton_analysis.models.types import Handedness


LEFT_RIGHT_PAIRS = ((1, 2), (3, 4), (5, 6), (7, 8), (9, 10), (11, 12), (13, 14), (15, 16))


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


def test_phase_constrained_dtw_preserves_anchors_and_student_holds() -> None:
    source = resample_sequence(_pose_sequence(frames=11), 64)
    phases = np.asarray((0, 13, 35, 46, 63), dtype=np.int64)
    source[13:25, 10] = source[13, 10]
    aligned_source = phase_align_sequence(source, phases)
    corrected = aligned_source.copy()
    corrected[:, 10, 2] += np.linspace(0.0, 0.8, len(corrected))

    restored = restore_phase_timing_dtw(
        corrected,
        aligned_source,
        source,
        phases,
    )

    np.testing.assert_allclose(
        restored[phases], corrected[CANONICAL_PHASE_INDICES], atol=1e-5
    )
    assert np.max(np.linalg.norm(np.diff(restored[13:25, 10], axis=0), axis=1)) < 0.08


def test_normalization_preserves_rotation_relative_to_preparation() -> None:
    sequence = _pose_sequence_2d(frames=2)
    angle = np.deg2rad(30.0)
    rotation = np.asarray(
        ((np.cos(angle), -np.sin(angle)), (np.sin(angle), np.cos(angle))),
        dtype=np.float32,
    )
    sequence[1, :, :2] = sequence[1, :, :2] @ rotation.T
    confidence = np.ones(sequence.shape[:2], dtype=np.float32)
    normalized, _ = normalize_skeleton_sequence(
        sequence, confidence, Handedness.RIGHT
    )
    shoulder_vector_0 = normalized[0, 6] - normalized[0, 5]
    shoulder_vector_1 = normalized[1, 6] - normalized[1, 5]
    assert abs(float(shoulder_vector_0[1])) < 1e-5
    assert abs(float(shoulder_vector_1[1])) > 0.2


def test_normalization_does_not_amplify_compressed_shoulders() -> None:
    sequence = _pose_sequence_2d()
    sequence[:, 5, 0] = -0.05
    sequence[:, 6, 0] = 0.05
    confidence = np.ones(sequence.shape[:2], dtype=np.float32)

    normalized, _ = normalize_skeleton_sequence(
        sequence, confidence, Handedness.RIGHT
    )

    assert float(np.ptp(normalized[..., 1])) < 10.0


def test_correction_distance_is_zero_for_identical_sequences() -> None:
    sequence = _pose_sequence()
    confidence = np.ones(sequence.shape[:2], dtype=np.float32)
    distance, components = correction_distance(sequence, sequence.copy(), confidence)
    assert distance < 1e-8
    assert all(value < 1e-8 for value in components.values())


def test_score_distance_increases_with_synthetic_corruption() -> None:
    sequence = _pose_sequence()
    confidence = np.ones(sequence.shape[:2], dtype=np.float32)
    pattern = np.linspace(-1.0, 1.0, sequence.size, dtype=np.float32).reshape(sequence.shape)
    small, _ = correction_distance(sequence, sequence + 0.02 * pattern, confidence)
    large, _ = correction_distance(sequence, sequence + 0.10 * pattern, confidence)
    assert 0.0 < small < large


def test_calibration_targets_separated_distributions() -> None:
    experts = np.asarray((0.01, 0.012, 0.014, 0.016), dtype=np.float64)
    beginners = np.asarray((0.08, 0.09, 0.10, 0.11), dtype=np.float64)
    calibration = fit_score_calibration(experts, beginners)
    expert_scores = calibration.score(experts)
    beginner_scores = calibration.score(beginners)
    assert 98.0 <= float(np.mean(expert_scores)) <= 100.0
    assert 40.0 <= float(np.mean(beginner_scores)) <= 50.0
    assert calibration.target_reachable


def test_quality_metrics_detect_bone_stretch_and_temporal_jitter() -> None:
    original = resample_sequence(_pose_sequence(), 64)
    confidence = np.ones(original.shape[:2], dtype=np.float32)
    stretched = original.copy()
    stretched[:, 10] += np.asarray((0.5, 0.0, 0.0), dtype=np.float32)
    stretched[1::2, 10, 1] += 0.25
    quality = correction_quality_metrics(original, stretched, confidence)
    assert quality["mean_relative_bone_change"] > 0.0
    assert quality["mean_correction_acceleration"] > 0.0


def test_bone_projection_restores_source_lengths() -> None:
    original = resample_sequence(_pose_sequence(), 64)
    stretched = original.copy()
    stretched[:, 9] += np.asarray((-0.4, 0.3, 0.0), dtype=np.float32)
    stretched[:, 10] += np.asarray((0.5, -0.2, 0.0), dtype=np.float32)
    projected = project_bone_lengths(original, stretched)
    confidence = np.ones(original.shape[:2], dtype=np.float32)
    quality = correction_quality_metrics(original, projected, confidence)
    assert quality["p95_relative_bone_change"] < 1e-3
    source_pelvis = (original[:, 11] + original[:, 12]) / 2.0
    projected_pelvis = (projected[:, 11] + projected[:, 12]) / 2.0
    np.testing.assert_allclose(projected_pelvis, source_pelvis, atol=1e-5)


def test_expert_euclidean_distance_identifies_nearest_reference() -> None:
    source = _pose_sequence()
    experts = np.stack((source + 0.2, source + 0.05))
    confidence = np.ones(source.shape[:2], dtype=np.float32)
    expert_confidence = np.ones(experts.shape[:3], dtype=np.float32)
    distances = expert_euclidean_distances(
        source, experts, confidence, expert_confidence
    )
    assert distances.shape == (2,)
    assert distances[1] < distances[0]
    np.testing.assert_allclose(distances[1], np.sqrt(3.0) * 0.05, atol=1e-6)


def test_bone_adapted_expert_selection_uses_skill_joint_weights() -> None:
    source = _pose_sequence()
    experts = np.stack((source.copy(), source.copy()))
    experts[0, :, 10, 0] += 0.3
    experts[1, :, 9, 0] -= 0.3
    confidence = np.ones(source.shape[:2], dtype=np.float32)
    expert_confidence = np.ones(experts.shape[:3], dtype=np.float32)
    joint_weights = np.zeros(17, dtype=np.float64)
    joint_weights[10] = 1.0

    index, adapted, selected_confidence, distance = select_bone_adapted_expert(
        source, experts, confidence, expert_confidence, joint_weights
    )

    assert index == 1
    assert adapted.shape == source.shape
    assert selected_confidence.shape == confidence.shape
    assert distance >= 0.0


def test_full_transition_separates_support_and_torso_lean() -> None:
    source = _pose_sequence()
    confidence = np.ones(source.shape[:2], dtype=np.float32)
    identical = full_transition_components(
        source,
        source,
        confidence,
        (11, 12, 13, 14, 15, 16),
        (5, 6, 11, 12),
    )
    assert identical == {
        "support_transition_distance": 0.0,
        "torso_lean_transition_distance": 0.0,
        "lunge_direction_distance": 0.0,
        "transition_distance": 0.0,
    }

    lower_body_difference = source.copy()
    lower_body_difference[:, (15, 16), 0] += np.linspace(
        0.0, 0.8, len(source)
    )[:, None]
    support = full_transition_components(
        source,
        lower_body_difference,
        confidence,
        (11, 12, 13, 14, 15, 16),
        (5, 6, 11, 12),
    )
    assert support["support_transition_distance"] > 0.0
    assert support["torso_lean_transition_distance"] == pytest.approx(0.0)

    upper_body_difference = source.copy()
    upper_body_difference[:, (5, 6), 2] += np.linspace(
        0.0, 1.0, len(source)
    )[:, None]
    lean = full_transition_components(
        source,
        upper_body_difference,
        confidence,
        (11, 12, 13, 14, 15, 16),
        (5, 6, 11, 12),
    )
    assert lean["support_transition_distance"] == pytest.approx(0.0)
    assert lean["torso_lean_transition_distance"] > 0.0
    assert lean["transition_distance"] == pytest.approx(
        0.35 * lean["torso_lean_transition_distance"]
    )


def test_lift_transition_penalizes_opposite_dominant_leg_direction() -> None:
    target = _pose_sequence(64)
    target[:, 16, 0] += np.linspace(0.0, 0.8, len(target))
    opposite = target.copy()
    opposite[:, 16, 0] -= 2.0 * np.linspace(0.0, 0.8, len(target))
    confidence = np.ones(target.shape[:2], dtype=np.float32)

    components = full_transition_components(
        opposite,
        target,
        confidence,
        (11, 12, 13, 14, 15, 16),
        (5, 6, 11, 12),
        16,
    )

    assert components["lunge_direction_distance"] == pytest.approx(1.0)
    assert components["transition_distance"] >= 0.35


def test_lift_training_loss_penalizes_opposite_lunge_direction() -> None:
    target = _pose_sequence(64)
    target[:, 16, 0] += np.linspace(0.0, 0.8, len(target))
    opposite = target.copy()
    opposite[:, 16, 0] -= 2.0 * np.linspace(0.0, 0.8, len(target))
    prediction = torch.tensor(opposite[None], requires_grad=True)
    confidence = torch.ones((1, len(target), 17), dtype=torch.float32)

    losses = sequence_training_losses(
        prediction,
        torch.tensor(target[None]),
        confidence,
        transition_weight=0.75,
        transition_joints=(11, 12, 13, 14, 15, 16),
        transition_lean_joints=(5, 6, 11, 12),
        transition_direction_joint=16,
    )
    losses["loss"].backward()

    assert losses["direction"].item() == pytest.approx(1.0)
    assert prediction.grad is not None
    assert torch.isfinite(prediction.grad).all()


def test_training_loss_penalizes_missing_torso_forward_lean() -> None:
    target = _pose_sequence()
    target[:, (5, 6), 2] += np.linspace(0.0, 0.8, len(target))[:, None]
    prediction = torch.tensor(_pose_sequence()[None], requires_grad=True)
    target_tensor = torch.tensor(target[None])
    confidence = torch.ones((1, len(target), 17), dtype=torch.float32)

    losses = sequence_training_losses(
        prediction,
        target_tensor,
        confidence,
        transition_weight=1.0,
        transition_joints=(11, 12, 13, 14, 15, 16),
        transition_lean_joints=(5, 6, 11, 12),
    )
    losses["loss"].backward()

    assert losses["transition"].item() > 0.0
    assert prediction.grad is not None
    assert torch.isfinite(prediction.grad).all()


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


def test_keypoint_components_attribute_wrist_correction() -> None:
    original = resample_sequence(_pose_sequence(), 64)
    corrected = original.copy()
    corrected[:, 10, 0] += np.linspace(0.0, 0.8, len(corrected))
    confidence = np.ones(original.shape[:2], dtype=np.float32)

    components = keypoint_correction_components(
        original, corrected, confidence
    )

    assert components["correction_distance"].shape == (17,)
    assert (
        components["correction_distance"][10]
        > components["correction_distance"][9]
    )
    assert components["position_distance"][10] > 0.0
    assert components["velocity_distance"][10] > 0.0
