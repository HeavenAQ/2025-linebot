from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from badminton_analysis.ml.smash_expert_scoring import (
    SmashVariant,
    allocate_smash_total_to_weighted_criteria,
    extract_smash_evidence,
    fit_smash_distribution,
    load_smash_distribution,
    save_smash_distribution,
    score_smash_evidence,
)


def test_smash_aggregate_attribution_preserves_total_and_rubric_caps() -> None:
    ratios = np.asarray((1.0, 1.0, 0.2, 0.8, 0.6, 1.0))
    maxima = np.asarray((10.0, 10.0, 20.0, 20.0, 20.0, 20.0))

    lower = allocate_smash_total_to_weighted_criteria(ratios, maxima, 54.0)
    higher = allocate_smash_total_to_weighted_criteria(ratios, maxima, 83.0)

    assert np.sum(lower) == pytest.approx(54.0)
    assert np.sum(higher) == pytest.approx(83.0)
    assert np.all(lower >= 0.0)
    assert np.all(higher <= maxima)
    np.testing.assert_array_equal(np.argsort(lower), np.argsort(ratios * maxima))


def _smash_pose(*, complete: bool = True) -> np.ndarray:
    pose = np.zeros((64, 17, 2), dtype=np.float32)
    pose[:] = np.asarray(
        [
            (0.0, 2.1), (-0.1, 2.15), (0.1, 2.15), (-0.2, 2.05),
            (0.2, 2.05), (-0.4, 1.45), (0.4, 1.45), (-0.7, 0.9),
            (0.7, 0.9), (-0.8, 0.35), (0.8, 0.35), (-0.3, 0.0),
            (0.3, 0.0), (-0.3, -1.0), (0.3, -1.0), (-0.3, -2.0),
            (0.3, -2.0),
        ],
        dtype=np.float32,
    )
    if not complete:
        return pose
    timeline = np.linspace(0.0, 1.0, 64, dtype=np.float32)
    lift = np.sin(np.pi * np.minimum(timeline / 0.55, 1.0))
    follow = np.clip((timeline - 0.50) / 0.50, 0.0, 1.0)
    pose[:, 7, 1] += 0.8 * lift
    pose[:, 9, 1] += 1.1 * lift
    pose[:, 8, 0] -= 0.9 * np.sin(np.pi * timeline)
    pose[:, 8, 1] += 0.9 * lift - 1.0 * follow
    pose[:, 10, 0] -= 1.3 * np.sin(np.pi * timeline) + 0.7 * follow
    pose[:, 10, 1] += 1.7 * lift - 2.0 * follow
    pose[:, 5, 0] += 0.25 * follow
    pose[:, 6, 0] -= 0.25 * follow
    return pose


def _variant() -> SmashVariant:
    return SmashVariant(
        name="test",
        envelope_policy="identity_support",
        decay=0.75,
        aggregation="geometric",
        checkpoint_profile="semantic_base",
    )


def test_smash_evidence_is_invariant_to_fixed_camera_similarity() -> None:
    pose = _smash_pose()
    confidence = np.ones((64, 17), dtype=np.float32)
    angle = np.deg2rad(21.0)
    rotation = np.asarray(
        ((np.cos(angle), -np.sin(angle)), (np.sin(angle), np.cos(angle))),
        dtype=np.float32,
    )
    transformed = 1.7 * (pose @ rotation.T) + np.asarray((4.0, -2.0))

    expected, _ = extract_smash_evidence(pose, confidence)
    actual, _ = extract_smash_evidence(transformed, confidence)

    np.testing.assert_allclose(actual, expected, atol=1e-5)


def test_expert_only_distribution_scores_incomplete_phase_sequence_lower() -> None:
    confidence = np.ones((64, 17), dtype=np.float32)
    experts = []
    for scale in (0.95, 1.0, 1.05):
        evidence, _ = extract_smash_evidence(scale * _smash_pose(), confidence)
        experts.append(evidence)
    distribution = fit_smash_distribution(
        np.stack(experts), ("coach-a", "coach-b", "coach-c"),
        policy="identity_support",
    )
    valid_evidence, valid_reliability = extract_smash_evidence(
        _smash_pose(), confidence
    )
    incomplete_evidence, incomplete_reliability = extract_smash_evidence(
        _smash_pose(complete=False), confidence
    )

    valid = score_smash_evidence(
        valid_evidence, valid_reliability, distribution, _variant()
    )
    incomplete = score_smash_evidence(
        incomplete_evidence, incomplete_reliability, distribution, _variant()
    )

    assert valid["total_score"] > incomplete["total_score"]
    assert valid["student_data_used_for_training_or_calibration"] is False
    elbow = next(
        item for item in incomplete["criteria"]
        if item["rule_reference"] == "elbow_forward"
    )
    assert elbow["ratio"] < 1.0


def test_smash_distribution_round_trip(tmp_path: Path) -> None:
    confidence = np.ones((64, 17), dtype=np.float32)
    evidence, _ = extract_smash_evidence(_smash_pose(), confidence)
    distribution = fit_smash_distribution(
        np.stack((evidence, evidence)), ("coach-a", "coach-b"),
        policy="identity_support",
    )
    destination = tmp_path / "smash_distribution.npz"

    save_smash_distribution(distribution, _variant(), destination)
    loaded_distribution, loaded_variant = load_smash_distribution(destination)

    np.testing.assert_allclose(loaded_distribution.lower, distribution.lower)
    np.testing.assert_allclose(loaded_distribution.upper, distribution.upper)
    np.testing.assert_allclose(loaded_distribution.scale, distribution.scale)
    assert loaded_variant == _variant()
