from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

import badminton_analysis.ml.expert_motion_backend as backend_module
from badminton_analysis.ml.expert_motion_backend import (
    _clip_level_rigid_target_alignment,
    _score_smash_correction,
    _serve_single_head_score,
)
from badminton_analysis.ml.skill_specs import get_skill_spec


def test_dual_window_alignment_is_one_rigid_transform() -> None:
    frames = 64
    source = np.zeros((frames, 17, 2), dtype=np.float32)
    source[:, :, 0] = np.arange(17, dtype=np.float32)
    source[:, :, 1] = np.arange(17, dtype=np.float32) * 0.5
    source += np.linspace(0.0, 0.2, frames, dtype=np.float32)[:, None, None]
    angle = np.deg2rad(18.0)
    rotation = np.asarray(
        ((np.cos(angle), -np.sin(angle)), (np.sin(angle), np.cos(angle))),
        dtype=np.float32,
    )
    target = source @ rotation + np.asarray((2.0, -1.0), dtype=np.float32)
    corrected = source.copy()
    corrected[:, 10, 0] += np.linspace(0.0, 1.0, frames)

    # The production fit interval is the model rubric's complete preparation
    # window (0:24 at 64 frames). A historical benchmark hard-coded 0:16 and
    # produced scores that did not match the reviewed overlay pipeline.
    preparation = next(
        phase
        for phase in get_skill_spec("serve").phase_windows
        if phase.name == "preparation"
    )
    start, end = preparation.bounds(frames)
    assert (start, end) == (0, 24)
    mapped = _clip_level_rigid_target_alignment(
        source, target, corrected, start=start, end=end
    )

    np.testing.assert_allclose(mapped, corrected @ rotation + (2.0, -1.0), atol=1e-5)
    # Rigid mapping preserves every generated bone length; it cannot copy a
    # different learner pose independently in each frame.
    np.testing.assert_allclose(
        np.linalg.norm(mapped[:, 10] - mapped[:, 8], axis=-1),
        np.linalg.norm(corrected[:, 10] - corrected[:, 8], axis=-1),
        atol=1e-5,
    )


def test_serve_single_head_preserves_original_rubric_and_total() -> None:
    ids = (
        "arms_raised",
        "racket_foot_weight",
        "weight_transfer",
        "hip_rotation",
        "wrist_flick",
        "shoulder_rotation",
    )
    maxima = (5.0, 5.0, 30.0, 10.0, 30.0, 20.0)
    criteria = [
        {"rule_reference": rule, "score": maximum * 0.8, "maximum": maximum}
        for rule, maximum in zip(ids, maxima, strict=True)
    ]

    result = _serve_single_head_score(
        {"criteria": criteria, "checklist_total_score": 62.5}
    )

    assert np.isclose(result["total_score"], 62.5)
    assert np.isclose(sum(item["score"] for item in result["criteria"]), 62.5)
    assert [item["maximum"] for item in result["criteria"]] == list(maxima)
    assert all(
        0.0 <= item["score"] <= item["maximum"] for item in result["criteria"]
    )


def test_serve_single_head_caps_transfer_when_required_cues_disagree() -> None:
    ids = (
        "arms_raised",
        "racket_foot_weight",
        "weight_transfer",
        "hip_rotation",
        "wrist_flick",
        "shoulder_rotation",
    )
    maxima = (5.0, 5.0, 30.0, 10.0, 30.0, 20.0)
    criteria = [
        {"rule_reference": rule, "score": maximum, "maximum": maximum}
        for rule, maximum in zip(ids, maxima, strict=True)
    ]
    transfer = criteria[2]
    transfer.update(
        {
            "strict_required_cue_distance": 0.4,
            "expert_tolerance": 0.2,
            "expert_robust_scale": 0.1,
        }
    )

    result = _serve_single_head_score(
        {"criteria": criteria, "checklist_total_score": 70.0}
    )
    by_id = {item["rule_reference"]: item for item in result["criteria"]}

    assert np.isclose(result["total_score"], 70.0)
    assert np.isclose(sum(item["score"] for item in result["criteria"]), 70.0)
    assert by_id["weight_transfer"]["score"] == pytest.approx(
        30.0 * np.exp(-2.0)
    )
    assert by_id["weight_transfer"]["strict_transfer_support_ratio"] == (
        pytest.approx(np.exp(-2.0))
    )


def test_serve_single_head_keeps_transfer_inside_expert_support() -> None:
    ids = (
        "arms_raised",
        "racket_foot_weight",
        "weight_transfer",
        "hip_rotation",
        "wrist_flick",
        "shoulder_rotation",
    )
    maxima = (5.0, 5.0, 30.0, 10.0, 30.0, 20.0)
    criteria = [
        {"rule_reference": rule, "score": maximum, "maximum": maximum}
        for rule, maximum in zip(ids, maxima, strict=True)
    ]
    criteria[2].update(
        {
            "strict_required_cue_distance": 0.1,
            "expert_tolerance": 0.2,
            "expert_robust_scale": 0.1,
        }
    )

    result = _serve_single_head_score(
        {"criteria": criteria, "checklist_total_score": 100.0}
    )
    by_id = {item["rule_reference"]: item for item in result["criteria"]}

    assert by_id["weight_transfer"]["score"] == pytest.approx(30.0)
    assert result["total_score"] == pytest.approx(100.0)


def test_serve_single_head_keeps_corrected_shoulder_height_as_arm_pass() -> None:
    ids = (
        "arms_raised",
        "racket_foot_weight",
        "weight_transfer",
        "hip_rotation",
        "wrist_flick",
        "shoulder_rotation",
    )
    maxima = (5.0, 5.0, 30.0, 10.0, 30.0, 20.0)
    criteria = [
        {"rule_reference": rule, "score": maximum, "maximum": maximum}
        for rule, maximum in zip(ids, maxima, strict=True)
    ]
    criteria[0]["passes_corrected_shoulder_height"] = True
    criteria[2].update(
        {
            "strict_required_cue_distance": 0.0,
            "expert_tolerance": 0.0,
            "expert_robust_scale": 0.1,
        }
    )

    result = _serve_single_head_score(
        {"criteria": criteria, "checklist_total_score": 50.0}
    )
    by_id = {item["rule_reference"]: item for item in result["criteria"]}

    assert by_id["arms_raised"]["score"] == pytest.approx(5.0)
    assert sum(item["score"] for item in result["criteria"]) == pytest.approx(
        50.0
    )


def test_smash_runtime_score_preserves_total_and_rubric_caps(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec = get_skill_spec("smash")
    ratios = (0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
    semantic = {
        "criteria": [
            {
                "rule_reference": rule.id,
                "ratio": ratio,
                "semantic_distance": 1.0 - ratio,
            }
            for rule, ratio in zip(spec.rules, ratios, strict=True)
        ],
        "total_score": 72.5,
        "score_method": "fixture",
    }
    monkeypatch.setattr(
        backend_module,
        "aligned_smash_evidence",
        lambda *args: (np.zeros(1), np.ones(1)),
    )
    monkeypatch.setattr(
        backend_module,
        "score_smash_evidence",
        lambda *args: semantic,
    )
    sample = SimpleNamespace(
        pose=np.zeros((64, 17, 2), dtype=np.float32),
        confidence=np.ones((64, 17), dtype=np.float32),
        phase_indices=np.asarray((0, 16, 32, 48, 63), dtype=np.int64),
    )
    correction = SimpleNamespace(
        aligned_student_pose=sample.pose,
        aligned_corrected_pose=sample.pose,
    )

    result = _score_smash_correction(
        {"references": []},
        sample,
        correction,
        distribution=SimpleNamespace(),
        variant=SimpleNamespace(),
        trajectory_scorer=None,
        spec=spec,
    )

    assert result["total_score"] == pytest.approx(72.5)
    assert sum(item["score"] for item in result["criteria"]) == pytest.approx(72.5)
    assert all(
        0.0 <= item["score"] <= item["maximum"]
        for item in result["criteria"]
    )
