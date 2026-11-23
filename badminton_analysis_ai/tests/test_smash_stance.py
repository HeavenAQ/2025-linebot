from types import SimpleNamespace
import json

import numpy as np
import pytest

from badminton_analysis.ml.smash.checkpoints import CRITERIA, score_source_evidence


def stance_pose():
    pose = np.zeros((32, 17, 2))
    pose[:, 5] = [-0.25, -1.0]
    pose[:, 6] = [0.25, -1.1]
    pose[:, 11] = [-0.20, 0]
    pose[:, 12] = [0.20, 0]
    pose[:, 15] = [-0.20, 1]
    pose[:, 16] = [0.20, 1.2]
    pose[:, 7] = [-0.45, -0.7]
    pose[:, 8] = [0.45, -0.7]
    pose[:, 9] = [-0.40, -1.2]
    pose[:, 10] = [0.40, -1.2]
    return pose


def score(pose, confidence=None, prior_rotation=0):
    if confidence is None:
        confidence = np.ones((len(pose), 17))
    intervals = {name: dict(start=0, anchor=8, end=12) for name in CRITERIA}
    intervals["arm_balance"] = dict(start=4, anchor=18, end=19)
    intervals["wrist_flick"] = dict(start=20, anchor=22, end=23)
    intervals["follow_through"] = dict(start=22, anchor=27, end=31)
    return score_source_evidence(
        poses=pose,
        confidence=confidence,
        corrected_pixels=pose.copy(),
        generated_frames=np.arange(len(pose)),
        intervals=intervals,
        contact=22,
        historical_points=np.array([5, prior_rotation, 5, 5, 5, 5.0]),
        calibration=SimpleNamespace(
            rotation_lower=(16.55, 12.08, 9.73),
            rotation_full=(22.17, 15.57, 11.66),
            preparation_height=0.61,
            balance_gap=0.22,
            shoulder_span_reference=0.46,
            endpoint_map=(0.13, 0.02),
            legacy_rotation_reference=0.19,
        ),
    )


def test_already_set_stance_earns_bounded_partial_credit():
    result = score(stance_pose())
    assert result["points"][1] == pytest.approx(10)
    evidence = result["evidence"]["body_rotation"]["sustained_stance"]
    assert evidence["source_interval"][1] < 22
    assert result["points"][4] == pytest.approx(7.5)


def test_partial_stance_does_not_reduce_existing_credit_with_unassessed_turn():
    pose = stance_pose()
    confidence = np.ones((len(pose), 17))
    confidence[0, 16] = 0.1
    result = score(pose, confidence, prior_rotation=8)
    assert result["points"][1] == pytest.approx(16)
    assert result["evidence"]["body_rotation"]["sustained_stance"]["credit"] == 0.5


def test_partial_stance_keeps_a_higher_legacy_fallback_score():
    pose = stance_pose()
    for i in range(1, len(pose)):
        angle = np.deg2rad(-min(i, 18) / 18 * 25)
        axis = np.array([np.cos(angle), np.sin(angle)]) * 0.25
        pose[i, 5] = [0, -1.05] - axis
        pose[i, 6] = [0, -1.05] + axis
    pose[23:, [5, 6], 0] *= 0.4
    confidence = np.ones((len(pose), 17))
    confidence[0, 16] = 0.1
    result = score(pose, confidence)
    assert result["evidence"]["body_rotation"]["fallback"] == "legacy_directional_span"
    assert result["points"][1] > 10


@pytest.mark.parametrize(
    "missing", ["torso", "base", "brief", "late", "confidence", "profile"]
)
def test_partial_stance_requires_sustained_pre_contact_torso_and_base(missing):
    pose = stance_pose()
    confidence = np.ones((len(pose), 17))
    if missing == "torso":
        pose[:, 6, 1] = pose[:, 5, 1]
    elif missing == "base":
        pose[:, 16, 1] = pose[:, 15, 1]
    elif missing == "brief":
        pose[8:, 16, 1] = pose[8:, 15, 1]
    elif missing == "late":
        pose[:23, 16, 1] = pose[:23, 15, 1]
    elif missing == "confidence":
        confidence[:, 16] = 0.1
    else:
        pose[:, 12, 0] = pose[:, 11, 0] + 0.01
    result = score(pose, confidence)
    assert result["points"][1] == pytest.approx(0)
    assert result["points"][4] == pytest.approx(7.5)
    if missing in ("confidence", "profile"):
        assert not result["evidence"]["body_rotation"]["sustained_stance"]["available"]


def test_short_missing_stretch_leaves_finite_stance_diagnostics():
    pose = stance_pose()
    pose[8:10, 5] = np.nan
    result = score(pose)
    assert result["points"][1] == pytest.approx(10)
    json.dumps(result["evidence"]["body_rotation"]["sustained_stance"], allow_nan=False)


@pytest.mark.parametrize("transform", ["mirror", "roll", "translate", "scale"])
def test_already_set_stance_credit_survives_camera_transforms(transform):
    pose = stance_pose()
    if transform == "mirror":
        pose[..., 0] *= -1
    elif transform == "roll":
        angle = np.deg2rad(25)
        matrix = np.array(
            [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
        )
        pose = pose @ matrix.T
    elif transform == "translate":
        pose += [450, 800]
    else:
        pose *= 230
    assert score(pose)["points"][1] == pytest.approx(10)
