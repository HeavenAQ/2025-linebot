from __future__ import annotations

import numpy as np
import pytest

from badminton_analysis.models.types import Skill
from service.renderer import _retarget_corrected_pose


def _angle_between(first: np.ndarray, second: np.ndarray) -> float:
    first_angle = np.arctan2(first[1], first[0])
    second_angle = np.arctan2(second[1], second[0])
    return abs(
        float(
            np.arctan2(
                np.sin(second_angle - first_angle),
                np.cos(second_angle - first_angle),
            )
        )
    )


def _cross_2d(first: np.ndarray, second: np.ndarray) -> float:
    """Scalar 2D cross product, including on NumPy 2.x."""
    return float(first[0] * second[1] - first[1] * second[0])


def _leg_pixels() -> tuple[np.ndarray, np.ndarray]:
    detected = np.zeros((17, 2), dtype=np.float32)
    corrected = np.zeros((17, 2), dtype=np.float32)
    detected[5], detected[6] = (75.0, 40.0), (125.0, 40.0)
    corrected[5], corrected[6] = (60.0, 25.0), (130.0, 55.0)
    detected[7], detected[9] = (60.0, 70.0), (50.0, 100.0)
    detected[8], detected[10] = (140.0, 70.0), (150.0, 100.0)
    corrected[7], corrected[9] = (20.0, 30.0), (0.0, 10.0)
    corrected[8], corrected[10] = (180.0, 30.0), (200.0, 10.0)
    for hip, knee, ankle, x in ((11, 13, 15, 80.0), (12, 14, 16, 120.0)):
        detected[hip] = (x, 100.0)
        detected[knee] = (x + 2.0, 150.0)
        detected[ankle] = (x, 205.0)
        corrected[hip] = (x + 4.0, 98.0)
        corrected[knee] = (x + 55.0, 145.0)
        corrected[ankle] = (x + 90.0, 190.0)
    return detected, corrected


def test_lift_rendering_moves_leg_directions_toward_expert_lunge() -> None:
    detected, corrected = _leg_pixels()

    result = _retarget_corrected_pose(corrected, detected, Skill.LIFT)
    rendered_lunge = result[16] - result[12]
    target_lunge = corrected[16] - corrected[12]

    assert np.sign(rendered_lunge[0]) == np.sign(target_lunge[0])
    assert result[16, 1] == pytest.approx(detected[16, 1], abs=1e-4)
    assert np.linalg.norm(result[14] - result[12]) == pytest.approx(
        np.linalg.norm(detected[14] - detected[12]), abs=1e-4
    )
    assert np.linalg.norm(result[16] - result[14]) == pytest.approx(
        np.linalg.norm(detected[16] - detected[14]), abs=1e-4
    )
    assert _angle_between(
        result[15] - result[13], detected[15] - detected[13]
    ) < 1e-6
    grounded_target = np.asarray(
        (target_lunge[0], detected[16, 1] - result[12, 1]), dtype=np.float32
    )
    target_cross = _cross_2d(grounded_target, corrected[14] - corrected[12])
    rendered_cross = _cross_2d(
        result[16] - result[12], result[14] - result[12]
    )
    assert np.sign(rendered_cross) == np.sign(target_cross)
    np.testing.assert_allclose(
        (result[11] + result[12]) * 0.5,
        (detected[11] + detected[12]) * 0.5,
    )


def test_serve_leg_correction_is_limited_to_twelve_degrees() -> None:
    detected, corrected = _leg_pixels()

    result = _retarget_corrected_pose(corrected, detected, Skill.SERVE)
    observed = detected[13] - detected[11]
    rendered = result[13] - result[11]

    assert np.linalg.norm(rendered) == pytest.approx(np.linalg.norm(observed))
    assert _angle_between(observed, rendered) <= np.deg2rad(12.0) + 1e-6
    assert _angle_between(observed, rendered) < _angle_between(
        observed, corrected[13] - corrected[11]
    )


def test_retargeting_keeps_corrected_arms_connected_and_detected_length() -> None:
    detected, corrected = _leg_pixels()

    result = _retarget_corrected_pose(corrected, detected, Skill.SMASH)

    for shoulder, elbow, wrist in ((5, 7, 9), (6, 8, 10)):
        assert np.linalg.norm(result[elbow] - result[shoulder]) == pytest.approx(
            np.linalg.norm(detected[elbow] - detected[shoulder])
        )
        assert np.linalg.norm(result[wrist] - result[elbow]) == pytest.approx(
            np.linalg.norm(detected[wrist] - detected[elbow])
        )


def test_serve_follow_through_can_move_dominant_forearm_across_body() -> None:
    detected, corrected = _leg_pixels()
    early = _retarget_corrected_pose(
        corrected, detected, Skill.SERVE, motion_progress=0.5
    )
    final = _retarget_corrected_pose(
        corrected, detected, Skill.SERVE, motion_progress=0.9
    )
    target_forearm = corrected[10] - corrected[8]

    assert _angle_between(final[10] - final[8], target_forearm) < _angle_between(
        early[10] - early[8], target_forearm
    )
