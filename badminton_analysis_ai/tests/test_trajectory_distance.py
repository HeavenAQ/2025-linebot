from __future__ import annotations

import numpy as np
import pytest

from badminton_analysis.ml.trajectory_distance import (
    constrained_dtw_cost,
    corrected_motion_distance,
    expert_residual_ratio,
    fit_serve_angle_manifold,
    fit_smash_motion_manifold,
    serve_angle_manifold_distance,
    smash_motion_manifold_distance,
)


def test_smash_motion_manifold_uses_signed_motion_and_other_identities() -> None:
    base = np.repeat(_pose(), 4, axis=0)
    experts = np.stack([base.copy() for _ in range(4)])
    for index in range(4):
        experts[index, :, 10, 0] += 0.04 * index
    subjects = ("a", "a", "b", "b")
    manifold = fit_smash_motion_manifold(experts, subjects)

    assert smash_motion_manifold_distance(experts[0], manifold) == pytest.approx(0.0)
    reversed_motion = experts[0].copy()
    reversed_motion[:, :, 0] *= -1.0
    assert smash_motion_manifold_distance(reversed_motion, manifold) > 0.1


def _pose() -> np.ndarray:
    pose = np.zeros((16, 17, 2), dtype=np.float32)
    pose[:, :, 1] = np.arange(17, dtype=np.float32)[None] * 0.1
    pose[:, 11, 0] = -0.2
    pose[:, 12, 0] = 0.2
    pose[:, 15, 0] = -0.3
    pose[:, 16, 0] = 0.3
    pose[:, 10, 0] = np.linspace(0.0, 1.0, len(pose))
    return pose


@pytest.mark.parametrize(
    "method",
    ("euclidean", "dtw", "derivative_dtw", "shape_dtw", "multi_feature_dtw"),
)
def test_corrected_motion_distance_is_zero_for_identical_motion(method: str) -> None:
    pose = _pose()
    assert corrected_motion_distance(
        pose,
        pose,
        joints=(6, 8, 10),
        start_fraction=0.0,
        end_fraction=1.0,
        method=method,
    ) == pytest.approx(0.0, abs=1e-12)


def test_constrained_dtw_tolerates_small_tempo_difference() -> None:
    source = np.linspace(0.0, 1.0, 16)[:, None]
    target = np.interp(
        np.linspace(0.0, 1.0, 16) ** 1.2,
        np.linspace(0.0, 1.0, 16),
        source[:, 0],
    )[:, None]
    dtw = constrained_dtw_cost(source, target, radius=3)
    euclidean = float(np.sqrt(np.mean(np.square(source - target))))
    assert dtw < euclidean


def test_constrained_dtw_rejects_disconnected_band() -> None:
    with pytest.raises(ValueError, match="cannot connect"):
        constrained_dtw_cost(np.zeros((4, 1)), np.zeros((9, 1)), radius=2)


def test_expert_residual_ratio_has_full_credit_tolerance() -> None:
    assert expert_residual_ratio(0.1, tolerance=0.2, scale=0.05) == 1.0
    assert expert_residual_ratio(0.25, tolerance=0.2, scale=0.05) == pytest.approx(
        np.exp(-1.0)
    )


def test_serve_angle_manifold_uses_other_identity_bounds() -> None:
    pose = np.repeat(_pose(), 4, axis=0)
    first = np.repeat(pose[None], 2, axis=0)
    second = np.repeat(pose[None], 2, axis=0)
    second[:, :, 10, 0] *= -1.0
    experts = np.concatenate((first, second), axis=0)
    manifold = fit_serve_angle_manifold(
        experts, ("first", "first", "second", "second")
    )

    assert manifold.expert_q80 >= 0.0
    assert manifold.expert_scale >= 0.5
    assert serve_angle_manifold_distance(experts[0], manifold) == pytest.approx(
        0.0, abs=1e-9
    )
