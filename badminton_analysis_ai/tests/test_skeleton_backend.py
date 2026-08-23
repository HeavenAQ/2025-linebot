from __future__ import annotations

import numpy as np
import pytest

from badminton_analysis.ml.skeleton_backend import tracking_to_normalized_sequence
from badminton_analysis.models.types import COCOKeypoints, Handedness, Skill
from badminton_analysis.services.video_analyzer import VideoAnalyzer


def _pose_sequence() -> np.ndarray:
    pose = np.zeros((6, 17, 3), dtype=np.float32)
    pose[:, 0] = (0.0, 3.2, 0.0)
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
    pose[:, 10, 0] += np.linspace(0.0, 0.8, 6)
    return pose


def _tracking(pose_3d: np.ndarray, pose_2d: np.ndarray) -> dict[str, object]:
    return {
        "original_landmarks": [
            {COCOKeypoints(joint): frame[joint] for joint in range(17)}
            for frame in pose_3d
        ],
        "body_landmarks_2d": [
            {COCOKeypoints(joint): frame[joint] for joint in range(17)}
            for frame in pose_2d
        ],
        "frames": [],
        "hand_positions": [frame[10] for frame in pose_2d],
        "elbow_positions": [frame[8] for frame in pose_2d],
        "time_intervals": [],
    }


def test_scoring_coordinates_come_from_3d_not_2d_projection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        VideoAnalyzer,
        "find_analysis_phases",
        classmethod(lambda cls, **kwargs: (0, 1, 2, 4, 5)),
    )
    pose_3d = _pose_sequence()
    projection_a = pose_3d[..., :2] * 100.0 + (320.0, 240.0)
    projection_b = projection_a[..., ::-1] * (0.6, -1.4) + (50.0, 900.0)

    sequence_a, confidence_a, window_a, phases_a = tracking_to_normalized_sequence(
        _tracking(pose_3d, projection_a),  # type: ignore[arg-type]
        Handedness.RIGHT,
        skill=Skill.LIFT,
        target_frames=6,
    )
    sequence_b, confidence_b, window_b, phases_b = tracking_to_normalized_sequence(
        _tracking(pose_3d, projection_b),  # type: ignore[arg-type]
        Handedness.RIGHT,
        skill=Skill.LIFT,
        target_frames=6,
    )

    np.testing.assert_allclose(sequence_a, sequence_b)
    np.testing.assert_allclose(confidence_a, confidence_b)
    assert window_a == window_b == (0, 2, 5)
    np.testing.assert_array_equal(phases_a, phases_b)
