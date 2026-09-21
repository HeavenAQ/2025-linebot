"""A serve cut short must reach the generator, or be refused in words.

The fixture is one expert serve's RF-DETR keypoints (142 frames, contact at
69), so these run the real EIMD-v3 model rather than a stand-in: the point is
that a reselected window still produces a grade, not merely a phase tuple.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from badminton_analysis.ml.expert_motion_backend import ExpertMotionGeneratorBackend
from badminton_analysis.models.constants import SERVE_MINIMUM_FOLLOW_THROUGH
from badminton_analysis.models.types import Handedness, Skill
from badminton_analysis.services.video_analyzer import VideoAnalyzer

FIXTURE = Path(__file__).parent / "fixtures" / "serve_expert_keypoints.npy"
MODEL_ROOT = Path(__file__).parents[1] / "models" / "error_isolated_motion"


@pytest.fixture(scope="module")
def keypoints() -> np.ndarray:
    return np.load(FIXTURE)


@pytest.fixture(scope="module")
def backend() -> ExpertMotionGeneratorBackend:
    if not (MODEL_ROOT / "serve" / "error_isolated_motion.pt").exists():
        pytest.skip("expert motion model is not available")
    return ExpertMotionGeneratorBackend(
        MODEL_ROOT, Skill.SERVE, candidates=8, seed=19, align_ankle_spine_view=True
    )


def tracking(points: np.ndarray) -> dict:
    return {
        "frames": [],
        "body_landmarks_2d": [
            {joint: (float(frame[joint][0]), float(frame[joint][1])) for joint in range(17)}
            for frame in points
        ],
        "body_keypoints_2d": [frame.astype(np.float64) for frame in points],
        "body_confidence_2d": [np.ones(17) for _ in points],
        "hand_positions": [(float(frame[10][0]), float(frame[10][1])) for frame in points],
        "elbow_positions": [(float(frame[8][0]), float(frame[8][1])) for frame in points],
    }


def phases(points: np.ndarray) -> tuple[int, int, int, int, int]:
    data = tracking(points)
    return VideoAnalyzer.find_analysis_phases(
        skill=Skill.SERVE,
        hand_positions=data["hand_positions"],
        elbow_positions=data["elbow_positions"],
    )


def test_a_complete_serve_is_unchanged_by_the_boundary(keypoints, backend, monkeypatch) -> None:
    """The guard must be invisible to a clip that already has a follow-through."""
    import badminton_analysis.services.video_analyzer as module

    guarded = phases(keypoints)
    monkeypatch.setattr(module, "SERVE_MINIMUM_FOLLOW_THROUGH", 0)
    assert guarded == phases(keypoints), "the boundary moved a complete stroke"
    monkeypatch.undo()

    result = backend.infer(tracking(keypoints), Handedness.RIGHT, "expert-serve.mp4")

    assert result.grade["total_grade"] == pytest.approx(100.0, abs=0.01)
    assert len(result.grade["grading_details"]) == 6


def test_a_serve_cut_just_after_contact_still_reaches_the_generator(keypoints, backend) -> None:
    """Contact at the edge is reselected, and the clip is graded, not refused."""
    clipped = keypoints[:72]
    start, preparation, contact, follow_through, end = phases(clipped)

    assert start < preparation < contact < follow_through < end
    assert contact + SERVE_MINIMUM_FOLLOW_THROUGH <= len(clipped) - 1

    result = backend.infer(tracking(clipped), Handedness.RIGHT, "clipped.mp4")

    assert np.isfinite(result.grade["total_grade"])
    assert len(result.grade["grading_details"]) == 6
    assert result.correction is not None, "the generator produced a corrected motion"


def test_a_serve_with_no_follow_through_is_refused_before_grading(keypoints, backend) -> None:
    """Nothing after contact means nothing to grade the follow-through on."""
    with pytest.raises(ValueError, match="cut off"):
        backend.infer(tracking(keypoints[:66]), Handedness.RIGHT, "truncated.mp4")


def test_the_grade_survives_footage_added_after_the_window(keypoints, backend) -> None:
    """Frames beyond the stroke must not move the score of the stroke itself."""
    padded = np.concatenate([keypoints, np.repeat(keypoints[-1:], 20, axis=0)], axis=0)

    whole = backend.infer(tracking(keypoints), Handedness.RIGHT, "a.mp4")
    extended = backend.infer(tracking(padded), Handedness.RIGHT, "b.mp4")

    assert extended.grade["total_grade"] == pytest.approx(
        whole.grade["total_grade"], abs=0.5
    )
