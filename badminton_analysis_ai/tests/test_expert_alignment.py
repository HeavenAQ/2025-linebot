from __future__ import annotations

import numpy as np
import pytest

from badminton_analysis.ml.expert_reference_bank import (
    ExpertReference,
    segmental_alignment,
)
from badminton_analysis.ml.skeleton_normalization import CANONICAL_PHASE_INDICES
from service.pipeline import _expert_alignment

FPS = 30.0
PHASES = (12, 27, 42, 51, 60)


def _pose(progress: np.ndarray) -> np.ndarray:
    """A 17-joint pose sequence that moves along one axis at a given pace."""
    return np.repeat(progress.astype(np.float32)[:, None, None], 17, axis=1).repeat(
        2, axis=2
    )


def _reference(skeleton: np.ndarray | None, *, phases: tuple[int, ...] = PHASES) -> ExpertReference:
    return ExpertReference(
        skill="serve",
        handedness="right",
        video_object_path="experts/v3/serve/videos/test.mp4",
        subject_id="test-expert",
        fps=FPS,
        analysis_window=(0, 70, 70),
        source_phase_indices=phases,
        distance=0.1,
        similarity=0.9,
        skeleton=skeleton,
    )


@pytest.fixture
def student() -> np.ndarray:
    return _pose(np.linspace(0.0, 1.0, 64))


@pytest.fixture
def expert() -> np.ndarray:
    # The same movement, but front-loaded: the expert covers most of the range
    # early where the learner is still winding up, which is exactly the tempo
    # difference a straight interpolation between checkpoints cannot express.
    return _pose(np.sqrt(np.linspace(0.0, 1.0, 64)))


def test_alignment_covers_every_learner_frame(student, expert) -> None:
    samples = segmental_alignment(_reference(expert), student)

    assert len(samples) == len(student)
    assert samples[0][0] == pytest.approx(0.0)
    assert samples[-1][0] == pytest.approx(1.0)


def test_alignment_never_rewinds_the_expert(student, expert) -> None:
    samples = segmental_alignment(_reference(expert), student)
    positions = [position for position, _ in samples]
    seconds = [value for _, value in samples]

    assert positions == sorted(positions)
    assert seconds == sorted(seconds)


def test_checkpoints_stay_where_the_expert_timeline_puts_them(student, expert) -> None:
    reference = _reference(expert)
    samples = segmental_alignment(reference, student)

    # The anchors are pinned, so each one lands on the same moment the expert
    # timeline reports for it. They agree to within the half frame that mapping
    # the clip's own frames into the 64-frame analysis window rounds away.
    for anchor, expected in zip(CANONICAL_PHASE_INDICES, reference.phase_seconds()):
        assert samples[int(anchor)][1] == pytest.approx(expected, abs=0.5 / FPS)


def test_between_checkpoints_the_expert_follows_its_own_tempo(student, expert) -> None:
    samples = segmental_alignment(_reference(expert), student)
    first, second = int(CANONICAL_PHASE_INDICES[0]), int(CANONICAL_PHASE_INDICES[1])
    start, end = samples[first][1], samples[second][1]

    # Halfway through the first phase the warp is somewhere other than halfway
    # through the expert's first phase -- which is the whole point of running
    # DTW inside the segment rather than drawing a line across it.
    middle = samples[(first + second) // 2][1]
    assert abs(middle - (start + end) / 2) > 0.5 / FPS


def test_a_bank_built_without_skeletons_yields_no_alignment(student) -> None:
    assert segmental_alignment(_reference(None), student) == ()


def test_a_pose_the_expert_cannot_be_compared_against_yields_no_alignment(expert) -> None:
    assert segmental_alignment(_reference(expert), np.zeros((32, 17, 2))) == ()
    assert segmental_alignment(_reference(expert), np.zeros((64, 17))) == ()


def test_phases_that_collapse_on_resampling_yield_no_alignment(student, expert) -> None:
    # Four checkpoints crowded into the first frames of a long clip round onto
    # the same resampled frame, which leaves no segment to warp.
    assert segmental_alignment(_reference(expert, phases=(0, 1, 2, 3, 400)), student) == ()


def test_alignment_failure_costs_the_alignment_and_not_the_analysis(expert, caplog) -> None:
    # A ragged pose is not something the warp can refuse politely; numpy raises
    # on it. The analysis still has to come back.
    assert _expert_alignment(_reference(expert), [[1.0, 2.0], [3.0]]) == ()
    assert "expert alignment unusable" in caplog.text


def test_alignment_matches_the_warp_when_it_succeeds(student, expert) -> None:
    reference = _reference(expert)

    assert _expert_alignment(reference, student) == segmental_alignment(reference, student)
