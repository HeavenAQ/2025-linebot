"""The serve window must leave room after contact for the follow-through."""

import numpy as np
import pytest

from badminton_analysis.models.constants import SERVE_MINIMUM_FOLLOW_THROUGH
from badminton_analysis.services.video_analyzer import VideoAnalyzer


def swing(length: int, contact: int, *, drop: float = 100.0) -> list[tuple[float, float]]:
    """A hand track whose y rises to a single low point at `contact`."""
    y = np.full(length, 10.0)
    rise = np.linspace(10.0, 10.0 + drop, contact + 1)
    y[: contact + 1] = rise
    if contact + 1 < length:
        y[contact + 1 :] = np.linspace(10.0 + drop, 10.0 + drop / 2, length - contact - 1)
    return [(float(x), float(value)) for x, value in enumerate(y)]


def test_a_contact_with_room_after_it_is_kept():
    track = swing(60, contact=30)

    peak = VideoAnalyzer._serve_peak_with_follow_through(track, start_frame=0, peak_frame=30)

    assert peak == 30, "a complete swing must not be reselected"


def test_a_contact_at_the_last_frame_falls_back_to_an_earlier_swing():
    # Two swings: one complete at frame 30, one cut off at the final frame.
    track = swing(60, contact=30)
    for index in range(50, 60):
        track[index] = (float(index), 10.0 + 8.0 * (index - 50))

    peak = VideoAnalyzer._serve_peak_with_follow_through(track, start_frame=0, peak_frame=59)

    assert peak + SERVE_MINIMUM_FOLLOW_THROUGH <= 59
    assert abs(peak - 30) <= 2, "the earlier complete swing is the one to analyse"


def test_a_single_cut_off_swing_is_refused_in_words():
    # The hand is still rising when the footage ends, and nothing precedes it.
    track = swing(40, contact=39)

    with pytest.raises(ValueError, match="cut off"):
        VideoAnalyzer._serve_peak_with_follow_through(track, start_frame=0, peak_frame=39)


def test_a_still_hand_is_not_mistaken_for_a_swing():
    track = [(float(index), 10.0) for index in range(40)]

    with pytest.raises(ValueError, match="cut off"):
        VideoAnalyzer._serve_peak_with_follow_through(track, start_frame=0, peak_frame=39)
