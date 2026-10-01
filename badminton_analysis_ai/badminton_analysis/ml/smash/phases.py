"""Where a smash's EIMD-v3 window ends: when the shoulders finish turning forward."""

from __future__ import annotations

from typing import (
    Sequence,
)
import numpy as np
from numpy.typing import NDArray
from badminton_analysis.models.constants import (
    IMPACT_FRAME_SEARCH_WINDOW_AFTER,
    IMPACT_FRAME_SEARCH_WINDOW_BEFORE,
)
from badminton_analysis.ml.skeleton_normalization import (
    refine_delayed_overhead_contact_phase_indices,
)
from badminton_analysis.services.video_analyzer import VideoAnalyzer


def _smash_rotation_completion(
    skeleton: NDArray[np.floating],
    confidence: NDArray[np.floating],
    *,
    start: int,
    contact: int,
    latest: int,
) -> int | None:
    """The frame after contact where the shoulders finish turning forward."""
    coordinates = np.asarray(skeleton, dtype=np.float64)
    seen = np.all(np.asarray(confidence, dtype=np.float64)[:, (5, 6)] > 0.3, axis=1)
    span = np.linalg.norm(coordinates[:, 5] - coordinates[:, 6], axis=-1)
    stance = slice(start, max(start + 1, contact - 10))
    stance_seen = seen[stance] & np.isfinite(span[stance])
    if not np.any(stance_seen):
        return None
    initial = float(np.median(span[stance][stance_seen]))
    if initial <= 1e-6:
        return None
    rotation = np.abs(span / initial - 1.0)
    kernel = np.ones(5, dtype=np.float64) / 5.0
    rotation = np.convolve(np.pad(rotation, (2, 2), mode="edge"), kernel, mode="valid")
    window = slice(contact, latest + 1)
    candidates = np.where(seen[window] & np.isfinite(rotation[window]), rotation[window], -np.inf)
    if not np.any(np.isfinite(candidates)):
        return None
    return contact + int(np.argmax(candidates))


def _smash_eimd_v3_phases(
    hand_positions: Sequence[Sequence[float]],
    elbow_positions: Sequence[Sequence[float]],
    skeleton: NDArray[np.floating] | None = None,
    confidence: NDArray[np.floating] | None = None,
) -> tuple[int, int, int, int, int]:
    """Reproduce the broad learner ending range used by EIMD-v3 smash."""
    start, _, acceleration_end = VideoAnalyzer.find_acc_analysis_window(
        list(hand_positions), list(elbow_positions)
    )
    hand = np.asarray(hand_positions, dtype=np.float64)
    elbow = np.asarray(elbow_positions, dtype=np.float64)
    contact = start + int(np.argmin(hand[start : acceleration_end + 1, 1]))
    start = max(0, contact - 2 * IMPACT_FRAME_SEARCH_WINDOW_BEFORE)
    minimum_follow_through = max(4, IMPACT_FRAME_SEARCH_WINDOW_AFTER // 2)
    # The lowest elbow after contact.
    end = min(
        len(hand) - 1,
        max(
            contact + int(np.argmax(elbow[contact:, 1])),
            acceleration_end,
            contact + minimum_follow_through,
        ),
    )
    completed = (
        _smash_rotation_completion(
            skeleton, confidence, start=start, contact=contact, latest=end
        )
        if skeleton is not None and confidence is not None
        else None
    )
    if completed is not None:
        # The stroke ends where the shoulders finish turning forward.
        end = max(completed, contact + 4)
    preparation = (start + contact) // 2
    follow_through = (contact + end) // 2
    if not start < preparation < contact < follow_through < end:
        raise ValueError("smash EIMD-v3 phases are not strictly increasing")
    return start, preparation, contact, follow_through, end


# -- The interface every skill's phases module provides ----------------------

# The frame rate the scorer was built on; the source is converted to it first.
REQUIRED_SOURCE_FPS = 30.0


def swing_positions(hand_positions, motion_skeleton, confidence, handedness):
    """The wrist (and shoulder) tracks the window detector reads."""
    return hand_positions, None


def eimd_v3_phases(phases, *, tracking, full_skeleton, full_confidence, motion_skeleton, handedness):
    """The window generation priors and the skill guard were built on: (phases, source)."""
    hand_positions = tracking.get("hand_positions")
    elbow_positions = tracking.get("elbow_positions")
    if not hand_positions or not elbow_positions:
        raise ValueError("smash EIMD-v3 phases require wrist and elbow tracks")
    return (
        _smash_eimd_v3_phases(hand_positions, elbow_positions, full_skeleton, full_confidence),
        "acceleration_ending_range_v4",
    )


def current_phases(phases, *, tracking, full_skeleton, full_confidence, motion_skeleton, handedness):
    """The window grading and display use: (phases, source)."""
    # The wrist-velocity stop the window detector found is the grading window.
    return phases, "acceleration_wrist_velocity_stop_v6"


def refine_phase_indices(pose, phase_indices, source):
    """Adjust the phases on the generator's 64-frame clock: (indices, source)."""
    refined = refine_delayed_overhead_contact_phase_indices(pose, phase_indices)
    if np.array_equal(refined, phase_indices):
        return phase_indices, source
    return refined, "acceleration_wrist_velocity_stop_delayed_contact_v7"


def alignment_contract(contract):
    """The name a sample records for how its phases were aligned."""
    if contract == "eimd_v3":
        return "overhead_acceleration_ending_range_v4"
    return "overhead_wrist_velocity_stop_v6"
