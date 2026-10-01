"""Where a serve starts, is struck and ends, on the learner's own clip."""

from __future__ import annotations

from typing import (
    Sequence,
)
import numpy as np
from numpy.typing import NDArray
from badminton_analysis.models.types import (
    Handedness,
)
from badminton_analysis.models.constants import (
    SERVE_WRIST_CONFIDENCE_FLOOR,
)
from badminton_analysis.services.video_analyzer import VideoAnalyzer


def _serve_hip_minimum_start(
    skeleton_2d: NDArray[np.floating],
    *,
    detected_start: int,
    acceleration: int,
    handedness: Handedness = Handedness.RIGHT,
    latest_start: int | None = None,
) -> int:
    """Anchor serve start at minimum canonical pelvis x before acceleration."""
    coordinates = np.asarray(skeleton_2d, dtype=np.float64)
    if coordinates.ndim != 3 or coordinates.shape[1:] != (17, 2):
        raise ValueError("skeleton_2d must have shape (T, 17, 2)")
    # A raw global minimum can occur during the swing itself.
    detected_span = acceleration - detected_start
    required_preparation = max(4, int(np.ceil(0.25 * detected_span)))
    percentage_cutoff = acceleration - required_preparation
    search_cutoff = (
        percentage_cutoff
        if latest_start is None
        else min(percentage_cutoff, latest_start)
    )
    search_end = min(len(coordinates), search_cutoff + 1)
    if not 0 <= detected_start < search_end:
        return detected_start

    hips = coordinates[:, (11, 12), 0]
    valid = np.isfinite(hips)
    counts = valid.sum(axis=1)
    pelvis_x = np.divide(
        np.where(valid, hips, 0.0).sum(axis=1),
        counts,
        out=np.full(len(coordinates), np.nan, dtype=np.float64),
        where=counts > 0,
    )
    observed = np.flatnonzero(np.isfinite(pelvis_x))
    if not len(observed):
        return detected_start
    pelvis_x = np.interp(np.arange(len(pelvis_x)), observed, pelvis_x[observed])
    kernel = np.ones(5, dtype=np.float64) / 5.0
    smoothed = np.convolve(np.pad(pelvis_x, (2, 2), mode="edge"), kernel, mode="valid")
    canonical_x = -smoothed if handedness == Handedness.LEFT else smoothed
    return detected_start + int(np.argmin(canonical_x[detected_start:search_end]))


def _serve_motion_onset_interval(
    skeleton_2d: NDArray[np.floating],
    *,
    detected_start: int,
    acceleration: int,
    handedness: Handedness = Handedness.RIGHT,
) -> tuple[int, int]:
    """Find stable preparation immediately preceding the main wrist episode."""
    coordinates = np.asarray(skeleton_2d, dtype=np.float64)
    if coordinates.ndim != 3 or coordinates.shape[1:] != (17, 2):
        raise ValueError("skeleton_2d must have shape (T, 17, 2)")
    if not 2 <= acceleration < len(coordinates):
        return detected_start, detected_start
    elbow, wrist = (7, 9) if handedness == Handedness.LEFT else (8, 10)
    relative_wrist = coordinates[:, wrist] - coordinates[:, elbow]
    trajectory = np.column_stack(
        [
            np.convolve(
                np.pad(relative_wrist[:, axis], (2, 2), mode="edge"),
                np.ones(5, dtype=np.float64) / 5.0,
                mode="valid",
            )
            for axis in range(2)
        ]
    )
    speed = np.linalg.norm(np.diff(trajectory, axis=0), axis=-1)
    pre_acceleration = speed[:acceleration]
    if not len(pre_acceleration) or not np.any(np.isfinite(pre_acceleration)):
        return detected_start, detected_start
    finite = np.where(np.isfinite(pre_acceleration), pre_acceleration, 0.0)
    kernel_width = min(7, len(finite))
    if kernel_width % 2 == 0:
        kernel_width -= 1
    kernel_width = max(kernel_width, 1)
    kernel = np.ones(kernel_width, dtype=np.float64) / kernel_width
    padding = kernel_width // 2
    coherent_speed = np.convolve(
        np.pad(finite, (padding, padding), mode="edge"),
        kernel,
        mode="valid",
    )
    episode = int(np.argmax(coherent_speed))
    peak_speed = float(coherent_speed[episode])
    lower_half = finite[finite <= np.quantile(finite, 0.5)]
    baseline = float(np.median(lower_half)) if len(lower_half) else 0.0
    noise_scale = (
        1.4826 * float(np.median(np.abs(lower_half - baseline)))
        if len(lower_half)
        else 0.0
    )
    threshold = max(0.15 * peak_speed, baseline + 3.0 * noise_scale, 1e-6)
    stable_frames = max(3, min(7, int(np.ceil(0.05 * acceleration))))
    onset = 0
    for end in range(episode, stable_frames - 1, -1):
        stable = coherent_speed[end - stable_frames : end]
        if float(np.mean(stable < threshold)) >= 0.8:
            onset = end
            break
    context = max(3, int(np.ceil(0.10 * max(acceleration - onset, 1))))
    search_start = min(detected_start, max(0, onset - context))
    return int(search_start), int(max(search_start, onset))


def _serve_shoulder_completion_phases(
    detected_phases: Sequence[int],
    skeleton_2d: NDArray[np.floating],
    handedness: Handedness,
    *,
    motion_skeleton_2d: NDArray[np.floating] | None = None,
) -> tuple[int, int, int, int, int]:
    """End serve at maximum shoulder angle after maximum acceleration."""
    phases = tuple(int(value) for value in detected_phases)
    if len(phases) != 5 or any(b <= a for a, b in zip(phases, phases[1:])):
        raise ValueError("serve phases must contain five increasing frames")
    coordinates = np.asarray(skeleton_2d, dtype=np.float64)
    if coordinates.ndim != 3 or coordinates.shape[1:] != (17, 2):
        raise ValueError("skeleton_2d must have shape (T, 17, 2)")
    motion_coordinates = (
        coordinates
        if motion_skeleton_2d is None
        else np.asarray(motion_skeleton_2d, dtype=np.float64)
    )
    if motion_coordinates.shape != coordinates.shape:
        raise ValueError("motion_skeleton_2d must match skeleton_2d")
    (
        start,
        detected_preparation,
        detected_peak,
        detected_follow_through,
        detected_end,
    ) = phases
    if handedness == Handedness.LEFT:
        shoulder, elbow, wrist, opposite_shoulder = 5, 7, 9, 6
    else:
        shoulder, elbow, wrist, opposite_shoulder = 6, 8, 10, 5

    kernel = np.ones(5, dtype=np.float64) / 5.0
    motion_relative_wrist = (
        motion_coordinates[:, wrist] - motion_coordinates[:, shoulder]
    )
    motion_smoothed_wrist = np.column_stack(
        [
            np.convolve(
                np.pad(motion_relative_wrist[:, axis], (2, 2), mode="edge"),
                kernel,
                mode="valid",
            )
            for axis in range(2)
        ]
    )
    forward_axes = (
        motion_coordinates[:, opposite_shoulder] - motion_coordinates[:, shoulder]
    )
    valid_axes = np.all(np.isfinite(forward_axes), axis=1)
    forward_axis = (
        np.median(forward_axes[valid_axes], axis=0)
        if np.any(valid_axes)
        else np.asarray((1.0, 0.0), dtype=np.float64)
    )
    if float(np.linalg.norm(forward_axis)) <= 1e-8:
        forward_axis = np.asarray((1.0, 0.0), dtype=np.float64)

    # Search the whole clip.
    directional_search_start = start + 2
    directional_search_stop = len(motion_smoothed_wrist) - 2
    if directional_search_stop <= directional_search_start:
        raise ValueError("serve directional acceleration range is too short")
    provisional_acceleration = directional_search_start + int(
        VideoAnalyzer._directional_acceleration_peak(
            motion_smoothed_wrist[
                directional_search_start : directional_search_stop + 1
            ],
            forward_axis=forward_axis,
        )
    )
    provisional_acceleration = int(
        np.clip(
            provisional_acceleration,
            directional_search_start,
            directional_search_stop,
        )
    )
    raw_onset_start, _ = _serve_motion_onset_interval(
        coordinates,
        detected_start=start,
        acceleration=provisional_acceleration,
        handedness=handedness,
    )
    onset_start, onset_end = _serve_motion_onset_interval(
        motion_coordinates,
        detected_start=start,
        acceleration=provisional_acceleration,
        handedness=handedness,
    )
    # Never fall back to acceleration magnitude here.
    acceleration = provisional_acceleration
    onset_start, onset_end = _serve_motion_onset_interval(
        motion_coordinates,
        detected_start=start,
        acceleration=acceleration,
        handedness=handedness,
    )
    start = _serve_hip_minimum_start(
        coordinates,
        detected_start=onset_start,
        acceleration=acceleration,
        handedness=handedness,
        latest_start=onset_end,
    )

    # The stroke finishes where the racket elbow is highest after contact.
    elbow_height = np.convolve(
        np.pad(-coordinates[:, elbow, 1], (2, 2), mode="edge"), kernel, mode="valid"
    )
    completion_start = acceleration + 2
    completion_end = len(coordinates) - 1
    if completion_start > completion_end:
        raise ValueError("serve shoulder-completion range is too short")
    completion = completion_start + int(
        np.nanargmax(elbow_height[completion_start : completion_end + 1])
    )
    preparation = start + max(1, int(round(0.45 * (acceleration - start))))
    preparation = min(preparation, acceleration - 1)
    follow_through = (acceleration + completion) // 2
    return start, preparation, acceleration, follow_through, completion


def _serve_eimd_v3_phases(
    detected_phases: Sequence[int],
    skeleton_2d: NDArray[np.floating],
    handedness: Handedness,
) -> tuple[int, int, int, int, int]:
    """Reproduce the phase contract used to train the EIMD-v3 serve model."""
    phases = tuple(int(value) for value in detected_phases)
    if len(phases) != 5 or any(b <= a for a, b in zip(phases, phases[1:])):
        raise ValueError("serve phases must contain five increasing frames")
    coordinates = np.asarray(skeleton_2d, dtype=np.float64)
    if coordinates.ndim != 3 or coordinates.shape[1:] != (17, 2):
        raise ValueError("skeleton_2d must have shape (T, 17, 2)")
    start, preparation, _, detected_follow_through, detected_end = phases
    if handedness == Handedness.LEFT:
        shoulder, elbow, wrist = 5, 7, 9
    else:
        shoulder, elbow, wrist = 6, 8, 10
    kernel = np.ones(5, dtype=np.float64) / 5.0
    relative_wrist = coordinates[:, wrist] - coordinates[:, shoulder]
    smoothed_wrist = np.column_stack(
        [
            np.convolve(
                np.pad(relative_wrist[:, axis], (2, 2), mode="edge"),
                kernel,
                mode="valid",
            )
            for axis in range(2)
        ]
    )
    acceleration_magnitude = np.linalg.norm(
        np.diff(smoothed_wrist, n=2, axis=0), axis=-1
    )
    acceleration_start = max(start + 2, preparation)
    acceleration_stop = min(detected_follow_through, detected_end - 2)
    if acceleration_stop < acceleration_start:
        raise ValueError("serve acceleration range is too short")
    acceleration = acceleration_start + int(
        np.nanargmax(acceleration_magnitude[acceleration_start - 1 : acceleration_stop])
    )
    # The stroke finishes where the racket elbow is highest after contact.
    elbow_height = np.convolve(
        np.pad(-coordinates[:, elbow, 1], (2, 2), mode="edge"), kernel, mode="valid"
    )
    completion_start = acceleration + 2
    if completion_start > detected_end:
        raise ValueError("serve shoulder-completion range is too short")
    completion = completion_start + int(
        np.nanargmax(elbow_height[completion_start : detected_end + 1])
    )
    preparation = max(start + 1, min(preparation, acceleration - 1))
    follow_through = (acceleration + completion) // 2
    return start, preparation, acceleration, follow_through, completion


def _serve_swing_positions(
    hand_positions: Sequence[Sequence[float]],
    skeleton: NDArray[np.floating],
    confidence: NDArray[np.floating],
    handedness: Handedness,
) -> tuple[list[tuple[float, float]], list[tuple[float, float]]]:
    """The racket wrist with poorly seen frames bridged, and its shoulder."""
    wrist, shoulder = (9, 5) if handedness == Handedness.LEFT else (10, 6)
    hand = np.asarray(hand_positions, dtype=np.float64).copy()
    seen = np.asarray(confidence, dtype=np.float64)[:, wrist] >= SERVE_WRIST_CONFIDENCE_FLOOR
    if 2 <= int(np.count_nonzero(seen)) < len(hand):
        frames = np.arange(len(hand))
        for axis in range(hand.shape[1]):
            hand[~seen, axis] = np.interp(frames[~seen], frames[seen], hand[seen, axis])
    shoulders = np.asarray(skeleton, dtype=np.float64)[:, shoulder]
    return (
        [tuple(map(float, point)) for point in hand],
        [tuple(map(float, point)) for point in shoulders],
    )


# -- The interface every skill's phases module provides ----------------------

# The scorer reads the upload at its own frame rate.
REQUIRED_SOURCE_FPS = None


def swing_positions(hand_positions, motion_skeleton, confidence, handedness):
    """The wrist (and shoulder) tracks the window detector reads."""
    if not hand_positions:
        return hand_positions, None
    return _serve_swing_positions(hand_positions, motion_skeleton, confidence, handedness)


def eimd_v3_phases(phases, *, tracking, full_skeleton, full_confidence, motion_skeleton, handedness):
    """The window generation priors and the skill guard were built on: (phases, source)."""
    return (
        _serve_eimd_v3_phases(phases, full_skeleton, handedness),
        "max_acceleration_shoulder_angle_v1",
    )


def current_phases(phases, *, tracking, full_skeleton, full_confidence, motion_skeleton, handedness):
    """The window grading and display use: (phases, source)."""
    return (
        _serve_shoulder_completion_phases(
            phases, full_skeleton, handedness, motion_skeleton_2d=motion_skeleton
        ),
        "across_body_directional_wrist_acceleration_v14",
    )


def refine_phase_indices(pose, phase_indices, source):
    """Adjust the phases on the generator's 64-frame clock: (indices, source)."""
    return phase_indices, source


def alignment_contract(contract):
    """The name a sample records for how its phases were aligned."""
    if contract == "eimd_v3":
        return "serve_max_acceleration_shoulder_angle_v1"
    return "serve_across_body_directional_wrist_acceleration_v14"
