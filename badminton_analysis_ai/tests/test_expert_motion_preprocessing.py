from __future__ import annotations

import numpy as np

from badminton_analysis.ml.expert_motion_preprocessing import (
    _serve_hip_minimum_start,
    _serve_motion_onset_interval,
    _serve_preparation_was_truncated,
    _serve_shoulder_completion_phases,
)
from badminton_analysis.models.types import Handedness


def test_serve_start_uses_minimum_smoothed_pelvis_x_before_acceleration() -> None:
    frames = 24
    skeleton = np.zeros((frames, 17, 2), dtype=np.float32)
    timeline = np.arange(frames, dtype=np.float32)
    pelvis_x = np.square(timeline - 8.0)
    skeleton[:, 11, 0] = pelvis_x - 0.2
    skeleton[:, 12, 0] = pelvis_x + 0.2

    start = _serve_hip_minimum_start(
        skeleton, detected_start=2, acceleration=16
    )

    assert start == 8


def test_serve_start_uses_visible_hip_when_other_hip_is_missing() -> None:
    skeleton = np.zeros((20, 17, 2), dtype=np.float32)
    pelvis_x = np.square(np.arange(20, dtype=np.float32) - 6.0)
    skeleton[:, 11, 0] = np.nan
    skeleton[:, 12, 0] = pelvis_x

    start = _serve_hip_minimum_start(
        skeleton, detected_start=1, acceleration=14
    )

    assert start == 6


def test_left_serve_start_uses_minimum_x_after_handedness_canonicalization() -> None:
    frames = 24
    right = np.zeros((frames, 17, 2), dtype=np.float32)
    timeline = np.arange(frames, dtype=np.float32)
    pelvis_x = np.square(timeline - 8.0)
    right[:, 11, 0] = pelvis_x - 0.2
    right[:, 12, 0] = pelvis_x + 0.2
    left = right.copy()
    left[..., 0] *= -1.0

    right_start = _serve_hip_minimum_start(
        right,
        detected_start=2,
        acceleration=16,
        handedness=Handedness.RIGHT,
    )
    left_start = _serve_hip_minimum_start(
        left,
        detected_start=2,
        acceleration=16,
        handedness=Handedness.LEFT,
    )

    assert right_start == 8
    assert left_start == right_start


def test_serve_start_rejects_late_secondary_minimum_that_collapses_preparation() -> None:
    frames = 24
    skeleton = np.zeros((frames, 17, 2), dtype=np.float32)
    pelvis_x = np.full(frames, 8.0, dtype=np.float32)
    pelvis_x[5:9] = 2.0
    pelvis_x[15:19] = 0.0
    skeleton[:, 11, 0] = pelvis_x - 0.2
    skeleton[:, 12, 0] = pelvis_x + 0.2

    start = _serve_hip_minimum_start(
        skeleton, detected_start=2, acceleration=18
    )

    # At least the final quarter of detected preparation remains before the
    # acceleration event, so the late swing minimum cannot become the start.
    assert start <= 14
    assert 18 - start >= 4


def test_serve_motion_onset_recovers_preparation_before_fixed_window() -> None:
    frames = 40
    skeleton = np.zeros((frames, 17, 2), dtype=np.float32)
    wrist_x = np.zeros(frames, dtype=np.float32)
    wrist_x[12:27] = np.linspace(0.0, 24.0, 15, dtype=np.float32)
    wrist_x[27:] = wrist_x[26]
    skeleton[:, 10, 0] = wrist_x

    search_start, onset = _serve_motion_onset_interval(
        skeleton,
        detected_start=22,
        acceleration=28,
    )

    assert search_start < 22
    assert 8 <= onset <= 16
    assert search_start <= onset


def test_serve_motion_onset_is_mirror_invariant() -> None:
    frames = 40
    right = np.zeros((frames, 17, 2), dtype=np.float32)
    wrist_x = np.zeros(frames, dtype=np.float32)
    wrist_x[12:27] = np.linspace(0.0, 24.0, 15, dtype=np.float32)
    wrist_x[27:] = wrist_x[26]
    right[:, 10, 0] = wrist_x
    left = right.copy()
    left[:, 9] = -right[:, 10]

    right_interval = _serve_motion_onset_interval(
        right,
        detected_start=22,
        acceleration=28,
        handedness=Handedness.RIGHT,
    )
    left_interval = _serve_motion_onset_interval(
        left,
        detected_start=22,
        acceleration=28,
        handedness=Handedness.LEFT,
    )

    assert left_interval == right_interval


def test_serve_truncation_gate_requires_raw_and_interpolated_evidence() -> None:
    # Long preparation is visible both before and after interpolation.
    assert _serve_preparation_was_truncated(
        detected_start=90,
        detected_peak=120,
        raw_onset_start=67,
        interpolated_onset_start=67,
    )
    # A detector gap hides part of the preparation, but raw motion still
    # supplies a four-frame minimum of independent evidence.
    assert _serve_preparation_was_truncated(
        detected_start=64,
        detected_peak=94,
        raw_onset_start=56,
        interpolated_onset_start=27,
    )
    # Do not replace a valid legacy contact anchor when a complete clip starts
    # exactly at the detected preparation boundary.
    assert not _serve_preparation_was_truncated(
        detected_start=32,
        detected_peak=62,
        raw_onset_start=32,
        interpolated_onset_start=32,
    )


def test_serve_contact_uses_across_body_direction_beyond_legacy_window() -> None:
    frames = 80
    right = np.zeros((frames, 17, 2), dtype=np.float32)
    right[:, 5] = np.asarray((2.0, 0.0), dtype=np.float32)
    right[:, 6] = np.asarray((0.0, 0.0), dtype=np.float32)
    right[:, 8] = np.asarray((0.0, 1.0), dtype=np.float32)
    right[:, 11] = np.asarray((1.5, 2.0), dtype=np.float32)
    right[:, 12] = np.asarray((0.5, 2.0), dtype=np.float32)
    wrist_x = np.zeros(frames, dtype=np.float32)
    wrist_x[10:21] = np.linspace(0.0, -24.0, 11, dtype=np.float32)
    wrist_x[21:31] = -24.0
    wrist_x[31:51] = -24.0 + 48.0 * np.square(
        np.linspace(0.0, 1.0, 20, dtype=np.float32)
    )
    wrist_x[51:] = wrist_x[50]
    right[:, 10, 0] = wrist_x
    right[:, 10, 1] = 1.0
    detected = (5, 10, 20, 25, 30)

    right_phases = _serve_shoulder_completion_phases(
        detected, right, Handedness.RIGHT
    )

    left = right.copy()
    left[..., 0] *= -1.0
    for first, second in ((5, 6), (7, 8), (9, 10), (11, 12), (13, 14), (15, 16)):
        left[:, (first, second)] = left[:, (second, first)]
    left_phases = _serve_shoulder_completion_phases(
        detected, left, Handedness.LEFT
    )

    assert right_phases[2] > detected[-1]
    assert right_phases[-1] > right_phases[2]
    assert left_phases == right_phases
