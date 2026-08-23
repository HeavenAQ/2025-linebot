from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np


def expert_source_frame_seconds(
    source_phase_frames: Sequence[int], video_fps: float
) -> tuple[float, ...]:
    """Time each checkpoint from the source frame the extractor recorded.

    Pose tracking drops frames it cannot land a skeleton on, so a tracked index
    is not a frame number and the two drift apart across the stroke. The dataset
    already resolves this into ``source_phase_indices``; using it is the only
    way to hit the right frame of the expert's video.
    """
    if video_fps <= 0:
        raise ValueError("expert video fps must be positive")
    frames = tuple(int(value) for value in source_phase_frames)
    if not frames or any(value < 0 for value in frames):
        raise ValueError(f"invalid expert source phase frames: {frames}")
    return tuple(float(frame) / video_fps for frame in frames)


def expert_phase_seconds(
    phase_indices: Sequence[int],
    analysis_window_frames: Sequence[int],
    video_fps: float,
    *,
    target_frames: int = 64,
) -> tuple[float, ...]:
    """Approximate checkpoint times for vectors that predate source frames.

    Interpolating across the analysis window assumes tracking kept every frame,
    which is only true when nothing was dropped. Prefer
    ``expert_source_frame_seconds``; this exists so catalog entries seeded
    before source frames were published still align roughly.
    """
    if video_fps <= 0:
        raise ValueError("expert video fps must be positive")
    if target_frames < 2:
        raise ValueError("target_frames must cover at least two frames")
    window = tuple(int(value) for value in analysis_window_frames)
    if len(window) != 3 or window[2] <= window[0]:
        raise ValueError(f"invalid expert analysis window: {window}")
    start, end = window[0], window[2]
    span = end - start
    return tuple(
        (start + span * min(max(int(index), 0), target_frames - 1) / (target_frames - 1))
        / video_fps
        for index in phase_indices
    )


def expert_vector_payload(
    path: Path, video_fps: float
) -> tuple[dict[str, Any], str]:
    with np.load(path, allow_pickle=False) as sample:
        skeleton = sample["skeleton_3d"].astype(np.float32)
        confidence = sample["confidence"].astype(np.float32)
        phases = sample["phase_indices"].astype(np.int32)
        analysis_window = sample["analysis_window"].astype(np.int32)
        source_phases = (
            sample["source_phase_indices"].astype(np.int32)
            if "source_phase_indices" in sample.files
            else None
        )
        handedness = str(sample["handedness"].item())
    if skeleton.shape != (64, 17, 3) or confidence.shape != (64, 17):
        raise ValueError(f"unexpected expert vector shape: {path}")
    if source_phases is not None and source_phases.shape != phases.shape:
        raise ValueError(f"expert source phases do not match phase indices: {path}")

    # The analysis window is in tracked indices, which only equal frame numbers
    # when tracking dropped nothing. Where the extractor published the real
    # source frames, time the window off those so the motion bounds and the
    # checkpoints inside them share one clock.
    if source_phases is not None:
        motion_start = float(source_phases[0]) / video_fps
        motion_end = float(source_phases[-1] + 1) / video_fps
    else:
        motion_start = float(analysis_window[0]) / video_fps
        motion_end = float(analysis_window[2] + 1) / video_fps

    payload: dict[str, Any] = {
        "vector": skeleton.reshape(-1).astype(float).tolist(),
        "vector_shape": list(skeleton.shape),
        "confidence": confidence.reshape(-1).astype(float).tolist(),
        "confidence_shape": list(confidence.shape),
        "phase_indices": phases.astype(int).tolist(),
        "analysis_window_frames": analysis_window.astype(int).tolist(),
        "motion_start_seconds": motion_start,
        "motion_end_seconds": motion_end,
    }
    if source_phases is not None:
        payload["source_phase_frames"] = source_phases.astype(int).tolist()
    return payload, handedness
