from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


def expert_vector_payload(
    path: Path, video_fps: float
) -> tuple[dict[str, Any], str]:
    with np.load(path, allow_pickle=False) as sample:
        skeleton = sample["skeleton_3d"].astype(np.float32)
        confidence = sample["confidence"].astype(np.float32)
        phases = sample["phase_indices"].astype(np.int32)
        analysis_window = sample["analysis_window"].astype(np.int32)
        handedness = str(sample["handedness"].item())
    if skeleton.shape != (64, 17, 3) or confidence.shape != (64, 17):
        raise ValueError(f"unexpected expert vector shape: {path}")
    return (
        {
            "vector": skeleton.reshape(-1).astype(float).tolist(),
            "vector_shape": list(skeleton.shape),
            "confidence": confidence.reshape(-1).astype(float).tolist(),
            "confidence_shape": list(confidence.shape),
            "phase_indices": phases.astype(int).tolist(),
            "analysis_window_frames": analysis_window.astype(int).tolist(),
            "motion_start_seconds": float(analysis_window[0]) / video_fps,
            "motion_end_seconds": float(analysis_window[2] + 1) / video_fps,
        },
        handedness,
    )
