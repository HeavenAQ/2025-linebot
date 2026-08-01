from pathlib import Path

import numpy as np

from badminton_analysis.ml.expert_vector import expert_vector_payload


def test_vector_payload_includes_expert_motion_window(tmp_path: Path) -> None:
    vector_path = tmp_path / "expert.npz"
    np.savez_compressed(
        vector_path,
        skeleton_3d=np.zeros((64, 17, 3), dtype=np.float32),
        confidence=np.ones((64, 17), dtype=np.float32),
        phase_indices=np.asarray((0, 16, 32, 48, 63), dtype=np.int32),
        analysis_window=np.asarray((30, 45, 74), dtype=np.int32),
        handedness=np.asarray("right"),
    )

    payload, handedness = expert_vector_payload(vector_path, video_fps=25.0)

    assert handedness == "right"
    assert payload["analysis_window_frames"] == [30, 45, 74]
    assert payload["motion_start_seconds"] == 1.2
    assert payload["motion_end_seconds"] == 3.0
