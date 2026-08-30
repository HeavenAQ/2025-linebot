from pathlib import Path

import numpy as np
import pytest

import service.renderer as renderer
from badminton_analysis.models.types import Handedness, Skill


def test_correction_video_contains_only_the_scored_analysis_range(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    written: list[np.ndarray] = []

    class Writer:
        def __init__(self, *args: object, **kwargs: object) -> None:
            del args, kwargs

        @staticmethod
        def fourcc(*args: str) -> int:
            del args
            return 0

        def isOpened(self) -> bool:
            return True

        def write(self, frame: np.ndarray) -> None:
            written.append(frame.copy())

        def release(self) -> None:
            pass

    monkeypatch.setattr(renderer.cv2, "VideoWriter", Writer)
    monkeypatch.setattr(renderer, "_draw_skeleton", lambda *args, **kwargs: None)
    monkeypatch.setattr(renderer, "_draw_header", lambda *args, **kwargs: None)
    monkeypatch.setattr(renderer, "_transcode_preserving_frame_rate", lambda *args: None)

    frames = [np.full((32, 32, 3), index, dtype=np.uint8) for index in range(5)]
    detected = np.zeros((5, 17, 2), dtype=np.float32)
    detected[..., 0] = np.arange(17, dtype=np.float32)
    detected[..., 1] = np.arange(17, dtype=np.float32) * 0.5
    confidence = np.ones((5, 17), dtype=np.float32)
    normalized = detected[1:4].copy()

    renderer.render_correction_video(
        tracking={
            "frames": frames,
            "original_landmarks": [{} for _ in frames],
            "body_keypoints_2d": detected,
            "body_confidence_2d": confidence,
            "hand_positions": [(0.0, 0.0) for _ in frames],
            "elbow_positions": [(0.0, 0.0) for _ in frames],
            "time_intervals": [0.0 for _ in frames],
        },
        original=normalized,
        corrected=normalized,
        confidence=np.ones((3, 17), dtype=np.float32),
        window=(1, 2, 3),
        handedness=Handedness.RIGHT,
        skill=Skill.SERVE,
        filename="fixture.mp4",
        score=100.0,
        output_path=tmp_path / "overlay.mp4",
        fps=30.0,
        generated_full_body=True,
    )

    assert len(written) == 3
    assert [int(frame[0, 0, 0]) for frame in written] == [1, 2, 3]
