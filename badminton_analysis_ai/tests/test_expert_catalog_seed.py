from pathlib import Path

import numpy as np

from badminton_analysis.ml.expert_vector import expert_vector_payload
from seed_expert_catalog import (
    LEGACY_VIDEO_DIRS,
    NSTC_EXPERT_COUNTS,
    _expert_sources,
)


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


def test_expert_sources_use_only_nstc_hand_directories(
    tmp_path: Path, monkeypatch
) -> None:
    skill = "clear"
    legacy_dir = tmp_path / LEGACY_VIDEO_DIRS[skill]
    legacy_dir.mkdir(parents=True)
    for index in range(50):
        (legacy_dir / f"legacy-{index}.mp4").touch()

    monkeypatch.setitem(NSTC_EXPERT_COUNTS, skill, {"left": 1, "right": 1})
    nstc_root = tmp_path / "training_videos" / "nstc" / skill
    for hand in ("left", "right"):
        directory = nstc_root / hand
        directory.mkdir(parents=True)
        (directory / "1.mp4").touch()
    ignored_directory = nstc_root / "person-name"
    ignored_directory.mkdir()
    (ignored_directory / "ignored.mp4").touch()

    sources = _expert_sources(tmp_path, skill)

    assert len(sources) == 52
    assert sources["legacy-0"].known_handedness == "right"
    assert sources["nstc_left_1"].known_handedness == "left"
    assert sources["nstc_right_1"].known_handedness == "right"
    assert "ignored" not in sources


def test_serve_expert_sources_exclude_legacy_team_directory(
    tmp_path: Path, monkeypatch
) -> None:
    skill = "serve"
    legacy_dir = tmp_path / "scoring_videos" / "發球" / "羽球隊同學"
    legacy_dir.mkdir(parents=True)
    (legacy_dir / "must-not-be-used.mp4").touch()

    monkeypatch.setitem(NSTC_EXPERT_COUNTS, skill, {"left": 1, "right": 1})
    nstc_root = tmp_path / "training_videos" / "nstc" / skill
    for hand in ("left", "right"):
        directory = nstc_root / hand
        directory.mkdir(parents=True)
        (directory / f"{hand}.mp4").touch()

    sources = _expert_sources(tmp_path, skill)

    assert set(sources) == {"nstc_left_left", "nstc_right_right"}
    assert {source.dataset_source for source in sources.values()} == {"nstc"}
