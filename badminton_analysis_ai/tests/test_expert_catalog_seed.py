from pathlib import Path

import numpy as np
import pytest

from badminton_analysis.ml.expert_vector import (
    expert_phase_seconds,
    expert_source_frame_seconds,
    expert_vector_payload,
)
from service.expert_catalog import ExpertRecord
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


def test_expert_phase_seconds_invert_the_window_resample() -> None:
    # Window frames 30-74 at 25fps: the normalized sequence is a uniform
    # resample of that span, so index 0 lands on the motion start and index 63
    # on its last frame.
    seconds = expert_phase_seconds((0, 16, 32, 48, 63), (30, 45, 74), 25.0)

    assert seconds[0] == 1.2
    assert seconds[-1] == 74 / 25.0
    assert list(seconds) == sorted(seconds)
    assert seconds[2] == pytest.approx((30 + 44 * 32 / 63) / 25.0)


def test_expert_phase_seconds_reject_unusable_metadata() -> None:
    with pytest.raises(ValueError):
        expert_phase_seconds((0, 63), (30, 45, 74), 0.0)
    with pytest.raises(ValueError):
        expert_phase_seconds((0, 63), (74, 45, 30), 25.0)


def test_expert_record_without_seeded_checkpoints_reports_none() -> None:
    record = ExpertRecord(
        expert_id="legacy",
        display_name="legacy",
        skill="clear",
        handedness="right",
        video_object_path="experts/v1/clear/videos/legacy.mp4",
        vector_object_path="experts/v2/clear/vectors/legacy.npz",
        duration_seconds=4.0,
        fps=25.0,
        width=1920,
        height=1080,
        motion_start_seconds=1.2,
        motion_end_seconds=3.0,
        phase_indices=(),
        analysis_window_frames=(),
        source_phase_frames=(),
    )

    assert record.phase_seconds() == ()


def _vector_npz(path: Path, **overrides) -> Path:
    payload = {
        "skeleton_3d": np.zeros((64, 17, 3), dtype=np.float32),
        "confidence": np.ones((64, 17), dtype=np.float32),
        "phase_indices": np.asarray((0, 16, 32, 48, 63), dtype=np.int32),
        "analysis_window": np.asarray((30, 45, 74), dtype=np.int32),
        "handedness": np.asarray("right"),
    }
    payload.update(overrides)
    np.savez_compressed(path, **payload)
    return path


def test_vector_payload_times_the_window_from_real_source_frames(
    tmp_path: Path,
) -> None:
    # Tracking dropped 10 frames inside the motion, so the tracked window
    # (30-74) sits well before the frames it actually came from (36-92).
    vector_path = _vector_npz(
        tmp_path / "expert.npz",
        source_phase_indices=np.asarray((36, 50, 66, 80, 92), dtype=np.int32),
    )

    payload, _ = expert_vector_payload(vector_path, video_fps=25.0)

    assert payload["source_phase_frames"] == [36, 50, 66, 80, 92]
    assert payload["motion_start_seconds"] == 36 / 25.0
    assert payload["motion_end_seconds"] == 93 / 25.0
    # The tracked window is still published for provenance, but no longer times
    # anything: reading it as frame numbers is what put playback a beat early.
    assert payload["analysis_window_frames"] == [30, 45, 74]


def test_vector_payload_falls_back_when_source_frames_are_absent(
    tmp_path: Path,
) -> None:
    payload, _ = expert_vector_payload(_vector_npz(tmp_path / "old.npz"), video_fps=25.0)

    assert "source_phase_frames" not in payload
    assert payload["motion_start_seconds"] == 1.2
    assert payload["motion_end_seconds"] == 3.0


def test_vector_payload_rejects_mismatched_source_frames(tmp_path: Path) -> None:
    vector_path = _vector_npz(
        tmp_path / "bad.npz",
        source_phase_indices=np.asarray((36, 50), dtype=np.int32),
    )

    with pytest.raises(ValueError):
        expert_vector_payload(vector_path, video_fps=25.0)


def _record(**overrides) -> ExpertRecord:
    fields = dict(
        expert_id="e",
        display_name="e",
        skill="serve",
        handedness="right",
        video_object_path="experts/v1/serve/videos/e.mp4",
        vector_object_path="experts/v2/serve/vectors/e.npz",
        duration_seconds=4.0,
        fps=25.0,
        width=1920,
        height=1080,
        motion_start_seconds=36 / 25.0,
        motion_end_seconds=93 / 25.0,
        phase_indices=(0, 16, 32, 48, 63),
        analysis_window_frames=(30, 45, 74),
        source_phase_frames=(36, 50, 66, 80, 92),
    )
    fields.update(overrides)
    return ExpertRecord(**fields)


def test_expert_record_prefers_source_frames_over_window_interpolation() -> None:
    record = _record()

    assert record.phase_seconds() == expert_source_frame_seconds(
        (36, 50, 66, 80, 92), 25.0
    )
    # Interpolating the tracked window would have put the contact keyframe at
    # 2.09s when the expert actually reaches it at 2.64s — half a second early,
    # which on a serve is most of the forward swing.
    interpolated = expert_phase_seconds((0, 16, 32, 48, 63), (30, 45, 74), 25.0)
    assert record.phase_seconds()[2] == pytest.approx(2.64)
    assert interpolated[2] == pytest.approx(2.0940, abs=1e-4)
    assert record.phase_seconds()[2] - interpolated[2] > 0.5


def test_expert_record_checkpoints_stay_inside_the_motion_window() -> None:
    record = _record()

    seconds = record.phase_seconds()
    assert seconds[0] >= record.motion_start_seconds
    assert seconds[-1] <= record.motion_end_seconds


def test_expert_record_without_source_frames_uses_the_window() -> None:
    record = _record(source_phase_frames=())

    assert record.phase_seconds() == expert_phase_seconds(
        (0, 16, 32, 48, 63), (30, 45, 74), 25.0
    )
