from __future__ import annotations

import json
import shutil
import subprocess

import cv2
import numpy as np
import pytest

from api.renderer import (
    _apply_fixed_hierarchical_placement,
    _constant_frame_rate_flag,
    _transcode_preserving_frame_rate,
    _complete_interpolated_display_confidence,
    _prepare_detected_pose_for_render,
    source_frame_rate,
)


def test_interpolated_joint_is_visible_without_synthesizing_elbow() -> None:
    confidence = np.ones((7, 17), dtype=np.float32)
    confidence[2:6, 8] = 0.0
    confidence[:, 10] = 0.0

    display = _complete_interpolated_display_confidence(confidence)

    np.testing.assert_allclose(display[2:6, 8], 0.0)
    np.testing.assert_allclose(display[:, 10], 0.0)
    np.testing.assert_allclose(display[:, 7], 1.0)


def test_renderer_preserves_measured_elbow_and_leaves_missing_elbow_absent() -> None:
    coordinates = np.zeros((3, 17, 2), dtype=np.float32)
    confidence = np.ones((3, 17), dtype=np.float32)
    coordinates[:, 6] = ((0, 0), (1, 0), (2, 0))
    coordinates[:, 10] = ((2, 0), (3, 0), (4, 0))
    # Frame 1 is an implausible bone-length outlier but is a real RF-DETR
    # measurement above the elbow-specific threshold and must not be replaced.
    coordinates[:, 8] = ((1, 1), (30, 20), (3, 1))
    confidence[:, 8] = (0.08, 0.07, 0.01)
    tracking = {
        "frames": [np.zeros((8, 8, 3), dtype=np.uint8) for _ in range(3)],
        "body_landmarks_2d": [{}, {}, {}],
        "body_keypoints_2d": list(coordinates),
        "body_confidence_2d": list(confidence),
        "hand_positions": [],
        "elbow_positions": [],
    }

    prepared, prepared_confidence = _prepare_detected_pose_for_render(tracking)

    np.testing.assert_allclose(prepared[:2, 8], coordinates[:2, 8])
    np.testing.assert_allclose(prepared_confidence[:, 8], (0.08, 0.07, 0.0))


def test_fixed_hierarchical_placement_preserves_corrected_motion() -> None:
    corrected = np.zeros((8, 17, 2), dtype=np.float32)
    corrected[:] = np.arange(17, dtype=np.float32)[None, :, None]
    corrected[:, :, 0] += np.linspace(0.0, 7.0, 8)[:, None]
    corrected[:, :, 1] += np.sin(np.linspace(0.0, 1.0, 8))[:, None]
    detected = corrected + np.asarray((10.0, 5.0), dtype=np.float32)
    detected[:, :15] += np.asarray((3.0, -2.0), dtype=np.float32)
    detected[:, :13] += np.asarray((-1.0, 4.0), dtype=np.float32)

    placed = _apply_fixed_hierarchical_placement(
        corrected,
        detected,
        np.ones((8, 17), dtype=np.float32),
        preparation_end=3,
    )

    np.testing.assert_allclose(placed, detected, atol=1e-6)
    np.testing.assert_allclose(
        np.diff(placed, axis=0), np.diff(corrected, axis=0), atol=1e-6
    )


@pytest.mark.skipif(
    shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
    reason="ffmpeg and ffprobe are required",
)
def test_transcode_preserves_exact_rational_frame_rate(tmp_path) -> None:
    raw_path = tmp_path / "raw.mp4"
    output_path = tmp_path / "output.mp4"
    writer = cv2.VideoWriter(
        str(raw_path), cv2.VideoWriter.fourcc(*"mp4v"), 30.0, (32, 32)
    )
    assert writer.isOpened()
    for frame_index in range(8):
        frame = np.full((32, 32, 3), frame_index * 20, dtype=np.uint8)
        writer.write(frame)
    writer.release()

    _transcode_preserving_frame_rate(raw_path, output_path, "30000/1001")

    metadata = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=r_frame_rate,avg_frame_rate,nb_frames",
            "-of",
            "json",
            str(output_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    stream = json.loads(metadata.stdout)["streams"][0]
    assert stream["r_frame_rate"] == "30000/1001"
    assert stream["avg_frame_rate"] == "30000/1001"
    assert int(stream["nb_frames"]) == 8
    assert source_frame_rate(output_path) == "30000/1001"


def test_constant_frame_rate_flag_falls_back_when_fps_mode_is_unknown(
    tmp_path, monkeypatch
):
    """An ffmpeg predating 5.0 must still get a constant frame rate.

    The container installs ffmpeg from its base image's distribution packages,
    which is older than the -fps_mode option; passing it there aborts the whole
    encode with "Option not found", which took down every analysis until the
    flag was chosen by asking the binary.
    """
    stub_dir = tmp_path / "bin"
    stub_dir.mkdir()
    stub = stub_dir / "ffmpeg"
    stub.write_text(
        "#!/bin/sh\n"
        'for arg in "$@"; do\n'
        '  if [ "$arg" = "-fps_mode" ]; then\n'
        "    echo 'Unrecognized option '\\''fps_mode'\\''.' >&2\n"
        "    echo 'Error splitting the argument list: Option not found' >&2\n"
        "    exit 1\n"
        "  fi\n"
        "done\n"
        "exit 0\n"
    )
    stub.chmod(0o755)
    monkeypatch.setenv("PATH", str(stub_dir))

    _constant_frame_rate_flag.cache_clear()
    try:
        assert _constant_frame_rate_flag() == "-vsync"
    finally:
        _constant_frame_rate_flag.cache_clear()


def test_constant_frame_rate_flag_prefers_fps_mode_when_supported(monkeypatch):
    _constant_frame_rate_flag.cache_clear()
    try:
        if shutil.which("ffmpeg") is None:
            pytest.skip("ffmpeg is not installed")
        assert _constant_frame_rate_flag() in {"-fps_mode", "-vsync"}
    finally:
        _constant_frame_rate_flag.cache_clear()
