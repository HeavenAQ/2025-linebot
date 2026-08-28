from __future__ import annotations

import json
import shutil
import subprocess

import cv2
import numpy as np
import pytest

from badminton_analysis.models.types import Skill
from service.renderer import (
    _align_smash_contact_timeline,
    _apply_constrained_hierarchical_placement,
    _apply_fixed_hierarchical_placement,
    _apply_smash_contact_leg_constraints,
    _retarget_corrected_pose,
    _constant_frame_rate_flag,
    _transcode_preserving_frame_rate,
    _normalized_to_source_frame,
    _ground_corrected_pose,
    _complete_interpolated_display_confidence,
    _ema_smooth_corrected_local_pose,
    _prepare_detected_pose_for_render,
    _transport_corrected_to_detected_pelvis,
    _transport_corrected_by_student_displacement,
    _smooth_corrected_bbox_placement,
    source_frame_rate,
)


def test_smash_contact_timeline_is_pinned_without_moving_endpoints() -> None:
    frames = 64
    pose = np.zeros((frames, 17, 2), dtype=np.float32)
    root = np.stack(
        (np.linspace(0.0, 4.0, frames), np.zeros(frames)), axis=-1
    ).astype(np.float32)
    pose[:, 6] = (0.0, 0.0)
    # A smooth racket-arm impulse centred well after the requested contact.
    time = np.arange(frames, dtype=np.float32)
    pose[:, 10, 0] = 12.0 / (1.0 + np.exp(-(time - 42.0) / 1.5))

    aligned_pose, aligned_root, details = _align_smash_contact_timeline(
        pose, root, target_index=30
    )

    assert details["contact_warp_applied"] is True
    assert abs(int(details["contact_event_after"]) - 30) <= 1
    np.testing.assert_allclose(aligned_pose[0], pose[0])
    np.testing.assert_allclose(aligned_pose[-1], pose[-1])
    np.testing.assert_allclose(aligned_root[0], root[0])
    np.testing.assert_allclose(aligned_root[-1], root[-1])


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
        "original_landmarks": [{}, {}, {}],
        "body_landmarks_2d": [{}, {}, {}],
        "body_keypoints_2d": list(coordinates),
        "body_confidence_2d": list(confidence),
        "hand_positions": [],
        "elbow_positions": [],
        "time_intervals": [0.0, 0.0, 0.0],
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


def test_constrained_hierarchical_placement_tracks_per_frame_translation() -> None:
    corrected = np.zeros((32, 17, 2), dtype=np.float32)
    corrected[:] = np.arange(17, dtype=np.float32)[None, :, None]
    detected = corrected.copy()
    detected[:, :, 0] += np.linspace(0.0, 12.0, len(detected))[:, None]
    detected[:, :, 1] += np.sin(np.linspace(0.0, np.pi, len(detected)))[:, None]

    placed = _apply_constrained_hierarchical_placement(
        corrected,
        detected,
        np.ones((len(detected), 17), dtype=np.float32),
        preparation_end=12,
    )

    assert placed[-1, 15, 0] > placed[0, 15, 0]
    np.testing.assert_allclose(
        placed[:, 10] - placed[:, 6],
        corrected[:, 10] - corrected[:, 6],
        atol=1e-6,
    )


def test_smash_root_transport_tracks_pelvis_without_changing_local_pose() -> None:
    timeline = 12
    corrected = np.zeros((timeline, 17, 2), dtype=np.float32)
    corrected[:] = np.stack(
        (np.arange(17, dtype=np.float32), np.arange(17, dtype=np.float32) * 2),
        axis=-1,
    )[None]
    corrected[:, :, 0] += np.linspace(0.0, 3.0, timeline)[:, None]
    detected = corrected.copy()
    detected[:, :, 0] += np.linspace(10.0, 80.0, timeline)[:, None]
    detected[:, :, 1] += np.linspace(-6.0, 18.0, timeline)[:, None]

    transported = _transport_corrected_to_detected_pelvis(corrected, detected)

    np.testing.assert_allclose(
        0.5 * (transported[:, 11] + transported[:, 12]),
        0.5 * (detected[:, 11] + detected[:, 12]),
        atol=1e-6,
    )


def test_smash_student_displacement_preserves_anchor_and_generated_motion() -> None:
    frames = 11
    corrected = np.zeros((frames, 17, 2), dtype=np.float32)
    corrected[:] = np.stack(
        (np.arange(17, dtype=np.float32), np.arange(17, dtype=np.float32)), axis=-1
    )[None]
    # Existing generated root/local motion must remain in the output.
    corrected[:, :, 0] += np.linspace(0.0, 5.0, frames)[:, None]
    detected = np.repeat(corrected[:1], frames, axis=0)
    travel = np.stack(
        (np.linspace(0.0, 40.0, frames), np.linspace(0.0, 10.0, frames)), axis=-1
    ).astype(np.float32)
    detected += travel[:, None]
    confidence = np.ones((frames, 17), dtype=np.float32)

    transported = _transport_corrected_by_student_displacement(
        corrected, detected, confidence
    )

    np.testing.assert_array_equal(transported[0], corrected[0])
    np.testing.assert_allclose(
        transported - transported[:, 11:12],
        corrected - corrected[:, 11:12],
        atol=5e-6,
    )
    # The student's displacement is added, not used to replace the generated
    # correction trajectory.
    np.testing.assert_allclose(
        transported[:, 11] - corrected[:, 11], travel, atol=1e-5
    )


def test_smash_local_ema_preserves_root_and_phase_endpoints() -> None:
    frames = 12
    pose = np.zeros((frames, 17, 2), dtype=np.float32)
    root = np.stack(
        (np.linspace(0.0, 20.0, frames), np.linspace(3.0, 9.0, frames)), axis=-1
    ).astype(np.float32)
    pose += root[:, None]
    pose[:, 5] = root
    pose[:, 7] = root + np.asarray((10.0, 0.0), dtype=np.float32)
    angles = np.asarray(
        (0.20, 0.35, 0.12, 0.38, 0.10, 0.40, 0.08, 0.42, 0.06, 0.44, 0.04, 0.46),
        dtype=np.float32,
    )
    pose[:, 9] = pose[:, 7] + 10.0 * np.stack(
        (np.cos(angles), np.sin(angles)), axis=-1
    )

    smoothed = _ema_smooth_corrected_local_pose(
        pose, alpha_current=0.85, reset_frames=(5, frames - 1)
    )

    np.testing.assert_allclose(
        0.5 * (smoothed[:, 11] + smoothed[:, 12]), root, atol=1e-6
    )
    np.testing.assert_array_equal(smoothed[0], pose[0])
    np.testing.assert_array_equal(smoothed[5], pose[5])
    np.testing.assert_array_equal(smoothed[-1], pose[-1])
    before_jitter = np.linalg.norm(np.diff(pose[:, 9] - root, n=2, axis=0))
    after_jitter = np.linalg.norm(np.diff(smoothed[:, 9] - root, n=2, axis=0))
    assert after_jitter < before_jitter


def test_smash_bbox_placement_smoothing_is_rigid_and_endpoint_preserving() -> None:
    frames = 20
    pose = np.zeros((frames, 17, 2), dtype=np.float32)
    template = np.stack(
        (np.arange(17, dtype=np.float32), 2.0 * np.arange(17, dtype=np.float32)),
        axis=-1,
    )
    pose[:] = template[None]
    pose[9:12] += np.asarray((120.0, -80.0), dtype=np.float32)

    smoothed = _smooth_corrected_bbox_placement(pose, alpha_current=0.65)

    np.testing.assert_array_equal(smoothed[0], pose[0])
    np.testing.assert_array_equal(smoothed[-1], pose[-1])
    np.testing.assert_allclose(
        smoothed - smoothed[:, 11:12],
        pose - pose[:, 11:12],
        atol=1e-5,
    )
    before_anchor = 0.5 * (
        pose[:, (5, 6, 11, 12)].min(axis=1)
        + pose[:, (5, 6, 11, 12)].max(axis=1)
    )
    after_anchor = 0.5 * (
        smoothed[:, (5, 6, 11, 12)].min(axis=1)
        + smoothed[:, (5, 6, 11, 12)].max(axis=1)
    )
    assert np.linalg.norm(np.diff(after_anchor, n=3, axis=0)) < np.linalg.norm(
        np.diff(before_anchor, n=3, axis=0)
    )


def test_smash_contact_leg_constraint_handles_swapped_ankle_labels() -> None:
    frames = 15
    corrected = np.zeros((frames, 17, 2), dtype=np.float32)
    detected = np.zeros_like(corrected)
    corrected[:, 11] = (40.0, 50.0)
    corrected[:, 12] = (60.0, 50.0)
    corrected[:, 13] = (38.0, 80.0)
    corrected[:, 14] = (62.0, 80.0)
    corrected[:, 15] = (36.0, 110.0)
    corrected[:, 16] = (64.0, 110.0)
    detected[:] = corrected
    detected[:, 15] = (34.0, 108.0)
    detected[:, 16] = (66.0, 108.0)
    # RF-DETR alternates semantic ankle IDs, but the screen-space pair is
    # stable and both feet are planted.
    detected[1::2, 15], detected[1::2, 16] = (
        detected[1::2, 16].copy(), detected[1::2, 15].copy()
    )
    confidence = np.ones((frames, 17), dtype=np.float32)

    constrained = _apply_smash_contact_leg_constraints(
        corrected, detected, confidence
    )

    expected = np.asarray(((34.0, 108.0), (66.0, 108.0)), dtype=np.float32)
    for frame in range(frames):
        actual = constrained[frame, (15, 16)]
        direct = np.sum(np.linalg.norm(actual - expected, axis=-1))
        crossed = np.sum(np.linalg.norm(actual - expected[::-1], axis=-1))
        assert min(direct, crossed) < 1e-3
    np.testing.assert_array_equal(constrained[:, :13], corrected[:, :13])
    np.testing.assert_allclose(
        np.linalg.norm(constrained[:, 13] - constrained[:, 11], axis=-1),
        np.linalg.norm(corrected[:, 13] - corrected[:, 11], axis=-1),
        atol=1e-4,
    )


def _angle_between(first: np.ndarray, second: np.ndarray) -> float:
    first_angle = np.arctan2(first[1], first[0])
    second_angle = np.arctan2(second[1], second[0])
    return abs(
        float(
            np.arctan2(
                np.sin(second_angle - first_angle),
                np.cos(second_angle - first_angle),
            )
        )
    )


def _cross_2d(first: np.ndarray, second: np.ndarray) -> float:
    """Scalar 2D cross product, including on NumPy 2.x."""
    return float(first[0] * second[1] - first[1] * second[0])


def _leg_pixels() -> tuple[np.ndarray, np.ndarray]:
    detected = np.zeros((17, 2), dtype=np.float32)
    corrected = np.zeros((17, 2), dtype=np.float32)
    detected[5], detected[6] = (75.0, 40.0), (125.0, 40.0)
    corrected[5], corrected[6] = (60.0, 25.0), (130.0, 55.0)
    detected[7], detected[9] = (60.0, 70.0), (50.0, 100.0)
    detected[8], detected[10] = (140.0, 70.0), (150.0, 100.0)
    corrected[7], corrected[9] = (20.0, 30.0), (0.0, 10.0)
    corrected[8], corrected[10] = (180.0, 30.0), (200.0, 10.0)
    for hip, knee, ankle, x in ((11, 13, 15, 80.0), (12, 14, 16, 120.0)):
        detected[hip] = (x, 100.0)
        detected[knee] = (x + 2.0, 150.0)
        detected[ankle] = (x, 205.0)
        corrected[hip] = (x + 4.0, 98.0)
        corrected[knee] = (x + 55.0, 145.0)
        corrected[ankle] = (x + 90.0, 190.0)
    return detected, corrected


def test_lift_rendering_moves_leg_directions_toward_expert_lunge() -> None:
    detected, corrected = _leg_pixels()

    result = _retarget_corrected_pose(corrected, detected, Skill.LIFT)
    rendered_lunge = result[16] - result[12]
    target_lunge = corrected[16] - corrected[12]

    assert np.sign(rendered_lunge[0]) == np.sign(target_lunge[0])
    assert result[16, 1] == pytest.approx(detected[16, 1], abs=1e-4)
    assert np.linalg.norm(result[14] - result[12]) == pytest.approx(
        np.linalg.norm(detected[14] - detected[12]), abs=1e-4
    )
    assert np.linalg.norm(result[16] - result[14]) == pytest.approx(
        np.linalg.norm(detected[16] - detected[14]), abs=1e-4
    )
    assert _angle_between(
        result[15] - result[13], detected[15] - detected[13]
    ) < 1e-6
    grounded_target = np.asarray(
        (target_lunge[0], detected[16, 1] - result[12, 1]), dtype=np.float32
    )
    target_cross = _cross_2d(grounded_target, corrected[14] - corrected[12])
    rendered_cross = _cross_2d(
        result[16] - result[12], result[14] - result[12]
    )
    assert np.sign(rendered_cross) == np.sign(target_cross)
    np.testing.assert_allclose(
        (result[11] + result[12]) * 0.5,
        (detected[11] + detected[12]) * 0.5,
    )


def test_serve_leg_correction_is_limited_to_twelve_degrees() -> None:
    detected, corrected = _leg_pixels()

    result = _retarget_corrected_pose(corrected, detected, Skill.SERVE)
    observed = detected[13] - detected[11]
    rendered = result[13] - result[11]

    assert np.linalg.norm(rendered) == pytest.approx(np.linalg.norm(observed))
    assert _angle_between(observed, rendered) <= np.deg2rad(12.0) + 1e-6
    assert _angle_between(observed, rendered) < _angle_between(
        observed, corrected[13] - corrected[11]
    )


def test_retargeting_keeps_corrected_arms_connected_and_detected_length() -> None:
    detected, corrected = _leg_pixels()

    result = _retarget_corrected_pose(corrected, detected, Skill.SMASH)

    for shoulder, elbow, wrist in ((5, 7, 9), (6, 8, 10)):
        assert np.linalg.norm(result[elbow] - result[shoulder]) == pytest.approx(
            np.linalg.norm(detected[elbow] - detected[shoulder])
        )
        assert np.linalg.norm(result[wrist] - result[elbow]) == pytest.approx(
            np.linalg.norm(detected[wrist] - detected[elbow])
        )


def test_serve_follow_through_can_move_dominant_forearm_across_body() -> None:
    detected, corrected = _leg_pixels()
    early = _retarget_corrected_pose(
        corrected, detected, Skill.SERVE, motion_progress=0.5
    )
    final = _retarget_corrected_pose(
        corrected, detected, Skill.SERVE, motion_progress=0.9
    )
    target_forearm = corrected[10] - corrected[8]

    assert _angle_between(final[10] - final[8], target_forearm) < _angle_between(
        early[10] - early[8], target_forearm
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


def test_normalized_correction_maps_back_to_source_motion_window() -> None:
    assert _normalized_to_source_frame(0, 64, 12, 43) == 12
    assert _normalized_to_source_frame(63, 64, 12, 43) == 43
    mapped = [
        _normalized_to_source_frame(index, 64, 12, 43) for index in range(64)
    ]
    assert mapped == sorted(mapped)
    assert set(mapped) == set(range(12, 44))


def test_full_body_correction_is_grounded_on_detected_support_ankle() -> None:
    detected, corrected = _leg_pixels()
    corrected += np.asarray((85.0, 110.0), dtype=np.float32)
    confidence = np.ones(17, dtype=np.float32)
    # Right ankle is lower in image coordinates and therefore load-bearing.
    detected[16, 1] = detected[15, 1] + 8.0

    grounded = _ground_corrected_pose(corrected, detected, confidence)

    np.testing.assert_allclose(grounded[16], detected[16])
    # Grounding is one rigid translation: expert joint configuration remains.
    np.testing.assert_allclose(
        grounded[10] - grounded[6], corrected[10] - corrected[6]
    )
    np.testing.assert_allclose(
        grounded[15] - grounded[16], corrected[15] - corrected[16]
    )


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
