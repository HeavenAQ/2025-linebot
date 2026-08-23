from __future__ import annotations

import subprocess
from functools import lru_cache
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from numpy.typing import NDArray
from PIL import Image, ImageDraw, ImageFont

from badminton_analysis.ml.skeleton_normalization import (
    interpolate_pose_sequence,
    landmark_dicts_to_array,
    resample_sequence,
)
from badminton_analysis.ml.skeleton_scoring import BONES
from badminton_analysis.models.types import Handedness, Skill, TrackingData

_LEFT_RIGHT_PAIRS = (
    (1, 2), (3, 4), (5, 6), (7, 8),
    (9, 10), (11, 12), (13, 14), (15, 16),
)
_LEG_CHAINS = ((11, 13, 15), (12, 14, 16))
_SERVE_MAX_LEG_CORRECTION_RADIANS = np.deg2rad(12.0)
_MAX_TORSO_CORRECTION_RADIANS = np.deg2rad(25.0)
_MAX_ARM_CORRECTION_RADIANS = np.deg2rad(55.0)
_SERVE_FOLLOW_THROUGH_START = 0.72
_SERVE_FOLLOW_THROUGH_MAX_UPPER_ARM_RADIANS = np.deg2rad(80.0)
_SERVE_FOLLOW_THROUGH_MAX_FOREARM_RADIANS = np.deg2rad(120.0)


@lru_cache(maxsize=1)
def _font() -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for candidate in (
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc",
        "/System/Library/Fonts/PingFang.ttc",
    ):
        if Path(candidate).exists():
            return ImageFont.truetype(candidate, 28)
    return ImageFont.load_default()


@lru_cache(maxsize=1)
def _feedback_font() -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for candidate in (
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/System/Library/Fonts/PingFang.ttc",
    ):
        if Path(candidate).exists():
            return ImageFont.truetype(candidate, 22)
    return ImageFont.load_default()


def _interpolate(
    coordinates: NDArray[np.floating], confidence: NDArray[np.floating]
) -> NDArray[np.float32]:
    values = np.asarray(coordinates, dtype=np.float64).copy()
    observed = np.asarray(confidence, dtype=np.float64)
    timeline = np.arange(len(values), dtype=np.float64)
    for joint in range(values.shape[1]):
        for dimension in range(values.shape[2]):
            series = values[:, joint, dimension]
            valid = (observed[:, joint] > 0) & np.isfinite(series)
            if not np.any(valid):
                values[:, joint, dimension] = 0.0
            elif np.count_nonzero(valid) == 1:
                values[:, joint, dimension] = series[valid][0]
            else:
                values[:, joint, dimension] = np.interp(
                    timeline, timeline[valid], series[valid]
                )
    return values.astype(np.float32)


def _canonicalize_left(
    coordinates: NDArray[np.float32], confidence: NDArray[np.float32]
) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
    result = coordinates.copy()
    result_confidence = confidence.copy()
    for left, right in _LEFT_RIGHT_PAIRS:
        result[:, [left, right]] = result[:, [right, left]]
        result_confidence[:, [left, right]] = result_confidence[:, [right, left]]
    return result, result_confidence


def _fit_affine(
    normalized: NDArray[np.float32],
    pixels: NDArray[np.float32],
    confidence: NDArray[np.float32],
) -> NDArray[np.float64]:
    valid = (
        (confidence > 0.05)
        & np.all(np.isfinite(normalized), axis=-1)
        & np.all(np.isfinite(pixels), axis=-1)
    )
    if np.count_nonzero(valid) < 3:
        raise ValueError("at least three visible joints are required for rendering")
    count = int(np.count_nonzero(valid))
    source = np.concatenate(
        (normalized[valid].astype(np.float64), np.ones((count, 1))), axis=1
    )
    transform, _, _, _ = np.linalg.lstsq(
        source, pixels[valid].astype(np.float64), rcond=None
    )
    return np.asarray(transform, dtype=np.float64)


def _map_to_pixels(
    normalized: NDArray[np.float32], transform: NDArray[np.float64]
) -> NDArray[np.float32]:
    homogeneous = np.concatenate(
        (normalized.astype(np.float64), np.ones((len(normalized), 1))), axis=1
    )
    return np.asarray(homogeneous @ transform, dtype=np.float32)


def _bounded_bone_vector(
    observed: NDArray[np.float32],
    target: NDArray[np.float32],
    maximum_angle: float,
) -> NDArray[np.float32]:
    length = float(np.linalg.norm(observed))
    if length <= 1e-6:
        return np.asarray(observed, dtype=np.float32)
    observed_angle = float(np.arctan2(observed[1], observed[0]))
    target_angle = float(np.arctan2(target[1], target[0]))
    difference = float(
        np.arctan2(
            np.sin(target_angle - observed_angle),
            np.cos(target_angle - observed_angle),
        )
    )
    angle = observed_angle + float(
        np.clip(difference, -maximum_angle, maximum_angle)
    )
    return np.asarray(
        (np.cos(angle) * length, np.sin(angle) * length), dtype=np.float32
    )


def _solve_two_bone_leg(
    hip: NDArray[np.float32],
    detected_knee: NDArray[np.float32],
    detected_ankle: NDArray[np.float32],
    target_hip: NDArray[np.float32],
    target_knee: NDArray[np.float32],
    target_ankle: NDArray[np.float32],
) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
    """Place a foot in the expert direction with detected segment lengths."""
    thigh_length = float(np.linalg.norm(detected_knee - hip))
    shin_length = float(np.linalg.norm(detected_ankle - detected_knee))
    target_thigh = float(np.linalg.norm(target_knee - target_hip))
    target_shin = float(np.linalg.norm(target_ankle - target_knee))
    if (
        thigh_length <= 1e-6
        or shin_length <= 1e-6
        or target_thigh + target_shin <= 1e-6
    ):
        return detected_knee.copy(), detected_ankle.copy()

    minimum_reach = abs(thigh_length - shin_length) + 1e-4
    maximum_reach = thigh_length + shin_length - 1e-4
    vertical = float(
        np.clip(target_ankle[1] - hip[1], -maximum_reach, maximum_reach)
    )
    horizontal_limit = float(
        np.sqrt(max(maximum_reach * maximum_reach - vertical * vertical, 0.0))
    )
    target_horizontal = float(target_ankle[0] - target_hip[0])
    horizontal = float(
        np.clip(target_horizontal, -horizontal_limit, horizontal_limit)
    )
    target_vector = np.asarray((horizontal, vertical), dtype=np.float64)
    target_length = float(np.linalg.norm(target_vector))
    if target_length < minimum_reach:
        minimum_horizontal = float(
            np.sqrt(max(minimum_reach * minimum_reach - vertical * vertical, 0.0))
        )
        sign = 1.0 if target_horizontal >= 0.0 else -1.0
        target_vector[0] = sign * minimum_horizontal
        target_length = float(np.linalg.norm(target_vector))
    if target_length <= 1e-6:
        return detected_knee.copy(), detected_ankle.copy()
    direction = target_vector / target_length
    reach = target_length
    ankle = np.asarray(hip, dtype=np.float64) + direction * reach
    along = (
        thigh_length * thigh_length
        - shin_length * shin_length
        + reach * reach
    ) / (2.0 * reach)
    height = float(np.sqrt(max(thigh_length * thigh_length - along * along, 0.0)))
    perpendicular = np.asarray((-direction[1], direction[0]), dtype=np.float64)
    target_knee_vector = np.asarray(target_knee - target_hip, dtype=np.float64)
    bend_cross = (
        target_vector[0] * target_knee_vector[1]
        - target_vector[1] * target_knee_vector[0]
    )
    bend_sign = 1.0 if bend_cross >= 0.0 else -1.0
    knee = (
        np.asarray(hip, dtype=np.float64)
        + direction * along
        + perpendicular * bend_sign * height
    )
    return knee.astype(np.float32), ankle.astype(np.float32)


def _retarget_corrected_pose(
    corrected_pixels: NDArray[np.float32],
    detected_pixels: NDArray[np.float32],
    skill: Skill,
    motion_progress: float = 0.0,
) -> NDArray[np.float32]:
    """Retarget projected directions onto a rooted, detected-length 2D skeleton."""
    corrected = np.asarray(corrected_pixels, dtype=np.float32)
    detected = np.asarray(detected_pixels, dtype=np.float32)
    result = detected.copy()

    detected_hip_center = (detected[11] + detected[12]) * 0.5
    corrected_hip_center = (corrected[11] + corrected[12]) * 0.5
    hip_half = _bounded_bone_vector(
        (detected[12] - detected[11]) * 0.5,
        (corrected[12] - corrected[11]) * 0.5,
        _MAX_TORSO_CORRECTION_RADIANS,
    )
    result[11] = detected_hip_center - hip_half
    result[12] = detected_hip_center + hip_half

    detected_shoulder_center = (detected[5] + detected[6]) * 0.5
    corrected_shoulder_center = (corrected[5] + corrected[6]) * 0.5
    result_shoulder_center = detected_hip_center + _bounded_bone_vector(
        detected_shoulder_center - detected_hip_center,
        corrected_shoulder_center - corrected_hip_center,
        _MAX_TORSO_CORRECTION_RADIANS,
    )
    shoulder_half = _bounded_bone_vector(
        (detected[6] - detected[5]) * 0.5,
        (corrected[6] - corrected[5]) * 0.5,
        _MAX_TORSO_CORRECTION_RADIANS,
    )
    result[5] = result_shoulder_center - shoulder_half
    result[6] = result_shoulder_center + shoulder_half

    for shoulder, elbow, wrist in ((5, 7, 9), (6, 8, 10)):
        is_serve_follow_through = (
            skill == Skill.SERVE
            and shoulder == 6
            and motion_progress >= _SERVE_FOLLOW_THROUGH_START
        )
        upper_arm_limit = (
            _SERVE_FOLLOW_THROUGH_MAX_UPPER_ARM_RADIANS
            if is_serve_follow_through
            else _MAX_ARM_CORRECTION_RADIANS
        )
        forearm_limit = (
            _SERVE_FOLLOW_THROUGH_MAX_FOREARM_RADIANS
            if is_serve_follow_through
            else _MAX_ARM_CORRECTION_RADIANS
        )
        result[elbow] = result[shoulder] + _bounded_bone_vector(
            detected[elbow] - detected[shoulder],
            corrected[elbow] - corrected[shoulder],
            upper_arm_limit,
        )
        result[wrist] = result[elbow] + _bounded_bone_vector(
            detected[wrist] - detected[elbow],
            corrected[wrist] - corrected[elbow],
            forearm_limit,
        )

    for hip, knee, ankle in _LEG_CHAINS:
        if skill == Skill.LIFT and ankle == 16:
            grounded_target_ankle = corrected[ankle].copy()
            grounded_target_ankle[1] = detected[ankle, 1]
            result[knee], result[ankle] = _solve_two_bone_leg(
                result[hip],
                detected[knee],
                detected[ankle],
                corrected[hip],
                corrected[knee],
                grounded_target_ankle,
            )
            continue
        maximum_leg_angle = (
            _SERVE_MAX_LEG_CORRECTION_RADIANS if skill == Skill.SERVE else 0.0
        )
        observed_thigh = detected[knee] - detected[hip]
        target_thigh = corrected[knee] - corrected[hip]
        result[knee] = result[hip] + _bounded_bone_vector(
            observed_thigh, target_thigh, maximum_leg_angle
        )
        observed_shin = detected[ankle] - detected[knee]
        target_shin = corrected[ankle] - corrected[knee]
        result[ankle] = result[knee] + _bounded_bone_vector(
            observed_shin, target_shin, maximum_leg_angle
        )

    head_translation = result_shoulder_center - detected_shoulder_center
    result[:5] = detected[:5] + head_translation
    return result


def _expand_display_confidence(
    confidence: NDArray[np.floating], radius: int = 2
) -> NDArray[np.float32]:
    values = np.asarray(confidence, dtype=np.float32)
    expanded = values.copy()
    for offset in range(1, radius + 1):
        expanded[offset:] = np.maximum(expanded[offset:], values[:-offset])
        expanded[:-offset] = np.maximum(expanded[:-offset], values[offset:])
    return expanded


def _draw_skeleton(
    frame: NDArray[np.uint8],
    coordinates: NDArray[np.float32],
    confidence: NDArray[np.float32],
    color: tuple[int, int, int],
    thickness: int,
) -> None:
    points = np.rint(coordinates).astype(np.int32)
    height, width = frame.shape[:2]
    for start, end in BONES:
        if confidence[start] <= 0.05 or confidence[end] <= 0.05:
            continue
        first, second = tuple(points[start]), tuple(points[end])
        if not (
            -width <= first[0] < 2 * width
            and -height <= first[1] < 2 * height
            and -width <= second[0] < 2 * width
            and -height <= second[1] < 2 * height
        ):
            continue
        cv2.line(frame, first, second, (12, 12, 12), thickness + 4, cv2.LINE_AA)
        cv2.line(frame, first, second, color, thickness, cv2.LINE_AA)
    for joint, point in enumerate(points):
        if confidence[joint] <= 0.05:
            continue
        cv2.circle(frame, tuple(point), thickness + 3, (12, 12, 12), -1, cv2.LINE_AA)
        cv2.circle(frame, tuple(point), thickness + 1, color, -1, cv2.LINE_AA)


def _draw_header(frame: NDArray[np.uint8], filename: str, score: float) -> None:
    width = min(frame.shape[1] - 24, 590)
    cv2.rectangle(frame, (12, 12), (width, 104), (18, 18, 18), -1)
    region = frame[12:62, 12:width]
    image = Image.fromarray(cv2.cvtColor(region, cv2.COLOR_BGR2RGB))
    label = f"{Path(filename).stem}  總分 {score:.1f}"
    ImageDraw.Draw(image).text((14, 4), label, font=_font(), fill=(245, 245, 245))
    frame[12:62, 12:width] = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
    cv2.line(frame, (28, 79), (68, 79), (255, 210, 30), 5, cv2.LINE_AA)
    cv2.putText(frame, "detected", (79, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (245, 245, 245), 1, cv2.LINE_AA)
    cv2.line(frame, (205, 79), (245, 79), (55, 225, 75), 5, cv2.LINE_AA)
    cv2.putText(frame, "corrected", (256, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (245, 245, 245), 1, cv2.LINE_AA)


def _draw_feedback(
    frame: NDArray[np.uint8],
    detected_pixels: NDArray[np.float32],
    issues: list[dict[str, Any]],
) -> None:
    height, width = frame.shape[:2]
    radius = max(20, round(min(height, width) * 0.025))
    for issue in issues:
        for joint_id in issue["joint_ids"]:
            point = detected_pixels[int(joint_id)]
            location = (int(round(point[0])), int(round(point[1])))
            if 0 <= location[0] < width and 0 <= location[1] < height:
                cv2.circle(frame, location, radius + 5, (15, 15, 15), 9, cv2.LINE_AA)
                cv2.circle(frame, location, radius, (40, 40, 245), 7, cv2.LINE_AA)

    panel_height = min(height // 3, 108 + 72 * len(issues))
    panel_top = height - panel_height
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, panel_top), (width, height), (15, 15, 15), -1)
    cv2.addWeighted(overlay, 0.9, frame, 0.1, 0.0, frame)
    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(image)
    draw.text((22, panel_top + 13), "教練指導暫停", font=_feedback_font(), fill=(255, 110, 95))
    y = panel_top + 50
    for index, issue in enumerate(issues, start=1):
        message = (
            f"{index}. {issue['title']} "
            f"{float(issue['criterion_score']):.1f}/"
            f"{float(issue['criterion_maximum']):.0f}分：{issue['feedback']}"
        )
        line = ""
        lines: list[str] = []
        for character in message:
            candidate = line + character
            if draw.textlength(candidate, font=_feedback_font()) <= width - 44:
                line = candidate
            else:
                lines.append(line)
                line = character
        if line:
            lines.append(line)
        for rendered_line in lines[:2]:
            draw.text((22, y), rendered_line, font=_feedback_font(), fill=(248, 248, 248))
            y += 28
        y += 7
    frame[:] = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)


def source_fps(video_path: Path) -> float:
    capture = cv2.VideoCapture(str(video_path))
    try:
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        return fps if np.isfinite(fps) and fps > 0 else 30.0
    finally:
        capture.release()


def probe_video(video_path: Path) -> dict[str, float | int]:
    capture = cv2.VideoCapture(str(video_path))
    try:
        fps = float(capture.get(cv2.CAP_PROP_FPS)) or 30.0
        frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        return {
            "fps": fps,
            "duration_seconds": frames / fps if fps > 0 else 0.0,
            "width": int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height": int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        }
    finally:
        capture.release()


def render_correction_video(
    *,
    tracking: TrackingData,
    original: NDArray[np.float32],
    corrected: NDArray[np.float32],
    confidence: NDArray[np.float32],
    window: tuple[int, int, int],
    handedness: Handedness,
    skill: Skill,
    filename: str,
    score: float,
    output_path: Path,
    fps: float,
    original_root: NDArray[np.float32] | None = None,
    corrected_root: NDArray[np.float32] | None = None,
    generated_full_body: bool = False,
    feedback: list[dict[str, Any]] | None = None,
    pause_seconds: float = 0.0,
) -> None:
    start, _, end = window
    target_frames = len(original)
    frame_indices = np.rint(np.linspace(start, end, target_frames)).astype(np.int64)
    selected = tracking["body_landmarks_2d"][start : end + 1]
    raw_2d, raw_confidence = landmark_dicts_to_array(selected, 2)
    raw_2d, raw_confidence = interpolate_pose_sequence(raw_2d, raw_confidence)
    if handedness == Handedness.LEFT:
        raw_2d, raw_confidence = _canonicalize_left(raw_2d, raw_confidence)
    pixel_2d = resample_sequence(raw_2d, target_frames)
    pixel_confidence = np.clip(
        resample_sequence(raw_confidence, target_frames), 0.0, 1.0
    )
    display_confidence = np.minimum(
        _expand_display_confidence(confidence),
        _expand_display_confidence(pixel_confidence),
    )
    original_root_values = (
        np.zeros((target_frames, 2), dtype=np.float32)
        if original_root is None
        else np.asarray(original_root, dtype=np.float32)
    )
    corrected_root_values = (
        original_root_values
        if corrected_root is None
        else np.asarray(corrected_root, dtype=np.float32)
    )
    if (
        original_root_values.shape != (target_frames, 2)
        or corrected_root_values.shape != (target_frames, 2)
    ):
        raise ValueError("root trajectories must have shape (T, 2)")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    raw_path = output_path.with_name(output_path.stem + ".raw.mp4")
    first_frame = tracking["frames"][int(frame_indices[0])]
    height, width = first_frame.shape[:2]
    writer = cv2.VideoWriter(
        str(raw_path), cv2.VideoWriter.fourcc(*"mp4v"), fps, (width, height)
    )
    if not writer.isOpened():
        raise RuntimeError(f"could not open output writer: {raw_path}")
    try:
        feedback_by_frame: dict[int, list[dict[str, Any]]] = {}
        for issue in feedback or []:
            feedback_by_frame.setdefault(int(issue["frame_index"]), []).append(issue)
        for target_index, frame_index in enumerate(frame_indices):
            frame = tracking["frames"][int(frame_index)].copy()
            mask = np.minimum(confidence[target_index], pixel_confidence[target_index])
            transform = _fit_affine(original[target_index], pixel_2d[target_index], mask)
            corrected_world = corrected[target_index] + (
                corrected_root_values[target_index]
                - original_root_values[target_index]
            )
            corrected_pixels = _map_to_pixels(corrected_world, transform)
            if not generated_full_body:
                corrected_pixels = _retarget_corrected_pose(
                    corrected_pixels,
                    pixel_2d[target_index],
                    skill,
                    target_index / max(len(frame_indices) - 1, 1),
                )
            display_mask = display_confidence[target_index]
            _draw_skeleton(
                frame, pixel_2d[target_index], display_mask, (255, 210, 30), 4
            )
            _draw_skeleton(
                frame, corrected_pixels, display_mask, (55, 225, 75), 3
            )
            _draw_header(frame, filename, score)
            issues = feedback_by_frame.get(target_index, [])
            if issues:
                _draw_feedback(frame, pixel_2d[target_index], issues)
            repetitions = 1 + (round(fps * pause_seconds) if issues else 0)
            for _ in range(repetitions):
                writer.write(frame)
    finally:
        writer.release()

    try:
        subprocess.run(
            [
                "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
                "-i", str(raw_path), "-an", "-c:v", "libx264", "-crf", "21",
                "-preset", "fast", "-pix_fmt", "yuv420p", "-movflags", "+faststart",
                str(output_path),
            ],
            check=True,
        )
    finally:
        raw_path.unlink(missing_ok=True)
