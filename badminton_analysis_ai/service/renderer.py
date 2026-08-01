from __future__ import annotations

import subprocess
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
from numpy.typing import NDArray
from PIL import Image, ImageDraw, ImageFont

from badminton_analysis.ml.skeleton_normalization import (
    landmark_dicts_to_array,
    resample_sequence,
)
from badminton_analysis.ml.skeleton_scoring import BONES
from badminton_analysis.models.types import Handedness, TrackingData

_LEFT_RIGHT_PAIRS = (
    (1, 2), (3, 4), (5, 6), (7, 8),
    (9, 10), (11, 12), (13, 14), (15, 16),
)


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
    issues: list[dict[str, object]],
) -> None:
    height, width = frame.shape[:2]
    radius = max(20, round(min(height, width) * 0.025))
    for issue in issues:
        for joint_id in issue["joint_ids"]:  # type: ignore[union-attr]
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
    filename: str,
    score: float,
    output_path: Path,
    fps: float,
    feedback: list[dict[str, object]] | None = None,
    pause_seconds: float = 0.0,
) -> None:
    start, _, end = window
    target_frames = len(original)
    frame_indices = np.rint(np.linspace(start, end, target_frames)).astype(np.int64)
    selected = tracking["body_landmarks_2d"][start : end + 1]
    raw_2d, raw_confidence = landmark_dicts_to_array(selected, 2)
    raw_2d = _interpolate(raw_2d, raw_confidence)
    if handedness == Handedness.LEFT:
        raw_2d, raw_confidence = _canonicalize_left(raw_2d, raw_confidence)
    pixel_2d = resample_sequence(raw_2d, target_frames)
    pixel_confidence = np.clip(
        resample_sequence(raw_confidence, target_frames), 0.0, 1.0
    )

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
        feedback_by_frame: dict[int, list[dict[str, object]]] = {}
        for issue in feedback or []:
            feedback_by_frame.setdefault(int(issue["frame_index"]), []).append(issue)
        for target_index, frame_index in enumerate(frame_indices):
            frame = tracking["frames"][int(frame_index)].copy()
            mask = np.minimum(confidence[target_index], pixel_confidence[target_index])
            transform = _fit_affine(original[target_index], pixel_2d[target_index], mask)
            corrected_pixels = _map_to_pixels(corrected[target_index], transform)
            _draw_skeleton(frame, pixel_2d[target_index], mask, (255, 210, 30), 4)
            _draw_skeleton(frame, corrected_pixels, mask, (55, 225, 75), 3)
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
