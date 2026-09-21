from __future__ import annotations

import json
import subprocess
from fractions import Fraction
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
from badminton_analysis.models.types import Handedness, TrackingData
from api.coaching_timeline import coaching_video_frame

_LEFT_RIGHT_PAIRS = (
    (1, 2),
    (3, 4),
    (5, 6),
    (7, 8),
    (9, 10),
    (11, 12),
    (13, 14),
    (15, 16),
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


def _expand_display_confidence(
    confidence: NDArray[np.floating], radius: int = 2
) -> NDArray[np.float32]:
    values = np.asarray(confidence, dtype=np.float32)
    expanded = values.copy()
    for offset in range(1, radius + 1):
        expanded[offset:] = np.maximum(expanded[offset:], values[:-offset])
        expanded[:-offset] = np.maximum(expanded[:-offset], values[offset:])
    return expanded


def _place_in_window(
    values: NDArray[np.floating], lead: int, span: int, total: int
) -> NDArray[np.floating]:
    """Put a model-space sequence back on the frames it was taken from.

    The model sees a 64-frame sample cut from its own phase window, which is
    usually shorter than the window being drawn. Stretching those frames across
    the whole render window plays the stroke at the wrong speed and puts the
    correction's contact somewhere the learner's contact is not. Resample onto
    the span it came from, hold the end poses either side, and let the display
    mask hide them.
    """
    placed = np.asarray(resample_sequence(values, span))
    if span == total:
        return placed
    head = np.repeat(placed[:1], lead, axis=0)
    tail = np.repeat(placed[-1:], total - lead - span, axis=0)
    return np.concatenate([head, placed, tail], axis=0)


def _complete_interpolated_display_confidence(
    confidence: NDArray[np.floating], reconstructed_confidence: float = 0.2
) -> NDArray[np.float32]:
    """Keep temporally reconstructed joints visible in review renders.

    ``interpolate_pose_sequence`` supplies finite coordinates for detector gaps
    and rejected limb outliers, but intentionally preserves their zero
    confidence for scoring.  Rendering with that same mask made an interpolated
    non-elbow joint disappear. Promote only the display copy when the joint has
    at least one real observation in this clip. Elbows are deliberately
    excluded: their lower detector threshold preserves measured coordinates,
    while truly missing elbow detections must remain hidden rather than being
    reconstructed. The scoring/model confidence is left untouched.
    """
    values = np.clip(np.asarray(confidence, dtype=np.float32), 0.0, 1.0)
    if values.ndim != 2 or values.shape[1] != 17:
        raise ValueError("display confidence must have shape (T, 17)")
    if not 0.05 < reconstructed_confidence <= 1.0:
        raise ValueError("reconstructed confidence must be in (0.05, 1]")
    observed_anywhere = np.any(values > 0.05, axis=0)
    completed = values.copy()
    missing = (completed <= 0.05) & observed_anywhere[None, :]
    missing[:, [7, 8]] = False
    completed[missing] = np.float32(reconstructed_confidence)
    return completed


def _prepare_detected_pose_for_render(
    tracking: TrackingData,
) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
    """Prepare detector coordinates without inventing elbow positions.

    Other joints keep the established outlier rejection/interpolation used by
    the review overlay. Elbows instead retain RF-DETR's measured coordinates at
    the lower 0.05 confidence cutoff. A below-threshold elbow remains absent.
    """
    dense_coordinates = tracking.get("body_keypoints_2d")
    dense_confidence = tracking.get("body_confidence_2d")
    if dense_coordinates is not None and dense_confidence is not None:
        measured = np.asarray(dense_coordinates, dtype=np.float32)
        measured_confidence = np.asarray(dense_confidence, dtype=np.float32)
        expected_frames = len(tracking["frames"])
        if measured.shape != (expected_frames, 17, 2) or measured_confidence.shape != (
            expected_frames,
            17,
        ):
            raise ValueError("dense detected pose must align with tracking frames")
    else:
        measured, measured_confidence = landmark_dicts_to_array(
            tracking["body_landmarks_2d"], 2
        )
    prepared, prepared_confidence = interpolate_pose_sequence(
        measured, measured_confidence
    )
    for elbow in (7, 8):
        accepted = (measured_confidence[:, elbow] > 0.05) & np.all(
            np.isfinite(measured[:, elbow]), axis=1
        )
        prepared[:, elbow] = measured[:, elbow]
        prepared_confidence[:, elbow] = np.where(
            accepted, measured_confidence[:, elbow], 0.0
        )
    return prepared.astype(np.float32), prepared_confidence.astype(np.float32)


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
    cv2.putText(
        frame,
        "detected",
        (79, 85),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (245, 245, 245),
        1,
        cv2.LINE_AA,
    )
    cv2.line(frame, (205, 79), (245, 79), (55, 225, 75), 5, cv2.LINE_AA)
    cv2.putText(
        frame,
        "corrected",
        (256, 85),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (245, 245, 245),
        1,
        cv2.LINE_AA,
    )


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
    draw.text(
        (22, panel_top + 13), "教練指導暫停", font=_feedback_font(), fill=(255, 110, 95)
    )
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
            draw.text(
                (22, y), rendered_line, font=_feedback_font(), fill=(248, 248, 248)
            )
            y += 28
        y += 7
    frame[:] = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)


def _normalized_frame_rate(value: str) -> str | None:
    try:
        rate = Fraction(value)
    except (ValueError, ZeroDivisionError):
        return None
    if rate <= 0:
        return None
    return f"{rate.numerator}/{rate.denominator}"


def source_frame_rate(video_path: Path) -> str:
    """Return the source stream's exact average frame rate as a fraction."""
    try:
        result = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=avg_frame_rate,r_frame_rate",
                "-of",
                "json",
                str(video_path),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        streams = json.loads(result.stdout).get("streams", [])
        if streams:
            for key in ("avg_frame_rate", "r_frame_rate"):
                rate = _normalized_frame_rate(str(streams[0].get(key, "")))
                if rate is not None:
                    return rate
    except (FileNotFoundError, subprocess.CalledProcessError, json.JSONDecodeError):
        pass

    capture = cv2.VideoCapture(str(video_path))
    try:
        fps = float(capture.get(cv2.CAP_PROP_FPS))
    finally:
        capture.release()
    if not np.isfinite(fps) or fps <= 0:
        return "30/1"
    rate = Fraction(fps).limit_denominator(100_000)
    return f"{rate.numerator}/{rate.denominator}"


def source_fps(video_path: Path) -> float:
    return float(Fraction(source_frame_rate(video_path)))


@lru_cache(maxsize=1)
def _constant_frame_rate_flag() -> str:
    """Name the flag this ffmpeg uses to force a constant frame rate.

    -fps_mode arrived in ffmpeg 5.0. The container takes ffmpeg from the base
    image's distribution packages, which are older than that and spell the
    same thing -vsync; a development machine is usually far newer and accepts
    both. Asking the binary keeps the two in step, and beats parsing a version
    string that varies between builds and forks.
    """
    probe = subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            "nullsrc=s=16x16:d=0.1",
            "-fps_mode",
            "cfr",
            "-f",
            "null",
            "-",
        ],
        capture_output=True,
    )
    return "-fps_mode" if probe.returncode == 0 else "-vsync"


def _transcode_preserving_frame_rate(
    raw_path: Path, output_path: Path, frame_rate: str
) -> None:
    """Encode every rendered frame at the source stream's exact rate."""
    normalized_rate = _normalized_frame_rate(frame_rate)
    if normalized_rate is None:
        raise ValueError(f"invalid frame rate: {frame_rate}")
    subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(raw_path),
            "-an",
            "-vf",
            f"setpts=N/(({normalized_rate})*TB)",
            "-r",
            normalized_rate,
            _constant_frame_rate_flag(),
            "cfr",
            "-c:v",
            "libx264",
            "-crf",
            "21",
            "-preset",
            "fast",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(output_path),
        ],
        check=True,
    )


def _apply_fixed_hierarchical_placement(
    corrected_pixels: NDArray[np.float32],
    detected_pixels: NDArray[np.float32],
    confidence: NDArray[np.floating],
    *,
    preparation_end: int,
) -> NDArray[np.float32]:
    """Apply one ankle, knee-chain, and hip-chain placement for a whole clip."""
    corrected = np.asarray(corrected_pixels, dtype=np.float32)
    detected = np.asarray(detected_pixels, dtype=np.float32)
    weights = np.asarray(confidence, dtype=np.float32)
    if corrected.shape != detected.shape or corrected.ndim != 3:
        raise ValueError("placement poses must share shape (T, 17, 2)")
    if weights.shape != corrected.shape[:2]:
        raise ValueError("placement confidence must have shape (T, 17)")
    if not 0 < preparation_end <= len(corrected):
        raise ValueError("invalid fixed-placement preparation window")
    prep = slice(0, preparation_end)
    ankle_scores = []
    for joint in (15, 16):
        visible = weights[prep, joint] > 0.05
        if np.any(visible):
            ankle_scores.append(
                (float(np.median(detected[prep, joint, 1][visible])), joint)
            )
    if not ankle_scores:
        return corrected.copy()
    support_ankle = max(ankle_scores)[1]
    ankle_visible = weights[prep, support_ankle] > 0.05
    ankle_delta = np.median(
        detected[prep, support_ankle][ankle_visible]
        - corrected[prep, support_ankle][ankle_visible],
        axis=0,
    )
    placed = corrected + ankle_delta

    knee_visible = np.minimum(weights[prep, 13], weights[prep, 14]) > 0.05
    if np.any(knee_visible):
        detected_knees = 0.5 * (detected[prep, 13] + detected[prep, 14])
        placed_knees = 0.5 * (placed[prep, 13] + placed[prep, 14])
        knee_delta = np.median(
            detected_knees[knee_visible] - placed_knees[knee_visible], axis=0
        )
        placed[:, :15] += knee_delta

    hip_visible = np.minimum(weights[prep, 11], weights[prep, 12]) > 0.05
    if np.any(hip_visible):
        detected_hips = 0.5 * (detected[prep, 11] + detected[prep, 12])
        placed_hips = 0.5 * (placed[prep, 11] + placed[prep, 12])
        hip_delta = np.median(
            detected_hips[hip_visible] - placed_hips[hip_visible], axis=0
        )
        placed[:, :13] += hip_delta
    return np.asarray(placed, dtype=np.float32)


def probe_video(video_path: Path) -> dict[str, float | int]:
    capture = cv2.VideoCapture(str(video_path))
    try:
        fps = source_fps(video_path)
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
    frame_rate: str | None = None,
    original_root: NDArray[np.float32] | None = None,
    corrected_root: NDArray[np.float32] | None = None,
    feedback: list[dict[str, Any]] | None = None,
    pause_seconds: float = 0.0,
    projected_corrected_pixels: NDArray[np.float32] | None = None,
    generated_source_window: tuple[int, int, int] | None = None,
) -> None:
    start, _, end = window
    target_frames = len(original)
    source_frame_count = len(tracking["frames"])
    if source_frame_count <= 0:
        raise ValueError("rendering requires at least one source frame")
    if not 0 <= start <= end < source_frame_count:
        raise ValueError("analysis window falls outside the source video")
    window_frame_count = end - start + 1
    if projected_corrected_pixels is not None:
        projected_corrected_pixels = np.asarray(
            projected_corrected_pixels, dtype=np.float32
        )
        if (
            projected_corrected_pixels.shape != (window_frame_count, 17, 2)
            or not np.isfinite(projected_corrected_pixels).all()
        ):
            raise ValueError(
                "Scored pixel overlay must exactly cover the render source window"
            )
    raw_2d, raw_confidence = _prepare_detected_pose_for_render(tracking)
    if handedness == Handedness.LEFT:
        raw_2d, raw_confidence = _canonicalize_left(raw_2d, raw_confidence)
    # The generated motion covers its own phase window, not the render window.
    generated_start, generated_end = start, end
    if generated_source_window is not None:
        generated_start = min(max(int(generated_source_window[0]), start), end)
        generated_end = min(max(int(generated_source_window[-1]), generated_start), end)
    generated_lead = generated_start - start
    generated_span = generated_end - generated_start + 1

    original_timeline = _place_in_window(
        original, generated_lead, generated_span, window_frame_count
    )
    corrected_timeline = _place_in_window(
        corrected, generated_lead, generated_span, window_frame_count
    )
    model_confidence = np.clip(
        _place_in_window(confidence, generated_lead, generated_span, window_frame_count),
        0.0,
        1.0,
    )
    model_display_confidence = _expand_display_confidence(model_confidence)
    expanded_detected_confidence = _expand_display_confidence(raw_confidence)
    # Do not spread an elbow observation into adjacent frames: that would make
    # a genuinely absent elbow appear at an unmeasured coordinate.
    expanded_detected_confidence[:, [7, 8]] = raw_confidence[:, [7, 8]]
    detected_display_confidence = _complete_interpolated_display_confidence(
        expanded_detected_confidence
    )
    original_root_values = _place_in_window(
        (
            np.zeros((target_frames, 2), dtype=np.float32)
            if original_root is None
            else np.asarray(original_root, dtype=np.float32)
        ),
        generated_lead,
        generated_span,
        window_frame_count,
    )
    corrected_root_values = (
        original_root_values
        if corrected_root is None
        else _place_in_window(
            np.asarray(corrected_root, dtype=np.float32),
            generated_lead,
            generated_span,
            window_frame_count,
        )
    )
    if original_root_values.shape != (
        window_frame_count,
        2,
    ) or corrected_root_values.shape != (window_frame_count, 2):
        raise ValueError("resampled root trajectories must have shape (T, 2)")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    raw_path = output_path.with_name(output_path.stem + ".raw.mp4")
    first_frame = tracking["frames"][0]
    height, width = first_frame.shape[:2]
    writer = cv2.VideoWriter(
        str(raw_path), cv2.VideoWriter.fourcc(*"mp4v"), fps, (width, height)
    )
    if not writer.isOpened():
        raise RuntimeError(f"could not open output writer: {raw_path}")
    if projected_corrected_pixels is not None:
        # Already phase-aligned, projected, smoothed, and transported by the
        # scorer. Never fit or smooth again in this presentation-only path.
        fixed_corrected_pixels = projected_corrected_pixels
        fixed_display_masks = detected_display_confidence[start : end + 1].copy()
        if generated_source_window is not None:
            # Show actual source evidence outside the generated window, not a
            # fabricated pose held at either end of it.
            fixed_display_masks[:generated_lead] = 0
            fixed_display_masks[generated_lead + generated_span :] = 0
    else:
        # Fit each generated frame onto the detected skeleton, then apply one
        # clip-level ankle/knee/hip placement.
        mapped = []
        masks = []
        for window_index in range(window_frame_count):
            frame_index = start + window_index
            detected_pixels = raw_2d[frame_index]
            mask = np.minimum(
                model_confidence[window_index], raw_confidence[frame_index]
            )
            transform = _fit_affine(
                original_timeline[window_index], detected_pixels, mask
            )
            corrected_world = corrected_timeline[window_index] + (
                corrected_root_values[window_index] - original_root_values[window_index]
            )
            mapped.append(_map_to_pixels(corrected_world, transform))
            masks.append(
                np.minimum(
                    model_display_confidence[window_index],
                    detected_display_confidence[frame_index],
                )
            )
        fixed_display_masks = np.asarray(masks, dtype=np.float32)
        if generated_source_window is not None:
            # Outside the model's own window there is no generated pose, only
            # the end one held in place; do not draw it.
            fixed_display_masks[:generated_lead] = 0
            fixed_display_masks[generated_lead + generated_span :] = 0
        fixed_corrected_pixels = _apply_fixed_hierarchical_placement(
            np.asarray(mapped, dtype=np.float32),
            raw_2d[start : end + 1],
            fixed_display_masks,
            preparation_end=max(1, int(np.ceil(0.375 * window_frame_count))),
        )
    try:
        feedback_by_frame: dict[int, list[dict[str, Any]]] = {}
        for issue in feedback or []:
            source_issue_frame = start + coaching_video_frame(
                issue, target_frames, end - start + 1
            )
            feedback_by_frame.setdefault(source_issue_frame, []).append(issue)
        # The API returns the same reviewable clip that was scored, not the
        # unanalysed lead-in/tail of the upload.  The localhost EIMD-v3 oracle
        # slices tracking to this inclusive range before calling the renderer;
        # iterating the range directly is equivalent while retaining the full
        # source indices needed for detected-pose and feedback lookup.
        for frame_index in range(start, end + 1):
            frame = tracking["frames"][frame_index].copy()
            detected_pixels = raw_2d[frame_index]
            _draw_skeleton(
                frame,
                detected_pixels,
                detected_display_confidence[frame_index],
                (255, 210, 30),
                4,
            )
            if start <= frame_index <= end:
                window_index = frame_index - start
                corrected_pixels = fixed_corrected_pixels[window_index]
                display_mask = fixed_display_masks[window_index]
                _draw_skeleton(frame, corrected_pixels, display_mask, (55, 225, 75), 3)
            _draw_header(frame, filename, score)
            issues = feedback_by_frame.get(frame_index, [])
            if issues:
                _draw_feedback(frame, detected_pixels, issues)
            repetitions = 1 + (round(fps * pause_seconds) if issues else 0)
            for _ in range(repetitions):
                writer.write(frame)
    finally:
        writer.release()

    try:
        _transcode_preserving_frame_rate(
            raw_path,
            output_path,
            frame_rate or f"{Fraction(fps).limit_denominator(100_000)}",
        )
    finally:
        raw_path.unlink(missing_ok=True)
