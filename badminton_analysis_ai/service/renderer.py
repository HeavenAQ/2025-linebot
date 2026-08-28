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
from badminton_analysis.ml.expert_phase_baseline import (
    apply_constrained_hierarchical_pose_placement,
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
        if (
            measured.shape != (expected_frames, 17, 2)
            or measured_confidence.shape != (expected_frames, 17)
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
        accepted = (
            (measured_confidence[:, elbow] > 0.05)
            & np.all(np.isfinite(measured[:, elbow]), axis=1)
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


def _transcode_preserving_frame_rate(
    raw_path: Path, output_path: Path, frame_rate: str
) -> None:
    """Encode every rendered frame at the source stream's exact rate."""
    normalized_rate = _normalized_frame_rate(frame_rate)
    if normalized_rate is None:
        raise ValueError(f"invalid frame rate: {frame_rate}")
    subprocess.run(
        [
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
            "-i", str(raw_path), "-an",
            "-vf", f"setpts=N/(({normalized_rate})*TB)",
            "-r", normalized_rate,
            "-fps_mode", "cfr",
            "-c:v", "libx264", "-crf", "21", "-preset", "fast",
            "-pix_fmt", "yuv420p", "-movflags", "+faststart",
            str(output_path),
        ],
        check=True,
    )


def _normalized_to_source_frame(
    normalized_frame: int,
    normalized_frame_count: int,
    start: int,
    end: int,
) -> int:
    if normalized_frame_count <= 1:
        return start
    progress = float(np.clip(normalized_frame, 0, normalized_frame_count - 1)) / (
        normalized_frame_count - 1
    )
    return int(round(start + progress * (end - start)))


def _ground_corrected_pose(
    corrected_pixels: NDArray[np.float32],
    detected_pixels: NDArray[np.float32],
    confidence: NDArray[np.floating],
) -> NDArray[np.float32]:
    """Translate the full generated pose onto the student's support ankle."""
    corrected = np.asarray(corrected_pixels, dtype=np.float32)
    detected = np.asarray(detected_pixels, dtype=np.float32)
    visible_ankles = [
        joint
        for joint in (15, 16)
        if confidence[joint] > 0.05
        and np.all(np.isfinite(corrected[joint]))
        and np.all(np.isfinite(detected[joint]))
    ]
    if not visible_ankles:
        return corrected.copy()
    # In image coordinates the lower visible ankle has the larger y value and
    # is the best 2D proxy for the load-bearing foot. Anchoring one ankle,
    # instead of the ankle midpoint, permits the expert prior to correct stance
    # width and weight transfer without allowing the whole body to float.
    support_ankle = max(visible_ankles, key=lambda joint: detected[joint, 1])
    translation = detected[support_ankle] - corrected[support_ankle]
    return np.asarray(corrected + translation, dtype=np.float32)


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


_FLICKER_CHAINS = ((5, 7, 9), (6, 8, 10), (11, 13, 15), (12, 14, 16))


def _transport_corrected_to_detected_pelvis(
    corrected_pixels: NDArray[np.float32],
    detected_pixels: NDArray[np.float32],
) -> NDArray[np.float32]:
    """Rigidly transport a corrected motion along the student's pelvis path.

    The expert correction supplies local pose, not where the moving student is
    located in the image.  A static clip-level ankle/knee/hip placement can
    therefore leave the green skeleton behind as the student translates.  Use
    the prepared (gap-filled) detected hip midpoint as the per-frame screen
    root and apply one shared translation to all 17 corrected joints.  Local
    joint vectors, angles, and bone lengths are consequently bit-identical.
    """
    corrected = np.asarray(corrected_pixels, dtype=np.float32)
    detected = np.asarray(detected_pixels, dtype=np.float32)
    if corrected.shape != detected.shape or corrected.ndim != 3 or corrected.shape[1:] != (17, 2):
        raise ValueError("pelvis transport poses must share shape (T, 17, 2)")
    corrected_pelvis = 0.5 * (corrected[:, 11] + corrected[:, 12])
    detected_pelvis = 0.5 * (detected[:, 11] + detected[:, 12])
    translation = detected_pelvis - corrected_pelvis
    return np.asarray(corrected + translation[:, None, :], dtype=np.float32)


def _transport_corrected_by_student_displacement(
    corrected_pixels: NDArray[np.float32],
    detected_pixels: NDArray[np.float32],
    confidence: NDArray[np.floating],
) -> NDArray[np.float32]:
    """Add only the student's global displacement to an anchored correction.

    ``corrected_pixels`` already contains the expert local motion, generated
    root trajectory, ankle--spine view projection, and the existing clip-level
    ankle/knee/hip placement.  Preserve that result exactly at frame zero and
    add one rigid translation equal to the student's smoothed pelvis movement
    from frame zero.  Torso centre and ankle midpoint are confidence fallbacks;
    per-frame foot motion is never the primary driver.
    """
    corrected = np.asarray(corrected_pixels, dtype=np.float32)
    detected = np.asarray(detected_pixels, dtype=np.float32)
    observed = np.asarray(confidence, dtype=np.float32)
    if corrected.shape != detected.shape or corrected.ndim != 3 or corrected.shape[1:] != (17, 2):
        raise ValueError("student displacement poses must share shape (T, 17, 2)")
    if observed.shape != corrected.shape[:2]:
        raise ValueError("student displacement confidence must have shape (T, 17)")

    pelvis = 0.5 * (detected[:, 11] + detected[:, 12])
    torso = 0.25 * (detected[:, 5] + detected[:, 6] + detected[:, 11] + detected[:, 12])
    ankles = 0.5 * (detected[:, 15] + detected[:, 16])
    pelvis_ok = np.minimum(observed[:, 11], observed[:, 12]) > 0.05
    torso_ok = np.minimum.reduce(observed[:, (5, 6, 11, 12)], axis=1) > 0.05
    ankles_ok = np.minimum(observed[:, 15], observed[:, 16]) > 0.05
    position = np.full((len(corrected), 2), np.nan, dtype=np.float64)
    position[pelvis_ok] = pelvis[pelvis_ok]
    fallback = ~pelvis_ok & torso_ok
    position[fallback] = torso[fallback]
    fallback = ~pelvis_ok & ~torso_ok & ankles_ok
    position[fallback] = ankles[fallback]
    valid = np.isfinite(position).all(axis=1)
    if not np.any(valid):
        return corrected.copy()
    timeline = np.arange(len(position))
    for axis in range(2):
        position[:, axis] = np.interp(
            timeline, timeline[valid], position[valid, axis]
        )
    # Reject isolated detector jitter without suppressing real player travel.
    smoothed = position.copy()
    padded = np.pad(position, ((2, 2), (0, 0)), mode="edge")
    for frame in range(len(position)):
        smoothed[frame] = np.median(padded[frame : frame + 5], axis=0)
    displacement = smoothed - smoothed[0]
    displacement[0] = 0.0
    return np.asarray(corrected + displacement[:, None], dtype=np.float32)


def _smooth_corrected_bbox_placement(
    corrected_pixels: NDArray[np.float32],
    *,
    alpha_current: float = 0.65,
) -> NDArray[np.float32]:
    """Stabilize correction placement with one rigid per-frame translation.

    Generated smash motions can contain a short, implausible excursion in the
    screen-space root even after their root-relative limb pose is repaired.
    Estimate placement from the body bounding-box centre (joints 5--16), apply
    a zero-phase EMA to that centre, and translate every joint by
    the same amount.  This deliberately leaves all local vectors, angles, and
    bone lengths unchanged.  Student displacement is added *after* this step,
    so the player's observed horizontal transport is never smoothed or lagged.

    The first and last frames remain exact to avoid changing the analysis
    window endpoints.
    """
    corrected = np.asarray(corrected_pixels, dtype=np.float32)
    if corrected.ndim != 3 or corrected.shape[1:] != (17, 2):
        raise ValueError("bbox placement smoothing requires shape (T, 17, 2)")
    if not 0.0 < alpha_current <= 1.0:
        raise ValueError("alpha_current must be in (0, 1]")
    if len(corrected) <= 2 or alpha_current >= 1.0:
        return corrected.copy()

    core = corrected[:, 5:17].astype(np.float64)
    anchor = 0.5 * (np.min(core, axis=1) + np.max(core, axis=1))
    forward = anchor.copy()
    for frame in range(1, len(anchor)):
        forward[frame] = (
            alpha_current * anchor[frame]
            + (1.0 - alpha_current) * forward[frame - 1]
        )
    backward = anchor.copy()
    for frame in range(len(anchor) - 2, -1, -1):
        backward[frame] = (
            alpha_current * anchor[frame]
            + (1.0 - alpha_current) * backward[frame + 1]
        )
    stable_anchor = 0.5 * (forward + backward)
    stable_anchor[0] = anchor[0]
    stable_anchor[-1] = anchor[-1]
    translation = stable_anchor - anchor
    return np.asarray(corrected + translation[:, None], dtype=np.float32)


def _dominant_wrist_acceleration_event(
    pose: NDArray[np.floating],
    *,
    target_index: int,
    search_before: int = 12,
    search_after: int = 20,
) -> int:
    """Locate the generated racket-arm acceleration event near contact."""
    values = np.asarray(pose, dtype=np.float64)
    if values.ndim != 3 or values.shape[1:] != (17, 2) or len(values) < 7:
        raise ValueError("contact event detection requires pose shape (T,17,2)")
    relative = values[:, 10] - values[:, 6]
    padded = np.pad(relative, ((2, 2), (0, 0)), mode="edge")
    kernel = np.asarray((1.0, 2.0, 3.0, 2.0, 1.0), dtype=np.float64) / 9.0
    smooth = np.stack(
        [np.convolve(padded[:, axis], kernel, mode="valid") for axis in range(2)],
        axis=-1,
    )
    acceleration = np.linalg.norm(np.diff(smooth, n=2, axis=0), axis=-1)
    left = max(0, int(target_index) - search_before - 1)
    right = min(len(acceleration), int(target_index) + search_after)
    if right <= left:
        return int(np.clip(target_index, 1, len(values) - 2))
    return left + int(np.argmax(acceleration[left:right])) + 1


def _sample_timeline_at_positions(
    sequence: NDArray[np.floating], positions: NDArray[np.floating]
) -> NDArray[np.float32]:
    values = np.asarray(sequence, dtype=np.float64)
    sample_positions = np.asarray(positions, dtype=np.float64)
    timeline = np.arange(len(values), dtype=np.float64)
    flattened = values.reshape(len(values), -1)
    sampled = np.stack(
        [
            np.interp(sample_positions, timeline, flattened[:, column])
            for column in range(flattened.shape[1])
        ],
        axis=-1,
    )
    return sampled.reshape((len(sample_positions), *values.shape[1:])).astype(
        np.float32
    )


def _align_smash_contact_timeline(
    corrected_pose: NDArray[np.float32],
    corrected_root: NDArray[np.float32],
    *,
    target_index: int,
    tolerance_frames: int = 1,
    maximum_shift_frames: int = 20,
) -> tuple[NDArray[np.float32], NDArray[np.float32], dict[str, int | bool]]:
    """Pin generated racket acceleration to the student's contact frame.

    The warp is monotonic and piecewise linear with fixed clip endpoints. It
    changes only when the generated contact differs by more than one frame and
    refuses implausibly distant events. Spatial correction and player travel
    are untouched.
    """
    pose = np.asarray(corrected_pose, dtype=np.float32)
    root = np.asarray(corrected_root, dtype=np.float32)
    if root.shape != (len(pose), 2):
        raise ValueError("corrected root must have shape (T,2)")
    target = int(np.clip(target_index, 1, len(pose) - 2))
    before = _dominant_wrist_acceleration_event(pose, target_index=target)
    shift = before - target
    apply = tolerance_frames < abs(shift) <= maximum_shift_frames
    if not apply:
        return pose.copy(), root.copy(), {
            "contact_event_before": before,
            "contact_event_target": target,
            "contact_event_after": before,
            "contact_warp_applied": False,
            "contact_shift_frames": shift,
        }
    aligned_pose, aligned_root = pose.copy(), root.copy()
    after = before
    # Acceleration is a second derivative, so resampling can move the detected
    # peak slightly even when the original peak position is mapped exactly.
    # Re-estimate and refine a few times; each pass remains monotonic with fixed
    # endpoints and the total shift remains bounded by the initial gate.
    for _ in range(6):
        if abs(after - target) <= tolerance_frames:
            break
        positions = np.interp(
            np.arange(len(pose), dtype=np.float64),
            np.asarray((0, target, len(pose) - 1), dtype=np.float64),
            np.asarray((0, after, len(pose) - 1), dtype=np.float64),
        )
        aligned_pose = _sample_timeline_at_positions(aligned_pose, positions)
        aligned_root = _sample_timeline_at_positions(aligned_root, positions)
        after = _dominant_wrist_acceleration_event(
            aligned_pose, target_index=target
        )
    return aligned_pose, aligned_root, {
        "contact_event_before": before,
        "contact_event_target": target,
        "contact_event_after": after,
        "contact_warp_applied": True,
        "contact_shift_frames": shift,
    }


def _ema_smooth_corrected_local_pose(
    corrected_pixels: NDArray[np.float32],
    *,
    alpha_current: float = 0.85,
    reset_frames: tuple[int, ...] = (),
) -> NDArray[np.float32]:
    """Apply a small causal EMA to pelvis-relative correction shape only.

    The pelvis/root trajectory is copied exactly.  Reset frames (preparation,
    contact, and completion boundaries) use the original local pose and restart
    EMA history, preventing phase-endpoint attenuation or cross-phase lag.
    """
    corrected = np.asarray(corrected_pixels, dtype=np.float32)
    if corrected.ndim != 3 or corrected.shape[1:] != (17, 2):
        raise ValueError("local-pose EMA requires shape (T, 17, 2)")
    if not 0.0 < alpha_current <= 1.0:
        raise ValueError("alpha_current must be in (0, 1]")
    if len(corrected) <= 1 or alpha_current >= 1.0:
        return corrected.copy()
    pelvis = 0.5 * (corrected[:, 11] + corrected[:, 12])
    local = corrected - pelvis[:, None]
    smoothed = local.copy()
    resets = {int(frame) for frame in reset_frames if 0 <= int(frame) < len(local)}
    resets.add(0)
    for frame in range(1, len(local)):
        if frame in resets:
            smoothed[frame] = local[frame]
        else:
            smoothed[frame] = (
                alpha_current * local[frame]
                + (1.0 - alpha_current) * smoothed[frame - 1]
            )
    # Retain the EMA directions but reconstruct each arm and leg using that
    # frame's already-repaired lengths. Averaging Cartesian joints otherwise
    # shortens bent limbs and can introduce a reach violation.
    for root, middle, endpoint in _FLICKER_CHAINS:
        first_lengths = np.linalg.norm(local[:, middle] - local[:, root], axis=-1)
        second_lengths = np.linalg.norm(local[:, endpoint] - local[:, middle], axis=-1)
        original_middle = smoothed[:, middle].copy()
        first = smoothed[:, middle] - smoothed[:, root]
        first /= np.maximum(np.linalg.norm(first, axis=-1, keepdims=True), 1e-6)
        smoothed[:, middle] = smoothed[:, root] + first_lengths[:, None] * first
        second = smoothed[:, endpoint] - original_middle
        second /= np.maximum(np.linalg.norm(second, axis=-1, keepdims=True), 1e-6)
        smoothed[:, endpoint] = smoothed[:, middle] + second_lengths[:, None] * second
    return np.asarray(pelvis[:, None] + smoothed, dtype=np.float32)


def _apply_smash_contact_leg_constraints(
    corrected_pixels: NDArray[np.float32],
    detected_pixels: NDArray[np.float32],
    confidence: NDArray[np.floating],
) -> NDArray[np.float32]:
    """Pin planted smash feet with a stable two-bone IK solve.

    Detector left/right ankle labels can alternate for several frames (EG1),
    so contact targets are treated as an unordered screen-space pair, assigned
    once to the two corrected legs, and median-filtered.  Only a near-ground,
    low-velocity target is pinned.  The hip and every upper-body joint remain
    untouched; the knee is the unique fixed-length two-bone solution on the
    corrected leg's existing bend branch.
    """
    corrected = np.asarray(corrected_pixels, dtype=np.float64)
    detected = np.asarray(detected_pixels, dtype=np.float64)
    observed = np.asarray(confidence, dtype=np.float64)
    if corrected.shape != detected.shape or corrected.ndim != 3 or corrected.shape[1:] != (17, 2):
        raise ValueError("contact constraint poses must share shape (T, 17, 2)")
    if observed.shape != corrected.shape[:2]:
        raise ValueError("contact constraint confidence must have shape (T, 17)")

    # Stabilize the unordered detected ankle pair against 15/16 label swaps.
    ankle_pair = detected[:, (15, 16)].copy()
    ordered = np.empty_like(ankle_pair)
    left_first = ankle_pair[:, 0, 0] <= ankle_pair[:, 1, 0]
    ordered[:, 0] = np.where(left_first[:, None], ankle_pair[:, 0], ankle_pair[:, 1])
    ordered[:, 1] = np.where(left_first[:, None], ankle_pair[:, 1], ankle_pair[:, 0])
    smoothed = ordered.copy()
    for frame in range(len(ordered)):
        lo, hi = max(0, frame - 2), min(len(ordered), frame + 3)
        smoothed[frame] = np.median(ordered[lo:hi], axis=0)
    target_kernel = np.asarray((0.1, 0.2, 0.4, 0.2, 0.1), dtype=np.float64)
    for track in range(2):
        for axis in range(2):
            padded = np.pad(smoothed[:, track, axis], (2, 2), mode="edge")
            smoothed[:, track, axis] = np.convolve(
                padded, target_kernel, mode="valid"
            )

    preparation = slice(0, max(2, int(np.ceil(0.375 * len(corrected)))))
    direct_cost = float(np.median(np.linalg.norm(corrected[preparation, 15] - smoothed[preparation, 0], axis=-1)) +
                        np.median(np.linalg.norm(corrected[preparation, 16] - smoothed[preparation, 1], axis=-1)))
    crossed_cost = float(np.median(np.linalg.norm(corrected[preparation, 15] - smoothed[preparation, 1], axis=-1)) +
                         np.median(np.linalg.norm(corrected[preparation, 16] - smoothed[preparation, 0], axis=-1)))
    track_for_joint = {15: 0, 16: 1} if direct_cost <= crossed_cost else {15: 1, 16: 0}

    torso = np.linalg.norm(
        0.5 * (detected[:, 5] + detected[:, 6])
        - 0.5 * (detected[:, 11] + detected[:, 12]), axis=-1,
    )
    scale = max(float(np.median(torso[np.isfinite(torso) & (torso > 1e-5)])), 1e-5)
    velocity = np.zeros((len(smoothed), 2), dtype=np.float64)
    if len(smoothed) > 1:
        velocity[1:-1] = 0.5 * np.linalg.norm(smoothed[2:] - smoothed[:-2], axis=-1)
        velocity[0] = np.linalg.norm(smoothed[1] - smoothed[0], axis=-1)
        velocity[-1] = np.linalg.norm(smoothed[-1] - smoothed[-2], axis=-1)
    ground_y = np.max(smoothed[:, :, 1], axis=1)
    contact = ((ground_y[:, None] - smoothed[:, :, 1]) <= 0.18 * scale) & (velocity <= 0.08 * scale)
    # A smash retains at least one load-bearing foot.  If the stricter pair
    # test rejects both, retain only the lower, slower candidate.
    for frame in range(len(contact)):
        if not np.any(contact[frame]):
            candidate = int(np.argmax(smoothed[frame, :, 1] - 0.5 * velocity[frame]))
            if velocity[frame, candidate] <= 0.12 * scale:
                contact[frame, candidate] = True
    # Contact onset/offset is not a one-frame switch in real motion.  A short
    # symmetric ramp prevents a binary IK snap while keeping full pinning over
    # every sustained planted interval.  IK is still solved at each ramped
    # target, so bone lengths remain fixed throughout the transition.
    kernel = target_kernel
    contact_weight = np.empty_like(contact, dtype=np.float64)
    for track in range(2):
        padded = np.pad(contact[:, track].astype(np.float64), (2, 2), mode="edge")
        contact_weight[:, track] = np.convolve(padded, kernel, mode="valid")

    result = corrected.copy()
    for hip, knee, ankle in _LEG_CHAINS:
        track = track_for_joint[ankle]
        thigh = float(np.median(np.linalg.norm(corrected[:, knee] - corrected[:, hip], axis=-1)))
        shin = float(np.median(np.linalg.norm(corrected[:, ankle] - corrected[:, knee], axis=-1)))
        if thigh <= 1e-5 or shin <= 1e-5:
            continue
        for frame in np.flatnonzero(contact_weight[:, track] > 1e-4):
            weight = float(contact_weight[frame, track])
            target = (
                (1.0 - weight) * corrected[frame, ankle]
                + weight * smoothed[frame, track]
            )
            vector = target - result[frame, hip]
            distance = float(np.linalg.norm(vector))
            if distance <= 1e-5:
                continue
            direction = vector / distance
            reach = float(np.clip(distance, abs(thigh - shin) + 1e-4, thigh + shin - 1e-4))
            ankle_target = result[frame, hip] + direction * reach
            along = (thigh * thigh - shin * shin + reach * reach) / (2.0 * reach)
            height = float(np.sqrt(max(thigh * thigh - along * along, 0.0)))
            perpendicular = np.asarray((-direction[1], direction[0]), dtype=np.float64)
            old_knee = corrected[frame, knee] - corrected[frame, hip]
            cross = direction[0] * old_knee[1] - direction[1] * old_knee[0]
            bend = 1.0 if cross >= 0.0 else -1.0
            result[frame, knee] = result[frame, hip] + direction * along + perpendicular * bend * height
            result[frame, ankle] = ankle_target
    return result.astype(np.float32)


def _repair_isolated_corrected_flickers(
    corrected_pixels: NDArray[np.float32],
    detected_pixels: NDArray[np.float32],
    confidence: NDArray[np.floating],
    *,
    maximum_interval: int = 3,
    residual_limit: float = 0.22,
    endpoint_gap_limit: float = 0.12,
) -> tuple[NDArray[np.float32], dict[str, Any]]:
    """Interpolate only bounded 1--3 frame corrected-chain flickers.

    The test is parent-relative and normalized by detected torso length. A
    segment is changed only when the vectors immediately before and after it
    agree, while the bounded interior departs sharply and reverses direction.
    Fast monotonic racket motion therefore remains untouched. If several chains
    fail in the same frame, the affine mapping itself flickered and the complete
    corrected frame is interpolated. Stable-bone projection runs only when at
    least one interval was actually rejected.
    """
    corrected = np.asarray(corrected_pixels, dtype=np.float64)
    detected = np.asarray(detected_pixels, dtype=np.float64)
    observed = np.asarray(confidence, dtype=np.float64)
    if corrected.shape != detected.shape or corrected.ndim != 3 or corrected.shape[1:] != (17, 2):
        raise ValueError("flicker repair poses must share shape (T, 17, 2)")
    if observed.shape != corrected.shape[:2]:
        raise ValueError("flicker repair confidence must have shape (T, 17)")
    torso = np.linalg.norm(
        0.5 * (detected[:, 5] + detected[:, 6])
        - 0.5 * (detected[:, 11] + detected[:, 12]), axis=1,
    )
    finite = torso[np.isfinite(torso) & (torso > 1e-5)]
    scale = float(np.median(finite)) if len(finite) else 1.0
    chain_frames: list[set[int]] = [set() for _ in _FLICKER_CHAINS]
    intervals: list[dict[str, Any]] = []
    for chain_index, (root, middle, endpoint) in enumerate(_FLICKER_CHAINS):
        vectors = np.stack(
            (corrected[:, middle] - corrected[:, root],
             corrected[:, endpoint] - corrected[:, middle]), axis=1,
        )
        candidates: list[tuple[int, int, float, float]] = []
        for start in range(1, len(corrected) - 1):
            for interval_length in range(1, maximum_interval + 1):
                stop = start + interval_length - 1
                if stop >= len(corrected) - 1:
                    break
                before, after = vectors[start - 1], vectors[stop + 1]
                endpoint_gap = float(np.max(np.linalg.norm(after - before, axis=-1)) / scale)
                if endpoint_gap >= endpoint_gap_limit:
                    continue
                alpha = np.linspace(1.0 / (interval_length + 1), interval_length / (interval_length + 1), interval_length)
                expected = before[None] + alpha[:, None, None] * (after - before)[None]
                residual = np.linalg.norm(vectors[start:stop + 1] - expected, axis=-1) / scale
                incoming = vectors[start:stop + 1] - vectors[start - 1:stop]
                outgoing = vectors[start + 1:stop + 2] - vectors[start:stop + 1]
                reversal = np.any(np.sum(incoming * outgoing, axis=-1) < 0.0)
                maximum_residual = float(np.max(residual))
                if maximum_residual > residual_limit and reversal:
                    candidates.append((start, interval_length, maximum_residual, endpoint_gap))
        # Prefer the narrowest explanation. A valid one-frame bounded outlier
        # must never turn into a three-frame rewrite merely because an earlier
        # search window also contains it.
        occupied: set[int] = set()
        for start, interval_length, maximum_residual, endpoint_gap in sorted(
            candidates, key=lambda item: (item[1], -item[2], item[0])
        ):
            stop = start + interval_length - 1
            if any(frame in occupied for frame in range(start, stop + 1)):
                continue
            chain_frames[chain_index].update(range(start, stop + 1))
            occupied.update(range(start, stop + 1))
            intervals.append({"chain": [root, middle, endpoint], "start": start, "end": stop,
                              "kind": "temporal_parent_vector",
                              "maximum_normalized_residual": maximum_residual,
                              "normalized_endpoint_gap": endpoint_gap})

    # Independent spatial feasibility: affine-fit failures can persist for a
    # few frames and therefore lack a clean velocity reversal. Rigid limb
    # lengths must stay close to their robust clip medians. Requiring either
    # two moderate violations or one extreme violation avoids treating normal
    # 2-D foreshortening as an invalid skeleton.
    segments = tuple((root, middle) for root, middle, _ in _FLICKER_CHAINS) + tuple(
        (middle, endpoint) for _, middle, endpoint in _FLICKER_CHAINS
    )
    lengths = np.stack(
        [np.linalg.norm(corrected[:, end] - corrected[:, start], axis=1)
         for start, end in segments], axis=1,
    )
    stable_lengths = np.median(lengths, axis=0)
    relative_length_error = np.abs(
        lengths / np.maximum(stable_lengths[None], 1e-5) - 1.0
    )
    raw_spatial = (
        (np.sum(relative_length_error > 0.25, axis=1) >= 2)
        | (np.max(relative_length_error, axis=1) > 0.40)
    )
    # Only bounded spatial bursts are repairable evidence. A long run may be
    # genuine camera foreshortening and is intentionally left untouched.
    spatial_frames: set[int] = set()
    cursor = 0
    while cursor < len(raw_spatial):
        if not raw_spatial[cursor]:
            cursor += 1
            continue
        stop = cursor
        while stop + 1 < len(raw_spatial) and raw_spatial[stop + 1]:
            stop += 1
        bounded = cursor > 0 and stop < len(raw_spatial) - 1
        if bounded and stop - cursor + 1 <= max(maximum_interval, 5):
            spatial_frames.update(range(cursor, stop + 1))
        cursor = stop + 1
    for frame in sorted(spatial_frames):
        intervals.append({
            "kind": "spatial_bone_length",
            "start": frame,
            "end": frame,
            "maximum_relative_bone_length_error": float(
                np.max(relative_length_error[frame])
            ),
        })

    # Isolated elbow/knee branch flips are spatially plausible in length but
    # topologically wrong. Neighbouring frames must agree on the bend branch.
    branch_frames: list[set[int]] = [set() for _ in _FLICKER_CHAINS]
    for chain_index, (root, middle, endpoint) in enumerate(_FLICKER_CHAINS):
        first = corrected[:, root] - corrected[:, middle]
        second = corrected[:, endpoint] - corrected[:, middle]
        cross = first[:, 0] * second[:, 1] - first[:, 1] * second[:, 0]
        for frame in range(1, len(corrected) - 1):
            if (
                (frame in chain_frames[chain_index] or frame in spatial_frames)
                and
                np.sign(cross[frame - 1]) == np.sign(cross[frame + 1])
                and np.sign(cross[frame]) != np.sign(cross[frame - 1])
                and abs(cross[frame]) > 0.03 * scale * scale
            ):
                branch_frames[chain_index].add(frame)
                intervals.append({"kind": "isolated_branch_flip", "chain": [root, middle, endpoint],
                                  "start": frame, "end": frame})

    flagged = np.zeros(corrected.shape[:2], dtype=bool)
    for chain_index, (root, middle, endpoint) in enumerate(_FLICKER_CHAINS):
        for frame in chain_frames[chain_index] | branch_frames[chain_index]:
            # A bounded limb flicker is repaired as one connected chain.  If
            # the shoulder/hip root is left at its rejected-frame location,
            # the fixed-length FK pass can recreate the very midpoint jump we
            # just removed (notably EG43's right leg).  Interpolating all three
            # joints keeps the local frame of the limb coherent, while the
            # assignment below still leaves every unflagged frame bit-exact.
            flagged[frame, (root, middle, endpoint)] = True
    for frame in range(len(corrected)):
        if frame in spatial_frames or sum(frame in values for values in chain_frames) >= 2:
            flagged[frame] = True
    if not np.any(flagged):
        return corrected.astype(np.float32), {"intervals": [], "flagged_frames": [], "flagged_joint_frames": 0}
    repaired = corrected.copy(); timeline = np.arange(len(repaired))
    for joint in range(17):
        bad = flagged[:, joint]; valid = ~bad & np.isfinite(repaired[:, joint]).all(axis=1)
        if not np.any(bad) or np.count_nonzero(valid) < 2:
            continue
        for axis in range(2):
            fill = np.interp(timeline, timeline[valid], repaired[valid, joint, axis])
            repaired[bad, joint, axis] = fill[bad]
    # Stable two-bone FK on rejected intervals only. Roots remain at the
    # interpolated shoulder/hip position; directions retain the bounded motion;
    # robust clip lengths eliminate the spatial reach failure.
    for chain_index, (root, middle, endpoint) in enumerate(_FLICKER_CHAINS):
        first_length = stable_lengths[chain_index]
        second_length = stable_lengths[len(_FLICKER_CHAINS) + chain_index]
        selected = flagged[:, middle] | flagged[:, endpoint]
        for frame in np.flatnonzero(selected):
            first_direction = repaired[frame, middle] - repaired[frame, root]
            first_direction /= max(float(np.linalg.norm(first_direction)), 1e-5)
            second_direction = repaired[frame, endpoint] - repaired[frame, middle]
            second_direction /= max(float(np.linalg.norm(second_direction)), 1e-5)
            repaired[frame, middle] = repaired[frame, root] + first_length * first_direction
            repaired[frame, endpoint] = repaired[frame, middle] + second_length * second_direction
    # Every valid frame is bit-identical to the pre-repair render timeline.
    output = corrected.copy()
    repaired_frames = np.any(flagged, axis=1)
    output[repaired_frames] = repaired[repaired_frames]
    return output.astype(np.float32), {
        "intervals": intervals,
        "flagged_frames": np.flatnonzero(np.any(flagged, axis=1)).tolist(),
        "flagged_joint_frames": int(np.sum(flagged)),
    }


def _repair_corrected_flickers_until_stable(
    corrected_pixels: NDArray[np.float32],
    detected_pixels: NDArray[np.float32],
    confidence: NDArray[np.floating],
    *,
    maximum_passes: int = 5,
) -> tuple[NDArray[np.float32], dict[str, Any]]:
    """Repeat bounded repair only when the repaired interval exposes an edge outlier."""
    output = np.asarray(corrected_pixels, dtype=np.float32)
    audits = []
    all_frames: set[int] = set()
    joint_frames = 0
    for pass_index in range(maximum_passes):
        output, audit = _repair_isolated_corrected_flickers(
            output, detected_pixels, confidence
        )
        if not audit["flagged_frames"]:
            return output, {
                "passes": audits,
                "intervals": [item for entry in audits for item in entry["intervals"]],
                "flagged_frames": sorted(all_frames),
                "flagged_joint_frames": joint_frames,
                "converged": True,
            }
        audit = {"pass": pass_index + 1, **audit}
        audits.append(audit)
        all_frames.update(audit["flagged_frames"])
        joint_frames += int(audit["flagged_joint_frames"])
    return output, {
        "passes": audits,
        "intervals": [item for entry in audits for item in entry["intervals"]],
        "flagged_frames": sorted(all_frames),
        "flagged_joint_frames": joint_frames,
        "converged": False,
    }


def _apply_constrained_hierarchical_placement(
    corrected_pixels: NDArray[np.float32],
    detected_pixels: NDArray[np.float32],
    confidence: NDArray[np.floating],
    *,
    preparation_end: int,
) -> NDArray[np.float32]:
    """Apply bounded, smooth per-frame ankle/knee/hip chain placement."""
    return apply_constrained_hierarchical_pose_placement(
        detected_pixels,
        corrected_pixels,
        start=0,
        end=preparation_end,
        confidence=confidence,
    )


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
    skill: Skill,
    filename: str,
    score: float,
    output_path: Path,
    fps: float,
    frame_rate: str | None = None,
    original_root: NDArray[np.float32] | None = None,
    corrected_root: NDArray[np.float32] | None = None,
    generated_full_body: bool = False,
    feedback: list[dict[str, Any]] | None = None,
    pause_seconds: float = 0.0,
    fixed_hierarchical_placement: bool = False,
    constrained_hierarchical_placement: bool = False,
) -> None:
    start, _, end = window
    target_frames = len(original)
    source_frame_count = len(tracking["frames"])
    if source_frame_count <= 0:
        raise ValueError("rendering requires at least one source frame")
    if not 0 <= start <= end < source_frame_count:
        raise ValueError("analysis window falls outside the source video")
    window_frame_count = end - start + 1
    raw_2d, raw_confidence = _prepare_detected_pose_for_render(tracking)
    if handedness == Handedness.LEFT:
        raw_2d, raw_confidence = _canonicalize_left(raw_2d, raw_confidence)
    original_timeline = resample_sequence(original, window_frame_count)
    corrected_timeline = resample_sequence(corrected, window_frame_count)
    model_confidence = np.clip(
        resample_sequence(confidence, window_frame_count), 0.0, 1.0
    )
    model_display_confidence = _expand_display_confidence(model_confidence)
    expanded_detected_confidence = _expand_display_confidence(raw_confidence)
    # Do not spread an elbow observation into adjacent frames: that would make
    # a genuinely absent elbow appear at an unmeasured coordinate.
    expanded_detected_confidence[:, [7, 8]] = raw_confidence[:, [7, 8]]
    detected_display_confidence = _complete_interpolated_display_confidence(
        expanded_detected_confidence
    )
    original_root_values = resample_sequence(
        (
            np.zeros((target_frames, 2), dtype=np.float32)
            if original_root is None
            else np.asarray(original_root, dtype=np.float32)
        ),
        window_frame_count,
    )
    corrected_root_values = resample_sequence(
        (
            original_root_values
            if corrected_root is None
            else np.asarray(corrected_root, dtype=np.float32)
        ),
        window_frame_count,
    )
    if (
        original_root_values.shape != (window_frame_count, 2)
        or corrected_root_values.shape != (window_frame_count, 2)
    ):
        raise ValueError("resampled root trajectories must have shape (T, 2)")
    if skill == Skill.SMASH:
        corrected_timeline, corrected_root_values, _ = _align_smash_contact_timeline(
            corrected_timeline,
            corrected_root_values,
            target_index=window[1] - start,
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    raw_path = output_path.with_name(output_path.stem + ".raw.mp4")
    first_frame = tracking["frames"][0]
    height, width = first_frame.shape[:2]
    writer = cv2.VideoWriter(
        str(raw_path), cv2.VideoWriter.fourcc(*"mp4v"), fps, (width, height)
    )
    if not writer.isOpened():
        raise RuntimeError(f"could not open output writer: {raw_path}")
    fixed_corrected_pixels: NDArray[np.float32] | None = None
    fixed_display_masks: NDArray[np.float32] | None = None
    if fixed_hierarchical_placement or constrained_hierarchical_placement:
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
                corrected_root_values[window_index]
                - original_root_values[window_index]
            )
            mapped.append(_map_to_pixels(corrected_world, transform))
            masks.append(
                np.minimum(
                    model_display_confidence[window_index],
                    detected_display_confidence[frame_index],
                )
            )
        fixed_display_masks = np.asarray(masks, dtype=np.float32)
        placement = (
            _apply_constrained_hierarchical_placement
            if constrained_hierarchical_placement
            else _apply_fixed_hierarchical_placement
        )
        fixed_corrected_pixels = placement(
            np.asarray(mapped, dtype=np.float32),
            raw_2d[start : end + 1],
            fixed_display_masks,
            preparation_end=max(1, int(np.ceil(0.375 * window_frame_count))),
        )
        if skill == Skill.SMASH:
            # Repair bounded generated-pose failures before any temporal EMA;
            # otherwise one impossible frame is blended into valid neighbours.
            # Parent-relative detection and fixed-length reconstruction do not
            # touch the later player-displacement transport.
            fixed_corrected_pixels, _ = _repair_corrected_flickers_until_stable(
                fixed_corrected_pixels,
                raw_2d[start : end + 1],
                fixed_display_masks,
            )
            fixed_corrected_pixels = _smooth_corrected_bbox_placement(
                fixed_corrected_pixels,
                alpha_current=0.65,
            )
            fixed_corrected_pixels = _transport_corrected_by_student_displacement(
                fixed_corrected_pixels,
                raw_2d[start : end + 1],
                fixed_display_masks,
            )
    try:
        feedback_by_frame: dict[int, list[dict[str, Any]]] = {}
        for issue in feedback or []:
            source_issue_frame = _normalized_to_source_frame(
                int(issue["frame_index"]), target_frames, start, end
            )
            feedback_by_frame.setdefault(source_issue_frame, []).append(issue)
        for frame_index in range(source_frame_count):
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
                if fixed_corrected_pixels is not None:
                    corrected_pixels = fixed_corrected_pixels[window_index]
                    assert fixed_display_masks is not None
                    display_mask = fixed_display_masks[window_index]
                else:
                    mask = np.minimum(
                        model_confidence[window_index], raw_confidence[frame_index]
                    )
                    transform = _fit_affine(
                        original_timeline[window_index], detected_pixels, mask
                    )
                    corrected_world = corrected_timeline[window_index] + (
                        corrected_root_values[window_index]
                        - original_root_values[window_index]
                    )
                    corrected_pixels = _map_to_pixels(corrected_world, transform)
                    if not generated_full_body:
                        corrected_pixels = _retarget_corrected_pose(
                            corrected_pixels,
                            detected_pixels,
                            skill,
                            window_index / max(window_frame_count - 1, 1),
                        )
                    display_mask = np.minimum(
                        model_display_confidence[window_index],
                        detected_display_confidence[frame_index],
                    )
                    corrected_pixels = _ground_corrected_pose(
                        corrected_pixels, detected_pixels, display_mask
                    )
                _draw_skeleton(
                    frame, corrected_pixels, display_mask, (55, 225, 75), 3
                )
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
