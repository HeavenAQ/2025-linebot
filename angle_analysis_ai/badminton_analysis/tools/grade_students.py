from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Sequence

import pandas as pd

from badminton_analysis.ml.handedness import estimate_handedness, interpolated_keypoint
from badminton_analysis.ml.skeleton_backend import SkeletonCorrectionBackend
from badminton_analysis.ml.skeleton_normalization import landmark_dicts_to_array
from badminton_analysis.ml.skill_specs import get_skill_spec, supported_skill_choices
from badminton_analysis.models.types import (
    COCOKeypoints,
    GradingDetail,
    Handedness,
    Skill,
    TrackingData,
)
from badminton_analysis.services.pose_detector import PoseDetector
from badminton_analysis.services.video_processor import VideoProcessor

VIDEO_EXTENSIONS = (".mp4", ".mov")
DEFAULT_MODEL_PATH = str(get_skill_spec(Skill.CLEAR).model_path)


def _flatten_details(details: list[GradingDetail]) -> dict[str, Any]:
    columns: dict[str, Any] = {}
    for index, detail in enumerate(details, start=1):
        columns[f"detail_{index}_desc"] = detail["description"]
        columns[f"detail_{index}_grade"] = detail["grade"]
    return columns


def _resolve_handedness(
    tracking: TrackingData, requested: str
) -> Handedness:
    if requested != "auto":
        return Handedness.convert_to_enum(requested)

    landmarks_2d = tracking.get("body_landmarks_2d")
    if not landmarks_2d:
        raise ValueError("cannot estimate handedness without aligned 2D landmarks")
    skeleton, confidence = landmark_dicts_to_array(landmarks_2d, 2)
    estimate = estimate_handedness(skeleton, confidence)
    if estimate.handedness is None:
        raise ValueError(
            "handedness is ambiguous; rerun with --handedness right or left"
        )
    return estimate.handedness


def _populate_dominant_motion(
    tracking: TrackingData, handedness: Handedness
) -> None:
    frame_count = len(tracking["original_landmarks"])
    if (
        len(tracking["hand_positions"]) == frame_count
        and len(tracking["elbow_positions"]) == frame_count
    ):
        return

    landmarks_2d = tracking.get("body_landmarks_2d")
    if not landmarks_2d or len(landmarks_2d) != frame_count:
        raise ValueError("aligned 2D landmarks are required for motion analysis")
    skeleton, confidence = landmark_dicts_to_array(landmarks_2d, 2)
    wrist = (
        COCOKeypoints.RIGHT_WRIST
        if handedness == Handedness.RIGHT
        else COCOKeypoints.LEFT_WRIST
    )
    elbow = (
        COCOKeypoints.RIGHT_ELBOW
        if handedness == Handedness.RIGHT
        else COCOKeypoints.LEFT_ELBOW
    )
    hand_positions = interpolated_keypoint(skeleton, confidence, wrist)
    elbow_positions = interpolated_keypoint(skeleton, confidence, elbow)
    if len(hand_positions) != frame_count or len(elbow_positions) != frame_count:
        raise ValueError("dominant wrist or elbow tracking is incomplete")
    tracking["hand_positions"] = list(hand_positions)
    tracking["elbow_positions"] = list(elbow_positions)


def grade_videos_in_dir(
    input_dir: str,
    output_dir: str,
    *,
    skill: Skill = Skill.CLEAR,
    model_path: str | None = None,
    calibration_path: str | None = None,
    handedness: str = "auto",
) -> tuple[list[dict[str, Any]], int]:
    source_dir = Path(input_dir)
    destination_dir = Path(output_dir)
    destination_dir.mkdir(parents=True, exist_ok=True)
    spec = get_skill_spec(skill)
    resolved_model_path = model_path or str(spec.model_path)
    videos = sorted(
        path
        for path in source_dir.iterdir()
        if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS
    )
    if not videos:
        raise ValueError(f"No videos found in input directory: {input_dir}")

    pose_detector = PoseDetector()
    backend = SkeletonCorrectionBackend(
        resolved_model_path,
        calibration_path=calibration_path,
    )
    rows: list[dict[str, Any]] = []
    failures = 0

    for video_path in videos:
        print(f"Processing: {video_path}")
        resolved_handedness: Handedness | None = None
        try:
            processor = VideoProcessor(
                str(video_path),
                video_path.name,
                str(destination_dir),
                pose_detector=pose_detector,
            )
            tracking = processor.process_frames(None)
            resolved_handedness = _resolve_handedness(tracking, handedness)
            _populate_dominant_motion(tracking, resolved_handedness)
            grade, window, diagnostics = backend.score(
                tracking, resolved_handedness, skill
            )
            row: dict[str, Any] = {
                "filename": video_path.name,
                "skill": spec.slug,
                "handedness": str(resolved_handedness),
                "status": "success",
                "error": "",
                "total_grade": grade["total_grade"],
                "start_frame": window[0],
                "peak_frame": window[1],
                "end_frame": window[2],
            }
            row.update(_flatten_details(grade["grading_details"]))
            row.update(diagnostics)
        except Exception as exc:
            print(f"  ERROR: {exc}")
            failures += 1
            row = {
                "filename": video_path.name,
                "skill": spec.slug,
                "handedness": (
                    str(resolved_handedness) if resolved_handedness is not None else ""
                ),
                "status": "error",
                "error": str(exc),
                "total_grade": 0,
                "start_frame": -1,
                "peak_frame": -1,
                "end_frame": -1,
            }
        rows.append(row)

    return rows, failures


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Score badminton videos with a skill-specific skeleton corrector",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--skill", choices=supported_skill_choices(), default="clear"
    )
    parser.add_argument("--model-path")
    parser.add_argument("--calibration-path")
    parser.add_argument(
        "--handedness",
        choices=("auto", "right", "left"),
        default="auto",
        help="Racket hand; auto compares normalized wrist acceleration",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not Path(args.input_dir).is_dir():
        print(f"Error: input directory not found: {args.input_dir}")
        return 1
    try:
        rows, failures = grade_videos_in_dir(
            args.input_dir,
            args.output_dir,
            skill=Skill.convert_to_enum(args.skill),
            model_path=args.model_path,
            calibration_path=args.calibration_path,
            handedness=args.handedness,
        )
    except (OSError, ValueError) as exc:
        print(f"Error: {exc}")
        return 1

    output_path = Path(args.output_dir) / "grading_results.csv"
    pd.DataFrame(rows).to_csv(output_path, index=False)
    print(
        f"Completed grading: processed={len(rows)} failed={failures} csv={output_path}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
