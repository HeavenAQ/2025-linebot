from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from badminton_analysis.ml.handedness import estimate_handedness, interpolated_keypoint
from badminton_analysis.ml.infer_skeleton_corrector import phase_grading_details
from badminton_analysis.ml.skeleton_backend import (
    SkeletonCorrectionBackend,
    tracking_to_normalized_sequence,
)
from badminton_analysis.ml.skeleton_normalization import landmark_dicts_to_array
from badminton_analysis.ml.skill_specs import SkillCorrectionSpec, get_skill_spec
from badminton_analysis.models.types import COCOKeypoints, Handedness, Skill, TrackingData
from badminton_analysis.services.pose_detector import PoseDetector
from badminton_analysis.services.video_processor import VideoProcessor

from service.renderer import render_correction_video, source_fps
from service.coaching import CoachingGenerator


@dataclass(frozen=True)
class PhaseResult:
    id: str
    label: str
    normalized_frame: int
    normalized_position: float
    timestamp_seconds: float


@dataclass(frozen=True)
class AnalysisResult:
    skill: Skill
    handedness: Handedness
    grade: dict[str, Any]
    diagnostics: dict[str, Any]
    expert_id: str
    expert_distance: float
    phases: tuple[PhaseResult, ...]
    overall_feedback: str
    coaching_problems: tuple[dict[str, Any], ...]
    pause_seconds: float
    output_path: Path


def _resolve_handedness(tracking: TrackingData, requested: str) -> Handedness:
    if requested in ("left", "right"):
        return Handedness.convert_to_enum(requested)
    body_2d = tracking.get("body_landmarks_2d")
    if not body_2d:
        raise ValueError("cannot estimate handedness without 2D landmarks")
    skeleton, confidence = landmark_dicts_to_array(body_2d, 2)
    estimate = estimate_handedness(skeleton, confidence)
    if estimate.handedness is None:
        raise ValueError("handedness is ambiguous; specify left or right")
    return estimate.handedness


def _populate_dominant_motion(
    tracking: TrackingData, handedness: Handedness
) -> None:
    body_2d = tracking.get("body_landmarks_2d")
    if not body_2d:
        raise ValueError("2D landmarks are required for motion analysis")
    skeleton, confidence = landmark_dicts_to_array(body_2d, 2)
    wrist = COCOKeypoints.RIGHT_WRIST if handedness == Handedness.RIGHT else COCOKeypoints.LEFT_WRIST
    elbow = COCOKeypoints.RIGHT_ELBOW if handedness == Handedness.RIGHT else COCOKeypoints.LEFT_ELBOW
    tracking["hand_positions"] = list(interpolated_keypoint(skeleton, confidence, wrist))
    tracking["elbow_positions"] = list(interpolated_keypoint(skeleton, confidence, elbow))


class SkeletonAnalysisPipeline:
    def __init__(
        self,
        model_root: Path,
        *,
        device: str = "auto",
        openai_model: str = "gpt-5.6-terra",
        pause_seconds: float = 2.0,
    ) -> None:
        self.pose_detector = PoseDetector()
        self.lock = threading.Lock()
        self.backends: dict[Skill, SkeletonCorrectionBackend] = {}
        self.coaching = CoachingGenerator(openai_model)
        self.pause_seconds = pause_seconds
        for skill in (Skill.SERVE, Skill.LIFT, Skill.CLEAR, Skill.SMASH):
            spec = get_skill_spec(skill)
            self.backends[skill] = SkeletonCorrectionBackend(
                model_root / f"{spec.model_stem}.pt", device=device
            )

    @property
    def loaded_skills(self) -> tuple[Skill, ...]:
        return tuple(self.backends)

    def analyze(
        self,
        *,
        video_path: Path,
        output_path: Path,
        filename: str,
        skill: Skill,
        requested_handedness: str,
    ) -> AnalysisResult:
        pipeline_started = time.perf_counter()
        with self.lock:
            pose_started = time.perf_counter()
            processor = VideoProcessor(
                str(video_path), filename, str(output_path.parent), self.pose_detector
            )
            tracking = processor.process_frames(None)
            pose_finished = time.perf_counter()
            handedness = _resolve_handedness(tracking, requested_handedness)
            _populate_dominant_motion(tracking, handedness)
            backend = self.backends[skill]
            skeleton, confidence, window, phases = tracking_to_normalized_sequence(
                tracking,
                handedness,
                skill=skill,
                target_frames=backend.target_frames,
            )
            preprocessing_finished = time.perf_counter()
            grade, corrected, diagnostics = backend.score_sequence(
                skeleton, confidence, phases
            )
            scoring_finished = time.perf_counter()
            fps = source_fps(video_path)
            spec: SkillCorrectionSpec = get_skill_spec(skill)
            criterion_values = phase_grading_details(
                skeleton,
                corrected,
                confidence,
                backend.calibration,
                float(grade["total_grade"]),
                spec,
            )
            correction_grade = {
                "score_method_zh_tw": (
                    "學生原始骨架與專家化修正骨架之加權差距，經專家與學生群組分布校準"
                ),
                "total_score": float(grade["total_grade"]),
                "correction_distance": float(diagnostics["correction_distance"]),
                "distance_components": {
                    key: float(diagnostics[key])
                    for key in (
                        "position_distance",
                        "angle_distance",
                        "velocity_distance",
                        "bone_length_distance",
                    )
                },
                "criteria": [
                    {
                        "name_zh_tw": rule.name_zh_tw,
                        "rule_reference": rule.id,
                        "score": float(value[2]),
                        "maximum": rule.maximum,
                        "correction_distance": float(value[1]),
                    }
                    for rule, value in zip(spec.rules, criterion_values, strict=True)
                ],
            }
            preview_path = output_path.with_name(output_path.stem + ".preview.mp4")
            render_correction_video(
                tracking=tracking,
                original=skeleton,
                corrected=corrected,
                confidence=confidence,
                window=window,
                handedness=handedness,
                filename=filename,
                score=float(grade["total_grade"]),
                output_path=preview_path,
                fps=fps,
            )
            preview_finished = time.perf_counter()
            coaching_payload = self.coaching.generate(
                video_path=preview_path,
                working_dir=output_path.parent,
                filename=filename,
                handedness=str(handedness),
                phase_indices=tuple(int(value) for value in phases),
                spec=spec,
                correction_grade=correction_grade,
            )
            coaching_finished = time.perf_counter()
            problems = coaching_payload["analysis"]["problems"]
            render_correction_video(
                tracking=tracking,
                original=skeleton,
                corrected=corrected,
                confidence=confidence,
                window=window,
                handedness=handedness,
                filename=filename,
                score=float(grade["total_grade"]),
                output_path=output_path,
                fps=fps,
                feedback=problems,
                pause_seconds=self.pause_seconds,
            )
            final_render_finished = time.perf_counter()
            preview_path.unlink(missing_ok=True)

        diagnostics.update(
            {
                "latency_pose_seconds": pose_finished - pose_started,
                "latency_preprocessing_seconds": preprocessing_finished - pose_finished,
                "latency_scoring_seconds": scoring_finished - preprocessing_finished,
                "latency_preview_render_seconds": preview_finished - scoring_finished,
                "latency_openai_seconds": coaching_finished - preview_finished,
                "latency_final_render_seconds": final_render_finished - coaching_finished,
                "latency_pipeline_seconds": final_render_finished - pipeline_started,
            }
        )

        expert_id = str(diagnostics.get("expert_reference_id", ""))
        if not expert_id:
            raise RuntimeError("checkpoint did not report a selected expert")
        duration = len(skeleton) / fps
        phase_results = tuple(
            PhaseResult(
                id=("start", "transition", "contact", "follow_through", "end")[index],
                label=spec.checkpoint_roles_zh_tw[index],
                normalized_frame=int(frame),
                normalized_position=float(frame) / max(1, len(skeleton) - 1),
                timestamp_seconds=float(frame) / fps,
            )
            for index, frame in enumerate(phases)
        )
        if phase_results[-1].timestamp_seconds > duration + 1.0 / fps:
            raise RuntimeError("phase timeline exceeds rendered video duration")
        return AnalysisResult(
            skill=skill,
            handedness=handedness,
            grade=grade,
            diagnostics=diagnostics,
            expert_id=expert_id,
            expert_distance=float(diagnostics["expert_reference_distance"]),
            phases=phase_results,
            overall_feedback=str(coaching_payload["analysis"]["overall_feedback"]),
            coaching_problems=tuple(problems),
            pause_seconds=self.pause_seconds,
            output_path=output_path,
        )
