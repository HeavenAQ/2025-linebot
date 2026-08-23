from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from badminton_analysis.ml.handedness import estimate_handedness, interpolated_keypoint
from badminton_analysis.ml.expert_reference_bank import (
    ExpertReference,
    ExpertReferenceBank,
)
from badminton_analysis.ml.expert_motion_backend import (
    ExpertMotionGeneratorBackend,
)
from badminton_analysis.ml.skeleton_normalization import landmark_dicts_to_array
from badminton_analysis.ml.skill_specs import SkillCorrectionSpec
from badminton_analysis.models.types import (
    COCOKeypoints,
    GradingOutcome,
    Handedness,
    Skill,
    TrackingData,
)
from badminton_analysis.services.pose_detector import PoseDetector
from badminton_analysis.services.video_processor import VideoProcessor

from service.renderer import render_correction_video, source_fps
from service.coaching import CoachingGenerator


LOGGER = logging.getLogger("badminton-analysis")


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
    grade: GradingOutcome
    diagnostics: dict[str, Any]
    expert_id: str
    expert_distance: float
    phases: tuple[PhaseResult, ...]
    overall_feedback: str
    coaching_problems: tuple[dict[str, Any], ...]
    pause_seconds: float
    output_path: Path
    skeleton_overlay_path: Path
    # The real demonstration closest to this learner's corrected motion, or
    # None when no expert clip exists for the skill.
    expert_reference: ExpertReference | None = None


def _correction_grade_context(
    grade: GradingOutcome,
    diagnostics: dict[str, Any],
    spec: SkillCorrectionSpec,
    criterion_values: list[tuple[str, float, float]],
) -> dict[str, Any]:
    component_names = (
        "position_distance",
        "angle_distance",
        "velocity_distance",
        "bone_length_distance",
        "support_transition_distance",
        "torso_lean_transition_distance",
        "lunge_direction_distance",
        "transition_distance",
    )
    generated_expert = (
        diagnostics.get("scorer") == "continuous_generated_expert_distribution_v1"
    )
    score_status = (
        "expert_only_generated_distribution"
        if generated_expert
        else "diagnostic_group_calibrated"
    )
    score_method = (
        "學生骨架與依其身形、站位座標及動作階段生成的專家全身骨架，逐項比較歐氏距離與目標關節角；"
        "分數容許範圍只由保留身分的專家動作分布校準"
        if generated_expert
        else (
            "學生原始骨架與專家化修正骨架之加權差距，經專家與學生群組分布校準；"
            "發球重心轉移另比較完整下肢支撐軌跡與軀幹前傾變化；"
            "挑球另比較持拍腳由預備至擊球的跨步方向"
        )
    )
    return {
        "score_method_zh_tw": score_method,
        "score_status": score_status,
        "total_score": float(grade["total_grade"]),
        "correction_distance": float(diagnostics["correction_distance"]),
        "distance_components": {
            key: float(diagnostics[key])
            for key in component_names
            if key in diagnostics
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


def _rule_anchor_frames(
    spec: SkillCorrectionSpec,
    phase_indices: tuple[int, ...],
    last_frame: int,
) -> list[int]:
    """Normalized frame each criterion is anchored to, in rule order."""
    if len(phase_indices) != 5 or any(
        first >= second for first, second in zip(phase_indices, phase_indices[1:])
    ):
        raise ValueError("checkpoint timeline requires five ordered phase indices")
    return [
        min(
            last_frame,
            max(0, int(phase_indices[rule.allowed_anchor_indices[-1]])),
        )
        for rule in spec.rules
    ]


def expert_phase_results(
    spec: SkillCorrectionSpec,
    *,
    phase_indices: tuple[int, ...],
    phase_seconds: tuple[float, ...],
    sequence_length: int,
) -> tuple[PhaseResult, ...]:
    """The expert's checkpoints, timestamped in the expert's own video.

    Built from the same rule anchors as the student timeline, so marker ``i``
    on either side is the same moment of the stroke and playback can map one
    onto the other segment by segment.
    """
    if sequence_length <= 0:
        raise ValueError("expert timeline requires a positive sequence length")
    if len(phase_seconds) != len(phase_indices):
        raise ValueError("expert timeline needs one timestamp per phase index")
    last_frame = sequence_length - 1
    frames = _rule_anchor_frames(spec, phase_indices, last_frame)
    return tuple(
        PhaseResult(
            id=rule.id,
            label=rule.name_zh_tw,
            normalized_frame=frame,
            normalized_position=float(frame) / max(1, last_frame),
            timestamp_seconds=float(
                phase_seconds[rule.allowed_anchor_indices[-1]]
            ),
        )
        for rule, frame in zip(spec.rules, frames, strict=True)
    )


def _qualitative_phase_results(
    spec: SkillCorrectionSpec,
    *,
    phase_indices: tuple[int, ...] = (0, 16, 32, 48, 63),
    sequence_length: int,
    fps: float,
) -> tuple[PhaseResult, ...]:
    if sequence_length <= 0 or fps <= 0:
        raise ValueError("checkpoint timeline requires a positive length and fps")
    last_frame = sequence_length - 1
    frames = _rule_anchor_frames(spec, phase_indices, last_frame)
    return tuple(
        PhaseResult(
            id=rule.id,
            label=rule.name_zh_tw,
            normalized_frame=frame,
            normalized_position=float(frame) / max(1, last_frame),
            timestamp_seconds=float(frame) / fps,
        )
        for rule, frame in zip(spec.rules, frames, strict=True)
    )


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
        expert_motion_model_root: Path,
        *,
        device: str = "auto",
        openai_model: str = "gpt-5.6-terra",
        pause_seconds: float = 2.0,
        expert_reference_bank: Path | None = None,
    ) -> None:
        self.pose_detector = PoseDetector()
        self.lock = threading.Lock()
        self.backends: dict[Skill, ExpertMotionGeneratorBackend] = {}
        self.coaching = CoachingGenerator(openai_model)
        self.pause_seconds = pause_seconds
        # The prior generates an idealised movement rather than copying an
        # expert, so the clip shown beside it is chosen by similarity instead.
        bank_path = expert_reference_bank or Path("models/expert_reference_bank.npz")
        self.expert_bank = ExpertReferenceBank(bank_path) if bank_path.exists() else None
        if self.expert_bank is None:
            LOGGER.warning("no expert reference bank at %s; comparison will have no video", bank_path)
        for skill in (Skill.SERVE, Skill.SMASH):
            self.backends[skill] = ExpertMotionGeneratorBackend(
                expert_motion_model_root,
                skill,
                device=device,
            )

    @property
    def loaded_skills(self) -> tuple[Skill, ...]:
        return tuple(self.backends)

    def analyze(
        self,
        *,
        video_path: Path,
        output_path: Path,
        skeleton_overlay_path: Path,
        filename: str,
        skill: Skill,
        requested_handedness: str,
    ) -> AnalysisResult:
        if skill not in self.backends:
            raise ValueError("only serve and smash are currently supported")
        pipeline_started = time.perf_counter()
        with self.lock:
            pose_started = time.perf_counter()
            processor = VideoProcessor(
                str(video_path), filename, str(output_path.parent), self.pose_detector
            )
            # Batched, so the pose pass runs on the cached TensorRT engine
            # rather than frame-by-frame in PyTorch. A request arrives with the
            # whole video already uploaded, so there is nothing to stream: the
            # constraint that kept the batched path to offline extraction --
            # needing the frames up front -- does not apply here. Both paths
            # pick the same person the same way and record through the same
            # getters; this one is roughly an order of magnitude faster per
            # frame, which on a GPU billed by the second is the whole point.
            tracking = processor.process_frames_batched(None)
            pose_finished = time.perf_counter()
            handedness = _resolve_handedness(tracking, requested_handedness)
            _populate_dominant_motion(tracking, handedness)
            backend = self.backends[skill]
            generated = backend.infer(tracking, handedness, filename)
            correction = generated.correction
            skeleton = correction.student.pose
            confidence = correction.student.confidence
            original_root = correction.student.root
            corrected = correction.corrected_pose
            corrected_root = correction.corrected_root
            window = generated.window
            phases = correction.student.phase_indices
            source_phase_frames = [
                int(generated.source_frame_indices[int(value)]) for value in phases
            ]
            preprocessing_finished = time.perf_counter()
            grade = generated.grade
            diagnostics = generated.diagnostics
            criterion_values = [
                (
                    str(item["name_zh_tw"]),
                    float(item["combined_distance"]),
                    float(item["score"]),
                )
                for item in generated.score["criteria"]
            ]
            scoring_finished = time.perf_counter()
            fps = source_fps(video_path)
            spec: SkillCorrectionSpec = backend.spec
            correction_grade = _correction_grade_context(
                grade, diagnostics, spec, criterion_values
            )
            render_correction_video(
                tracking=tracking,
                original=skeleton,
                corrected=corrected,
                original_root=original_root,
                corrected_root=corrected_root,
                confidence=confidence,
                window=window,
                handedness=handedness,
                skill=skill,
                filename=filename,
                score=float(grade["total_grade"]),
                output_path=skeleton_overlay_path,
                fps=fps,
                generated_full_body=True,
            )
            overlay_finished = time.perf_counter()
            coaching_payload = self.coaching.generate(
                video_path=skeleton_overlay_path,
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
                original_root=original_root,
                corrected_root=corrected_root,
                confidence=confidence,
                window=window,
                handedness=handedness,
                skill=skill,
                filename=filename,
                score=float(grade["total_grade"]),
                output_path=output_path,
                fps=fps,
                feedback=problems,
                pause_seconds=self.pause_seconds,
                generated_full_body=True,
            )
            final_render_finished = time.perf_counter()

        diagnostics.update(
            {
                "source_fps": fps,
                "analysis_window_start_frame": int(window[0]),
                "analysis_window_peak_frame": int(window[1]),
                "analysis_window_end_frame": int(window[2]),
                "normalized_phase_indices": [int(value) for value in phases],
                "source_phase_frames": source_phase_frames,
                "latency_pose_seconds": pose_finished - pose_started,
                "latency_preprocessing_seconds": preprocessing_finished - pose_finished,
                "latency_scoring_seconds": scoring_finished - preprocessing_finished,
                "latency_preview_render_seconds": overlay_finished - scoring_finished,
                "latency_skeleton_overlay_render_seconds": (
                    overlay_finished - scoring_finished
                ),
                "latency_coaching_total_seconds": coaching_finished - overlay_finished,
                "latency_llm_inference_seconds": float(
                    coaching_payload["latency_llm_inference_seconds"]
                ),
                "latency_coaching_preparation_seconds": (
                    coaching_finished
                    - overlay_finished
                    - float(coaching_payload["latency_llm_inference_seconds"])
                ),
                "latency_final_render_seconds": final_render_finished - coaching_finished,
                "latency_pipeline_seconds": final_render_finished - pipeline_started,
                "pose_execution_provider": self.pose_detector.execution_provider,
                "pose_active_execution_providers": (
                    self.pose_detector.active_execution_providers
                ),
                # Detection runs through Torch-TensorRT now, not onnxruntime,
                # so this reflects whether the compiled engine served the run
                # rather than which ORT provider was selected.
                "pose_tensorrt_active": float(self.pose_detector.tensorrt_active),
            }
        )

        expert_id = str(diagnostics.get("expert_reference_id", ""))
        if not expert_id:
            raise RuntimeError("checkpoint did not report a selected expert")
        duration = len(skeleton) / fps
        phase_results = _qualitative_phase_results(
            spec,
            phase_indices=tuple(int(value) for value in phases),
            sequence_length=len(skeleton),
            fps=fps,
        )
        if phase_results[-1].timestamp_seconds > duration + 1.0 / fps:
            raise RuntimeError("phase timeline exceeds rendered video duration")
        # Match on the canonical-space correction, which is the space the bank
        # was built in, so no renormalization is needed.
        expert_reference = None
        if self.expert_bank is not None:
            expert_reference = self.expert_bank.select(
                correction.aligned_corrected_pose,
                skill=str(skill),
                handedness=str(handedness),
            )
            if expert_reference is not None:
                diagnostics["expert_reference_similarity"] = expert_reference.similarity
                diagnostics["expert_reference_pose_distance"] = expert_reference.distance

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
            skeleton_overlay_path=skeleton_overlay_path,
            expert_reference=expert_reference,
        )
