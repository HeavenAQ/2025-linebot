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
    segmental_alignment,
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
    output_path: Path
    # The real demonstration closest to this learner's corrected motion, or
    # None when no expert clip exists for the skill.
    expert_reference: ExpertReference | None = None
    # (normalized_position, expert_seconds) pairs mapping this learner's motion
    # onto that expert's clock between the checkpoints. Empty when it could not
    # be built, which costs playback its dense alignment and nothing else.
    expert_alignment: tuple[tuple[float, float], ...] = ()


def _expert_alignment(
    reference: ExpertReference,
    corrected_pose: Any,
) -> tuple[tuple[float, float], ...]:
    """Where each of the learner's frames falls in the matched expert's clip.

    Playback anchors on the checkpoints either way; this fills in the movement
    between them by warping the poses instead of assuming both performers hold
    the same relative tempo inside a phase. A clip the warp cannot handle costs
    playback that refinement, not the analysis, so it degrades to nothing and
    the player interpolates between checkpoints as before.
    """
    try:
        return segmental_alignment(reference, corrected_pose)
    except (ValueError, IndexError) as exc:
        LOGGER.warning(
            "expert alignment unusable expert=%s error=%s", reference.subject_id, exc
        )
        return ()


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
        expert_reference_bank: Path | None = None,
    ) -> None:
        self.pose_detector = PoseDetector()
        self.lock = threading.Lock()
        self.backends: dict[Skill, ExpertMotionGeneratorBackend] = {}
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
            scoring_finished = time.perf_counter()
            fps = source_fps(video_path)
            spec: SkillCorrectionSpec = backend.spec
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
                generated_full_body=True,
            )
            render_finished = time.perf_counter()

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
                "latency_render_seconds": render_finished - scoring_finished,
                "latency_pipeline_seconds": render_finished - pipeline_started,
                "pose_execution_provider": self.pose_detector.execution_provider,
                "pose_active_execution_providers": (
                    self.pose_detector.active_execution_providers
                ),
                # Whether the compiled TensorRT engine served this run, as
                # opposed to falling back to PyTorch.
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
        expert_alignment: tuple[tuple[float, float], ...] = ()
        if self.expert_bank is not None:
            expert_reference = self.expert_bank.select(
                correction.aligned_corrected_pose,
                skill=str(skill),
                handedness=str(handedness),
            )
            if expert_reference is not None:
                diagnostics["expert_reference_similarity"] = expert_reference.similarity
                diagnostics["expert_reference_pose_distance"] = expert_reference.distance
                expert_alignment = _expert_alignment(
                    expert_reference, correction.aligned_corrected_pose
                )

        return AnalysisResult(
            skill=skill,
            handedness=handedness,
            grade=grade,
            diagnostics=diagnostics,
            expert_id=expert_id,
            expert_distance=float(diagnostics["expert_reference_distance"]),
            phases=phase_results,
            output_path=output_path,
            expert_reference=expert_reference,
            expert_alignment=expert_alignment,
        )
