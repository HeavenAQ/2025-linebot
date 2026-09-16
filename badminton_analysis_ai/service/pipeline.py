from __future__ import annotations

import logging
import os
import tempfile
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from badminton_analysis.ml.handedness import estimate_handedness, interpolated_keypoint
from badminton_analysis.ml.expert_reference_bank import (
    ExpertReference,
    ExpertReferenceBank,
    segmental_alignment,
)
from badminton_analysis.ml.expert_motion_backend import (
    ExpertMotionGeneratorBackend,
)
from badminton_analysis.ml.skeleton_normalization import (
    tracking_body_arrays,
)
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

from service.renderer import render_correction_video, source_fps, source_frame_rate
from service.coaching import CoachingGenerator
from service.pose_batcher import PoseBatcher

LOGGER = logging.getLogger("badminton-analysis")


class SkillMismatchError(ValueError):
    """The requested stroke contradicts the detected temporal motion support."""


@dataclass(frozen=True)
class PhaseResult:
    id: str
    label: str
    normalized_frame: int
    normalized_position: float
    timestamp_seconds: float
    start_seconds: float | None = None
    end_seconds: float | None = None


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
    # The real demonstration closest to this learner's corrected motion, or
    # None when no expert clip exists for the skill.
    expert_reference: ExpertReference | None = None
    # (normalized_position, expert_seconds) pairs mapping this learner's motion
    # onto that expert's clock between the checkpoints. Empty when it could not
    # be built, which costs playback its dense alignment and nothing else.
    expert_alignment: tuple[tuple[float, float], ...] = ()


def _correction_grade_context(
    grade: GradingOutcome,
    diagnostics: dict[str, Any],
    spec: SkillCorrectionSpec,
    criterion_values: list[tuple[str, float, float]],
) -> dict[str, Any]:
    component_names = ("position_distance", "angle_distance")
    # Every supported scorer is fitted from expert data only; learner recordings
    # never calibrate a tolerance, so the prompt must not say otherwise.
    score_status = "expert_only_generated_distribution"
    score_method = (
        "學生骨架與依其身形、站位座標及動作階段生成的專家全身骨架，逐項比較歐氏距離與目標關節角；"
        "分數容許範圍只由保留身分的專家動作分布校準"
    )
    if spec.skill == Skill.SERVE:
        score_method += "；發球重心轉移另比較預備至完成的下肢支撐與軀幹前傾變化"
    if diagnostics.get("scorer") == "smash_local_checkpoint_graph_geometry_v20260913":
        score_status = "frozen_checkpoint_calibration"
        score_method = (
            "殺球按指定檢核區間評分，權重為5/20/5/20/30/20。預備與平衡採凍結骨架圖模型，"
            "再以手腕高度及連續五幀的持拍手過低限制得分；轉體比較下肢、軀幹及肩軸角度變化；"
            "手肘與手腕沿用凍結語意及軌跡評分；收拍保留最佳終點，並以起終肩寬變化限制得分。"
            "只根據所附實際評分區間與量測解釋分數，不得以未觀察到的幀推斷動作。"
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


def _checkpoint_phase_range(spec, rule, phase_indices, phase_times, sequence_length):
    """Map a semantic window to the source clock, including its display anchor.

    Anchor indices locate checkpoint buttons; they are not interval bounds.
    Explicit scored evidence takes precedence over this phase-window fallback.
    """
    window = next(item for item in spec.phase_windows if item.name == rule.phase)
    last = max(1, sequence_length - 1)
    anchor = phase_indices[rule.allowed_anchor_indices[-1]]
    start = min(window.start_fraction * last, anchor)
    end = max(window.end_fraction * last, anchor)
    return tuple(
        float(value) for value in np.interp([start, end], phase_indices, phase_times)
    )


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
            timestamp_seconds=float(phase_seconds[rule.allowed_anchor_indices[-1]]),
            start_seconds=_checkpoint_phase_range(
                spec, rule, phase_indices, phase_seconds, sequence_length
            )[0],
            end_seconds=_checkpoint_phase_range(
                spec, rule, phase_indices, phase_seconds, sequence_length
            )[1],
        )
        for rule, frame in zip(spec.rules, frames, strict=True)
    )


def _source_qualitative_phase_results(
    spec: SkillCorrectionSpec,
    *,
    phase_indices: tuple[int, ...],
    source_phase_frames: list[int],
    normalized_sequence_length: int,
    source_sequence_length: int,
    analysis_window_start_frame: int,
    analysis_window_end_frame: int,
    fps: float,
    checkpoint_source_frames: dict[str, int] | None = None,
    checkpoint_evidence: dict | None = None,
) -> tuple[PhaseResult, ...]:
    if source_sequence_length <= 0 or normalized_sequence_length <= 0 or fps <= 0:
        raise ValueError("source checkpoint timeline requires positive dimensions")
    if len(source_phase_frames) != len(phase_indices):
        raise ValueError("source timeline needs one source frame per phase")
    normalized_frames = _rule_anchor_frames(
        spec, phase_indices, normalized_sequence_length - 1
    )
    if not 0 <= analysis_window_start_frame <= analysis_window_end_frame:
        raise ValueError("analysis window must be an ordered inclusive range")
    if analysis_window_end_frame >= source_sequence_length:
        raise ValueError("analysis window exceeds the source sequence")
    last_local_frame = analysis_window_end_frame - analysis_window_start_frame

    def evidence_range(rule):
        item = (checkpoint_evidence or {}).get(rule.id)
        if item:
            start, end = item.get("replay_source_interval", item["source_interval"])
        else:
            start, end = _checkpoint_phase_range(
                spec,
                rule,
                phase_indices,
                source_phase_frames,
                normalized_sequence_length,
            )
        start = max(analysis_window_start_frame, min(start, analysis_window_end_frame))
        end = max(start, min(end, analysis_window_end_frame))
        return (
            (start - analysis_window_start_frame) / fps,
            (end - analysis_window_start_frame) / fps,
        )

    return tuple(
        PhaseResult(
            id=rule.id,
            label=rule.name_zh_tw,
            normalized_frame=normalized_frame,
            start_seconds=evidence_range(rule)[0],
            end_seconds=evidence_range(rule)[1],
            normalized_position=float(
                min(
                    max(source_frame - analysis_window_start_frame, 0),
                    last_local_frame,
                )
            )
            / max(1, last_local_frame),
            timestamp_seconds=float(
                min(
                    max(source_frame - analysis_window_start_frame, 0),
                    last_local_frame,
                )
            )
            / fps,
        )
        for rule, normalized_frame, source_frame in zip(
            spec.rules,
            normalized_frames,
            (
                (
                    checkpoint_source_frames[rule.id]
                    if checkpoint_source_frames is not None
                    else source_phase_frames[rule.allowed_anchor_indices[-1]]
                )
                for rule in spec.rules
            ),
            strict=True,
        )
    )


def _resolve_handedness(tracking: TrackingData, requested: str) -> Handedness:
    if requested in ("left", "right"):
        return Handedness.convert_to_enum(requested)
    body_2d = tracking.get("body_landmarks_2d")
    if not body_2d:
        raise ValueError("cannot estimate handedness without 2D landmarks")
    skeleton, confidence = tracking_body_arrays(tracking)
    estimate = estimate_handedness(skeleton, confidence)
    if estimate.handedness is None:
        raise ValueError("handedness is ambiguous; specify left or right")
    return estimate.handedness


def _populate_dominant_motion(tracking: TrackingData, handedness: Handedness) -> None:
    body_2d = tracking.get("body_landmarks_2d")
    if not body_2d:
        raise ValueError("2D landmarks are required for motion analysis")
    skeleton, confidence = tracking_body_arrays(tracking)
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
    tracking["hand_positions"] = list(
        interpolated_keypoint(skeleton, confidence, wrist)
    )
    tracking["elbow_positions"] = list(
        interpolated_keypoint(skeleton, confidence, elbow)
    )


def _dump_pose_arrays(
    prefix: str,
    filename: str,
    skill: Skill,
    handedness: Handedness,
    tracking: TrackingData,
    skeleton: Any,
    confidence: Any,
    root: Any,
    window: Any,
    phase_indices: Any,
    source_frame_indices: Any,
    pose_backend: str,
) -> None:
    """Write this run's pose arrays to GCS for an offline joint-by-joint diff.

    A grade is reproducible from the pose, but the pose never leaves the
    container, so a deployed run and an offline one can only be compared
    through the single number they both end at -- which cannot say which joint
    moved. The keys here match the offline extraction cache so the two load
    side by side without translation.

    Off unless ANALYSIS_POSE_DUMP_PREFIX is set, and never allowed to fail a
    request: this exists to explain a bad grade, not to cause one.
    """
    try:
        from service.storage import ObjectStorage

        with tempfile.TemporaryDirectory() as raw_directory:
            local = Path(raw_directory) / "pose.npz"
            payload: dict[str, Any] = {
                "skeleton": np.asarray(skeleton, dtype=np.float32),
                "confidence": np.asarray(confidence, dtype=np.float32),
                "root_trajectory": np.asarray(root, dtype=np.float32),
                "phase_indices": np.asarray(phase_indices, dtype=np.int64),
                "analysis_window": np.asarray(window, dtype=np.int64),
                "source_frame_indices": np.asarray(
                    source_frame_indices, dtype=np.int64
                ),
                # Names ("serve", "right"), as the offline extraction cache and
                # Skill/Handedness.convert_to_enum expect -- not IntEnum values.
                "skill": str(skill),
                "handedness": str(handedness),
                "video_name": filename,
                "pose_backend": pose_backend,
            }
            keypoints = tracking.get("body_keypoints_2d")
            if keypoints is not None:
                payload["source_skeleton_2d"] = np.asarray(
                    keypoints, dtype=np.float32
                )
            scores = tracking.get("body_confidence_2d")
            if scores is not None:
                payload["source_confidence"] = np.asarray(scores, dtype=np.float32)
            np.savez_compressed(local, **payload)

            bucket = os.getenv("GCS_BUCKET_NAME", "")
            storage = ObjectStorage(os.getenv("GCP_PROJECT_ID", ""), bucket)
            stem = Path(filename).stem
            stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
            object_path = f"{prefix.strip('/')}/{stamp}-{stem}.npz"
            storage.upload_file(
                local, object_path, content_type="application/octet-stream"
            )
            LOGGER.info("pose dump written to gs://%s/%s", bucket, object_path)
    except Exception:  # noqa: BLE001 - diagnostics must never fail a request
        LOGGER.exception("pose dump failed; the analysis itself is unaffected")


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
        self.pose_batcher = PoseBatcher(self.pose_detector, self.lock)
        self.backends: dict[Skill, ExpertMotionGeneratorBackend] = {}
        self.coaching = CoachingGenerator(openai_model)
        self.pause_seconds = pause_seconds
        # The prior generates an idealised movement rather than copying an
        # expert, so the clip shown beside it is chosen by similarity instead.
        bank_path = expert_reference_bank or Path("models/expert_reference_bank.npz")
        if not bank_path.exists():
            raise FileNotFoundError(
                "expert reference bank is required for temporal skill validation: "
                f"{bank_path}"
            )
        self.expert_bank = ExpertReferenceBank(bank_path)
        for skill in (Skill.SERVE, Skill.SMASH):
            self.backends[skill] = ExpertMotionGeneratorBackend(
                expert_motion_model_root,
                skill,
                device=device,
                # This is part of the frozen EIMD-v3 inference contract.  Keep
                # it explicit here so a helper's default cannot silently make
                # serving diverge from calibration or the review cohort.
                candidates=8,
                seed=19,
                current_smash=(skill == Skill.SMASH),
                # Serve retains its preparation-window view alignment. The
                # current smash path instead projects once at its first source
                # frame and supplies scored pixels directly to the renderer.
                align_ankle_spine_view=True,
            )

    @property
    def loaded_skills(self) -> tuple[Skill, ...]:
        return tuple(self.backends)

    def warmup(self):
        with self.lock:
            if not getattr(self, "_warmed", False):
                self.pose_detector.get_poses_batch(
                    [np.zeros((480, 640, 3), dtype=np.uint8)]
                )
                self._warmed = True

    def analyze(
        self,
        *,
        video_path: Path,
        output_path: Path,
        skeleton_overlay_path: Path,
        filename: str,
        skill: Skill,
        requested_handedness: str,
        skip_coaching: bool = False,
    ) -> AnalysisResult:
        if skill not in self.backends:
            raise ValueError("only serve and smash are currently supported")
        pipeline_started = time.perf_counter()
        if skill == Skill.SMASH:
            from service.smash_source import normalize_smash_source

            video_path = normalize_smash_source(
                video_path,
                output_path.with_name(output_path.stem + ".source-30fps.mp4"),
            )
        pose_started = time.perf_counter()
        processor = VideoProcessor(
            str(video_path),
            self.pose_batcher.request_detector(),
        )
        # The whole upload is available, so poses are extracted in batches:
        # the cached TensorRT engine on CUDA, RF-DETR predict() on MPS/CPU.
        # The pose batcher may merge frames from concurrent requests.
        tracking = processor.process_frames_batched(None)
        pose_finished = time.perf_counter()
        handedness = _resolve_handedness(tracking, requested_handedness)
        _populate_dominant_motion(tracking, handedness)
        backend = self.backends[skill]
        # The request label is untrusted.  Validate it against expert-only
        # temporal support after pose/handedness extraction and before the
        # requested generator can steer itself from an out-of-distribution
        # phase sequence. The independently frozen label-support bank
        # retains its EIMD-v3 windows; grading uses its own matched contract.
        prepared = backend.prepare(tracking, handedness, filename)
        support_prepared = (
            backend.prepare_skill_support(tracking, handedness, filename)
            if hasattr(backend, "prepare_skill_support")
            else prepared
        )
        alternative_skill = Skill.SMASH if skill == Skill.SERVE else Skill.SERVE
        try:
            alternative = self.backends[alternative_skill]
            support_prepare = getattr(
                alternative, "prepare_skill_support", alternative.prepare
            )
            alternative_prepared = support_prepare(tracking, handedness, filename)
        except ValueError as exc:
            # This is a conservative rejection-only guard. If the other
            # stroke cannot form a valid five-phase hypothesis, it has not
            # won temporal support and the requested analysis continues.
            alternative_prepared = None
            LOGGER.info(
                "alternative skill hypothesis unavailable requested=%s "
                "alternative=%s error=%s",
                skill,
                alternative_skill,
                exc,
            )
        skill_support = (
            self.expert_bank.temporal_skill_support(
                support_prepared[0].pose,
                alternative_prepared[0].pose,
                requested_skill=str(skill),
            )
            if alternative_prepared is not None
            else None
        )
        if skill_support is not None and skill_support.mismatch:
            raise SkillMismatchError(
                f"requested {skill_support.requested_skill} conflicts with "
                f"{skill_support.alternative_skill} temporal motion support "
                f"(advantage={skill_support.alternative_advantage:.6f}, "
                f"margin={skill_support.rejection_margin:.6f})"
            )
        with self.lock:
            generated = backend.infer(
                tracking,
                handedness,
                filename,
                prepared=prepared,
                fps=source_fps(video_path),
            )
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
        diagnostics.update(
            {
                "skill_consistency_gate_active": 1.0,
                "alternative_skill_phase_hypothesis_available": float(
                    alternative_prepared is not None
                ),
            }
        )
        if skill_support is not None:
            diagnostics.update(
                {
                    "requested_skill_support_distance": (
                        skill_support.requested_distance
                    ),
                    "alternative_skill_support_distance": (
                        skill_support.alternative_distance
                    ),
                    "alternative_skill_support_advantage": (
                        skill_support.alternative_advantage
                    ),
                    "skill_consistency_rejection_margin": (
                        skill_support.rejection_margin
                    ),
                }
            )
        criterion_values = [
            (
                str(item["name_zh_tw"]),
                float(item["combined_distance"]),
                float(item["score"]),
            )
            for item in generated.score["criteria"]
        ]
        # A grade that disagrees with the offline run is invisible in the
        # response, which carries only the total and the Chinese criterion
        # names. Log the scorer identity, every criterion, and the gate
        # state so a divergence can be located from the deploy log alone.
        LOGGER.info(
            "grade skill=%s scorer=%s pose_backend=%s total=%.4f criteria=%s diagnostics=%s",
            skill,
            diagnostics.get("scorer", "unknown"),
            self.pose_detector.execution_provider,
            float(grade["total_grade"]),
            [
                (name, round(distance, 6), round(score, 4))
                for name, distance, score in criterion_values
            ],
            {
                key: round(float(value), 6)
                for key, value in sorted(diagnostics.items())
                if isinstance(value, (int, float))
            },
        )
        dump_prefix = os.getenv("ANALYSIS_POSE_DUMP_PREFIX", "").strip()
        if dump_prefix:
            _dump_pose_arrays(
                dump_prefix,
                filename,
                skill,
                handedness,
                tracking,
                skeleton,
                confidence,
                original_root,
                window,
                phases,
                generated.source_frame_indices,
                self.pose_detector.execution_provider,
            )
        scoring_finished = time.perf_counter()
        frame_rate = source_frame_rate(video_path)
        fps = source_fps(video_path)
        spec: SkillCorrectionSpec = backend.spec
        correction_grade = _correction_grade_context(
            grade, diagnostics, spec, criterion_values
        )
        if "checkpoint_evidence" in generated.score:
            correction_grade["checkpoint_evidence"] = generated.score[
                "checkpoint_evidence"
            ]
            correction_grade["checkpoint_measurements"] = generated.score[
                "checkpoint_measurements"
            ]
            correction_grade["generated_source_window"] = generated.score[
                "generation_window"
            ]
            correction_grade["overlay_padding_policy"] = (
                "No generated skeleton before coverage; after coverage the final local pose is held for display only."
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
            filename=filename,
            score=float(grade["total_grade"]),
            output_path=skeleton_overlay_path,
            projected_corrected_pixels=generated.corrected_pixels,
            generated_source_window=generated.score.get("generation_window"),
            fps=fps,
            frame_rate=frame_rate,
        )
        overlay_finished = time.perf_counter()
        # Coaching is the only stage that leaves this machine: it uploads
        # sampled frames of the learner to a third-party model. Skipping it
        # is therefore not an optimisation but a guarantee -- no image of
        # this learner is sent anywhere. Everything that decides the grade
        # has already happened above, and all of it is local.
        if skip_coaching:
            coaching_payload = {
                "analysis": {"problems": [], "overall_feedback": ""},
                "latency_llm_inference_seconds": 0.0,
            }
        else:
            coaching_payload = self.coaching.generate(
                video_path=skeleton_overlay_path,
                working_dir=output_path.parent,
                filename=filename,
                handedness=str(handedness),
                phase_indices=tuple(int(value) for value in phases),
                normalized_sequence_length=len(skeleton),
                output_frame_count=int(window[2] - window[0] + 1),
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
            filename=filename,
            score=float(grade["total_grade"]),
            output_path=output_path,
            projected_corrected_pixels=generated.corrected_pixels,
            generated_source_window=generated.score.get("generation_window"),
            fps=fps,
            frame_rate=frame_rate,
            feedback=problems,
            pause_seconds=self.pause_seconds,
        )
        final_render_finished = time.perf_counter()

        diagnostics.update(
            {
                "source_fps": fps,
                "source_frame_count": len(tracking["frames"]),
                "normalized_sequence_length": len(skeleton),
                "analysis_window_start_frame": int(window[0]),
                "analysis_window_peak_frame": int(window[1]),
                "analysis_window_end_frame": int(window[2]),
                "normalized_phase_indices": [int(value) for value in phases],
                "source_phase_frames": source_phase_frames,
                "latency_pose_seconds": pose_finished - pose_started,
                "latency_preprocessing_seconds": preprocessing_finished - pose_finished,
                "latency_scoring_seconds": scoring_finished - preprocessing_finished,
                "latency_preview_render_seconds": overlay_finished - scoring_finished,
                "latency_coaching_total_seconds": coaching_finished - overlay_finished,
                "latency_llm_inference_seconds": float(
                    coaching_payload["latency_llm_inference_seconds"]
                ),
                "latency_coaching_preparation_seconds": (
                    coaching_finished
                    - overlay_finished
                    - float(coaching_payload["latency_llm_inference_seconds"])
                ),
                "latency_final_render_seconds": final_render_finished
                - coaching_finished,
                "latency_pipeline_seconds": final_render_finished - pipeline_started,
                # Whether the compiled TensorRT engine served this run, as
                # opposed to falling back to PyTorch.
                "pose_tensorrt_active": float(self.pose_detector.tensorrt_active),
            }
        )

        expert_id = str(diagnostics.get("expert_reference_id", ""))
        if not expert_id:
            raise RuntimeError("checkpoint did not report a selected expert")
        duration = len(tracking["frames"]) / fps
        phase_results = _source_qualitative_phase_results(
            spec,
            phase_indices=tuple(int(value) for value in phases),
            source_phase_frames=source_phase_frames,
            normalized_sequence_length=len(skeleton),
            source_sequence_length=len(tracking["frames"]),
            analysis_window_start_frame=int(window[0]),
            analysis_window_end_frame=int(window[2]),
            checkpoint_source_frames=generated.score.get("checkpoint_source_frames"),
            checkpoint_evidence=generated.score.get("checkpoint_evidence"),
            fps=fps,
        )
        if phase_results[-1].timestamp_seconds > duration + 1.0 / fps:
            raise RuntimeError("phase timeline exceeds rendered video duration")
        # Match on the canonical-space correction, which is the space the bank
        # was built in, so no renormalization is needed.
        expert_alignment: tuple[tuple[float, float], ...] = ()
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
            overall_feedback=str(coaching_payload["analysis"]["overall_feedback"]),
            coaching_problems=tuple(problems),
            pause_seconds=self.pause_seconds,
            expert_reference=expert_reference,
            expert_alignment=expert_alignment,
        )
