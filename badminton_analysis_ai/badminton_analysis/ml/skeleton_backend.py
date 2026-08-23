from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch

from badminton_analysis.ml.infer_skeleton_corrector import (
    load_corrector,
    phase_grading_details,
    predict_correction_with_reference,
)
from badminton_analysis.ml.skeleton_normalization import (
    interpolate_pose_sequence,
    landmark_dicts_to_array,
    normalize_skeleton_sequence,
    resample_detected_phase_indices,
    resample_phase_indices,
    resample_sequence,
)
from badminton_analysis.ml.skeleton_scoring import (
    ScoreCalibration,
    correction_distance,
    correction_quality_metrics,
)
from badminton_analysis.ml.skill_specs import get_skill_spec, validate_checkpoint_spec
from badminton_analysis.models.types import (
    GradingDetail,
    GradingOutcome,
    Handedness,
    Skill,
    TrackingData,
)
from badminton_analysis.services.video_analyzer import VideoAnalyzer


def tracking_to_normalized_sequence(
    tracking: TrackingData,
    handedness: Handedness,
    *,
    skill: Skill = Skill.CLEAR,
    target_frames: int = 64,
) -> tuple[np.ndarray, np.ndarray, tuple[int, int, int], np.ndarray]:
    landmarks_3d = tracking["original_landmarks"]
    landmarks_2d = tracking.get("body_landmarks_2d")
    tracked_frames = len(landmarks_3d)
    if tracked_frames < 5:
        raise ValueError("fewer than five tracked frames")
    if landmarks_2d is None or len(landmarks_2d) != tracked_frames:
        raise ValueError("aligned 2D body landmarks are unavailable")
    detected_phases = VideoAnalyzer.find_analysis_phases(
        skill=skill,
        hand_positions=tracking["hand_positions"],
        elbow_positions=tracking["elbow_positions"],
    )
    start, peak, end = (
        detected_phases[0],
        detected_phases[2],
        detected_phases[4],
    )
    start = max(0, min(tracked_frames - 1, int(start)))
    peak = max(start, min(tracked_frames - 1, int(peak)))
    end = max(peak, min(tracked_frames - 1, int(end)))
    if end - start < 4:
        raise ValueError(f"analysis window is too short: {(start, peak, end)}")

    skeleton_3d, confidence_3d = landmark_dicts_to_array(
        landmarks_3d[start : end + 1], 3
    )
    skeleton_2d, confidence_2d = landmark_dicts_to_array(
        landmarks_2d[start : end + 1], 2
    )
    _, confidence_2d = interpolate_pose_sequence(skeleton_2d, confidence_2d)
    confidence = np.minimum(confidence_3d, confidence_2d)
    normalized, normalized_confidence = normalize_skeleton_sequence(
        skeleton_3d, confidence, handedness
    )
    sequence = resample_sequence(normalized, target_frames)
    resampled_confidence = np.clip(
        resample_sequence(normalized_confidence, target_frames), 0.0, 1.0
    )
    phases = (
        resample_detected_phase_indices(detected_phases, target_frames)
        if skill == Skill.LIFT
        else resample_phase_indices((start, peak, end), target_frames)
    )
    return sequence, resampled_confidence, (start, peak, end), phases


class SkeletonCorrectionBackend:
    def __init__(
        self,
        model_path: str | Path,
        *,
        calibration_path: str | Path | None = None,
        device: str = "auto",
    ) -> None:
        self.model_path = Path(model_path)
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        self.model, checkpoint = load_corrector(self.model_path, self.device)
        self.checkpoint = checkpoint
        self.spec = get_skill_spec(str(checkpoint.get("skill", "clear")))
        validate_checkpoint_spec(checkpoint, self.spec)
        self.target_frames = int(checkpoint.get("sequence_frames", 64))
        self.phase_aligned = bool(checkpoint.get("phase_aligned", False))
        self.correction_strength = float(checkpoint.get("inference_strength", 1.0))
        self.reference_guidance = float(checkpoint.get("reference_guidance", 0.0))
        reference_files = checkpoint.get(
            "expert_reference_files",
            checkpoint.get("expert_training_files", ()),
        )
        self.expert_training_files = tuple(str(value) for value in reference_files)
        self.expert_reference_handedness = tuple(
            str(value).lower()
            for value in checkpoint.get("expert_reference_handedness", ())
        )
        if self.expert_training_files and (
            len(self.expert_reference_handedness)
            != len(self.expert_training_files)
        ):
            raise ValueError(
                "checkpoint expert handedness metadata does not match its reference bank"
            )
        self.inference_session: Any | None = None
        self.inference_providers: tuple[str, ...] = ()
        execution_provider = os.getenv(
            "SKELETON_EXECUTION_PROVIDER", "pytorch"
        ).lower()
        if execution_provider != "pytorch":
            self.inference_session = self._load_onnx_session(execution_provider)
            self.inference_providers = tuple(
                self.inference_session.get_providers()
            )
        calibration_file = (
            Path(calibration_path)
            if calibration_path is not None
            else self.model_path.with_suffix(".calibration.json")
        )
        if not calibration_file.exists():
            raise ValueError(
                f"calibration file not found: {calibration_file}; run feasibility inference first"
            )
        calibration_values = json.loads(calibration_file.read_text(encoding="utf-8"))
        self.calibration = ScoreCalibration(**calibration_values)

    def _load_onnx_session(self, execution_provider: str) -> Any:
        import onnxruntime as ort

        onnx_path = self.model_path.with_suffix(".onnx")
        if not onnx_path.exists():
            raise ValueError(f"correction ONNX model not found: {onnx_path}")
        device_id = int(os.getenv("ONNXRUNTIME_DEVICE_ID", "0"))
        available = set(ort.get_available_providers())
        if execution_provider == "tensorrt":
            if "TensorrtExecutionProvider" not in available:
                raise RuntimeError("TensorRT correction provider is unavailable")
            cache_root = Path(
                os.getenv(
                    "POSE_TENSORRT_CACHE_DIR",
                    "/app/models/tensorrt-cache",
                )
            )
            cache_path = cache_root / "correctors" / self.spec.slug
            cache_path.mkdir(parents=True, exist_ok=True)
            providers: list[Any] = [
                (
                    "TensorrtExecutionProvider",
                    {
                        "device_id": device_id,
                        "trt_engine_cache_enable": True,
                        "trt_engine_cache_path": str(cache_path),
                        "trt_engine_cache_prefix": (
                            f"skeleton_{self.spec.slug}"
                        ),
                        "trt_fp16_enable": True,
                        "trt_engine_hw_compatible": True,
                        "trt_timing_cache_enable": True,
                    },
                ),
                ("CUDAExecutionProvider", {"device_id": device_id}),
                "CPUExecutionProvider",
            ]
        elif execution_provider == "cuda":
            if "CUDAExecutionProvider" not in available:
                raise RuntimeError("CUDA correction provider is unavailable")
            providers = [
                ("CUDAExecutionProvider", {"device_id": device_id}),
                "CPUExecutionProvider",
            ]
        else:
            raise ValueError(
                "SKELETON_EXECUTION_PROVIDER must be pytorch, tensorrt, or cuda"
            )
        session = ort.InferenceSession(str(onnx_path), providers=providers)
        active = tuple(session.get_providers())
        expected = (
            "TensorrtExecutionProvider"
            if execution_provider == "tensorrt"
            else "CUDAExecutionProvider"
        )
        if not active or active[0] != expected:
            raise RuntimeError(
                f"{expected} did not activate for {self.spec.slug}: {active}"
            )
        return session

    def score(
        self,
        tracking: TrackingData,
        handedness: Handedness,
        skill: Skill,
    ) -> tuple[GradingOutcome, tuple[int, int, int], dict[str, Any]]:
        if skill != self.spec.skill:
            raise ValueError(
                f"checkpoint supports {self.spec.slug}, but requested skill is {skill}"
            )
        skeleton, confidence, window, phases = tracking_to_normalized_sequence(
            tracking,
            handedness,
            skill=skill,
            target_frames=self.target_frames,
        )
        grade, _, diagnostics = self.score_sequence(
            skeleton, confidence, phases, handedness
        )
        return grade, window, diagnostics

    def score_sequence(
        self,
        skeleton: np.ndarray,
        confidence: np.ndarray,
        phases: np.ndarray,
        handedness: Handedness,
    ) -> tuple[GradingOutcome, np.ndarray, dict[str, Any]]:
        requested_handedness = str(handedness).lower()
        allowed_reference_indices = np.flatnonzero(
            np.asarray(self.expert_reference_handedness)
            == requested_handedness
        )
        if not len(allowed_reference_indices):
            raise ValueError(
                f"no {requested_handedness}-handed expert reference is available "
                f"for {self.spec.slug}"
            )
        corrected, reference_index, reference_distance = (
            predict_correction_with_reference(
                self.model,
                skeleton,
                confidence,
                self.device,
                phases if self.phase_aligned else None,
                self.correction_strength,
                self.reference_guidance,
                joint_weights=self.spec.joint_weights_array,
                inference_session=self.inference_session,
                allowed_reference_indices=allowed_reference_indices,
                transition_weight=self.spec.transition_weight,
                transition_joints=self.spec.transition_joints,
                transition_lean_joints=self.spec.transition_lean_joints,
                transition_direction_joint=self.spec.transition_direction_joint,
                use_dtw_timing=self.spec.skill == Skill.LIFT,
            )
        )
        total_distance, components = correction_distance(
            skeleton,
            corrected,
            confidence,
            joint_weights=self.spec.joint_weights_array,
            transition_weight=self.spec.transition_weight,
            transition_joints=self.spec.transition_joints,
            transition_lean_joints=self.spec.transition_lean_joints,
            transition_direction_joint=self.spec.transition_direction_joint,
        )
        quality = correction_quality_metrics(skeleton, corrected, confidence)
        total_grade = float(self.calibration.score(total_distance))
        details = [
            GradingDetail(description=description, grade=grade)
            for description, _, grade in phase_grading_details(
                skeleton,
                corrected,
                confidence,
                self.calibration,
                total_grade,
                self.spec,
            )
        ]
        diagnostics: dict[str, Any] = {
            "correction_distance": total_distance,
            **components,
            **quality,
            "model_path": str(self.model_path),
            "scorer": "skeleton-correction",
            "skeleton_execution_provider": (
                self.inference_providers[0]
                if self.inference_providers
                else "PyTorch"
            ),
            "skeleton_tensorrt_active": float(
                bool(self.inference_providers)
                and self.inference_providers[0] == "TensorrtExecutionProvider"
            ),
            "expert_handedness_match": 1.0,
        }
        if reference_index is not None:
            diagnostics["expert_reference_index"] = reference_index
            diagnostics["expert_reference_distance"] = reference_distance
            if reference_index < len(self.expert_training_files):
                expert_filename = self.expert_training_files[reference_index]
                diagnostics["expert_reference_filename"] = expert_filename
                diagnostics["expert_reference_id"] = Path(expert_filename).stem
        return (
            GradingOutcome(total_grade=total_grade, grading_details=details),
            corrected,
            diagnostics,
        )
