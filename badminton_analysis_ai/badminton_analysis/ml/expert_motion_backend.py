from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from badminton_analysis.ml.error_isolated_motion import (
    ErrorIsolatedMotionBundle,
    correct_student_motion_error_isolated,
    load_error_isolated_bundle,
)
from badminton_analysis.ml.expert_motion_generator import ExpertCorrection
from badminton_analysis.ml.expert_motion_preprocessing import (
    prepare_expert_motion_sample,
)
from badminton_analysis.ml.expert_phase_baseline import (
    ExpertPhaseModel,
    load_expert_phase_model,
    score_expert_correction,
)
from badminton_analysis.models.types import (
    GradingDetail,
    GradingOutcome,
    Handedness,
    Skill,
    TrackingData,
)


# The bundle validates that the checkpoint declares this method on load, but
# does not keep it as a field, so the diagnostic names it directly.
EIMD_METHOD = "expert_only_error_isolated_motion_diffusion"


@dataclass(frozen=True)
class GeneratedMotionInference:
    grade: GradingOutcome
    score: dict[str, Any]
    correction: ExpertCorrection
    window: tuple[int, int, int]
    source_frame_indices: NDArray[np.int64]
    diagnostics: dict[str, Any]


class ExpertMotionGeneratorBackend:
    """Frozen expert-only EIMD inference for serve and smash.

    The checkpoint certifies expert-only training: the loader rejects any
    bundle whose ``student_data_used`` flag is set, so a model trained on
    learner recordings cannot reach grading.
    """

    target_frames = 64

    def __init__(
        self,
        model_root: str | Path,
        skill: Skill,
        *,
        device: str = "auto",
        candidates: int = 16,
        seed: int = 19,
    ) -> None:
        if skill not in {Skill.SERVE, Skill.SMASH}:
            raise ValueError("generated expert motion supports serve and smash")
        if candidates < 1:
            raise ValueError("generator candidate count must be positive")
        root = Path(model_root) / str(skill)
        self.skill = skill
        self.model_path = root / "error_isolated_motion.pt"
        self.score_model_path = root / "expert_score_model.npz"
        self.bundle: ErrorIsolatedMotionBundle = load_error_isolated_bundle(
            self.model_path, device=device
        )
        self.score_model: ExpertPhaseModel = load_expert_phase_model(
            self.score_model_path
        )
        self.spec = self.score_model.spec
        expected_ids = tuple(rule.id for rule in self.spec.rules)
        model_ids = tuple(str(value) for value in self.score_model.criterion_ids)
        if model_ids != expected_ids:
            raise ValueError(
                f"{skill} scoring criteria do not match checkpoint: "
                f"runtime={expected_ids}, checkpoint={model_ids}"
            )
        if self.bundle.skill != str(skill) or self.score_model.skill != str(skill):
            raise ValueError(f"error-isolated checkpoint skill mismatch for {skill}")
        self.candidates = candidates
        self.seed = seed

    def infer(
        self,
        tracking: TrackingData,
        handedness: Handedness,
        filename: str,
    ) -> GeneratedMotionInference:
        sample, window, source_indices = prepare_expert_motion_sample(
            tracking,
            handedness,
            self.skill,
            filename,
            target_frames=self.target_frames,
        )
        correction = correct_student_motion_error_isolated(
            self.bundle,
            sample,
            candidates=self.candidates,
            seed=self.seed,
        )
        # Scored against the expert phase model with the checkpoint's own
        # canonical phases, the same way the reference grader does.
        score = score_expert_correction(
            self.score_model,
            correction,
            canonical_phase_indices=self.bundle.canonical_phase_indices,
        )
        criteria = score["criteria"]
        grade = GradingOutcome(
            total_grade=float(score["total_score"]),
            grading_details=[
                GradingDetail(
                    description=str(item["name_zh_tw"]),
                    grade=float(item["score"]),
                )
                for item in criteria
            ],
        )
        references = score.get("references", [])
        primary_reference = references[0] if references else {}
        diagnostics: dict[str, Any] = {
            "correction_distance": float(
                np.mean([item["combined_distance"] for item in criteria])
            ),
            "position_distance": float(
                np.mean([item["euclidean_distance"] for item in criteria])
            ),
            "angle_distance": float(
                np.mean([item["target_angle_distance"] for item in criteria])
            ),
            "expert_reference_id": Path(
                str(primary_reference.get("file", "generated-expert-prior"))
            ).stem,
            "expert_reference_distance": float(
                primary_reference.get("stance_distance", 0.0)
            ),
            "model_path": str(self.model_path),
            "scorer": str(score["score_method"]),
            "generator_method": EIMD_METHOD,
            "student_data_used_for_training": False,
            "skeleton_execution_provider": str(
                next(self.bundle.network.parameters()).device
            ),
            "skeleton_tensorrt_active": 0.0,
            "expert_wrist_velocity_limit": float(
                self.bundle.expert_wrist_velocity_limit
            ),
            "wrist_velocity_limited": float(
                correction.maximum_wrist_velocity_after is not None
            ),
        }
        optional_metrics = {
            "maximum_wrist_velocity_before": correction.maximum_wrist_velocity_before,
            "maximum_wrist_velocity_after": correction.maximum_wrist_velocity_after,
            "maximum_body_velocity_before": correction.maximum_body_velocity_before,
            "maximum_body_velocity_after": correction.maximum_body_velocity_after,
        }
        diagnostics.update(
            {
                key: float(value)
                for key, value in optional_metrics.items()
                if value is not None
            }
        )
        return GeneratedMotionInference(
            grade=grade,
            score=score,
            correction=correction,
            window=window,
            source_frame_indices=source_indices,
            diagnostics=diagnostics,
        )
