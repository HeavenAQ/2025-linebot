"""Expert-only EIMD inference for every skill: generate, score, report."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from badminton_analysis.ml.eimd.inference import (
    ErrorIsolatedMotionBundle,
    load_error_isolated_bundle,
)
from badminton_analysis.ml.motion.samples import (
    ExpertCorrection,
    ExpertPhaseModel,
    MotionSample,
    load_expert_phase_model,
)
from badminton_analysis.ml.skill import (
    PreparedSample,
    ScoringContext,
    skill_definition,
    supported_skills,
    supported_skills_text,
)
from badminton_analysis.models.types import (
    GradingDetail,
    GradingOutcome,
    Handedness,
    Skill,
    TrackingData,
)

# The bundle validates this method on load but does not keep it as a field.
EIMD_METHOD = "expert_only_error_isolated_motion_diffusion"


@dataclass(frozen=True)
class GeneratedMotionInference:
    grade: GradingOutcome
    score: dict[str, Any]
    correction: ExpertCorrection
    window: tuple[int, int, int]
    source_frame_indices: NDArray[np.int64]
    diagnostics: dict[str, Any]
    corrected_pixels: NDArray[np.float32] | None = None


class ExpertMotionGeneratorBackend:
    """Frozen expert-only EIMD inference; the loader rejects learner-trained bundles."""

    target_frames = 64

    def __init__(
        self,
        model_root: str | Path,
        skill: Skill,
        *,
        device: str = "auto",
        candidates: int = 8,
        seed: int = 19,
        align_ankle_spine_view: bool = False,
        current_scorer: bool = False,
    ) -> None:
        if skill not in supported_skills():
            raise ValueError(f"generated expert motion supports {supported_skills_text()}")
        if candidates < 1:
            raise ValueError("generator candidate count must be positive")
        root = Path(model_root) / str(skill)
        self.skill = skill
        self.definition = skill_definition(skill)
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
        self.align_ankle_spine_view = align_ankle_spine_view
        self.scorer = self.definition.create_scorer(
            root=root,
            model_path=self.model_path,
            bundle=self.bundle,
            score_model=self.score_model,
            candidates=candidates,
            seed=seed,
            align_ankle_spine_view=align_ankle_spine_view,
            target_frames=self.target_frames,
            current_scorer=current_scorer,
        )

    def prepare(
        self,
        tracking: TrackingData,
        handedness: Handedness,
        filename: str,
    ) -> PreparedSample:
        """Build the exact generation sample used by both gate and inference."""
        return self.definition.prepare_sample(
            tracking,
            handedness,
            filename,
            target_frames=self.target_frames,
            phase_contract=self.scorer.generation_contract,
        )

    def prepare_skill_support(
        self, tracking: TrackingData, handedness: Handedness, filename: str
    ) -> PreparedSample:
        """The skill guard was calibrated on EIMD-v3 windows; never score from this."""
        return self.definition.prepare_sample(
            tracking,
            handedness,
            filename,
            target_frames=self.target_frames,
            phase_contract="eimd_v3",
        )

    def infer(
        self,
        tracking: TrackingData,
        handedness: Handedness,
        filename: str,
        *,
        prepared: PreparedSample | None = None,
        fps: float = 30.0,
    ) -> GeneratedMotionInference:
        sample, window, source_indices = (
            prepared
            if prepared is not None
            else self.prepare(tracking, handedness, filename)
        )
        scored = self.scorer.score(
            ScoringContext(
                tracking=tracking,
                handedness=handedness,
                filename=filename,
                sample=sample,
                window=window,
                source_indices=source_indices,
                correction=self.scorer.generate(sample),
                fps=fps,
            )
        )
        score, correction = scored.score, scored.correction
        scoring_sample: MotionSample = scored.scoring_sample
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
            "generation_candidates": float(self.candidates),
            "generation_seed": float(self.seed),
            "phase_source": sample.phase_source,
            "phase_alignment_contract": sample.alignment_contract,
            "grading_phase_source": scoring_sample.phase_source,
            "grading_phase_alignment_contract": scoring_sample.alignment_contract,
            "dual_window_scoring_active": float(scoring_sample is not sample),
            "student_data_used_for_training": False,
            "raw_expert_motion_score": float(score["total_score"]),
            "post_hoc_score_calibration_active": 0.0,
            "current_smash_checkpoint_scorer_active": float(
                scored.checkpoint_scorer_active
            ),
            "ankle_spine_view_alignment_active": float(
                scored.view_rotation is not None or scored.checkpoint_scorer_active
            ),
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
        trajectory_diagnostics = score.get("trajectory_diagnostics")
        if isinstance(trajectory_diagnostics, dict):
            for key in (
                "manifold_distance",
                "manifold_ratio",
                "fused_criterion_count",
            ):
                if key in trajectory_diagnostics:
                    diagnostics[f"smash_trajectory_{key}"] = float(
                        trajectory_diagnostics[key]
                    )
            diagnostics["smash_trajectory_gate_active"] = float(
                bool(
                    trajectory_diagnostics.get("outside_extreme_expert_support", False)
                )
            )
        if scored.view_rotation is not None:
            rotation = scored.view_rotation
            diagnostics["ankle_spine_view_rotation_degrees"] = float(
                np.degrees(np.arctan2(rotation[1, 0], rotation[0, 0]))
            )
        return GeneratedMotionInference(
            grade=grade,
            score=score,
            correction=correction,
            window=scored.window,
            source_frame_indices=source_indices,
            diagnostics=diagnostics,
            corrected_pixels=scored.corrected_pixels,
        )
