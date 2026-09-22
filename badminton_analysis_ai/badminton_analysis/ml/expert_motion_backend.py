from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from badminton_analysis.ml.error_isolated_motion import (
    ErrorIsolatedMotionBundle,
    correct_student_motion_error_isolated,
    load_error_isolated_bundle,
)
from badminton_analysis.ml.expert_motion_preprocessing import (
    prepare_expert_motion_sample,
)
from badminton_analysis.ml.expert_phase_baseline import (
    ExpertCorrection,
    ExpertPhaseModel,
    MotionSample,
    align_expert_correction_to_ankle_spine_view,
    load_expert_phase_model,
    score_expert_correction,
)
from badminton_analysis.ml.smash_expert_scoring import (
    SmashDistribution,
    SmashVariant,
    allocate_smash_total_to_weighted_criteria,
    aligned_smash_evidence,
    load_smash_distribution,
    score_smash_evidence,
)
from badminton_analysis.ml.skill_specs import SkillCorrectionSpec
from badminton_analysis.ml.trajectory_distance import (
    SmashTrajectoryScorer,
    apply_smash_trajectory_score,
    load_smash_trajectory_scorer,
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


def _clip_level_rigid_target_alignment(
    generation_student: NDArray[np.floating],
    scoring_student: NDArray[np.floating],
    generated_target: NDArray[np.floating],
    *,
    start: int,
    end: int,
) -> NDArray[np.float32]:
    """Map an EIMD target into the grading view with one rigid transform.

    This deliberately cannot follow the learner per frame.  It estimates one
    rotation and translation from preparation torso/leg joints, then applies
    that same transform to the complete generated motion.
    """
    source = np.asarray(generation_student, dtype=np.float64)
    target = np.asarray(scoring_student, dtype=np.float64)
    corrected = np.asarray(generated_target, dtype=np.float64)
    if source.shape != target.shape or source.shape != corrected.shape:
        raise ValueError("dual-window poses must have matching shapes")
    if source.ndim != 3 or source.shape[1:] != (17, 2):
        raise ValueError("dual-window poses must have shape (T, 17, 2)")
    if not 0 <= start < end <= len(source):
        raise ValueError("invalid dual-window preparation interval")
    joints = np.asarray((5, 6, 11, 12, 13, 14, 15, 16), dtype=np.int64)
    source_points = source[start:end, joints].reshape(-1, 2)
    target_points = target[start:end, joints].reshape(-1, 2)
    source_center = np.mean(source_points, axis=0)
    target_center = np.mean(target_points, axis=0)
    left, _, right = np.linalg.svd(
        (source_points - source_center).T @ (target_points - target_center)
    )
    rotation = left @ right
    if float(np.linalg.det(rotation)) < 0.0:
        left[:, -1] *= -1.0
        rotation = left @ right
    return ((corrected - source_center) @ rotation + target_center).astype(np.float32)


def _dual_window_scoring_correction(
    generation: ExpertCorrection,
    scoring_scaffold: ExpertCorrection,
    *,
    start: int,
    end: int,
) -> ExpertCorrection:
    """Pair the robust grading window with the EIMD-v3 generated target."""
    mapped_target = _clip_level_rigid_target_alignment(
        generation.aligned_student_pose,
        scoring_scaffold.aligned_student_pose,
        generation.aligned_corrected_pose,
        start=start,
        end=end,
    )
    return replace(
        scoring_scaffold,
        aligned_corrected_pose=mapped_target,
        corrected_pose=mapped_target,
    )


def _serve_single_head_score(score: dict[str, Any]) -> dict[str, Any]:
    """Grade each serve checkpoint from its own evidence and add them up.

    A checkpoint's grade is its maximum times its own ratio: the expert floor
    where its own joints move within the experts' range over its own frames,
    and outside that range the lower of the floor and its corrected-skeleton
    residual. Nothing about any other checkpoint enters it.

    The total is the validated six-item checklist, which is what the raters'
    workbook is measured against, and is not the sum of the six grades: a
    checklist where one checkpoint fails is worth less than the points the
    others carry, and that judgement is what agrees with the raters. Each
    grade says how that checkpoint was performed; the total says how the serve
    was graded.
    """
    criteria = [dict(item) for item in score["criteria"]]
    for item in criteria:
        maximum = float(item["maximum"])
        ratio = float(
            item.get("raw_checkpoint_ratio", float(item["score"]) / max(maximum, 1e-8))
        )
        # Within the experts' own range for this checkpoint, the expert floor
        # decides; outside it, the checkpoint must also agree with the learner's
        # corrected skeleton -- the same switch the whole-serve score makes,
        # taken on this checkpoint's own joints and frames.
        residual = item.get("corrected_skeleton_residual_ratio")
        inside_expert_range = float(item.get("checkpoint_distance", 0.0)) <= float(
            item.get("checkpoint_expert_q80", float("inf"))
        )
        if residual is not None and not inside_expert_range:
            ratio = min(ratio, float(residual))
        if item["rule_reference"] == "weight_transfer":
            # Passing on one alternative cue while the strict all-cues distance
            # disagrees caps the transfer on its own evidence.
            strict = float(
                item.get("strict_required_cue_distance", item.get("combined_distance", 0.0))
            )
            tolerance = float(item.get("expert_tolerance", strict))
            scale = max(float(item.get("expert_robust_scale", 1.0)), 1e-8)
            support = float(np.exp(-max(0.0, strict - tolerance) / scale))
            item["strict_transfer_support_ratio"] = support
            item["strict_transfer_attribution_cap"] = maximum * support
            ratio = min(ratio, support)
        if item["rule_reference"] == "arms_raised" and bool(
            item.get("passes_corrected_shoulder_height", False)
        ):
            ratio = 1.0
        item["own_checkpoint_ratio"] = float(np.clip(ratio, 0.0, 1.0))
        item["within_expert_range"] = bool(inside_expert_range)
        item["raw_weighted_score"] = float(item["score"])
        item["score"] = maximum * item["own_checkpoint_ratio"]
        item["aggregate_attributed_score"] = float(item["score"])
    checkpoint_sum = float(sum(float(item["score"]) for item in criteria))
    total = float(score["checklist_total_score"])
    return {
        **score,
        "criteria": criteria,
        "raw_weighted_total_score": float(
            sum(float(item["raw_weighted_score"]) for item in criteria)
        ),
        "independent_checkpoint_sum": checkpoint_sum,
        "weighted_total_score": total,
        "total_score": total,
        "single_head_attribution_policy": (
            "independent_checkpoints_own_evidence_validated_checklist_total"
        ),
    }


def _score_smash_correction(
    base_score: dict[str, Any],
    sample: Any,
    correction: ExpertCorrection,
    *,
    distribution: SmashDistribution,
    variant: SmashVariant,
    trajectory_scorer: SmashTrajectoryScorer | None,
    spec: SkillCorrectionSpec,
) -> dict[str, Any]:
    """Apply the frozen smash scorer to the correction shown by the renderer.

    Keeping this as one runtime function lets the service and the cohort parity
    verifier share criterion attribution as well as the aggregate score.
    """
    evidence, reliability = aligned_smash_evidence(
        sample.pose,
        sample.confidence,
        sample.phase_indices,
    )
    semantic_score = score_smash_evidence(
        evidence,
        reliability,
        distribution,
        variant,
    )
    if trajectory_scorer is not None:
        semantic_score = apply_smash_trajectory_score(
            semantic_score,
            correction.aligned_student_pose,
            correction.aligned_corrected_pose,
            trajectory_scorer,
        )
    rules = {rule.id: rule for rule in spec.rules}
    semantic_criteria = []
    for item in semantic_score["criteria"]:
        rule = rules[str(item["rule_reference"])]
        semantic_criteria.append(
            {
                **item,
                "name_zh_tw": rule.name_zh_tw,
                "raw_checkpoint_ratio": float(item["ratio"]),
                "raw_weighted_score": (float(rule.maximum) * float(item["ratio"])),
                "maximum": float(rule.maximum),
                "euclidean_distance": float(item["semantic_distance"]),
                "target_angle_distance": 0.0,
                "combined_distance": float(item["semantic_distance"]),
            }
        )
    semantic_total = float(semantic_score["total_score"])
    attributed_scores = allocate_smash_total_to_weighted_criteria(
        np.asarray(
            [item["raw_checkpoint_ratio"] for item in semantic_criteria],
            dtype=np.float64,
        ),
        np.asarray(
            [item["maximum"] for item in semantic_criteria],
            dtype=np.float64,
        ),
        semantic_total,
    )
    for item, attributed in zip(semantic_criteria, attributed_scores, strict=True):
        item["score"] = float(attributed)
        item["aggregate_attributed_score"] = float(attributed)
    attributed_total = float(sum(item["score"] for item in semantic_criteria))
    return {
        **base_score,
        **semantic_score,
        "criteria": semantic_criteria,
        "checklist_total_score": semantic_total,
        "raw_weighted_total_score": float(
            sum(item["raw_weighted_score"] for item in semantic_criteria)
        ),
        "weighted_total_score": attributed_total,
        "total_score": attributed_total,
        "score_reference_policy": (
            "expert_only_identity_distribution_frozen_inference"
        ),
        "post_hoc_human_score_scale_calibration": False,
    }


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
        candidates: int = 8,
        seed: int = 19,
        align_ankle_spine_view: bool = False,
        current_smash: bool = False,
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
        semantic_score_path = root / "expert_semantic_score_model.npz"
        self.smash_semantic_distribution: SmashDistribution | None = None
        self.smash_semantic_variant: SmashVariant | None = None
        self.smash_trajectory_scorer: SmashTrajectoryScorer | None = None
        # The current smash scorer loads its own semantic model from
        # checkpoint_scorer_v1; the root one serves only the non-current path.
        if skill == Skill.SMASH and not current_smash and semantic_score_path.exists():
            (
                self.smash_semantic_distribution,
                self.smash_semantic_variant,
            ) = load_smash_distribution(semantic_score_path)
        trajectory_score_path = root / "expert_trajectory_score_model.npz"
        if skill == Skill.SMASH and trajectory_score_path.exists():
            self.smash_trajectory_scorer = load_smash_trajectory_scorer(
                trajectory_score_path
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
        self.current_smash_scorer = None
        if current_smash and skill == Skill.SMASH:
            from badminton_analysis.ml.smash_current_runtime import CurrentSmashScorer

            self.current_smash_scorer = CurrentSmashScorer(
                root / "checkpoint_scorer_v1",
                generator_path=self.model_path,
                trajectory_path=trajectory_score_path,
                device=next(self.bundle.network.parameters()).device,
                candidates=candidates,
                seed=seed,
            )

    def prepare(
        self,
        tracking: TrackingData,
        handedness: Handedness,
        filename: str,
    ) -> tuple[MotionSample, tuple[int, int, int], NDArray[np.int64]]:
        """Build the exact generation sample used by both gate and inference."""
        return prepare_expert_motion_sample(
            tracking,
            handedness,
            self.skill,
            filename,
            target_frames=self.target_frames,
            phase_contract=(
                "current" if self.current_smash_scorer is not None else "eimd_v3"
            ),
        )

    def infer(
        self,
        tracking: TrackingData,
        handedness: Handedness,
        filename: str,
        *,
        prepared: (
            tuple[MotionSample, tuple[int, int, int], NDArray[np.int64]] | None
        ) = None,
        fps: float = 30.0,
    ) -> GeneratedMotionInference:
        sample, window, source_indices = (
            prepared
            if prepared is not None
            else self.prepare(tracking, handedness, filename)
        )
        scoring_sample = sample
        if self.skill == Skill.SERVE:
            scoring_sample, _, _ = prepare_expert_motion_sample(
                tracking,
                handedness,
                self.skill,
                filename,
                target_frames=self.target_frames,
                phase_contract="current",
            )
        correction = correct_student_motion_error_isolated(
            self.bundle,
            sample,
            candidates=self.candidates,
            seed=self.seed,
        )
        raw_correction = correction
        corrected_pixels = None
        view_rotation = None
        if self.align_ankle_spine_view and self.current_smash_scorer is None:
            preparation = next(
                window
                for window in self.spec.phase_windows
                if window.name == "preparation"
            )
            view_start, view_end = preparation.bounds(
                len(correction.aligned_student_pose)
            )
            correction, view_rotation = align_expert_correction_to_ankle_spine_view(
                correction,
                start=view_start,
                end=view_end,
            )
        scoring_correction = correction
        if scoring_sample is not sample:
            scoring_correction = correct_student_motion_error_isolated(
                self.bundle,
                scoring_sample,
                candidates=self.candidates,
                seed=self.seed,
            )
            preparation = next(
                phase
                for phase in self.spec.phase_windows
                if phase.name == "preparation"
            )
            scoring_start, scoring_end = preparation.bounds(
                len(correction.aligned_student_pose)
            )
            if self.align_ankle_spine_view:
                scoring_correction, _ = align_expert_correction_to_ankle_spine_view(
                    scoring_correction,
                    start=scoring_start,
                    end=scoring_end,
                )
            scoring_correction = _dual_window_scoring_correction(
                correction,
                scoring_correction,
                start=scoring_start,
                end=scoring_end,
            )
        if self.current_smash_scorer is None:
            # Scored against the expert phase model with the checkpoint's own
            # canonical phases, the same way the reference grader does.
            score = score_expert_correction(
                self.score_model,
                scoring_correction,
            )
            if self.skill == Skill.SERVE:
                score = _serve_single_head_score(score)
        if (
            self.smash_semantic_distribution is not None
            and self.smash_semantic_variant is not None
            and self.current_smash_scorer is None
        ):
            score = _score_smash_correction(
                score,
                sample,
                correction,
                distribution=self.smash_semantic_distribution,
                variant=self.smash_semantic_variant,
                trajectory_scorer=self.smash_trajectory_scorer,
                spec=self.spec,
            )
        if self.current_smash_scorer is not None:
            from badminton_analysis.ml.skeleton_normalization import (
                tracking_body_arrays,
            )

            full, full_confidence = tracking_body_arrays(tracking)
            score, corrected_pixels, window = self.current_smash_scorer.score(
                sample=sample,
                correction=raw_correction,
                source_phases=source_indices[sample.phase_indices],
                native_phases=self.bundle.canonical_phase_indices,
                window=window,
                poses=full,
                confidence=full_confidence,
                handedness=handedness,
                spec=self.spec,
                trajectory_scorer=self.smash_trajectory_scorer,
                score_baseline=_score_smash_correction,
                fps=fps,
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
                self.current_smash_scorer is not None
            ),
            "ankle_spine_view_alignment_active": float(
                view_rotation is not None or self.current_smash_scorer is not None
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
        if view_rotation is not None:
            diagnostics["ankle_spine_view_rotation_degrees"] = float(
                np.degrees(np.arctan2(view_rotation[1, 0], view_rotation[0, 0]))
            )
        return GeneratedMotionInference(
            grade=grade,
            score=score,
            correction=correction,
            window=window,
            source_frame_indices=source_indices,
            diagnostics=diagnostics,
            corrected_pixels=corrected_pixels,
        )

    def prepare_skill_support(self, tracking, handedness, filename):
        """The unchanged skill-label bank was calibrated with EIMD-v3 windows.

        Do not feed the new grading windows to that independently frozen gate.
        This hypothesis never supplies score or rendering frame indices.
        """
        return prepare_expert_motion_sample(
            tracking,
            handedness,
            self.skill,
            filename,
            target_frames=self.target_frames,
            phase_contract="eimd_v3",
        )
