from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Literal

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
from badminton_analysis.ml.skeleton_normalization import phase_align_sequence
from badminton_analysis.ml.skeleton_scoring import (
    TORSO_WIDTH_BONES,
    project_stable_bone_lengths,
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
        (source_points - source_center).T
        @ (target_points - target_center)
    )
    rotation = left @ right
    if float(np.linalg.det(rotation)) < 0.0:
        left[:, -1] *= -1.0
        rotation = left @ right
    return ((corrected - source_center) @ rotation + target_center).astype(
        np.float32
    )


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
    """Expose the validated six-item checklist on the original rubric scale."""
    total = float(score["checklist_total_score"])
    criteria = [dict(item) for item in score["criteria"]]
    by_id = {str(item["rule_reference"]): item for item in criteria}
    preserved = ("racket_foot_weight", "weight_transfer", "shoulder_rotation")
    flexible = ("arms_raised", "hip_rotation", "wrist_flick")
    attributed: dict[str, float] = {
        criterion: min(
            float(by_id[criterion]["maximum"]),
            max(0.0, float(by_id[criterion]["score"])),
        )
        for criterion in preserved
    }
    transfer = by_id["weight_transfer"]
    strict_distance = float(
        transfer.get(
            "strict_required_cue_distance",
            transfer.get("combined_distance", 0.0),
        )
    )
    tolerance = float(transfer.get("expert_tolerance", strict_distance))
    robust_scale = max(float(transfer.get("expert_robust_scale", 1.0)), 1e-8)
    transfer_support_ratio = float(
        np.exp(-max(0.0, strict_distance - tolerance) / robust_scale)
    )
    transfer_cap = float(transfer["maximum"]) * transfer_support_ratio
    attributed["weight_transfer"] = min(
        attributed["weight_transfer"], transfer_cap
    )
    preserved_sum = float(sum(attributed.values()))
    if preserved_sum > total:
        scale = total / max(preserved_sum, 1e-8)
        attributed = {key: value * scale for key, value in attributed.items()}
        attributed.update({key: 0.0 for key in flexible})
    else:
        remaining = min(
            total - preserved_sum,
            sum(float(by_id[key]["maximum"]) for key in flexible),
        )
        active = list(flexible)
        weights = {
            key: max(float(by_id[key]["score"]), 1e-12) for key in flexible
        }
        while active and remaining > 1e-12:
            weight_sum = sum(weights[key] for key in active)
            capped = []
            for key in active:
                proposal = remaining * weights[key] / weight_sum
                maximum = float(by_id[key]["maximum"])
                if proposal >= maximum - 1e-12:
                    attributed[key] = maximum
                    remaining -= maximum
                    capped.append(key)
            if not capped:
                for key in active:
                    attributed[key] = remaining * weights[key] / weight_sum
                remaining = 0.0
                break
            active = [key for key in active if key not in capped]
        attributed.update({key: attributed.get(key, 0.0) for key in flexible})
    arms = by_id["arms_raised"]
    if bool(arms.get("passes_corrected_shoulder_height", False)):
        # The semantic shoulder-height rule is authoritative for this one
        # preparation checkpoint.  Reattribute existing total points rather
        # than inflating the aggregate: the score distribution stays on the
        # validated single head while the UI cannot call a passed arm position
        # a failure merely because another criterion consumed the allocation.
        target = float(arms["maximum"])
        needed = max(0.0, target - attributed.get("arms_raised", 0.0))
        for donor in (
            "hip_rotation",
            "wrist_flick",
            "shoulder_rotation",
            "racket_foot_weight",
            "weight_transfer",
        ):
            transfer_points = min(needed, attributed.get(donor, 0.0))
            attributed[donor] = attributed.get(donor, 0.0) - transfer_points
            attributed["arms_raised"] = (
                attributed.get("arms_raised", 0.0) + transfer_points
            )
            needed -= transfer_points
            if needed <= 1e-12:
                break
    raw_weighted_total = float(sum(float(item["score"]) for item in criteria))
    for item in criteria:
        item["raw_weighted_score"] = float(item["score"])
        item["score"] = float(attributed[str(item["rule_reference"])])
        item["aggregate_attributed_score"] = float(item["score"])
        if item["rule_reference"] == "weight_transfer":
            item["strict_transfer_support_ratio"] = transfer_support_ratio
            item["strict_transfer_attribution_cap"] = transfer_cap
        if item["rule_reference"] == "arms_raised" and bool(
            item.get("passes_corrected_shoulder_height", False)
        ):
            item["single_head_attribution_override"] = (
                "corrected_shoulder_height_semantic_pass"
            )
    attributed_total = float(sum(float(item["score"]) for item in criteria))
    return {
        **score,
        "criteria": criteria,
        "raw_weighted_total_score": raw_weighted_total,
        "weighted_total_score": attributed_total,
        "total_score": attributed_total,
        "single_head_attribution_policy": (
            "strict_transfer_cap_then_bounded_semantic_allocate"
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
                "raw_weighted_score": (
                    float(rule.maximum) * float(item["ratio"])
                ),
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
    for item, attributed in zip(
        semantic_criteria, attributed_scores, strict=True
    ):
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


_SERVE_CORRECTION_CHAINS: dict[str, tuple[int, ...]] = {
    "arms_raised": (5, 6, 7, 8, 9, 10),
    "racket_foot_weight": (11, 12, 13, 14, 15, 16),
    "weight_transfer": (5, 6, 11, 12, 13, 14, 15, 16),
    "hip_rotation": (5, 6, 11, 12, 13, 14, 15, 16),
    "wrist_flick": (6, 8, 10),
    "shoulder_rotation": (5, 6, 8, 10, 11, 12),
}


def _smooth_interval_weight(
    frame_count: int, start: int, end: int, strength: float
) -> NDArray[np.float32]:
    values = np.zeros(frame_count, dtype=np.float32)
    values[start:end] = np.float32(strength)
    ramp = min(4, start, frame_count - end)
    if ramp:
        transition = 0.5 - 0.5 * np.cos(
            np.linspace(0.0, np.pi, ramp + 2, dtype=np.float64)[1:-1]
        )
        values[start - ramp : start] = np.float32(strength) * transition
        values[end : end + ramp] = np.float32(strength) * transition[::-1]
    return values


def apply_score_conditioned_correction(
    correction: ExpertCorrection,
    score: dict[str, Any],
    spec: SkillCorrectionSpec,
    *,
    canonical_phase_indices: NDArray[np.integer],
) -> ExpertCorrection:
    """Apply generated motion only where expert-distribution evidence is weak.

    Scoring and generation deliberately have separate responsibilities.  A
    full-credit checkpoint is already a valid expert-distribution movement,
    so replacing it with one stochastic diffusion style creates a false
    visual correction (the crossing/missing green elbow case).  Deficient
    checkpoints receive a smooth, chain-coherent blend toward the generated
    expert motion.  This policy depends only on generic checkpoint scores, not
    on filenames, cohorts, or human validation labels.
    """
    if spec.slug != "serve":
        return correction
    criteria = {
        str(item["rule_reference"]): item for item in score["criteria"]
    }

    def blend(
        student: NDArray[np.floating],
        generated: NDArray[np.floating],
        confidence: NDArray[np.floating],
    ) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
        frame_count = len(student)
        alpha = np.zeros((frame_count, 17), dtype=np.float32)
        root_alpha = np.zeros(frame_count, dtype=np.float32)
        for detail, rule in zip(spec.details, spec.rules, strict=True):
            item = criteria[rule.id]
            maximum = max(float(item["maximum"]), 1e-8)
            ratio = np.clip(
                float(
                    item.get(
                        "raw_checkpoint_ratio",
                        float(item["score"]) / maximum,
                    )
                ),
                0.0,
                1.0,
            )
            # A small no-op margin avoids imperceptible detector noise making
            # a nominally correct green skeleton shimmer around the student.
            strength = float(np.clip((0.98 - ratio) / 0.78, 0.0, 1.0))
            if strength <= 0.0:
                continue
            start, end = detail.bounds(frame_count)
            interval = _smooth_interval_weight(frame_count, start, end, strength)
            joints = _SERVE_CORRECTION_CHAINS.get(
                rule.id, tuple(detail.joints or rule.measured_joints)
            )
            alpha[:, list(joints)] = np.maximum(
                alpha[:, list(joints)], interval[:, None]
            )
            if rule.id in {
                "racket_foot_weight",
                "weight_transfer",
                "hip_rotation",
            }:
                root_alpha = np.maximum(root_alpha, interval)
        if not np.any(alpha > 0.0):
            return (
                np.asarray(student, dtype=np.float32).copy(),
                root_alpha,
            )
        placed = np.asarray(student, dtype=np.float32) + alpha[..., None] * (
            np.asarray(generated, dtype=np.float32)
            - np.asarray(student, dtype=np.float32)
        )
        placed = project_stable_bone_lengths(
            student,
            placed,
            confidence,
            # Shoulder and hip spans contract in the image when the player
            # rotates away from the camera.  Treating those projected spans
            # like rigid limb lengths forces a frontal clip median onto
            # side-on frames and can tear the corrected torso/arms apart.
            expert_length_bones=TORSO_WIDTH_BONES,
            preserve_target_pelvis=True,
            preserve_direction_chains=((5, 7, 9), (6, 8, 10)),
        )
        return placed, root_alpha

    output_confidence = correction.student.confidence
    corrected, root_alpha = blend(
        correction.student.pose,
        correction.corrected_pose,
        output_confidence,
    )
    aligned_confidence = phase_align_sequence(
        output_confidence,
        correction.student.phase_indices,
        canonical_indices=canonical_phase_indices,
    )
    aligned_corrected, aligned_root_alpha = blend(
        correction.aligned_student_pose,
        correction.aligned_corrected_pose,
        aligned_confidence,
    )
    corrected_root = correction.student.root + root_alpha[:, None] * (
        correction.corrected_root - correction.student.root
    )
    aligned_corrected_root = (
        correction.aligned_student_root
        + aligned_root_alpha[:, None]
        * (
            correction.aligned_corrected_root
            - correction.aligned_student_root
        )
    )
    return replace(
        correction,
        corrected_pose=corrected,
        corrected_root=corrected_root.astype(np.float32),
        aligned_corrected_pose=aligned_corrected,
        aligned_corrected_root=aligned_corrected_root.astype(np.float32),
    )


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
        candidates: int = 8,
        seed: int = 19,
        align_ankle_spine_view: bool = False,
        hierarchical_placement_mode: Literal["fixed", "constrained"] = "fixed",
        generation_phase_contract: Literal["current", "eimd_v3"] = "eimd_v3",
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
        self.smash_semantic_score_model_path: Path | None = None
        self.smash_semantic_distribution: SmashDistribution | None = None
        self.smash_semantic_variant: SmashVariant | None = None
        self.smash_trajectory_score_model_path: Path | None = None
        self.smash_trajectory_scorer: SmashTrajectoryScorer | None = None
        if skill == Skill.SMASH and semantic_score_path.exists():
            (
                self.smash_semantic_distribution,
                self.smash_semantic_variant,
            ) = load_smash_distribution(semantic_score_path)
            self.smash_semantic_score_model_path = semantic_score_path
        trajectory_score_path = root / "expert_trajectory_score_model.npz"
        if skill == Skill.SMASH and trajectory_score_path.exists():
            self.smash_trajectory_scorer = load_smash_trajectory_scorer(
                trajectory_score_path
            )
            self.smash_trajectory_score_model_path = trajectory_score_path
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
        if hierarchical_placement_mode not in {"fixed", "constrained"}:
            raise ValueError(
                f"unsupported hierarchical placement mode: {hierarchical_placement_mode}"
            )
        self.hierarchical_placement_mode = hierarchical_placement_mode
        self.generation_phase_contract = generation_phase_contract

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
            phase_contract=self.generation_phase_contract,
        )

    def infer(
        self,
        tracking: TrackingData,
        handedness: Handedness,
        filename: str,
        *,
        prepared: tuple[
            MotionSample, tuple[int, int, int], NDArray[np.int64]
        ]
        | None = None,
    ) -> GeneratedMotionInference:
        sample, window, source_indices = (
            prepared
            if prepared is not None
            else self.prepare(tracking, handedness, filename)
        )
        scoring_sample = sample
        if self.skill == Skill.SERVE and self.generation_phase_contract == "eimd_v3":
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
        view_rotation = None
        if self.align_ankle_spine_view:
            preparation = next(
                window for window in self.spec.phase_windows
                if window.name == "preparation"
            )
            view_start, view_end = preparation.bounds(
                len(correction.aligned_student_pose)
            )
            correction, view_rotation = align_expert_correction_to_ankle_spine_view(
                correction,
                start=view_start,
                end=view_end,
                placement_mode=self.hierarchical_placement_mode,
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
                phase for phase in self.spec.phase_windows
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
                    placement_mode=self.hierarchical_placement_mode,
                )
            scoring_correction = _dual_window_scoring_correction(
                correction,
                scoring_correction,
                start=scoring_start,
                end=scoring_end,
            )
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
        # EIMD-v3 is the approved visible motion.  The older score-conditioned
        # blend was a presentation policy that could pull a valid generated
        # follow-through back toward the learner (notably removing the serve
        # forward lean).  Keep it only for the legacy/current generation
        # contract; scoring itself has already completed above.
        if self.generation_phase_contract == "current":
            correction = apply_score_conditioned_correction(
                correction,
                score,
                self.spec,
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
            "skeleton_execution_provider": str(
                next(self.bundle.network.parameters()).device
            ),
            "skeleton_tensorrt_active": 0.0,
            "ankle_spine_view_alignment_active": float(view_rotation is not None),
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
                    trajectory_diagnostics.get(
                        "outside_extreme_expert_support", False
                    )
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
        )
