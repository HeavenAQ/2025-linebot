"""The serve scorer: runs one serve grade end to end."""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Sequence

from numpy.typing import NDArray
import numpy as np

from badminton_analysis.ml.motion.samples import ExpertCorrection, ExpertPhaseModel
from badminton_analysis.ml.motion.view import _aligned, _EPS
from badminton_analysis.ml.serve.checkpoints import (
    _angles,
    _serve_arms_at_corrected_shoulder_evidence,
    _SERVE_CORRECTION_ELBOW_ALLOWANCE_DEGREES,
    _SERVE_CORRECTION_ELBOW_SCALE_DEGREES,
    _SERVE_CORRECTION_SHOULDER_TURN_ALLOWANCE_DEGREES,
    _SERVE_CORRECTION_SHOULDER_TURN_SCALE_DEGREES,
    _serve_hip_rotation_components,
    _serve_qualitative_factor,
    _serve_required_motion_factor,
    _serve_transfer_against_correction,
    _serve_transfer_correction_residuals,
    _serve_weight_transfer_components,
    _serve_wrist_action_components,
    _serve_wrist_correction_residuals,
)
from badminton_analysis.ml.serve.checkpoints import (
    _serve_image_elbow_between_shoulders,
    _serve_image_elbow_opening,
    _serve_image_hip_lean,
    _serve_qualitative_pose_evidence,
)
from badminton_analysis.ml.serve.experts import (
    _serve_checkpoint_manifold,
    _serve_expert_envelope,
    _serve_expert_envelope_components,
    _serve_expert_qualitative_envelope,
)
from badminton_analysis.ml.skeleton_normalization import (
    CANONICAL_PHASE_INDICES,
    phase_align_sequence,
)
from badminton_analysis.ml.skeleton_scoring import ANGLE_TRIPLETS
from badminton_analysis.ml.skill_specs import (
    motion_completion_bounds,
    SkillCorrectionSpec,
)
from badminton_analysis.ml.motion.view import (
    align_expert_correction_to_ankle_spine_view,
)
from badminton_analysis.ml.skill import ScoredMotion, ScoringContext, SkillScorer


def criterion_distance_components(
    source: NDArray[np.floating],
    target: NDArray[np.floating],
    confidence: NDArray[np.floating],
    *,
    start: int,
    end: int,
    joints: Sequence[int],
    joint_weights: NDArray[np.floating],
) -> dict[str, float]:
    """Return separate Euclidean and target-angle distances for one rubric."""
    first = np.asarray(source, dtype=np.float64)[start:end]
    second = np.asarray(target, dtype=np.float64)[start:end]
    observed = np.asarray(confidence, dtype=np.float64)[start:end]
    selected = np.asarray(tuple(dict.fromkeys(int(value) for value in joints)))
    weights = observed[:, selected] * np.asarray(joint_weights)[selected][None]
    denominator = float(weights.sum())
    euclidean = (
        float(
            np.sum(
                np.linalg.norm(first[:, selected] - second[:, selected], axis=-1)
                * weights
            )
            / denominator
        )
        if denominator > _EPS
        else 0.0
    )
    selected_set = set(selected.tolist())
    triplets = tuple(item for item in ANGLE_TRIPLETS if set(item) <= selected_set)
    angle = 0.0
    if triplets:
        indices = np.asarray(triplets, dtype=np.int64)
        angle_mask = (
            observed[:, indices[:, 0]]
            * observed[:, indices[:, 1]]
            * observed[:, indices[:, 2]]
        )
        angle_denominator = float(angle_mask.sum())
        if angle_denominator > _EPS:
            delta = np.abs(_angles(first, triplets) - _angles(second, triplets)) / np.pi
            angle = float(np.sum(delta * angle_mask) / angle_denominator)
    return {
        "euclidean_distance": euclidean,
        "target_angle_distance": angle,
        "combined_distance": euclidean + 0.5 * angle,
    }


def _criterion_components_for_spec(
    spec: SkillCorrectionSpec,
    source_pose: NDArray[np.float32],
    source_root: NDArray[np.float32],
    target_pose: NDArray[np.float32],
    target_root: NDArray[np.float32],
    confidence: NDArray[np.float32],
    *,
    serve_expert_envelope: dict[str, dict[str, Any]] | None = None,
    image_evidence: dict[str, float] | None = None,
) -> list[dict[str, float]]:
    frame_count = len(source_pose)
    if not (
        len(source_root)
        == len(target_pose)
        == len(target_root)
        == len(confidence)
        == frame_count
    ):
        raise ValueError("criterion inputs must use the same motion length")
    output = []
    for detail, rule in zip(spec.details, spec.rules, strict=True):
        joints = detail.joints or rule.measured_joints
        source = source_pose
        target = target_pose
        start, end = detail.bounds(frame_count)
        if (
            spec.slug == "serve"
            and serve_expert_envelope is not None
            and rule.id in serve_expert_envelope
        ):
            components = _serve_expert_envelope_components(
                rule.id,
                source_pose,
                source_root,
                confidence,
                serve_expert_envelope,
                image_evidence,
            )
        elif spec.slug == "serve" and rule.id == "weight_transfer":
            components = _serve_weight_transfer_components(
                source_pose,
                source_root,
                target_pose,
                target_root,
                confidence,
            )
        elif spec.slug == "serve" and rule.id == "hip_rotation":
            components = _serve_hip_rotation_components(
                source_pose, target_pose, confidence
            )
        elif spec.slug == "serve" and rule.id == "wrist_flick":
            components = _serve_wrist_action_components(
                source_pose,
                target_pose,
                confidence,
                start=start,
                end=end,
            )
        else:
            components = criterion_distance_components(
                source,
                target,
                confidence,
                start=start,
                end=end,
                joints=joints,
                joint_weights=spec.joint_weights_array,
            )
        if detail.metric == "serve_follow_through_cross_body" and not (
            spec.slug == "serve"
            and serve_expert_envelope is not None
            and rule.id in serve_expert_envelope
        ):
            # Endpoint detection is noisy, especially across pose backends.
            completion_start, completion_end = motion_completion_bounds(
                frame_count, 0.875, 1.0
            )
            terminal = criterion_distance_components(
                source,
                target,
                confidence,
                start=completion_start,
                end=completion_end,
                joints=joints,
                joint_weights=spec.joint_weights_array,
            )
            required = (5, 6, 8, 10)
            terminal_confidence = confidence[completion_start:completion_end]
            valid = np.prod(terminal_confidence[:, list(required)], axis=1) > 0.2
            if np.any(valid):
                source_frames = source[completion_start:completion_end]
                target_frames = target[completion_start:completion_end]
                source_shoulder_center = 0.5 * (
                    source_frames[:, 5] + source_frames[:, 6]
                )
                target_shoulder_center = 0.5 * (
                    target_frames[:, 5] + target_frames[:, 6]
                )
                source_forearm_offset = (
                    0.75 * source_frames[:, 8, 0]
                    + 0.25 * source_frames[:, 10, 0]
                    - source_shoulder_center[:, 0]
                )
                target_forearm_offset = (
                    0.75 * target_frames[:, 8, 0]
                    + 0.25 * target_frames[:, 10, 0]
                    - target_shoulder_center[:, 0]
                )
                cross_body_deficiency = float(
                    np.median(
                        np.maximum(
                            source_forearm_offset[valid] - target_forearm_offset[valid],
                            0.0,
                        )
                    )
                )
            else:
                cross_body_deficiency = 0.0
            terminal_angle = float(terminal["target_angle_distance"])
            terminal_euclidean = float(terminal["euclidean_distance"])
            components = {
                "euclidean_distance": terminal_euclidean,
                "target_angle_distance": terminal_angle,
                "combined_distance": max(
                    cross_body_deficiency,
                    terminal_euclidean + 0.5 * terminal_angle,
                ),
                "window_euclidean_distance": float(components["euclidean_distance"]),
                "window_target_angle_distance": float(
                    components["target_angle_distance"]
                ),
                "completion_start_fraction": 0.875,
                "completion_end_fraction": 1.0,
                "cross_body_deficiency": cross_body_deficiency,
            }
        output.append(components)
    return output


def _aggregate_qualitative_checkpoint_ratios(
    ratios: np.ndarray,
    *,
    power: float = 1.0 / 3.0,
    floor: float = 1e-3,
) -> float:
    """Aggregate equally important checkpoints as a fixed soft conjunction."""

    values = np.asarray(ratios, dtype=np.float64)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("checkpoint ratios must be a non-empty vector")
    if power <= 0.0:
        raise ValueError("generalized-mean power must be positive")
    bounded = np.maximum(np.clip(values, 0.0, 1.0), floor)
    return float(np.mean(bounded**power) ** (1.0 / power))


def _serve_checklist_aggregation(
    criteria: list[dict[str, Any]],
) -> tuple[float, str, float, float, bool]:
    """Aggregate serve checkpoints without validation-video exceptions."""

    by_rule = {str(item["rule_reference"]): item for item in criteria}
    ordered_ids = (
        "arms_raised",
        "racket_foot_weight",
        "weight_transfer",
        "hip_rotation",
        "wrist_flick",
        "shoulder_rotation",
    )
    ratios = np.asarray(
        [
            float(by_rule[rule_id]["score"])
            / max(float(by_rule[rule_id]["maximum"]), _EPS)
            for rule_id in ordered_ids
        ],
        dtype=np.float64,
    )
    preparation = by_rule["arms_raised"]
    support_limit = float(preparation["expert_tolerance"]) + 2.5 * max(
        float(preparation["expert_robust_scale"]), 1e-3
    )
    isolated_preparation_deviation = bool(
        ratios[0] < 0.20
        and float(np.min(ratios[1:])) >= 0.90
        and float(preparation["generated_target_distance"]) <= support_limit
    )
    if isolated_preparation_deviation:
        return (
            float(100.0 * np.mean(ratios)),
            "expert_supported_isolated_preparation_additive_v4",
            1.0,
            0.0,
            True,
        )
    power = 1.0 / 3.0
    floor = 1e-3
    return (
        float(
            100.0
            * _aggregate_qualitative_checkpoint_ratios(
                ratios,
                power=power,
                floor=floor,
            )
        ),
        "soft_conjunctive_qualitative_checkpoints_v4",
        power,
        floor,
        False,
    )


def _serve_corrected_residual_checklist(
    model: ExpertPhaseModel,
    correction: ExpertCorrection,
    criteria: list[dict[str, Any]],
    *,
    semantic_total: float,
    semantic_policy: str,
    semantic_power: float,
    semantic_floor: float,
    manifold_pose: NDArray[np.floating] | None = None,
) -> tuple[float, str, float, float, NDArray[np.float64], dict[str, Any]]:
    """Fuse semantic evidence with the learner-to-correction residual."""

    tolerances = model.criterion_residual_tolerances
    scales = model.criterion_residual_scales
    semantic_ratios = np.asarray(
        [float(item["score"]) / max(float(item["maximum"]), _EPS) for item in criteria],
        dtype=np.float64,
    )
    if (
        tolerances is None
        or scales is None
        or np.asarray(tolerances).shape != semantic_ratios.shape
        or np.asarray(scales).shape != semantic_ratios.shape
    ):
        return (
            semantic_total,
            semantic_policy,
            semantic_power,
            semantic_floor,
            semantic_ratios,
            {"corrected_residual_fusion_active": False},
        )

    from badminton_analysis.ml.trajectory_distance import (
        corrected_motion_distance,
        expert_residual_ratio,
        serve_angle_manifold_distance,
    )

    costs = []
    residual_ratios = []
    for criterion, rule, tolerance, scale in zip(
        model.spec.details,
        model.spec.rules,
        tolerances,
        scales,
        strict=True,
    ):
        cost = corrected_motion_distance(
            correction.aligned_student_pose,
            correction.aligned_corrected_pose,
            joints=tuple(criterion.joints or rule.measured_joints),
            start_fraction=criterion.start_fraction,
            end_fraction=criterion.end_fraction,
            method="euclidean",
        )
        costs.append(cost)
        residual_ratios.append(
            expert_residual_ratio(
                cost,
                tolerance=float(tolerance),
                scale=float(scale),
            )
        )
    residual = np.asarray(residual_ratios, dtype=np.float64)
    # The elbow that stays bent through contact holds wherever the wrist is read from.
    residual *= np.asarray(
        [float(item.get("correction_elbow_factor", 1.0)) for item in criteria]
    )
    for item, cost, ratio in zip(criteria, costs, residual, strict=True):
        item["corrected_skeleton_euclidean_cost"] = float(cost)
        item["corrected_skeleton_residual_ratio"] = float(ratio)

    manifold = model.serve_angle_manifold
    assert manifold is not None
    manifold_distance = serve_angle_manifold_distance(
        (correction.aligned_student_pose if manifold_pose is None else manifold_pose),
        manifold,
    )
    low = np.flatnonzero(semantic_ratios < 0.20)
    isolated_wrist_uncertainty = bool(
        len(low) == 1
        and low[0] == 4
        and float(np.min(semantic_ratios[[0, 1, 2, 3, 5]])) >= 0.40
    )
    isolated_preparation_style = bool(
        len(low) == 1 and low[0] == 0 and manifold_distance <= manifold.expert_q80
    )
    diagnostics: dict[str, Any] = {
        "corrected_residual_fusion_active": True,
        "trajectory_manifold_distance": manifold_distance,
        "trajectory_manifold_expert_q80": manifold.expert_q80,
        "trajectory_manifold_expert_scale": manifold.expert_scale,
        "isolated_wrist_observability_fallback": isolated_wrist_uncertainty,
        "isolated_preparation_style_fallback": isolated_preparation_style,
    }
    if isolated_wrist_uncertainty or isolated_preparation_style:
        return (
            float(100.0 * np.mean(semantic_ratios)),
            "additive_isolated_pose_observability_v1",
            1.0,
            0.0,
            semantic_ratios,
            diagnostics,
        )
    every_checkpoint_passed = bool(np.min(semantic_ratios) >= 1.0 - 1e-3)
    diagnostics["every_checkpoint_passed"] = every_checkpoint_passed
    if manifold_distance <= manifold.expert_q80 or every_checkpoint_passed:
        # A serve that satisfies all six checkpoints on expert evidence is a complete serve.
        return (
            semantic_total,
            f"expert_manifold_supported_{semantic_policy}",
            semantic_power,
            semantic_floor,
            semantic_ratios,
            diagnostics,
        )

    effective = np.minimum(semantic_ratios, residual)
    # COCO-17 cannot see true wrist flexion.
    effective[4] = residual[4]
    power = 1.0 / 3.0
    floor = 1e-3
    checklist = float(
        100.0
        * _aggregate_qualitative_checkpoint_ratios(effective, power=power, floor=floor)
    )
    structural_indices = np.asarray((0, 1, 2, 3, 5), dtype=np.int64)
    novelty_factor = 1.0
    if float(np.min(effective[structural_indices])) < 0.20:
        standardized_novelty = max(
            0.0,
            (manifold_distance - manifold.expert_q80) / manifold.expert_scale,
        )
        novelty_factor = float(np.exp(-0.75 * standardized_novelty))
        checklist *= novelty_factor
    diagnostics["trajectory_novelty_factor"] = novelty_factor
    return (
        checklist,
        "corrected_euclidean_residual_outside_expert_manifold_v1",
        power,
        floor,
        effective,
        diagnostics,
    )


def score_expert_correction(
    model: ExpertPhaseModel,
    correction: ExpertCorrection,
    *,
    canonical_phase_indices: NDArray[np.integer] = CANONICAL_PHASE_INDICES,
    image_evidence: dict[str, float] | None = None,
) -> dict[str, Any]:
    spec = model.spec
    confidence = np.clip(
        phase_align_sequence(
            correction.student.confidence,
            correction.student.phase_indices,
            canonical_indices=canonical_phase_indices,
        ),
        0.0,
        1.0,
    )
    components = _criterion_components_for_spec(
        spec,
        correction.aligned_student_pose,
        correction.aligned_student_root,
        correction.aligned_corrected_pose,
        correction.aligned_corrected_root,
        confidence,
    )
    if model.criterion_metric_version in {
        "serve_subject_pattern_trajectory_v4",
        "serve_expert_distribution_v6",
    }:
        # The diffusion bundle may use skill-specific canonical anchors.
        semantic_pose, semantic_confidence, semantic_root = _aligned(correction.student)
        semantic_components = _criterion_components_for_spec(
            spec,
            semantic_pose,
            semantic_root,
            semantic_pose,
            semantic_root,
            semantic_confidence,
            serve_expert_envelope=_serve_expert_envelope(model),
            image_evidence=image_evidence,
        )
        for index, rule in enumerate(spec.rules):
            if rule.id in {
                "arms_raised",
                "racket_foot_weight",
                "weight_transfer",
                "hip_rotation",
                "wrist_flick",
                "shoulder_rotation",
            }:
                semantic = semantic_components[index]
                generated_distance = float(components[index]["combined_distance"])
                if rule.id == "weight_transfer":
                    semantic = _serve_transfer_against_correction(
                        semantic,
                        _serve_transfer_correction_residuals(
                            semantic_pose,
                            phase_align_sequence(
                                correction.corrected_pose,
                                correction.student.phase_indices,
                            ),
                        ),
                        float(model.criterion_tolerances[index]),
                    )
                if rule.id == "wrist_flick":
                    semantic = {
                        **semantic,
                        **_serve_wrist_correction_residuals(
                            semantic_pose,
                            phase_align_sequence(
                                correction.corrected_pose,
                                correction.student.phase_indices,
                            ),
                        ),
                    }
                if rule.id == "wrist_flick":
                    semantic = {
                        **semantic,
                        **_serve_wrist_correction_residuals(
                            semantic_pose,
                            phase_align_sequence(
                                correction.corrected_pose,
                                correction.student.phase_indices,
                            ),
                        ),
                    }
                detail = spec.details[index]
                start, end = detail.bounds(len(semantic_pose))
                checkpoint_manifold = _serve_checkpoint_manifold(
                    model, rule.id, tuple(detail.joints or rule.measured_joints), start, end
                )
                from badminton_analysis.ml.trajectory_distance import (
                    SERVE_CHECKPOINT_HELD_OUT,
                    serve_checkpoint_distance,
                )

                semantic = {
                    **semantic,
                    "checkpoint_distance": serve_checkpoint_distance(
                        np.asarray(semantic_pose, dtype=np.float64), checkpoint_manifold
                    ),
                    "checkpoint_expert_q80": float(
                        np.quantile(
                            SERVE_CHECKPOINT_HELD_OUT[
                                (
                                    tuple(checkpoint_manifold.triplets),
                                    checkpoint_manifold.start,
                                    checkpoint_manifold.end,
                                )
                            ],
                            0.80,
                        )
                    ),
                }
                components[index] = {
                    **semantic,
                    "generated_target_distance": generated_distance,
                    "semantic_envelope_distance": float(semantic["combined_distance"]),
                    "selected_expert_evidence": (
                        "expert_only_identity_distribution"
                        if model.criterion_metric_version
                        == "serve_expert_distribution_v6"
                        else "nearest_expert_subject_pattern"
                    ),
                }
    criteria = []
    qualitative_evidence: dict[str, float] | None = None
    qualitative_envelope: dict[str, dict[str, float]] | None = None
    if model.criterion_metric_version == "serve_dominant_chain_coupled_v5":
        qualitative_evidence = _serve_qualitative_pose_evidence(
            correction.aligned_student_pose,
            correction.aligned_student_root,
        )
        qualitative_envelope = _serve_expert_qualitative_envelope(model)
    for index, (rule, component) in enumerate(zip(spec.rules, components, strict=True)):
        tolerance = float(model.criterion_tolerances[index])
        scale = float(model.criterion_scales[index])
        distance = component["combined_distance"]
        excess = max(0.0, distance - tolerance)
        ratio = float(np.exp(-excess / max(scale, 1e-3)))
        if "correction_elbow_at_contact_shortfall_degrees" in component:
            component["correction_elbow_factor"] = float(
                np.exp(
                    -max(
                        0.0,
                        component["correction_elbow_at_contact_shortfall_degrees"]
                        - _SERVE_CORRECTION_ELBOW_ALLOWANCE_DEGREES,
                    )
                    / _SERVE_CORRECTION_ELBOW_SCALE_DEGREES
                )
            )
            ratio *= component["correction_elbow_factor"]
        if "correction_shoulder_turn_shortfall_degrees" in component:
            shoulder_factor = float(
                np.exp(
                    -max(
                        0.0,
                        component["correction_shoulder_turn_shortfall_degrees"]
                        - _SERVE_CORRECTION_SHOULDER_TURN_ALLOWANCE_DEGREES,
                    )
                    / _SERVE_CORRECTION_SHOULDER_TURN_SCALE_DEGREES
                )
            )
            component["correction_shoulder_turn_factor"] = shoulder_factor
            component["correction_elbow_factor"] = (
                component.get("correction_elbow_factor", 1.0) * shoulder_factor
            )
            ratio *= shoulder_factor
        qualitative_factor = 1.0
        qualitative_diagnostics: dict[str, float | str] = {}
        if qualitative_evidence is not None and qualitative_envelope is not None:
            evidence_name = (
                "simultaneous_arm_elevation"
                if rule.id == "arms_raised"
                else (
                    "preparation_stance_width"
                    if rule.id
                    in {
                        "racket_foot_weight",
                        "shoulder_rotation",
                    }
                    else None
                )
            )
            if rule.id in {"weight_transfer", "hip_rotation"}:
                stance_factor = _serve_qualitative_factor(
                    float(qualitative_evidence["preparation_stance_width"]),
                    qualitative_envelope["preparation_stance_width"],
                )
                movement_factors = {
                    name: _serve_required_motion_factor(
                        float(qualitative_evidence[name]),
                        qualitative_envelope[name],
                    )
                    for name in (
                        "pelvis_loading_shift",
                        "dominant_chain_excursion",
                        "coordinated_hip_rotation",
                    )
                }
                # All three are prerequisites.
                transfer_magnitude_factor = float(
                    np.prod(list(movement_factors.values()))
                )
                qualitative_factor = stance_factor * transfer_magnitude_factor
                qualitative_diagnostics = {
                    "qualitative_evidence": ("coordinated_absolute_weight_transfer"),
                    "qualitative_evidence_factor": qualitative_factor,
                    "preparation_stance_factor": stance_factor,
                    "pelvis_loading_shift": float(
                        qualitative_evidence["pelvis_loading_shift"]
                    ),
                    "pelvis_loading_shift_factor": movement_factors[
                        "pelvis_loading_shift"
                    ],
                    "dominant_chain_change_magnitude": float(
                        qualitative_evidence["dominant_chain_change"]
                    ),
                    "dominant_chain_excursion": float(
                        qualitative_evidence["dominant_chain_excursion"]
                    ),
                    "dominant_chain_excursion_factor": movement_factors[
                        "dominant_chain_excursion"
                    ],
                    "hip_rotation_excursion": float(
                        qualitative_evidence["hip_rotation_excursion"]
                    ),
                    "transfer_rotation_correlation": float(
                        qualitative_evidence["transfer_rotation_correlation"]
                    ),
                    "coordinated_hip_rotation": float(
                        qualitative_evidence["coordinated_hip_rotation"]
                    ),
                    "coordinated_hip_rotation_factor": movement_factors[
                        "coordinated_hip_rotation"
                    ],
                    "root_transfer_distance": float(
                        qualitative_evidence["root_transfer_distance"]
                    ),
                    "transfer_magnitude_factor": transfer_magnitude_factor,
                    "qualitative_calibration_policy": (
                        "expert_identity_coordinated_transfer_envelope_only"
                    ),
                }
            elif evidence_name is not None:
                evidence_value = float(qualitative_evidence[evidence_name])
                evidence_calibration = qualitative_envelope[evidence_name]
                qualitative_factor = _serve_qualitative_factor(
                    evidence_value, evidence_calibration
                )
                qualitative_diagnostics = {
                    "qualitative_evidence": evidence_name,
                    "qualitative_evidence_value": evidence_value,
                    "qualitative_expert_lower": float(
                        evidence_calibration["expert_lower"]
                    ),
                    "qualitative_expert_scale": float(
                        evidence_calibration["expert_scale"]
                    ),
                    "qualitative_evidence_factor": qualitative_factor,
                    "qualitative_calibration_policy": (
                        "expert_identity_lower_envelope_only"
                    ),
                }
        criteria.append(
            {
                "name_zh_tw": rule.name_zh_tw,
                "rule_reference": rule.id,
                "score": rule.maximum * ratio * qualitative_factor,
                "maximum": rule.maximum,
                "expert_tolerance": tolerance,
                "expert_robust_scale": scale,
                "standardized_excess": excess / max(scale, 1e-3),
                **component,
                **qualitative_diagnostics,
            }
        )
    if model.criterion_metric_version == "serve_expert_distribution_v6":
        arm_evidence = _serve_arms_at_corrected_shoulder_evidence(
            correction.aligned_student_pose,
            correction.aligned_corrected_pose,
            confidence,
        )
        arms_item = next(
            item for item in criteria if item["rule_reference"] == "arms_raised"
        )
        arms_item.update(arm_evidence)
        if bool(arm_evidence["passes_corrected_shoulder_height"]):
            arms_item["score_before_corrected_shoulder_height_pass"] = float(
                arms_item["score"]
            )
            arms_item["score"] = float(arms_item["maximum"])
            arms_item["arms_raised_policy"] = (
                "both_hands_at_or_near_corrected_shoulder_height"
            )
    if model.criterion_metric_version == "serve_expert_distribution_v6":
        by_rule = {item["rule_reference"]: item for item in criteria}
        dynamic_completion_gate = min(
            float(by_rule[rule_id]["score"])
            / max(float(by_rule[rule_id]["maximum"]), _EPS)
            for rule_id in (
                "weight_transfer",
                "wrist_flick",
            )
        )
        for rule_id in ("hip_rotation", "shoulder_rotation"):
            item = by_rule[rule_id]
            if item.get("semantic_cue_aggregation") not in {
                "contraction_and_orientation_camera_robust",
                "shoulder_contraction_or_cross_body_completion",
            }:
                continue
            tolerance = float(item["expert_tolerance"])
            scale = max(float(item["expert_robust_scale"]), 1e-3)
            strict_distance = float(item["strict_required_cue_distance"])
            strict_ratio = float(np.exp(-max(0.0, strict_distance - tolerance) / scale))
            alternative_ratio = float(item["score"]) / max(float(item["maximum"]), _EPS)
            generated_distance = float(item["generated_target_distance"])
            generated_tolerance = max(tolerance, 0.5 * scale)
            generated_ratio = float(
                np.exp(-max(0.0, generated_distance - generated_tolerance) / scale)
            )
            supported_alternative_ratio = max(
                alternative_ratio,
                generated_ratio,
            )
            selected_ratio = max(
                strict_ratio,
                supported_alternative_ratio * dynamic_completion_gate,
            )
            item["score"] = float(item["maximum"]) * selected_ratio
            item["camera_robust_alternative_ratio"] = alternative_ratio
            item["generated_agreement_ratio"] = generated_ratio
            item["generated_agreement_tolerance"] = generated_tolerance
            item["supported_camera_evidence_ratio"] = supported_alternative_ratio
            item["strict_required_cue_ratio"] = strict_ratio
            item["serve_motion_completeness_gate"] = dynamic_completion_gate
            item["motion_completeness_gate_policy"] = (
                "weight_transfer_and_wrist_dynamic_completion"
            )
            item["selected_camera_evidence_ratio"] = selected_ratio
    weighted_total = float(sum(item["score"] for item in criteria))
    criterion_ratios = np.asarray(
        [float(item["score"]) / max(float(item["maximum"]), _EPS) for item in criteria],
        dtype=np.float64,
    )
    arithmetic_checklist_total = float(100.0 * np.mean(criterion_ratios))
    isolated_preparation_deviation = False
    residual_fusion_diagnostics: dict[str, Any] = {
        "corrected_residual_fusion_active": False
    }
    if model.criterion_metric_version == "serve_expert_distribution_v6":
        (
            checklist_total,
            total_aggregation,
            aggregation_power,
            aggregation_floor,
            isolated_preparation_deviation,
        ) = _serve_checklist_aggregation(criteria)
        (
            checklist_total,
            total_aggregation,
            aggregation_power,
            aggregation_floor,
            effective_checklist_ratios,
            residual_fusion_diagnostics,
        ) = _serve_corrected_residual_checklist(
            model,
            correction,
            criteria,
            semantic_total=checklist_total,
            semantic_policy=total_aggregation,
            semantic_power=aggregation_power,
            semantic_floor=aggregation_floor,
            manifold_pose=semantic_pose,
        )
        bounded = np.maximum(
            np.clip(effective_checklist_ratios, 0.0, 1.0),
            aggregation_floor,
        )
        if aggregation_power == 1.0:
            contribution_weights = effective_checklist_ratios / max(
                float(np.sum(effective_checklist_ratios)), _EPS
            )
        else:
            transformed = bounded**aggregation_power
            contribution_weights = transformed / max(float(np.sum(transformed)), _EPS)
        for index, (item, ratio) in enumerate(
            zip(criteria, criterion_ratios, strict=True)
        ):
            # Product grading retains the original qualitative rubric (5/5/30/10/30/20).
            item["raw_checkpoint_ratio"] = float(ratio)
            item["effective_checklist_ratio"] = float(effective_checklist_ratios[index])
            item["checklist_score_contribution"] = float(
                checklist_total * contribution_weights[index]
            )
            item["checklist_maximum"] = 100.0 / len(criteria)
    else:
        checklist_total = arithmetic_checklist_total
        aggregation_power = 1.0
        aggregation_floor = 0.0
        total_aggregation = "equal_qualitative_checkpoint_mean_v1"
    return {
        "filename": correction.student.video_name,
        "skill": model.skill,
        "handedness": correction.student.handedness,
        "score_method": (
            "expert_only_generated_projection_residual_v4"
            if model.criterion_metric_version
            == "expert_generated_projection_residual_v4"
            else (
                "expert_only_coordinated_transfer_coupled_v4"
                if model.criterion_metric_version == "serve_dominant_chain_coupled_v5"
                else (
                    (
                        "expert_only_identity_distribution_v6"
                        if model.criterion_metric_version
                        == "serve_expert_distribution_v6"
                        else "expert_only_subject_pattern_trajectory_v8"
                    )
                    if model.criterion_metric_version
                    in {
                        "serve_subject_pattern_trajectory_v4",
                        "serve_expert_distribution_v6",
                    }
                    else (
                        "expert_only_semantic_criterion_tolerance_v2"
                        if model.criterion_metric_version
                        == "serve_semantic_motion_features_v2"
                        else "expert_only_held_out_identity_tolerance_v1"
                    )
                )
            )
        ),
        "correction_policy": (
            "full_body_generated_expert_projection_energy"
            if model.criterion_metric_version
            == "expert_generated_projection_residual_v4"
            else "full_body_coherent_expert_phase_projection"
        ),
        "score_reference_policy": (
            "subject_balanced_generated_projection_residual"
            if model.criterion_metric_version
            == "expert_generated_projection_residual_v4"
            else (
                "generated_expert_dominant_chain_and_rotation_coupling"
                if model.criterion_metric_version == "serve_dominant_chain_coupled_v5"
                else (
                    (
                        "expert_identity_held_out_checkpoint_distribution"
                        if model.criterion_metric_version
                        == "serve_expert_distribution_v6"
                        else "nearest_subject_checkpoint_trajectory_pattern"
                    )
                    if model.criterion_metric_version
                    in {
                        "serve_subject_pattern_trajectory_v4",
                        "serve_expert_distribution_v6",
                    }
                    else "generated_correction_distance"
                )
            )
        ),
        "limitations": (
            ["wrist_action_uses_coco17_distal_arm_motion_proxy"]
            if model.criterion_metric_version
            in {
                "serve_subject_pattern_trajectory_v4",
                "serve_dominant_chain_coupled_v5",
                "serve_expert_distribution_v6",
            }
            else []
        ),
        # The product grade and expert-validation checklist are intentionally both reported.
        "total_score": weighted_total,
        "weighted_total_score": weighted_total,
        "checklist_total_score": checklist_total,
        "checklist_score_0_6": checklist_total * 6.0 / 100.0,
        "arithmetic_checklist_total_score": arithmetic_checklist_total,
        "weighted_coaching_total_score": weighted_total,
        "total_aggregation": total_aggregation,
        "aggregation_power": aggregation_power,
        "aggregation_floor": aggregation_floor,
        "isolated_preparation_deviation": isolated_preparation_deviation,
        **residual_fusion_diagnostics,
        "criteria": criteria,
        "references": [
            {
                "file": str(model.expert_files[index]),
                "subject_id": str(model.expert_subject_ids[index]),
                "identity_level": str(model.expert_identity_levels[index]),
                "alignment_contract": str(model.expert_alignment_contracts[index]),
                "weight": float(weight),
                "stance_distance": float(distance),
            }
            for index, weight, distance in zip(
                correction.reference_indices,
                correction.reference_weights,
                correction.reference_distances,
                strict=True,
            )
        ],
    }


def _clip_level_rigid_target_alignment(
    generation_student: NDArray[np.floating],
    scoring_student: NDArray[np.floating],
    generated_target: NDArray[np.floating],
    *,
    start: int,
    end: int,
) -> NDArray[np.float32]:
    """Map an EIMD target into the grading view with one rigid transform."""
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


def _serve_single_head_score(
    score: dict[str, Any],
    hip_lean: dict[str, float] | None = None,
    elbow_opening: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Grade each serve checkpoint from its own evidence and add them up."""
    criteria = [dict(item) for item in score["criteria"]]
    for item in criteria:
        maximum = float(item["maximum"])
        ratio = float(
            item.get("raw_checkpoint_ratio", float(item["score"]) / max(maximum, 1e-8))
        )
        # Within the experts' own range for this checkpoint, the expert floor decides; outside it.
        residual = item.get("corrected_skeleton_residual_ratio")
        inside_expert_range = float(item.get("checkpoint_distance", 0.0)) <= float(
            item.get("checkpoint_expert_q80", float("inf"))
        )
        if residual is not None and not inside_expert_range:
            ratio = min(ratio, float(residual))
        if item["rule_reference"] == "weight_transfer":
            # Passing on one alternative cue while the strict all-cues distance disagrees caps the transfer on its own evidence.
            strict = float(
                item.get("strict_required_cue_distance", item.get("combined_distance", 0.0))
            )
            tolerance = float(item.get("expert_tolerance", strict))
            scale = max(float(item.get("expert_robust_scale", 1.0)), 1e-8)
            support = float(np.exp(-max(0.0, strict - tolerance) / scale))
            item["strict_transfer_support_ratio"] = support
            item["strict_transfer_attribution_cap"] = maximum * support
            ratio = min(ratio, support)
            if hip_lean is not None:
                # Hips stacked over the ankles at the finish: the weight never went onto the front foot.
                item.update(hip_lean)
                ratio = min(ratio, hip_lean["image_hip_lean_factor"])
        if item["rule_reference"] == "wrist_flick" and elbow_opening is not None:
            # An elbow that only folds into contact pushes the racket instead of flicking it.
            item.update(elbow_opening)
            ratio = min(ratio, elbow_opening["image_elbow_opening_factor"])
        if item["rule_reference"] == "arms_raised" and bool(
            item.get("passes_corrected_shoulder_height", False)
        ):
            ratio = 1.0
        item["own_checkpoint_ratio"] = float(np.clip(ratio, 0.0, 1.0))
        item["within_expert_range"] = bool(inside_expert_range)
        item["raw_weighted_score"] = float(item["score"])
        item["score"] = maximum * item["own_checkpoint_ratio"]
        item["aggregate_attributed_score"] = float(item["score"])
    total = float(sum(float(item["score"]) for item in criteria))
    return {
        **score,
        "criteria": criteria,
        "raw_weighted_total_score": float(
            sum(float(item["raw_weighted_score"]) for item in criteria)
        ),
        "weighted_total_score": total,
        "total_score": total,
        "single_head_attribution_policy": "independent_checkpoints_own_evidence_sum",
    }


class ServeScorer(SkillScorer):
    """Expert phase model on the robust grading window, one head per checkpoint."""

    def score(self, context: ScoringContext) -> ScoredMotion:
        scoring_sample, scoring_window, scoring_frames = self.definition.prepare_sample(
            context.tracking,
            context.handedness,
            context.filename,
            target_frames=self.target_frames,
            phase_contract="current",
        )
        correction, view_rotation = self.view_aligned(context.correction)
        scoring_correction = self.generate(scoring_sample)
        scoring_start, scoring_end = self.preparation_bounds(
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
        # Cues the pre-processed sample cannot carry are read on the video's own keypoints.
        keypoints = np.asarray(context.tracking["body_keypoints_2d"], dtype=np.float64)
        observed = np.asarray(context.tracking["body_confidence_2d"], dtype=np.float64)
        side = context.handedness.name.lower()
        hip_lean = _serve_image_hip_lean(keypoints, observed, scoring_frames, side)
        elbow_opening = _serve_image_elbow_opening(
            keypoints, scoring_window, side, context.fps
        )
        elbow_between = _serve_image_elbow_between_shoulders(
            keypoints, observed, scoring_window, side
        )
        # Scored against the expert phase model with the checkpoint's own canonical phases.
        score = _serve_single_head_score(
            score_expert_correction(
                self.score_model,
                scoring_correction,
                image_evidence={"terminal_elbow_between_shoulders": -elbow_between},
            ),
            hip_lean,
            elbow_opening,
        )
        return ScoredMotion(
            score=score,
            window=context.window,
            scoring_sample=scoring_sample,
            correction=correction,
            view_rotation=view_rotation,
        )


# -- The interface every skill's scorer module provides ----------------------

LIMIT_GENERATED_WRIST_VELOCITY = True


def create_scorer(definition, *, current_scorer=True, **config):
    return ServeScorer(definition, **config)

