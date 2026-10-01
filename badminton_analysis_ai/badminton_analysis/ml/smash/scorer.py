"""The smash scorer: runs one smash grade end to end."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any
import hashlib
import json

import numpy as np
import torch

from badminton_analysis.ml.motion.samples import ExpertCorrection
from badminton_analysis.ml.motion.view import (
    ankle_spine_view_rotation,
    apply_fixed_hierarchical_pose_placement,
    project_pose_to_student_view,
)
from badminton_analysis.ml.skeleton_normalization import phase_align_sequence
from badminton_analysis.ml.skill_specs import SkillCorrectionSpec
from badminton_analysis.ml.smash.experts import (
    align_contacts,
    observed_cost,
    observed_features,
    transfer_intervals,
)
from badminton_analysis.ml.smash.coaching import build_checkpoint_evidence
from badminton_analysis.ml.smash.experts import (
    aggregate,
    checkpoint_index_map,
    checkpoint_reference_at_source_frames,
    checkpoint_windows,
    CheckpointMetricGraph,
    infer,
)
from badminton_analysis.ml.motion.view import (
    first_frame_ankle_spine_map,
    first_frame_standing_offsets,
    smooth_corrected_bbox_placement,
    transport_corrected_by_student_displacement,
)
from badminton_analysis.ml.smash.checkpoints import (
    CurrentSmashCalibration,
    LEGACY_MAXIMA,
    MAXIMA,
    score_source_evidence,
)
from badminton_analysis.ml.smash.experts import (
    aligned_smash_evidence,
    allocate_smash_total_to_weighted_criteria,
    load_smash_distribution,
    score_smash_evidence,
    SmashDistribution,
    SmashVariant,
)
from badminton_analysis.ml.trajectory_distance import (
    apply_smash_trajectory_score,
    SmashTrajectoryScorer,
)
from badminton_analysis.ml.serve.scorer import score_expert_correction
from badminton_analysis.ml.skeleton_normalization import tracking_body_arrays
from badminton_analysis.ml.skill import ScoredMotion, ScoringContext, SkillScorer
from badminton_analysis.ml.trajectory_distance import load_smash_trajectory_scorer


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def canonical_source(poses, confidence, handedness):
    """Match normalization's anatomical swap, without inventing a camera flip."""
    p, c = np.asarray(poses).copy(), np.asarray(confidence).copy()
    if str(handedness) == "left":
        for left, right in (
            (1, 2),
            (3, 4),
            (5, 6),
            (7, 8),
            (9, 10),
            (11, 12),
            (13, 14),
            (15, 16),
        ):
            p[:, [left, right]] = p[:, [right, left]]
            c[:, [left, right]] = c[:, [right, left]]
    elif str(handedness) != "right":
        raise ValueError("Explicit handedness required")
    return p, c


class CurrentSmashScorer:
    """Loads only frozen runtime artifacts, with a checked dependency contract."""

    def __init__(
        self, root, *, generator_path, trajectory_path, device, candidates, seed
    ):
        self.root = Path(root)
        manifest = json.loads((self.root / "manifest.json").read_text())
        required = {
            "calibration.json",
            "checkpoint_reference.npz",
            "metric_graph.pt",
            "expert_semantic_score_model.npz",
        }
        if set(manifest) != required:
            raise ValueError("Incomplete current smash artifact manifest")
        for name, expected in manifest.items():
            if sha256(self.root / name) != expected:
                raise ValueError(f"Current smash artifact hash mismatch: {name}")
        self.contract = json.loads((self.root / "calibration.json").read_text())
        generation = self.contract["generation"]
        if (candidates, seed) != (generation["candidates"], generation["seed"]):
            raise ValueError("Smash candidates/seed differ from calibration")
        if sha256(generator_path) != generation["checkpoint_sha256"]:
            raise ValueError("Smash generator differs from calibration")
        if sha256(trajectory_path) != self.contract["trajectory_sha256"]:
            raise ValueError("Smash trajectory scorer differs from calibration")
        self.calibration = CurrentSmashCalibration.load(self.root / "calibration.json")
        self.distribution, self.variant = load_smash_distribution(
            self.root / "expert_semantic_score_model.npz"
        )
        checkpoint = torch.load(
            self.root / "metric_graph.pt", weights_only=True, map_location="cpu"
        )
        self.graph = CheckpointMetricGraph(**checkpoint["config"]).to(device).eval()
        self.graph.load_state_dict(checkpoint["state_dict"])
        self.support = np.asarray(self.contract["graph_support"], dtype=float)
        with np.load(self.root / "checkpoint_reference.npz", allow_pickle=False) as z:
            if float(z["fps"]) != 30:
                raise ValueError("Checkpoint template must use 30fps source indices")
            self.reference = observed_features(z["poses"], z["confidence"])

    def historical_heads(
        self,
        sample,
        correction,
        source_phases,
        native_phases,
        spec,
        trajectory_scorer,
        score_baseline,
    ):
        """Exact frozen graph + semantic/trajectory baseline before new caps."""
        legacy_spec = replace(
            spec,
            rules=tuple(
                replace(rule, maximum=float(maximum))
                for rule, maximum in zip(spec.rules, LEGACY_MAXIMA, strict=True)
            ),
        )
        common = np.array([0, 25, 50, 60, 63])
        p = phase_align_sequence(
            sample.pose, sample.phase_indices, canonical_indices=common
        )
        q = phase_align_sequence(
            correction.aligned_corrected_pose, native_phases, canonical_indices=common
        )
        start, end = next(
            w for w in spec.phase_windows if w.name == "preparation"
        ).bounds(64)
        rotation = ankle_spine_view_rotation(p, q, start=start, end=end)
        q = project_pose_to_student_view(p, q, rotation)
        q = apply_fixed_hierarchical_pose_placement(p, q, start=start, end=end)
        baseline = score_baseline(
            {},
            sample,
            SimpleNamespace(aligned_student_pose=p, aligned_corrected_pose=q),
            distribution=self.distribution,
            variant=self.variant,
            trajectory_scorer=trajectory_scorer,
            spec=legacy_spec,
        )
        points = np.array([r["score"] for r in baseline["criteria"]])
        indices, ids, _ = checkpoint_index_map(
            sample.phase_indices, [r["anchors"] for r in self.contract["graph_rules"]]
        )
        observed, confidence = checkpoint_windows(
            sample.pose, sample.confidence, indices
        )
        queries = np.interp(indices, sample.phase_indices, source_phases)
        target, _ = checkpoint_reference_at_source_frames(
            correction.aligned_corrected_pose,
            native_phases,
            source_phases,
            queries,
            "kinematic",
        )
        probability = infer(
            self.graph, observed[None], confidence[None], target[None], ids
        )
        ratios = aggregate(probability, ids, [0, 2])[0] / self.support
        if not np.isfinite(ratios[[0, 2]]).all():
            raise ValueError("Non-finite checkpoint graph prediction")
        points[[0, 2]] = LEGACY_MAXIMA[[0, 2]] * np.clip(ratios[[0, 2]], 0, 1)
        return baseline, points

    def score(
        self,
        *,
        sample,
        correction,
        source_phases,
        native_phases,
        window,
        poses,
        confidence,
        handedness,
        spec,
        trajectory_scorer,
        score_baseline,
        fps,
    ):
        if fps != 30.0:
            raise ValueError("Normalize smash video to 30fps before pose extraction")
        full, full_c = canonical_source(poses, confidence, handedness)
        baseline, prior = self.historical_heads(
            sample,
            correction,
            source_phases,
            native_phases,
            spec,
            trajectory_scorer,
            score_baseline,
        )
        start, peak, end = window
        frames = np.arange(start, end + 1)
        queries = np.repeat(frames[:, None], 5, axis=1)
        q = checkpoint_reference_at_source_frames(
            correction.aligned_corrected_pose,
            native_phases,
            source_phases,
            queries,
            "kinematic",
        )[0][:, 0]
        matrix, translation, _ = first_frame_ankle_spine_map(
            q[0], full[start], full_c[start]
        )
        q = q @ matrix + translation
        q += first_frame_standing_offsets(q[0], full[start], full_c[start])
        local = smooth_corrected_bbox_placement(
            q.astype(np.float32), alpha_current=0.65
        )
        displayed = transport_corrected_by_student_displacement(
            local, full[frames], full_c[frames]
        )
        template = self.contract["intervals"]
        mapping, _, _ = align_contacts(
            observed_cost(self.reference, observed_features(full, full_c)),
            template["wrist_flick"]["anchor"],
            int(source_phases[2]),
        )
        intervals = transfer_intervals(template, mapping)
        result = score_source_evidence(
            poses=full,
            confidence=full_c,
            corrected_pixels=displayed,
            generated_frames=frames,
            intervals=intervals,
            contact=int(source_phases[2]),
            historical_points=prior,
            calibration=self.calibration,
            fps=fps,
            endpoint_acceleration=peak,
        )
        # Grade the full evidence first, then crop presentation at the selected endpoint.
        render_end = int(result["selected_endpoint"])
        if not start <= peak <= render_end < len(full):
            raise ValueError("Scored smash endpoint falls outside the render window")
        displacement = transport_corrected_by_student_displacement(
            np.zeros_like(full, dtype=np.float32), full, full_c
        )[:, 0].astype(float)
        tail = displayed[-1] + (displacement[end + 1 :] - displacement[end])[:, None]
        extended = np.concatenate((displayed, tail))
        render_pixels = np.concatenate(
            (np.repeat(displayed[:1], start, axis=0), extended)
        )[: render_end + 1]
        checkpoint_frames = {
            key: int(value["anchor"]) for key, value in intervals.items()
        }
        checkpoint_frames["follow_through"] = result["selected_endpoint"]
        checkpoint_frames["wrist_flick"] = int(source_phases[2])
        evidence = build_checkpoint_evidence(
            intervals=intervals,
            window_start=0,
            window_end=render_end,
            initial_frame=start,
            balance_anchor=result["balance_anchor"],
            selected_endpoint=result["selected_endpoint"],
            balance_measurement=result["evidence"]["arm_balance"],
        )
        criteria = [
            dict(
                rule_reference=row["rule_reference"],
                name_zh_tw=row["name_zh_tw"],
                score=float(points),
                maximum=float(maximum),
                ratio=float(points / maximum),
                euclidean_distance=row["euclidean_distance"],
                target_angle_distance=row["target_angle_distance"],
                combined_distance=row["combined_distance"],
                distance_basis="historical_semantic_diagnostic_not_current_score",
            )
            for row, points, maximum in zip(
                baseline["criteria"], result["points"], MAXIMA, strict=True
            )
        ]
        score = dict(
            criteria=criteria,
            total_score=result["total"],
            weighted_total_score=result["total"],
            historical_score=baseline,
            references=baseline.get("references", []),
            trajectory_diagnostics=baseline.get("trajectory_diagnostics", {}),
            score_reference_policy="frozen_expert_checkpoint_calibration",
            score_method="smash_local_checkpoint_graph_geometry_v20260913",
            checkpoint_evidence=evidence,
            checkpoint_source_frames=checkpoint_frames,
            checkpoint_measurements=result["evidence"],
            generation_window=list(window),
            scorer_contract=self.contract["version"],
        )
        return score, render_pixels.astype(np.float32), (0, peak, render_end)


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
    """Apply the frozen smash scorer to the correction shown by the renderer."""
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


# -- The interface every skill's scorer module provides ----------------------

LIMIT_GENERATED_WRIST_VELOCITY = False


class SmashScorer(SkillScorer):
    """Checkpoint graph, geometry and endpoint rules on source-clock evidence."""

    generation_contract = "current"

    def __init__(self, definition, **config):
        super().__init__(definition, **config)
        trajectory_path = self.root / "expert_trajectory_score_model.npz"
        self.trajectory_scorer = (
            load_smash_trajectory_scorer(trajectory_path)
            if trajectory_path.exists()
            else None
        )
        self.checkpoint_scorer = CurrentSmashScorer(
            self.root / "checkpoint_scorer_v1",
            generator_path=self.model_path,
            trajectory_path=trajectory_path,
            device=next(self.bundle.network.parameters()).device,
            candidates=self.candidates,
            seed=self.seed,
        )

    def score(self, context: ScoringContext) -> ScoredMotion:
        full, full_confidence = tracking_body_arrays(context.tracking)
        score, corrected_pixels, window = self.checkpoint_scorer.score(
            sample=context.sample,
            correction=context.correction,
            source_phases=context.source_indices[context.sample.phase_indices],
            native_phases=self.bundle.canonical_phase_indices,
            window=context.window,
            poses=full,
            confidence=full_confidence,
            handedness=context.handedness,
            spec=self.spec,
            trajectory_scorer=self.trajectory_scorer,
            score_baseline=_score_smash_correction,
            fps=context.fps,
        )
        return ScoredMotion(
            score=score,
            window=window,
            scoring_sample=context.sample,
            correction=context.correction,
            corrected_pixels=corrected_pixels,
            checkpoint_scorer_active=True,
        )


class LegacySmashScorer(SkillScorer):
    """The EIMD-v3 semantic scorer, kept to replay the reviewed cohort."""

    def __init__(self, definition, **config):
        super().__init__(definition, **config)
        semantic_path = self.root / "expert_semantic_score_model.npz"
        self.distribution, self.variant = (
            load_smash_distribution(semantic_path)
            if semantic_path.exists()
            else (None, None)
        )
        trajectory_path = self.root / "expert_trajectory_score_model.npz"
        self.trajectory_scorer = (
            load_smash_trajectory_scorer(trajectory_path)
            if trajectory_path.exists()
            else None
        )

    def score(self, context: ScoringContext) -> ScoredMotion:
        correction, view_rotation = self.view_aligned(context.correction)
        score = score_expert_correction(self.score_model, correction)
        if self.distribution is not None and self.variant is not None:
            score = _score_smash_correction(
                score,
                context.sample,
                correction,
                distribution=self.distribution,
                variant=self.variant,
                trajectory_scorer=self.trajectory_scorer,
                spec=self.spec,
            )
        return ScoredMotion(
            score=score,
            window=context.window,
            scoring_sample=context.sample,
            correction=correction,
            view_rotation=view_rotation,
        )


def create_scorer(definition, *, current_scorer=True, **config):
    return (SmashScorer if current_scorer else LegacySmashScorer)(definition, **config)

