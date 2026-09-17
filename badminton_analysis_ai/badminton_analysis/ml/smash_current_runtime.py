"""Frozen September smash pipeline shared by serving and offline validation.

Only source poses and the generated correction enter grading. Rendering and
coaching consume the returned source-clock decisions instead of refitting them.
"""

from dataclasses import replace
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from badminton_analysis.ml.expert_phase_baseline import (
    ankle_spine_view_rotation,
    apply_fixed_hierarchical_pose_placement,
    project_pose_to_student_view,
)
from badminton_analysis.ml.skeleton_normalization import phase_align_sequence
from badminton_analysis.ml.smash_current_alignment import (
    align_contacts,
    observed_cost,
    observed_features,
    transfer_intervals,
)
from badminton_analysis.ml.smash_current_graph import (
    CheckpointMetricGraph,
    aggregate,
    checkpoint_index_map,
    checkpoint_reference_at_source_frames,
    checkpoint_windows,
    infer,
)
from badminton_analysis.ml.smash_current_placement import (
    first_frame_ankle_spine_map,
    first_frame_standing_offsets,
    smooth_corrected_bbox_placement,
    transport_corrected_by_student_displacement,
)
from badminton_analysis.ml.smash_current_scoring import (
    LEGACY_MAXIMA,
    MAXIMA,
    CurrentSmashCalibration,
    score_source_evidence,
)
from badminton_analysis.ml.smash_coaching_evidence import build_checkpoint_evidence
from badminton_analysis.ml.smash_expert_scoring import load_smash_distribution


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
        # Grade the full evidence first, then crop presentation at the selected
        # endpoint. Never resample motion or recompute scores on the cropped clip.
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
