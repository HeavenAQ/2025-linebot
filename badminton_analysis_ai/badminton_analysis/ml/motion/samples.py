"""Motion archives, the frozen expert phase model and the correction record."""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import Any
import numpy as np
from numpy.typing import NDArray
from badminton_analysis.ml.skeleton_normalization import CANONICAL_PHASE_INDICES
from badminton_analysis.ml.skill_specs import SkillCorrectionSpec, get_skill_spec
from badminton_analysis.ml.video_annotations import expert_subject_identity


@dataclass(frozen=True)
class MotionSample:
    path: Path
    pose: NDArray[np.float32]
    confidence: NDArray[np.float32]
    root: NDArray[np.float32]
    foot_contacts: NDArray[np.float32]
    phase_indices: NDArray[np.int64]
    handedness: str
    skill: str
    video_name: str
    subject_id: str
    phase_source: str
    alignment_contract: str
    identity_level: str


@dataclass(frozen=True)
class ExpertPhaseModel:
    skill: str
    expert_pose: NDArray[np.float32]
    expert_confidence: NDArray[np.float32]
    expert_root: NDArray[np.float32]
    expert_foot_contacts: NDArray[np.float32]
    expert_features: NDArray[np.float32]
    feature_mean: NDArray[np.float32]
    feature_scale: NDArray[np.float32]
    expert_handedness: NDArray[np.str_]
    expert_files: NDArray[np.str_]
    expert_subject_ids: NDArray[np.str_]
    expert_identity_levels: NDArray[np.str_]
    expert_alignment_contracts: NDArray[np.str_]
    criterion_ids: NDArray[np.str_]
    criterion_tolerances: NDArray[np.float32]
    criterion_scales: NDArray[np.float32]
    top_k: int
    criterion_metric_version: str = "generic_joint_distance_v1"
    criterion_residual_tolerances: NDArray[np.float32] | None = None
    criterion_residual_scales: NDArray[np.float32] | None = None

    @property
    def spec(self) -> SkillCorrectionSpec:
        return get_skill_spec(self.skill)

    @cached_property
    def serve_angle_manifold(self):
        if self.skill != "serve":
            return None
        from badminton_analysis.ml.trajectory_distance import (
            fit_serve_angle_manifold,
        )

        return fit_serve_angle_manifold(self.expert_pose, self.expert_subject_ids)


@dataclass(frozen=True)
class ExpertCorrection:
    student: MotionSample
    aligned_student_pose: NDArray[np.float32]
    aligned_student_root: NDArray[np.float32]
    aligned_corrected_pose: NDArray[np.float32]
    aligned_corrected_root: NDArray[np.float32]
    corrected_pose: NDArray[np.float32]
    corrected_root: NDArray[np.float32]
    aligned_corrected_contacts: NDArray[np.float32]
    corrected_contacts: NDArray[np.float32]
    expert_prototype_pose: NDArray[np.float32]
    expert_prototype_root: NDArray[np.float32]
    reference_indices: NDArray[np.int64]
    reference_weights: NDArray[np.float32]
    reference_distances: NDArray[np.float32]
    timing_interpolation_method: str = "student_phase_timing"
    timing_sample_positions: NDArray[np.float32] | None = None
    wrist_velocity_limit: float | None = None
    maximum_wrist_velocity_before: float | None = None
    maximum_wrist_velocity_after: float | None = None
    maximum_body_velocity_before: float | None = None
    maximum_body_velocity_after: float | None = None


def _scalar_string(values: Any, key: str, default: str) -> str:
    if key not in values:
        return default
    value = values[key]
    return str(value.item() if hasattr(value, "item") else value)


def _pose_key(archive: Any, dimensions: int) -> str:
    if dimensions == 2:
        candidates = ("skeleton", "skeleton_2d")
    elif dimensions == 3:
        candidates = ("skeleton_3d",)
    else:
        raise ValueError("dimensions must be 2 or 3")
    for key in candidates:
        if key in archive:
            return key
    raise ValueError(f"archive does not contain a {dimensions}D skeleton")


def load_motion_sample(path: str | Path, *, dimensions: int = 2) -> MotionSample:
    """Load either the current or legacy skeleton archive schema."""
    source = Path(path)
    with np.load(source, allow_pickle=False) as archive:
        key = _pose_key(archive, dimensions)
        pose = np.asarray(archive[key], dtype=np.float32)
        confidence = np.asarray(archive["confidence"], dtype=np.float32)
        phases = np.asarray(archive["phase_indices"], dtype=np.int64)
        root_dimensions = pose.shape[-1]
        root = np.asarray(
            archive.get(
                "root_trajectory",
                np.zeros((len(pose), root_dimensions), dtype=np.float32),
            ),
            dtype=np.float32,
        )
        foot_contacts = np.asarray(
            archive.get(
                "foot_contacts",
                np.zeros((len(pose), 2), dtype=np.float32),
            ),
            dtype=np.float32,
        )
        handedness = _scalar_string(archive, "handedness", "right").lower()
        skill = _scalar_string(archive, "skill", "")
        video_name = _scalar_string(archive, "video_name", source.name)
        subject_id = _scalar_string(
            archive,
            "subject_id",
            expert_subject_identity(video_name),
        )
        if video_name.lower().startswith("expert-"):
            subject_id = expert_subject_identity(video_name)
        phase_source = _scalar_string(archive, "phase_source", "legacy_unversioned")
        has_subject_id = "subject_id" in archive
    if pose.ndim != 3 or pose.shape[1] != 17:
        raise ValueError(f"{source}: pose must have shape (T, 17, D)")
    if pose.shape[-1] != dimensions:
        raise ValueError(f"{source}: expected {dimensions}D pose")
    if confidence.shape != pose.shape[:2]:
        raise ValueError(f"{source}: confidence must have shape (T, 17)")
    if root.shape != (len(pose), dimensions):
        raise ValueError(f"{source}: root trajectory must have shape (T, D)")
    if foot_contacts.shape != (len(pose), 2):
        raise ValueError(f"{source}: foot contacts must have shape (T, 2)")
    if phases.shape != (5,) or np.any(np.diff(phases) <= 0):
        raise ValueError(f"{source}: five strictly increasing phases are required")
    if phases[0] < 0 or phases[-1] >= len(pose):
        raise ValueError(f"{source}: phase anchors are outside the pose sequence")
    if handedness not in {"left", "right"}:
        raise ValueError(f"{source}: handedness must be left or right")
    if phase_source == "acceleration_ending_range_v4":
        alignment_contract = "overhead_asymmetric_ending_range_v4"
    elif phase_source in {
        "acceleration_wrist_velocity_stop_v6",
        "acceleration_wrist_velocity_stop_delayed_contact_v7",
    }:
        alignment_contract = "overhead_wrist_velocity_stop_v6"
    elif skill == "serve" and phase_source in {"detected", "legacy_unversioned"}:
        alignment_contract = "serve_detector_proxy_anchors_v1"
    else:
        alignment_contract = phase_source
    return MotionSample(
        path=source,
        pose=pose,
        confidence=np.clip(confidence, 0.0, 1.0),
        root=root,
        foot_contacts=np.clip(foot_contacts, 0.0, 1.0),
        phase_indices=phases,
        handedness=handedness,
        skill=skill,
        video_name=video_name,
        subject_id=subject_id,
        phase_source=phase_source,
        alignment_contract=alignment_contract,
        identity_level="subject" if has_subject_id else "archive_fallback",
    )


def load_expert_phase_model(path: str | Path) -> ExpertPhaseModel:
    with np.load(path, allow_pickle=False) as archive:
        if int(archive["format_version"].item()) != 1:
            raise ValueError("unsupported expert phase model format")
        if not np.array_equal(
            archive["canonical_phase_indices"], CANONICAL_PHASE_INDICES
        ):
            raise ValueError("model canonical phases do not match this runtime")
        expert_pose = np.asarray(archive["expert_pose"], dtype=np.float32)
        expert_foot_contacts = (
            np.asarray(archive["expert_foot_contacts"], dtype=np.float32)
            if "expert_foot_contacts" in archive
            else np.zeros((*expert_pose.shape[:2], 2), dtype=np.float32)
        )
        return ExpertPhaseModel(
            skill=str(archive["skill"].item()),
            expert_pose=expert_pose,
            expert_confidence=np.asarray(
                archive["expert_confidence"], dtype=np.float32
            ),
            expert_root=np.asarray(archive["expert_root"], dtype=np.float32),
            expert_foot_contacts=expert_foot_contacts,
            expert_features=np.asarray(archive["expert_features"], dtype=np.float32),
            feature_mean=np.asarray(archive["feature_mean"], dtype=np.float32),
            feature_scale=np.asarray(archive["feature_scale"], dtype=np.float32),
            expert_handedness=np.asarray(archive["expert_handedness"]),
            expert_files=np.asarray(archive["expert_files"]),
            expert_subject_ids=np.asarray(archive["expert_subject_ids"]),
            expert_identity_levels=np.asarray(archive["expert_identity_levels"]),
            expert_alignment_contracts=np.asarray(
                archive["expert_alignment_contracts"]
            ),
            criterion_ids=np.asarray(archive["criterion_ids"]),
            criterion_tolerances=np.asarray(
                archive["criterion_tolerances"], dtype=np.float32
            ),
            criterion_scales=np.asarray(archive["criterion_scales"], dtype=np.float32),
            top_k=int(archive["top_k"].item()),
            criterion_metric_version=_scalar_string(
                archive,
                "criterion_metric_version",
                "generic_joint_distance_v1",
            ),
            criterion_residual_tolerances=(
                np.asarray(archive["criterion_residual_tolerances"], dtype=np.float32)
                if "criterion_residual_tolerances" in archive
                and archive["criterion_residual_tolerances"].size
                else None
            ),
            criterion_residual_scales=(
                np.asarray(archive["criterion_residual_scales"], dtype=np.float32)
                if "criterion_residual_scales" in archive
                and archive["criterion_residual_scales"].size
                else None
            ),
        )
