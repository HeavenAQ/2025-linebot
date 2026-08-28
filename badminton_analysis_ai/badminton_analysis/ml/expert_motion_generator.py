"""Expert-only generative motion prior and personalized full-body inference.

Training is deliberately restricted to expert archives.  Student motion is
used only after a checkpoint has been frozen, to obtain static morphology,
preparation stance, phase timing, and the source camera transform.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Sequence

import numpy as np
from numpy.typing import NDArray
import torch

from badminton_analysis.ml.expert_phase_baseline import (
    ExpertCorrection,
    MotionSample,
)
from badminton_analysis.ml.kinematic_retargeting import (
    COCO_PARENTS,
    implicit_pelvis,
    parent_offsets,
    stable_parent_lengths,
)
from badminton_analysis.ml.skeleton_normalization import (
    CANONICAL_PHASE_INDICES,
    phase_align_sequence,
)


FRAMES = 64
JOINTS = 17
DIRECTION_DIM = JOINTS * 2
ROOT_DIM = 2
CONTACT_DIM = 2
STATE_DIM = DIRECTION_DIM + ROOT_DIM + CONTACT_DIM
MORPHOLOGY_DIM = JOINTS
# Static coordinate/stance conditioning deliberately excludes face, arms,
# elbows, and wrists. Those joints must come from the expert distribution so a
# learner's missing hand raise cannot be preserved as a target constraint.
STANCE_JOINTS = np.asarray((5, 6, 11, 12, 13, 14, 15, 16), dtype=np.int64)
STANCE_DIM = len(STANCE_JOINTS) * 2
CONDITIONING_POLICIES = ("selective", "full_pose", "morphology_only")
_EPS = 1e-8
_DOMINANT_SHOULDER = 6
_DOMINANT_WRIST = 10
_WRIST_VELOCITY_SAFETY_MARGIN = 1.05


@dataclass(frozen=True)
class MotionFeatures:
    state: NDArray[np.float32]
    morphology: NDArray[np.float32]
    stance: NDArray[np.float32]
    handedness: NDArray[np.float32]
    body_scale: float




def _aligned(
    sample: MotionSample,
    *,
    canonical_phase_indices: NDArray[np.integer] = CANONICAL_PHASE_INDICES,
) -> tuple[NDArray[np.float32], ...]:
    return (
        phase_align_sequence(
            sample.pose,
            sample.phase_indices,
            canonical_indices=canonical_phase_indices,
        ).astype(np.float32),
        np.clip(
            phase_align_sequence(
                sample.confidence,
                sample.phase_indices,
                canonical_indices=canonical_phase_indices,
            ),
            0.0,
            1.0,
        ).astype(np.float32),
        phase_align_sequence(
            sample.root,
            sample.phase_indices,
            canonical_indices=canonical_phase_indices,
        ).astype(np.float32),
        np.clip(
            phase_align_sequence(
                sample.foot_contacts,
                sample.phase_indices,
                canonical_indices=canonical_phase_indices,
            ),
            0.0,
            1.0,
        ).astype(np.float32),
    )


def _unit(values: NDArray[np.floating]) -> NDArray[np.float32]:
    array = np.asarray(values, dtype=np.float64)
    norms = np.linalg.norm(array, axis=-1, keepdims=True)
    return np.divide(
        array,
        norms,
        out=np.zeros_like(array),
        where=norms > _EPS,
    ).astype(np.float32)


def conditioning_stance_joints(policy: str) -> NDArray[np.int64]:
    """Return joints exposed to the generator for a named research ablation.

    ``selective`` is the proposed method: it exposes shoulders and lower-body
    preparation geometry while withholding arms, wrists and face. ``full_pose``
    is the leakage-prone baseline. ``morphology_only`` uses a constant stance
    token so only anatomy and handedness remain informative.
    """
    if policy == "selective":
        return STANCE_JOINTS.copy()
    if policy == "full_pose":
        return np.arange(JOINTS, dtype=np.int64)
    if policy == "morphology_only":
        return np.empty(0, dtype=np.int64)
    raise ValueError(
        f"unsupported conditioning policy {policy!r}; expected one of "
        f"{CONDITIONING_POLICIES}"
    )


def motion_features(
    sample: MotionSample,
    *,
    conditioning_policy: str = "selective",
    stance_joint_ids: Sequence[int] | None = None,
    canonical_phase_indices: NDArray[np.integer] = CANONICAL_PHASE_INDICES,
) -> MotionFeatures:
    """Encode one clip without using identity- or score-derived supervision."""
    canonical = np.asarray(canonical_phase_indices, dtype=np.int64)
    if canonical.shape != (5,) or canonical[0] != 0 or canonical[-1] != FRAMES - 1:
        raise ValueError("canonical phase indices must span 64 frames")
    if np.any(np.diff(canonical) <= 0):
        raise ValueError("canonical phase indices must be strictly increasing")
    pose, confidence, root, contacts = _aligned(
        sample, canonical_phase_indices=canonical
    )
    offsets = parent_offsets(pose)
    directions = _unit(offsets)
    preparation_end = int(canonical[1]) + 1
    # Morphology is estimated only from the preparation/standing interval.
    # A faulty or occluded swing must not change the body used by the prior.
    lengths = stable_parent_lengths(
        pose[:preparation_end], confidence[:preparation_end]
    )
    valid_lengths = lengths[np.isfinite(lengths) & (lengths > _EPS)]
    body_scale = float(np.median(valid_lengths)) if len(valid_lengths) else 1.0
    morphology = (lengths / max(body_scale, _EPS)).astype(np.float32)
    preparation = directions[:preparation_end]
    policy_joints = conditioning_stance_joints(conditioning_policy)
    joints = (
        policy_joints
        if stance_joint_ids is None
        else np.asarray(tuple(stance_joint_ids), dtype=np.int64)
    )
    if np.any((joints < 0) | (joints >= JOINTS)):
        raise ValueError("stance joint ids must be valid COCO joint indices")
    # A one-value constant token keeps the morphology-only architecture valid
    # without injecting any pose information through a learned stance branch.
    stance = (
        _unit(np.median(preparation[:, joints], axis=0)).reshape(-1)
        if len(joints)
        else np.zeros(1, dtype=np.float32)
    )
    root_delta = (root - root[:1]) / max(body_scale, _EPS)
    state = np.concatenate(
        (directions.reshape(FRAMES, -1), root_delta, contacts), axis=-1
    ).astype(np.float32)
    handedness = np.asarray(
        (1.0, 0.0) if sample.handedness == "right" else (0.0, 1.0),
        dtype=np.float32,
    )
    if not np.isfinite(state).all():
        raise ValueError(f"{sample.path}: non-finite expert motion state")
    return MotionFeatures(state, morphology, stance, handedness, body_scale)






def dominant_wrist_velocities(
    pose: NDArray[np.floating],
) -> NDArray[np.float32]:
    """Return root-invariant dominant-wrist displacement per output frame."""
    values = np.asarray(pose, dtype=np.float64)
    if values.shape != (FRAMES, JOINTS, 2):
        raise ValueError("pose must have shape (64, 17, 2)")
    relative_wrist = (
        values[:, _DOMINANT_WRIST] - values[:, _DOMINANT_SHOULDER]
    )
    return np.linalg.norm(np.diff(relative_wrist, axis=0), axis=-1).astype(
        np.float32
    )


def mean_joint_velocities(
    pose: NDArray[np.floating], root: NDArray[np.floating]
) -> NDArray[np.float32]:
    """Return mean full-body velocity, including generated root movement."""
    values = np.asarray(pose, dtype=np.float64)
    root_values = np.asarray(root, dtype=np.float64)
    if values.shape != (FRAMES, JOINTS, 2) or root_values.shape != (FRAMES, 2):
        raise ValueError("pose/root must have shapes (64, 17, 2)/(64, 2)")
    world = values + root_values[:, None, :]
    return np.linalg.norm(np.diff(world, axis=0), axis=-1).mean(1).astype(
        np.float32
    )


def expert_wrist_velocity_limit(
    samples: Sequence[MotionSample],
    *,
    safety_margin: float = _WRIST_VELOCITY_SAFETY_MARGIN,
) -> float:
    """Set a serve jump ceiling solely from observed expert demonstrations."""
    if not samples:
        raise ValueError("expert wrist calibration requires samples")
    if safety_margin < 1.0:
        raise ValueError("wrist velocity safety margin cannot be below one")
    maxima = [
        float(np.max(dominant_wrist_velocities(sample.pose)))
        for sample in samples
    ]
    limit = safety_margin * max(maxima)
    if not np.isfinite(limit) or limit <= 0.0:
        raise ValueError("expert wrist velocity limit must be finite and positive")
    return float(limit)










def _device(value: str) -> torch.device:
    if value == "auto":
        value = "mps" if torch.backends.mps.is_available() else "cpu"
    if value == "mps" and not torch.backends.mps.is_available():
        raise ValueError("MPS was requested but is unavailable")
    return torch.device(value)




















def _fk_from_directions(
    directions: NDArray[np.floating], lengths: NDArray[np.floating]
) -> NDArray[np.float32]:
    direction = _unit(directions)
    bone_lengths = np.asarray(lengths, dtype=np.float64).copy()
    output = np.zeros((len(direction), JOINTS, 2), dtype=np.float64)
    hip_axis = _unit(direction[:, 12] - direction[:, 11]).astype(np.float64)
    hip_half_width = 0.5 * (bone_lengths[11] + bone_lengths[12])
    output[:, 11] = -hip_half_width * hip_axis
    output[:, 12] = hip_half_width * hip_axis
    for joint, parent in enumerate(COCO_PARENTS):
        if joint in (11, 12):
            continue
        anchor = np.zeros((len(direction), 2)) if parent < 0 else output[:, parent]
        output[:, joint] = anchor + bone_lengths[joint] * direction[:, joint]
    return output.astype(np.float32)


def smooth_generated_motion_state(
    directions: NDArray[np.floating],
    root: NDArray[np.floating],
    contacts: NDArray[np.floating],
    *,
    passes: int = 2,
) -> tuple[NDArray[np.float32], NDArray[np.float32], NDArray[np.float32]]:
    """Remove high-frequency pose jitter before student-length FK.

    A symmetric binomial filter has zero phase delay, so semantic events do not
    move earlier or later. Filtering unit directions rather than Cartesian
    joints lets FK restore exact, stable bone lengths afterward.
    """
    if passes < 0:
        raise ValueError("passes cannot be negative")
    direction_values = np.asarray(directions, dtype=np.float64).copy()
    root_values = np.asarray(root, dtype=np.float64).copy()
    contact_values = np.asarray(contacts, dtype=np.float64).copy()
    kernel = np.asarray((1.0, 4.0, 6.0, 4.0, 1.0), dtype=np.float64) / 16.0

    def filtered(values: NDArray[np.float64]) -> NDArray[np.float64]:
        flattened = values.reshape(len(values), -1)
        padded = np.pad(flattened, ((2, 2), (0, 0)), mode="edge")
        output = np.stack(
            [
                np.convolve(padded[:, index], kernel, mode="valid")
                for index in range(flattened.shape[1])
            ],
            axis=-1,
        )
        return output.reshape(values.shape)

    for _ in range(passes):
        direction_values = filtered(direction_values)
        root_values = filtered(root_values)
    if passes:
        contact_values = filtered(contact_values)
    root_values -= root_values[:1]
    smoothed_directions = _unit(direction_values)
    degenerate = np.linalg.norm(direction_values, axis=-1) <= _EPS
    if np.any(degenerate):
        fallback = _unit(np.asarray(directions, dtype=np.float64))
        smoothed_directions[degenerate] = fallback[degenerate]
    return (
        smoothed_directions,
        root_values.astype(np.float32),
        np.clip(contact_values, 0.0, 1.0).astype(np.float32),
    )


def _sample_at_positions(
    sequence: NDArray[np.floating], positions: NDArray[np.floating]
) -> NDArray[np.float32]:
    values = np.asarray(sequence, dtype=np.float64)
    sample_positions = np.asarray(positions, dtype=np.float64)
    if len(values) != FRAMES or sample_positions.shape != (FRAMES,):
        raise ValueError("continuous timing samples must contain 64 frames")
    timeline = np.arange(FRAMES, dtype=np.float64)
    flattened = values.reshape(FRAMES, -1)
    sampled = np.stack(
        [
            np.interp(sample_positions, timeline, flattened[:, column])
            for column in range(flattened.shape[1])
        ],
        axis=-1,
    )
    return sampled.reshape(values.shape).astype(np.float32)


def _sample_pose_kinematically(
    sequence: NDArray[np.floating], positions: NDArray[np.floating]
) -> NDArray[np.float32]:
    """Interpolate edge directions/lengths, then reconstruct exact 2D FK."""
    values = np.asarray(sequence, dtype=np.float64)
    offsets = parent_offsets(values)
    lengths = np.linalg.norm(offsets, axis=-1)
    directions = _unit(offsets)
    sampled_directions = _unit(_sample_at_positions(directions, positions))
    sampled_lengths = _sample_at_positions(lengths, positions)
    pelvis = _sample_at_positions(implicit_pelvis(values), positions)
    output = np.empty_like(values)
    for joint, parent in enumerate(COCO_PARENTS):
        anchor = pelvis if parent < 0 else output[:, parent]
        output[:, joint] = (
            anchor
            + sampled_lengths[:, joint, None] * sampled_directions[:, joint]
        )
    return output.astype(np.float32)


def _rate_limited_sample_positions(
    aligned_pose: NDArray[np.floating],
    aligned_root: NDArray[np.floating],
    phase_indices: NDArray[np.integer],
    *,
    wrist_arc_step_limit: float,
    body_arc_step_limit: float,
    canonical_phase_indices: NDArray[np.integer] = CANONICAL_PHASE_INDICES,
) -> NDArray[np.float32]:
    """Advance pre-jump poses under wrist and whole-body arc ceilings."""
    if wrist_arc_step_limit <= 0.0 or body_arc_step_limit <= 0.0:
        raise ValueError("wrist/body arc step limits must be positive")
    phases = np.asarray(phase_indices, dtype=np.float64)
    if phases.shape != (5,) or np.any(np.diff(phases) <= 0):
        raise ValueError("phase indices must contain five increasing anchors")
    canonical = np.asarray(canonical_phase_indices, dtype=np.float64)
    if canonical.shape != (5,) or canonical[0] != 0 or canonical[-1] != FRAMES - 1:
        raise ValueError("canonical phase indices must span 64 frames")
    if np.any(np.diff(canonical) <= 0):
        raise ValueError("canonical phase indices must be strictly increasing")
    timeline = np.arange(FRAMES, dtype=np.float64)
    original_positions = np.interp(timeline, phases, canonical)

    pose = np.asarray(aligned_pose, dtype=np.float64)
    relative_wrist = (
        pose[:, _DOMINANT_WRIST] - pose[:, _DOMINANT_SHOULDER]
    )
    wrist_steps = np.linalg.norm(np.diff(relative_wrist, axis=0), axis=-1)
    body_steps = mean_joint_velocities(pose, aligned_root).astype(np.float64)
    normalized_steps = np.maximum(
        wrist_steps / wrist_arc_step_limit,
        body_steps / body_arc_step_limit,
    )
    cumulative_arc = np.concatenate(
        (
            np.zeros(1, dtype=np.float64),
            np.cumsum(normalized_steps),
        )
    )
    if cumulative_arc[-1] > FRAMES - 1 + 1e-6:
        raise ValueError("expert motion path cannot fit inside the velocity ceilings")
    original_arc = np.interp(original_positions, timeline, cumulative_arc)
    limited_arc = original_arc.copy()
    # Work backward from the exact ending. Raising an earlier arc position
    # makes the generated correction begin its swing sooner, while every
    # student event at and after the original jump remains fixed.
    for frame in range(FRAMES - 2, -1, -1):
        limited_arc[frame] = max(
            limited_arc[frame],
            limited_arc[frame + 1] - 1.0,
        )
    limited_arc = np.maximum.accumulate(limited_arc)

    unique_arc, unique_indices = np.unique(cumulative_arc, return_index=True)
    if len(unique_arc) < 2:
        return original_positions.astype(np.float32)
    positions = np.interp(
        limited_arc, unique_arc, timeline[unique_indices]
    ).astype(np.float32)
    positions[0] = 0.0
    positions[-1] = float(FRAMES - 1)
    return positions


def limit_correction_wrist_velocity(
    correction: ExpertCorrection,
    maximum_velocity: float,
    *,
    canonical_phase_indices: NDArray[np.integer] = CANONICAL_PHASE_INDICES,
    output_phase_indices: NDArray[np.integer] | None = None,
) -> ExpertCorrection:
    """Rate-limit only a serve correction that exceeds the expert ceiling."""
    before = float(
        np.max(dominant_wrist_velocities(correction.corrected_pose))
    )
    if correction.student.skill != "serve" or before <= maximum_velocity:
        return correction
    if not np.isfinite(maximum_velocity) or maximum_velocity <= 0.0:
        raise ValueError("maximum wrist velocity must be finite and positive")

    source_pose = np.asarray(correction.corrected_pose, dtype=np.float32)
    source_root = np.asarray(correction.corrected_root, dtype=np.float32)
    source_contacts = np.asarray(correction.corrected_contacts, dtype=np.float32)
    body_before = float(
        np.max(mean_joint_velocities(source_pose, source_root))
    )
    wrist_arc_limit = float(maximum_velocity)
    body_arc_limit = body_before
    best: tuple[
        NDArray[np.float32],
        NDArray[np.float32],
        NDArray[np.float32],
        NDArray[np.float32],
        float,
        float,
    ] | None = None
    timing_phases = (
        correction.student.phase_indices
        if output_phase_indices is None
        else np.asarray(output_phase_indices, dtype=np.int64)
    )
    for _ in range(8):
        positions = _rate_limited_sample_positions(
            correction.aligned_corrected_pose,
            correction.aligned_corrected_root,
            timing_phases,
            wrist_arc_step_limit=wrist_arc_limit,
            body_arc_step_limit=body_arc_limit,
            canonical_phase_indices=canonical_phase_indices,
        )
        pose = _sample_pose_kinematically(
            correction.aligned_corrected_pose, positions
        )
        root = _sample_at_positions(
            correction.aligned_corrected_root, positions
        )
        contacts = np.clip(
            _sample_at_positions(correction.aligned_corrected_contacts, positions),
            0.0,
            1.0,
        ).astype(np.float32)
        for frame in (0, FRAMES - 1):
            pose[frame] = source_pose[frame]
            root[frame] = source_root[frame]
            contacts[frame] = source_contacts[frame]
        after = float(np.max(dominant_wrist_velocities(pose)))
        body_after = float(np.max(mean_joint_velocities(pose, root)))
        best = (pose, root, contacts, positions, after, body_after)
        wrist_passes = after <= maximum_velocity * (1.0 + 1e-5)
        body_passes = body_after <= body_before * (1.0 + 1e-5)
        if wrist_passes and body_passes:
            break
        if not wrist_passes:
            wrist_arc_limit *= 0.98 * maximum_velocity / after
        if not body_passes:
            body_arc_limit *= 0.98 * body_before / body_after
    if (
        best is None
        or best[-2] > maximum_velocity * (1.0 + 1e-5)
        or best[-1] > body_before * (1.0 + 1e-5)
    ):
        raise RuntimeError("could not satisfy the expert motion velocity ceilings")
    pose, root, contacts, positions, after, body_after = best
    return replace(
        correction,
        corrected_pose=pose,
        corrected_root=root,
        corrected_contacts=contacts,
        timing_interpolation_method=(
            "expert_wrist_velocity_limited_arc_interpolation_v1"
        ),
        timing_sample_positions=positions,
        wrist_velocity_limit=float(maximum_velocity),
        maximum_wrist_velocity_before=before,
        maximum_wrist_velocity_after=after,
        maximum_body_velocity_before=body_before,
        maximum_body_velocity_after=body_after,
    )






def project_to_expert_motion_subspace(
    generated: NDArray[np.floating],
    expert_states: NDArray[np.floating],
    *,
    maximum_rank: int = 8,
) -> NDArray[np.float32]:
    """Project samples into a bounded low-rank expert motion distribution.

    Tiny expert banks leave diffusion weakly constrained in thousands of
    sequence dimensions.  The complete expert trajectories define a compact
    affine subspace; diffusion still chooses the coordinate in that subspace,
    but coordinates are bounded by the observed expert range and radius.  No
    single exemplar is selected or copied.
    """
    samples = np.asarray(generated, dtype=np.float64)
    experts = np.asarray(expert_states, dtype=np.float64)
    if samples.ndim != 3 or samples.shape[1:] != (FRAMES, STATE_DIM):
        raise ValueError("generated states must have shape (K, 64, state_dim)")
    if experts.ndim != 3 or experts.shape[1:] != samples.shape[1:]:
        raise ValueError("expert states must have shape (N, 64, state_dim)")
    flattened_experts = experts.reshape(len(experts), -1)
    center = flattened_experts.mean(axis=0)
    centered_experts = flattened_experts - center
    rank = min(maximum_rank, len(experts) - 1)
    if rank < 1:
        return np.broadcast_to(center, (len(samples), len(center))).reshape(
            samples.shape
        ).astype(np.float32)
    _, singular_values, right = np.linalg.svd(centered_experts, full_matrices=False)
    usable = int(min(rank, np.sum(singular_values > 1e-6)))
    basis = right[:usable]
    expert_coordinates = centered_experts @ basis.T
    coordinates = (samples.reshape(len(samples), -1) - center) @ basis.T
    lower = expert_coordinates.min(axis=0)
    upper = expert_coordinates.max(axis=0)
    coordinates = np.clip(coordinates, lower, upper)
    expert_radius = np.linalg.norm(expert_coordinates, axis=1)
    radius = np.linalg.norm(coordinates, axis=1)
    maximum_radius = max(float(np.quantile(expert_radius, 0.95)), _EPS)
    coordinates *= np.minimum(1.0, maximum_radius / np.maximum(radius, _EPS))[:, None]
    projected = center + coordinates @ basis
    return projected.reshape(samples.shape).astype(np.float32)




