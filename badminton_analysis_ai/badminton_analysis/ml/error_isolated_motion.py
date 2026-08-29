"""Expert-only Error-Isolated Motion Diffusion (EIMD).

EIMD learns which joint-phase conditioning tokens are trustworthy by
synthetically corrupting expert motions while retaining the clean expert
sequence as the target. Learner data is never accepted by training functions.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
import torch
from torch import Tensor

from badminton_analysis.ml.expert_motion_generator import (
    CONTACT_DIM,
    DIRECTION_DIM,
    FRAMES,
    JOINTS,
    ROOT_DIM,
    STATE_DIM,
    _aligned,
    _device,
    _fk_from_directions,
    _unit,
    dominant_wrist_velocities,
    expert_wrist_velocity_limit,
    limit_correction_wrist_velocity,
    motion_features,
    project_to_expert_motion_subspace,
    smooth_generated_motion_state,
)
from badminton_analysis.ml.expert_phase_baseline import (
    ExpertCorrection,
    MotionSample,
    _retarget_root_with_contacts,
)
from badminton_analysis.ml.kinematic_retargeting import (
    retarget_expert_canonical_2d_fk,
)
from badminton_analysis.ml.models.error_isolated_motion_diffusion import (
    ErrorIsolatedMotionDenoiser,
)
from badminton_analysis.ml.models.expert_motion_diffusion import (
    linear_diffusion_schedule,
)
from badminton_analysis.ml.skeleton_normalization import (
    CANONICAL_PHASE_INDICES,
    restore_phase_timing,
)


CONDITION_PHASES = 4
CORRUPTION_TYPES = ("rotation", "occlusion", "reflection", "phase_shift")
_EPS = 1e-8


@dataclass(frozen=True)
class ErrorIsolationLossWeights:
    denoising: float = 1.0
    reconstruction: float = 0.10
    invariance: float = 0.20
    reliability: float = 0.15
    condition_fidelity: float = 0.10
    velocity: float = 0.05
    acceleration: float = 0.05
    direction_unit: float = 0.02
    contacts: float = 0.05


@dataclass(frozen=True)
class CorruptionCurriculum:
    start_probability: float = 0.12
    end_probability: float = 0.45
    warmup_fraction: float = 0.70
    minimum_rotation_radians: float = 0.60
    maximum_rotation_radians: float = 2.60
    clean_direction_noise: float = 0.015


@dataclass(frozen=True)
class ErrorIsolatedMotionBundle:
    skill: str
    network: ErrorIsolatedMotionDenoiser
    diffusion_steps: int
    state_mean: NDArray[np.float32]
    state_scale: NDArray[np.float32]
    morphology_mean: NDArray[np.float32]
    morphology_scale: NDArray[np.float32]
    canonical_lengths: NDArray[np.float32]
    expert_states: NDArray[np.float32]
    expert_files: NDArray[np.str_]
    expert_subject_ids: NDArray[np.str_]
    training_manifest_sha256: str
    canonical_phase_indices: NDArray[np.int64]
    expert_canonical_wrist_velocity_limit: float
    expert_canonical_joint_velocity_limit: float
    expert_phase_duration_min: NDArray[np.int64]
    expert_phase_duration_max: NDArray[np.int64]
    expert_wrist_velocity_limit: float
    loss_weights: ErrorIsolationLossWeights
    corruption_curriculum: CorruptionCurriculum


def _validate_canonical_phase_indices(
    indices: NDArray[np.integer],
) -> NDArray[np.int64]:
    canonical = np.asarray(indices, dtype=np.int64)
    if canonical.shape != (5,):
        raise ValueError("canonical phase indices must contain five anchors")
    if canonical[0] != 0 or canonical[-1] != FRAMES - 1:
        raise ValueError("canonical phase indices must span 64 frames")
    if np.any(np.diff(canonical) <= 0):
        raise ValueError("canonical phase indices must be strictly increasing")
    return canonical


def condition_phase_bounds(
    canonical_phase_indices: NDArray[np.integer],
) -> tuple[tuple[int, int], ...]:
    """Convert five inclusive anchors into four non-overlapping slices."""
    canonical = _validate_canonical_phase_indices(canonical_phase_indices)
    starts = np.concatenate((np.asarray((0,), dtype=np.int64), canonical[1:-1] + 1))
    ends = canonical[1:] + 1
    return tuple(
        (int(start), int(end)) for start, end in zip(starts, ends, strict=True)
    )


CONDITION_PHASE_BOUNDS = condition_phase_bounds(CANONICAL_PHASE_INDICES)






def stabilize_phase_timing(
    phase_indices: NDArray[np.integer],
    duration_min: NDArray[np.integer],
    duration_max: NDArray[np.integer],
    *,
    canonical_phase_indices: NDArray[np.integer],
) -> NDArray[np.int64]:
    """Project an impossible learner schedule onto the expert timing envelope."""
    phases = _validate_canonical_phase_indices(phase_indices)
    lower = np.asarray(duration_min, dtype=np.int64)
    upper = np.asarray(duration_max, dtype=np.int64)
    if lower.shape != (4,) or upper.shape != (4,):
        raise ValueError("phase-duration bounds must contain four intervals")
    if np.any(lower <= 0) or np.any(upper < lower):
        raise ValueError("phase-duration bounds must be positive and ordered")
    target = np.diff(phases)
    # Minor phase-detector variation is harmless and should preserve the
    # learner/video alignment.  Projection is reserved for a severely
    # compressed preparation interval, the failure that skips the generated
    # expert motion from its beginning directly into the swing ending.
    severely_compressed = bool(np.any(target[:2] < 0.5 * lower[:2]))
    if not severely_compressed:
        return phases
    canonical_duration = np.diff(
        _validate_canonical_phase_indices(canonical_phase_indices)
    )
    feasible = [
        np.asarray(candidate, dtype=np.int64)
        for candidate in product(
            *(range(int(low), int(high) + 1) for low, high in zip(lower, upper))
        )
        if sum(candidate) == FRAMES - 1
    ]
    if not feasible:
        raise ValueError("expert phase-duration envelope has no 64-frame schedule")
    durations = min(
        feasible,
        key=lambda candidate: (
            int(np.square(candidate - target).sum()),
            int(np.square(candidate - canonical_duration).sum()),
            tuple(int(value) for value in candidate),
        ),
    )
    return np.concatenate(
        (np.asarray((0,), dtype=np.int64), np.cumsum(durations))
    )


def phase_joint_condition(
    sample: MotionSample,
    *,
    canonical_phase_indices: NDArray[np.integer] = CANONICAL_PHASE_INDICES,
) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
    """Summarize aligned motion into four joint-aware semantic phase tokens."""
    canonical = _validate_canonical_phase_indices(canonical_phase_indices)
    pose, confidence, _, _ = _aligned(
        sample, canonical_phase_indices=canonical
    )
    from badminton_analysis.ml.kinematic_retargeting import parent_offsets

    directions = _unit(parent_offsets(pose))
    phase_directions = []
    phase_confidence = []
    for start, end in condition_phase_bounds(canonical):
        weights = np.clip(confidence[start:end], 0.0, 1.0)
        weighted = np.sum(directions[start:end] * weights[..., None], axis=0)
        total = np.sum(weights, axis=0)
        mean = np.divide(
            weighted,
            total[:, None],
            out=np.zeros_like(weighted),
            where=total[:, None] > _EPS,
        )
        phase_directions.append(_unit(mean))
        phase_confidence.append(np.mean(weights, axis=0))
    return (
        np.stack(phase_directions).astype(np.float32),
        np.stack(phase_confidence).astype(np.float32),
    )


















def sample_error_isolated_diffusion(
    network: ErrorIsolatedMotionDenoiser,
    directions: Tensor,
    confidence: Tensor,
    morphology: Tensor,
    handedness: Tensor,
    *,
    candidates: int,
    diffusion_steps: int,
    sampling_steps: int = 25,
    seed: int = 19,
) -> tuple[Tensor, Tensor]:
    if candidates < 1:
        raise ValueError("candidates must be positive")
    device = next(network.parameters()).device
    with torch.no_grad():
        phase_tokens, reliability_logits = network.encode_condition(
            directions.to(device),
            confidence.to(device),
            morphology.to(device),
            handedness.to(device),
        )
    phase_tokens = phase_tokens.expand(candidates, -1, -1)
    cpu_generator = torch.Generator(device="cpu").manual_seed(seed)
    state = torch.randn(
        candidates,
        FRAMES,
        STATE_DIM,
        generator=cpu_generator,
        device="cpu",
    ).to(device)
    schedule = linear_diffusion_schedule(diffusion_steps, device=device)
    indices = torch.linspace(diffusion_steps - 1, 0, sampling_steps, device=device)
    indices = torch.unique_consecutive(indices.round().long())
    with torch.no_grad():
        for position, time_value in enumerate(indices):
            time = torch.full(
                (candidates,),
                int(time_value),
                device=device,
                dtype=torch.long,
            )
            predicted_noise = network.denoise(state, time, phase_tokens)
            alpha_bar = schedule["alpha_bar"][time_value]
            clean = (
                state - torch.sqrt(1.0 - alpha_bar) * predicted_noise
            ) / torch.sqrt(alpha_bar)
            clean = clean.clamp(-7.0, 7.0)
            if position == len(indices) - 1:
                state = clean
            else:
                previous_alpha_bar = schedule["alpha_bar"][indices[position + 1]]
                state = (
                    torch.sqrt(previous_alpha_bar) * clean
                    + torch.sqrt(1.0 - previous_alpha_bar) * predicted_noise
                )
    return state, torch.sigmoid(reliability_logits)








def load_error_isolated_bundle(
    path: str | Path, *, device: str = "auto"
) -> ErrorIsolatedMotionBundle:
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    format_version = int(checkpoint.get("format_version", -1))
    if format_version not in (1, 2, 3):
        raise ValueError("unsupported error-isolated checkpoint format")
    if checkpoint.get("student_data_used", True):
        raise ValueError("checkpoint does not certify expert-only training")
    expected_method = f"expert_only_error_isolated_motion_diffusion_v{format_version}"
    if checkpoint.get("method") != expected_method:
        raise ValueError("checkpoint method is not EIMD")
    canonical = _validate_canonical_phase_indices(
        np.asarray(
            checkpoint.get(
                "canonical_phase_indices", CANONICAL_PHASE_INDICES.tolist()
            ),
            dtype=np.int64,
        )
    )
    if tuple(tuple(pair) for pair in checkpoint["condition_phase_bounds"]) != (
        condition_phase_bounds(canonical)
    ):
        raise ValueError("checkpoint phase-conditioning contract changed")
    network = ErrorIsolatedMotionDenoiser(**checkpoint["network_config"])
    network.load_state_dict(checkpoint["state_dict"])
    network.to(_device(device)).eval()

    def array(key: str) -> NDArray[np.float32]:
        return checkpoint[key].detach().cpu().numpy().astype(np.float32)

    loss_values = dict(checkpoint["loss_weights"])
    if format_version == 1:
        loss_values.setdefault("condition_fidelity", 0.0)
    return ErrorIsolatedMotionBundle(
        skill=str(checkpoint["skill"]),
        network=network,
        diffusion_steps=int(checkpoint["diffusion_steps"]),
        state_mean=array("state_mean"),
        state_scale=array("state_scale"),
        morphology_mean=array("morphology_mean"),
        morphology_scale=array("morphology_scale"),
        canonical_lengths=array("canonical_lengths"),
        expert_states=array("expert_states"),
        expert_files=np.asarray(checkpoint["expert_files"]),
        expert_subject_ids=np.asarray(checkpoint["expert_subject_ids"]),
        training_manifest_sha256=str(checkpoint["training_manifest_sha256"]),
        canonical_phase_indices=canonical,
        expert_canonical_wrist_velocity_limit=float(
            checkpoint.get(
                "expert_canonical_wrist_velocity_limit",
                checkpoint["expert_wrist_velocity_limit"],
            )
        ),
        expert_canonical_joint_velocity_limit=float(
            checkpoint.get("expert_canonical_joint_velocity_limit", float("inf"))
        ),
        expert_phase_duration_min=np.asarray(
            checkpoint.get("expert_phase_duration_min", (1, 1, 1, 1)),
            dtype=np.int64,
        ),
        expert_phase_duration_max=np.asarray(
            checkpoint.get("expert_phase_duration_max", (60, 60, 60, 60)),
            dtype=np.int64,
        ),
        expert_wrist_velocity_limit=float(
            checkpoint["expert_wrist_velocity_limit"]
        ),
        loss_weights=ErrorIsolationLossWeights(**loss_values),
        corruption_curriculum=CorruptionCurriculum(
            **checkpoint["corruption_curriculum"]
        ),
    )


def _inference_features(
    bundle: ErrorIsolatedMotionBundle,
    sample: MotionSample,
    *,
    condition_sample: MotionSample | None = None,
) -> tuple[dict[str, NDArray[np.float32]], float]:
    """Separate target morphology from the optional motion condition.

    ``condition_sample`` is a research-only positive-control hook. Normal
    inference leaves it unset. Supplying another same-skill expert lets the
    evaluation test valid-condition responsiveness while holding the target
    person's morphology and camera coordinates fixed.
    """
    condition_source = sample if condition_sample is None else condition_sample
    if condition_source.skill != sample.skill:
        raise ValueError("condition sample must have the same skill as target sample")
    base = motion_features(
        sample,
        conditioning_policy="full_pose",
        canonical_phase_indices=bundle.canonical_phase_indices,
    )
    directions, confidence = phase_joint_condition(
        condition_source,
        canonical_phase_indices=bundle.canonical_phase_indices,
    )
    return (
        {
            "morphology": (
                base.morphology - bundle.morphology_mean
            )
            / bundle.morphology_scale,
            "handedness": base.handedness,
            "condition_directions": directions,
            "condition_confidence": confidence,
        },
        base.body_scale,
    )




def _apply_reliable_condition_guidance(
    generated: NDArray[np.float32],
    condition_directions: NDArray[np.float32],
    condition_confidence: NDArray[np.float32],
    reliability: NDArray[np.float32],
    state_mean: NDArray[np.float32],
    state_scale: NDArray[np.float32],
    *,
    strength: float,
    reliability_threshold: float = 0.5,
    canonical_phase_indices: NDArray[np.integer] = CANONICAL_PHASE_INDICES,
) -> NDArray[np.float32]:
    """Guide only high-reliability directions before manifold projection."""
    if not 0.0 <= strength <= 1.0:
        raise ValueError("condition guidance strength must be in [0, 1]")
    if not 0.0 <= reliability_threshold < 1.0:
        raise ValueError("reliability threshold must be in [0, 1)")
    if strength == 0.0:
        return generated
    raw = generated * state_scale[None] + state_mean[None]
    directions = raw[..., :DIRECTION_DIM].reshape(
        len(raw), FRAMES, JOINTS, 2
    )
    # Reliability below chance receives no guidance. Above chance, trust rises
    # linearly and is also bounded by observation confidence.
    trust = np.clip(
        (reliability - reliability_threshold)
        / (1.0 - reliability_threshold),
        0.0,
        1.0,
    )
    trust *= np.clip(condition_confidence, 0.0, 1.0)
    for phase, (start, end) in enumerate(
        condition_phase_bounds(canonical_phase_indices)
    ):
        weight = (strength * trust[phase])[None, None, :, None]
        target = condition_directions[phase][None, None]
        directions[:, start:end] = _unit(
            (1.0 - weight) * directions[:, start:end] + weight * target
        )
    raw[..., :DIRECTION_DIM] = directions.reshape(
        len(raw), FRAMES, DIRECTION_DIM
    )
    return ((raw - state_mean[None]) / state_scale[None]).astype(np.float32)


def _candidate_canonical_velocities(
    generated: NDArray[np.float32],
    bundle: ErrorIsolatedMotionBundle,
) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
    """Measure generated candidates after the same smoothing used at output."""
    wrist = []
    joint = []
    for standardized in np.asarray(generated, dtype=np.float32):
        state = standardized * bundle.state_scale + bundle.state_mean
        directions = _unit(
            state[:, :DIRECTION_DIM].reshape(FRAMES, JOINTS, 2)
        )
        root = state[:, DIRECTION_DIM : DIRECTION_DIM + ROOT_DIM]
        contacts = np.clip(state[:, -CONTACT_DIM:], 0.0, 1.0)
        directions, _, _ = smooth_generated_motion_state(
            directions, root, contacts
        )
        pose = _fk_from_directions(directions, bundle.canonical_lengths)
        wrist.append(float(np.max(dominant_wrist_velocities(pose))))
        joint.append(
            float(np.linalg.norm(np.diff(pose, axis=0), axis=-1).max())
        )
    return np.asarray(wrist, dtype=np.float32), np.asarray(
        joint, dtype=np.float32
    )


def correct_student_motion_error_isolated(
    bundle: ErrorIsolatedMotionBundle,
    sample: MotionSample,
    *,
    candidates: int = 8,
    seed: int = 19,
    condition_sample: MotionSample | None = None,
    project_to_manifold: bool = True,
    condition_guidance_strength: float = 0.0,
    condition_guidance_threshold: float = 0.5,
) -> ExpertCorrection:
    """Generate a gated expert target and retarget it through learner FK."""
    if sample.skill != bundle.skill:
        raise ValueError(
            f"student skill {sample.skill!r} does not match {bundle.skill!r}"
        )
    features, body_scale = _inference_features(
        bundle, sample, condition_sample=condition_sample
    )
    generated, reliability = sample_error_isolated_diffusion(
        bundle.network,
        torch.as_tensor(features["condition_directions"])[None],
        torch.as_tensor(features["condition_confidence"])[None],
        torch.as_tensor(features["morphology"])[None],
        torch.as_tensor(features["handedness"])[None],
        candidates=candidates,
        diffusion_steps=bundle.diffusion_steps,
        seed=seed,
    )
    generated_array = generated.detach().cpu().numpy().astype(np.float32)
    generated_array = _apply_reliable_condition_guidance(
        generated_array,
        features["condition_directions"],
        features["condition_confidence"],
        reliability[0].detach().cpu().numpy().astype(np.float32),
        bundle.state_mean,
        bundle.state_scale,
        strength=condition_guidance_strength,
        reliability_threshold=condition_guidance_threshold,
        canonical_phase_indices=bundle.canonical_phase_indices,
    )
    if project_to_manifold:
        generated_array = project_to_expert_motion_subspace(
            generated_array, bundle.expert_states
        )
    manifold = np.mean(
        np.square(
            generated_array[:, None] - bundle.expert_states[None]
        ),
        axis=(2, 3),
    ).min(axis=1)
    acceleration = np.diff(
        generated_array[:, :, :DIRECTION_DIM], n=2, axis=1
    )
    smoothness = np.mean(np.square(acceleration), axis=(1, 2))
    candidate_wrist_velocity, candidate_joint_velocity = (
        _candidate_canonical_velocities(generated_array, bundle)
    )
    wrist_limit = bundle.expert_canonical_wrist_velocity_limit
    joint_limit = bundle.expert_canonical_joint_velocity_limit
    admissible = (
        candidate_wrist_velocity <= wrist_limit * (1.0 + 1e-5)
    ) & (candidate_joint_velocity <= joint_limit * (1.0 + 1e-5))
    objective = manifold + 0.025 * smoothness
    if np.any(admissible):
        objective = np.where(admissible, objective, np.inf)
    else:
        wrist_excess = np.maximum(
            candidate_wrist_velocity / max(wrist_limit, _EPS) - 1.0,
            0.0,
        )
        joint_excess = np.maximum(
            candidate_joint_velocity / max(joint_limit, _EPS) - 1.0,
            0.0,
        )
        objective = objective + 10.0 * (
            np.square(wrist_excess) + np.square(joint_excess)
        )
    selected = int(np.argmin(objective))
    standardized = generated_array[selected]
    state = standardized * bundle.state_scale + bundle.state_mean
    directions = _unit(
        state[:, :DIRECTION_DIM].reshape(FRAMES, JOINTS, 2)
    )
    normalized_root = state[
        :, DIRECTION_DIM : DIRECTION_DIM + ROOT_DIM
    ]
    contacts = np.clip(state[:, -CONTACT_DIM:], 0.0, 1.0).astype(np.float32)
    directions, normalized_root, contacts = smooth_generated_motion_state(
        directions, normalized_root, contacts
    )
    generated_pose = _fk_from_directions(
        directions, bundle.canonical_lengths
    )

    aligned_pose, aligned_confidence, aligned_root, _ = _aligned(
        sample,
        canonical_phase_indices=bundle.canonical_phase_indices,
    )
    corrected = retarget_expert_canonical_2d_fk(
        aligned_pose,
        generated_pose,
        aligned_confidence,
        np.ones_like(aligned_confidence),
        root_trajectory=np.zeros((FRAMES, 2), dtype=np.float32),
    )
    root_delta = normalized_root * body_scale
    corrected_root = aligned_root[:1] + root_delta - root_delta[:1]
    canonical_root = normalized_root * float(
        np.median(bundle.canonical_lengths)
    )
    corrected_root = _retarget_root_with_contacts(
        corrected, corrected_root, contacts, generated_pose
    )
    distances = np.mean(
        np.square(bundle.expert_states - standardized[None]), axis=(1, 2)
    )
    nearest = np.argsort(distances, kind="stable")[: min(3, len(distances))]
    denominator = max(float(np.median(distances[nearest])), 1e-4)
    weights = np.exp(-distances[nearest] / denominator)
    weights /= weights.sum()
    output_phase_indices = stabilize_phase_timing(
        sample.phase_indices,
        bundle.expert_phase_duration_min,
        bundle.expert_phase_duration_max,
        canonical_phase_indices=bundle.canonical_phase_indices,
    )
    correction = ExpertCorrection(
        student=sample,
        aligned_student_pose=aligned_pose,
        aligned_student_root=aligned_root,
        aligned_corrected_pose=corrected,
        aligned_corrected_root=corrected_root.astype(np.float32),
        corrected_pose=restore_phase_timing(
            corrected,
            output_phase_indices,
            canonical_indices=bundle.canonical_phase_indices,
        ),
        corrected_root=restore_phase_timing(
            corrected_root,
            output_phase_indices,
            canonical_indices=bundle.canonical_phase_indices,
        ),
        aligned_corrected_contacts=contacts,
        corrected_contacts=restore_phase_timing(
            contacts,
            output_phase_indices,
            canonical_indices=bundle.canonical_phase_indices,
        ),
        expert_prototype_pose=generated_pose,
        expert_prototype_root=canonical_root.astype(np.float32),
        reference_indices=nearest.astype(np.int64),
        reference_weights=weights.astype(np.float32),
        reference_distances=np.sqrt(distances[nearest]).astype(np.float32),
    )
    if sample.skill == "serve" and np.isfinite(
        bundle.expert_wrist_velocity_limit
    ):
        correction = limit_correction_wrist_velocity(
            correction,
            bundle.expert_wrist_velocity_limit,
            canonical_phase_indices=bundle.canonical_phase_indices,
            output_phase_indices=output_phase_indices,
        )
    return correction

