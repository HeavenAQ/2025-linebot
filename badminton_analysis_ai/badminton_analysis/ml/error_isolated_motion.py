"""Expert-only Error-Isolated Motion Diffusion (EIMD).

EIMD learns which joint-phase conditioning tokens are trustworthy by
synthetically corrupting expert motions while retaining the clean expert
sequence as the target. Learner data is never accepted by training functions.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from itertools import product
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from numpy.typing import NDArray
import torch
from torch import Tensor
from torch.nn import functional as F

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
    assert_expert_only_training_data,
    dominant_wrist_velocities,
    expert_wrist_velocity_limit,
    limit_correction_wrist_velocity,
    motion_features,
    prepare_training_arrays,
    project_to_expert_motion_subspace,
    smooth_generated_motion_state,
    training_manifest,
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


def expert_canonical_phase_indices(
    samples: Sequence[MotionSample],
) -> NDArray[np.int64]:
    """Estimate the canonical phase schedule from training experts only."""
    if not samples:
        raise ValueError("canonical timing requires expert samples")
    phases = np.stack(
        [_validate_canonical_phase_indices(sample.phase_indices) for sample in samples]
    )
    canonical = np.rint(np.median(phases, axis=0)).astype(np.int64)
    canonical[0] = 0
    canonical[-1] = FRAMES - 1
    for index in range(1, len(canonical)):
        canonical[index] = max(canonical[index], canonical[index - 1] + 1)
    for index in range(len(canonical) - 2, -1, -1):
        canonical[index] = min(canonical[index], canonical[index + 1] - 1)
    return _validate_canonical_phase_indices(canonical)


def expert_phase_duration_bounds(
    samples: Sequence[MotionSample],
) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
    """Return per-phase duration bounds observed in training experts."""
    if not samples:
        raise ValueError("phase-duration bounds require expert samples")
    durations = np.stack(
        [
            np.diff(_validate_canonical_phase_indices(sample.phase_indices))
            for sample in samples
        ]
    )
    return durations.min(axis=0), durations.max(axis=0)


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


def prepare_error_isolated_arrays(
    samples: Sequence[MotionSample],
) -> tuple[dict[str, NDArray[np.float32]], dict[str, NDArray[np.float32]]]:
    """Create clean expert targets and expert-derived conditioning arrays."""
    if not samples:
        raise ValueError("error-isolated training requires expert samples")
    canonical = expert_canonical_phase_indices(samples)
    base, metadata = prepare_training_arrays(
        samples,
        conditioning_policy="full_pose",
        canonical_phase_indices=canonical,
    )
    metadata["canonical_phase_indices"] = canonical
    conditions = [
        phase_joint_condition(sample, canonical_phase_indices=canonical)
        for sample in samples
    ]
    arrays = {
        "state": base["state"],
        "morphology": base["morphology"],
        "handedness": base["handedness"],
        "condition_directions": np.stack([item[0] for item in conditions]),
        "condition_confidence": np.stack([item[1] for item in conditions]),
    }
    return arrays, metadata


def curriculum_probability(
    step: int, total_steps: int, curriculum: CorruptionCurriculum
) -> float:
    if total_steps < 1 or not 0 <= step < total_steps:
        raise ValueError("curriculum step must be inside the training range")
    warmup_steps = max(1, round(total_steps * curriculum.warmup_fraction))
    progress = min(step / warmup_steps, 1.0)
    return float(
        curriculum.start_probability
        + progress
        * (curriculum.end_probability - curriculum.start_probability)
    )


def corrupt_expert_condition(
    directions: Tensor,
    confidence: Tensor,
    *,
    probability: float,
    curriculum: CorruptionCurriculum,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Apply mixed joint/phase corruptions while preserving a reliability label."""
    if directions.ndim != 4 or directions.shape[-3:] != (
        CONDITION_PHASES,
        JOINTS,
        2,
    ):
        raise ValueError("directions must have shape (B, 4, 17, 2)")
    if confidence.shape != directions.shape[:-1]:
        raise ValueError("confidence must match direction joint tokens")
    if not 0.0 <= probability <= 1.0:
        raise ValueError("corruption probability must be in [0, 1]")
    batch = len(directions)
    device = directions.device
    joint_weights = torch.ones(JOINTS, device=device)
    joint_weights[[0, 1, 2, 3, 4]] = 0.65
    joint_weights[[7, 8, 9, 10]] = 1.35
    token_probability = (probability * joint_weights).clamp_max(0.85)
    mask = torch.rand(batch, CONDITION_PHASES, JOINTS, device=device)
    mask = mask < token_probability[None, None]
    empty = ~mask.flatten(1).any(dim=1)
    if probability > 0.0 and torch.any(empty):
        random_tokens = torch.randint(
            CONDITION_PHASES * JOINTS,
            (int(empty.sum()),),
            device=device,
        )
        rows = torch.nonzero(empty, as_tuple=False).flatten()
        mask[rows, random_tokens // JOINTS, random_tokens % JOINTS] = True

    noisy = directions + curriculum.clean_direction_noise * torch.randn_like(
        directions
    )
    noisy = F.normalize(noisy, dim=-1, eps=1e-6)
    corrupted = noisy.clone()
    corrupted_confidence = confidence.clone()
    type_value = torch.rand(mask.shape, device=device)
    type_index = torch.full(mask.shape, -1, device=device, dtype=torch.long)

    rotation = mask & (type_value < 0.50)
    angle_magnitude = curriculum.minimum_rotation_radians + torch.rand(
        mask.shape, device=device
    ) * (
        curriculum.maximum_rotation_radians
        - curriculum.minimum_rotation_radians
    )
    sign = torch.where(
        torch.rand(mask.shape, device=device) < 0.5,
        -torch.ones((), device=device),
        torch.ones((), device=device),
    )
    angle = angle_magnitude * sign
    cosine, sine = torch.cos(angle), torch.sin(angle)
    rotated = torch.stack(
        (
            noisy[..., 0] * cosine - noisy[..., 1] * sine,
            noisy[..., 0] * sine + noisy[..., 1] * cosine,
        ),
        dim=-1,
    )
    corrupted = torch.where(rotation[..., None], rotated, corrupted)
    type_index[rotation] = 0

    occlusion = mask & (type_value >= 0.50) & (type_value < 0.72)
    corrupted = torch.where(occlusion[..., None], torch.zeros_like(corrupted), corrupted)
    corrupted_confidence = torch.where(
        occlusion, torch.zeros_like(corrupted_confidence), corrupted_confidence
    )
    type_index[occlusion] = 1

    reflection = mask & (type_value >= 0.72) & (type_value < 0.86)
    reflected = corrupted.clone()
    reflected[..., 0] = -reflected[..., 0]
    corrupted = torch.where(reflection[..., None], reflected, corrupted)
    type_index[reflection] = 2

    phase_shift = mask & (type_value >= 0.86)
    shifted = torch.roll(noisy, shifts=1, dims=1)
    corrupted = torch.where(phase_shift[..., None], shifted, corrupted)
    type_index[phase_shift] = 3
    reliability = (~mask).to(directions.dtype)
    return corrupted, corrupted_confidence, reliability, type_index


def _raw_state(
    standardized: Tensor, state_mean: Tensor, state_scale: Tensor
) -> Tensor:
    return standardized * state_scale[None, None] + state_mean[None, None]


def _kinematic_losses(predicted: Tensor, target: Tensor) -> dict[str, Tensor]:
    direction = predicted[..., :DIRECTION_DIM].reshape(
        len(predicted), FRAMES, JOINTS, 2
    )
    velocity = predicted[:, 1:] - predicted[:, :-1]
    target_velocity = target[:, 1:] - target[:, :-1]
    acceleration = velocity[:, 1:] - velocity[:, :-1]
    target_acceleration = target_velocity[:, 1:] - target_velocity[:, :-1]
    return {
        "velocity": F.smooth_l1_loss(velocity, target_velocity),
        "acceleration": F.smooth_l1_loss(
            acceleration, target_acceleration
        ),
        "direction_unit": torch.mean(
            torch.square(torch.linalg.vector_norm(direction, dim=-1) - 1.0)
        ),
        "contacts": F.smooth_l1_loss(
            predicted[..., -CONTACT_DIM:].contiguous(),
            target[..., -CONTACT_DIM:].contiguous(),
        ),
    }


def _phase_direction_summary(
    state: Tensor,
    phase_bounds: tuple[tuple[int, int], ...],
) -> Tensor:
    directions = state[..., :DIRECTION_DIM].reshape(
        len(state), FRAMES, JOINTS, 2
    )
    return torch.stack(
        [
            F.normalize(directions[:, start:end].mean(dim=1), dim=-1, eps=1e-6)
            for start, end in phase_bounds
        ],
        dim=1,
    )


def _condition_fidelity_loss(
    predicted_state: Tensor,
    clean_condition: Tensor,
    reliability: Tensor,
    confidence: Tensor,
    phase_bounds: tuple[tuple[int, int], ...],
) -> Tensor:
    predicted_condition = _phase_direction_summary(predicted_state, phase_bounds)
    valid = reliability * (confidence > 0.10).to(reliability.dtype)
    error = F.smooth_l1_loss(
        predicted_condition, clean_condition, reduction="none"
    ).mean(dim=-1)
    return torch.sum(error * valid) / valid.sum().clamp_min(1.0)


def train_error_isolated_diffusion(
    arrays: dict[str, NDArray[np.float32]],
    metadata: dict[str, NDArray[np.float32]],
    *,
    steps: int,
    batch_size: int,
    learning_rate: float,
    diffusion_steps: int,
    device: str = "auto",
    seed: int = 19,
    model_dim: int = 96,
    layers: int = 4,
    diagnostic_layers: int = 2,
    fusion_layers: int = 2,
    use_reliability_gate: bool = True,
    loss_weights: ErrorIsolationLossWeights | None = None,
    curriculum: CorruptionCurriculum | None = None,
) -> tuple[ErrorIsolatedMotionDenoiser, list[dict[str, float]]]:
    """Fit EIMD on paired clean/corrupted views of expert motion only."""
    if steps < 1 or batch_size < 1:
        raise ValueError("steps and batch size must be positive")
    weights = loss_weights or ErrorIsolationLossWeights()
    corruption = curriculum or CorruptionCurriculum()
    torch.manual_seed(seed)
    np_rng = np.random.default_rng(seed)
    target_device = _device(device)
    network = ErrorIsolatedMotionDenoiser(
        morphology_dim=int(arrays["morphology"].shape[1]),
        model_dim=model_dim,
        layers=layers,
        diagnostic_layers=diagnostic_layers,
        fusion_layers=fusion_layers,
        use_reliability_gate=use_reliability_gate,
    ).to(target_device)
    optimizer = torch.optim.AdamW(
        network.parameters(), learning_rate, weight_decay=1e-4
    )
    schedule = linear_diffusion_schedule(diffusion_steps, device=target_device)
    phase_bounds = condition_phase_bounds(metadata["canonical_phase_indices"])
    state_mean = torch.as_tensor(metadata["state_mean"], device=target_device)
    state_scale = torch.as_tensor(metadata["state_scale"], device=target_device)
    history: list[dict[str, float]] = []
    network.train()
    record_interval = max(steps // 50, 1)
    for step in range(steps):
        indices = np_rng.integers(0, len(arrays["state"]), size=batch_size)
        clean = torch.as_tensor(arrays["state"][indices], device=target_device)
        morphology = torch.as_tensor(
            arrays["morphology"][indices], device=target_device
        )
        morphology = morphology + 0.08 * torch.randn_like(morphology)
        handedness = torch.as_tensor(
            arrays["handedness"][indices], device=target_device
        )
        directions = torch.as_tensor(
            arrays["condition_directions"][indices], device=target_device
        )
        confidence = torch.as_tensor(
            arrays["condition_confidence"][indices], device=target_device
        )
        probability = curriculum_probability(step, steps, corruption)
        (
            corrupted_directions,
            corrupted_confidence,
            corrupted_reliability,
            _,
        ) = corrupt_expert_condition(
            directions,
            confidence,
            probability=probability,
            curriculum=corruption,
        )
        clean_reliability = (confidence > 0.10).to(clean.dtype)
        corrupted_reliability = corrupted_reliability * clean_reliability

        diffusion_time = torch.randint(
            diffusion_steps, (batch_size,), device=target_device
        )
        noise = torch.randn_like(clean)
        alpha = schedule["sqrt_alpha_bar"][diffusion_time, None, None]
        sigma = schedule["sqrt_one_minus_alpha_bar"][
            diffusion_time, None, None
        ]
        noisy = alpha * clean + sigma * noise
        clean_prediction, clean_logits = network(
            noisy,
            diffusion_time,
            directions,
            confidence,
            morphology,
            handedness,
        )
        corrupted_prediction, corrupted_logits = network(
            noisy,
            diffusion_time,
            corrupted_directions,
            corrupted_confidence,
            morphology,
            handedness,
        )
        denoising = 0.5 * (
            F.mse_loss(clean_prediction, noise)
            + F.mse_loss(corrupted_prediction, noise)
        )
        predicted_clean_view = (
            noisy - sigma * clean_prediction
        ) / alpha.clamp_min(1e-4)
        predicted_corrupted_view = (
            noisy - sigma * corrupted_prediction
        ) / alpha.clamp_min(1e-4)
        reconstruction = F.smooth_l1_loss(predicted_corrupted_view, clean)
        invariance = F.smooth_l1_loss(
            predicted_corrupted_view, predicted_clean_view.detach()
        )
        reliability = 0.5 * (
            F.binary_cross_entropy_with_logits(
                clean_logits, clean_reliability
            )
            + F.binary_cross_entropy_with_logits(
                corrupted_logits, corrupted_reliability
            )
        )
        raw_prediction = _raw_state(
            predicted_corrupted_view, state_mean, state_scale
        )
        raw_target = _raw_state(clean, state_mean, state_scale)
        kinematic = _kinematic_losses(raw_prediction, raw_target)
        clean_fidelity = _condition_fidelity_loss(
            _raw_state(predicted_clean_view, state_mean, state_scale),
            directions,
            clean_reliability,
            confidence,
            phase_bounds,
        )
        corrupted_fidelity = _condition_fidelity_loss(
            raw_prediction,
            directions,
            corrupted_reliability,
            confidence,
            phase_bounds,
        )
        condition_fidelity = 0.5 * (clean_fidelity + corrupted_fidelity)
        loss = (
            weights.denoising * denoising
            + weights.reconstruction * reconstruction
            + weights.invariance * invariance
            + weights.reliability * reliability
            + weights.condition_fidelity * condition_fidelity
            + weights.velocity * kinematic["velocity"]
            + weights.acceleration * kinematic["acceleration"]
            + weights.direction_unit * kinematic["direction_unit"]
            + weights.contacts * kinematic["contacts"]
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(network.parameters(), 1.0)
        optimizer.step()
        if step % record_interval == 0 or step == steps - 1:
            history.append(
                {
                    "step": float(step),
                    "loss": float(loss.detach().cpu()),
                    "denoising": float(denoising.detach().cpu()),
                    "reconstruction": float(reconstruction.detach().cpu()),
                    "invariance": float(invariance.detach().cpu()),
                    "reliability": float(reliability.detach().cpu()),
                    "condition_fidelity": float(
                        condition_fidelity.detach().cpu()
                    ),
                    "velocity": float(kinematic["velocity"].detach().cpu()),
                    "acceleration": float(
                        kinematic["acceleration"].detach().cpu()
                    ),
                    "corruption_probability": probability,
                }
            )
    return network.eval(), history


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


def expert_canonical_velocity_limits(
    samples: Sequence[MotionSample],
    canonical_phase_indices: NDArray[np.integer],
) -> tuple[float, float]:
    """Return strict wrist and any-joint ceilings from aligned experts."""
    canonical = _validate_canonical_phase_indices(canonical_phase_indices)
    wrist_maxima = []
    joint_maxima = []
    for sample in samples:
        pose, _, _, _ = _aligned(
            sample, canonical_phase_indices=canonical
        )
        wrist_maxima.append(float(np.max(dominant_wrist_velocities(pose))))
        joint_maxima.append(
            float(np.linalg.norm(np.diff(pose, axis=0), axis=-1).max())
        )
    wrist_limit = max(wrist_maxima)
    joint_limit = max(joint_maxima)
    if min(wrist_limit, joint_limit) <= 0.0 or not np.isfinite(
        (wrist_limit, joint_limit)
    ).all():
        raise ValueError("expert canonical velocity limits must be finite and positive")
    return wrist_limit, joint_limit


def build_error_isolated_bundle(
    *,
    skill: str,
    network: ErrorIsolatedMotionDenoiser,
    diffusion_steps: int,
    arrays: dict[str, NDArray[np.float32]],
    metadata: dict[str, NDArray[np.float32]],
    samples: Sequence[MotionSample],
    loss_weights: ErrorIsolationLossWeights | None = None,
    corruption_curriculum: CorruptionCurriculum | None = None,
) -> ErrorIsolatedMotionBundle:
    canonical = _validate_canonical_phase_indices(
        metadata["canonical_phase_indices"]
    )
    canonical_wrist_limit, canonical_joint_limit = (
        expert_canonical_velocity_limits(samples, canonical)
    )
    duration_min, duration_max = expert_phase_duration_bounds(samples)
    return ErrorIsolatedMotionBundle(
        skill=skill,
        network=network,
        diffusion_steps=diffusion_steps,
        state_mean=metadata["state_mean"],
        state_scale=metadata["state_scale"],
        morphology_mean=metadata["morphology_mean"],
        morphology_scale=metadata["morphology_scale"],
        canonical_lengths=metadata["canonical_lengths"],
        expert_states=arrays["state"],
        expert_files=np.asarray([sample.path.name for sample in samples]),
        expert_subject_ids=np.asarray(
            [sample.subject_id for sample in samples]
        ),
        training_manifest_sha256=training_manifest(samples),
        canonical_phase_indices=canonical,
        expert_canonical_wrist_velocity_limit=canonical_wrist_limit,
        expert_canonical_joint_velocity_limit=canonical_joint_limit,
        expert_phase_duration_min=duration_min,
        expert_phase_duration_max=duration_max,
        expert_wrist_velocity_limit=expert_wrist_velocity_limit(samples),
        loss_weights=loss_weights or ErrorIsolationLossWeights(),
        corruption_curriculum=corruption_curriculum or CorruptionCurriculum(),
    )


def save_error_isolated_bundle(
    bundle: ErrorIsolatedMotionBundle, path: str | Path
) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "format_version": 3,
            "method": "expert_only_error_isolated_motion_diffusion_v3",
            "skill": bundle.skill,
            "diffusion_steps": bundle.diffusion_steps,
            "network_config": bundle.network.config(),
            "state_dict": bundle.network.state_dict(),
            "state_mean": torch.from_numpy(bundle.state_mean),
            "state_scale": torch.from_numpy(bundle.state_scale),
            "morphology_mean": torch.from_numpy(bundle.morphology_mean),
            "morphology_scale": torch.from_numpy(bundle.morphology_scale),
            "canonical_lengths": torch.from_numpy(bundle.canonical_lengths),
            "expert_states": torch.from_numpy(bundle.expert_states),
            "expert_files": bundle.expert_files.tolist(),
            "expert_subject_ids": bundle.expert_subject_ids.tolist(),
            "training_manifest_sha256": bundle.training_manifest_sha256,
            "canonical_phase_indices": bundle.canonical_phase_indices.tolist(),
            "expert_canonical_wrist_velocity_limit": (
                bundle.expert_canonical_wrist_velocity_limit
            ),
            "expert_canonical_joint_velocity_limit": (
                bundle.expert_canonical_joint_velocity_limit
            ),
            "expert_phase_duration_min": bundle.expert_phase_duration_min.tolist(),
            "expert_phase_duration_max": bundle.expert_phase_duration_max.tolist(),
            "expert_wrist_velocity_limit": bundle.expert_wrist_velocity_limit,
            "condition_phase_bounds": condition_phase_bounds(
                bundle.canonical_phase_indices
            ),
            "corruption_types": CORRUPTION_TYPES,
            "loss_weights": asdict(bundle.loss_weights),
            "corruption_curriculum": asdict(bundle.corruption_curriculum),
            "student_data_used": False,
        },
        destination,
    )


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


def conditioning_reliability(
    bundle: ErrorIsolatedMotionBundle, sample: MotionSample
) -> NDArray[np.float32]:
    features, _ = _inference_features(bundle, sample)
    device = next(bundle.network.parameters()).device
    with torch.no_grad():
        _, logits = bundle.network.encode_condition(
            torch.as_tensor(features["condition_directions"], device=device)[None],
            torch.as_tensor(features["condition_confidence"], device=device)[None],
            torch.as_tensor(features["morphology"], device=device)[None],
            torch.as_tensor(features["handedness"], device=device)[None],
        )
    return torch.sigmoid(logits)[0].cpu().numpy().astype(np.float32)


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
    candidates: int = 16,
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


def validate_expert_training_inputs(
    samples: Sequence[MotionSample], expert_root: str | Path
) -> None:
    """Public fail-closed training guard used by the EIMD command."""
    assert_expert_only_training_data(samples, expert_root)
