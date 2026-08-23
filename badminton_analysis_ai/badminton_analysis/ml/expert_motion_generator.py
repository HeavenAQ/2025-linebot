"""Expert-only generative motion prior and personalized full-body inference.

Training is deliberately restricted to expert archives.  Student motion is
used only after a checkpoint has been frozen, to obtain static morphology,
preparation stance, phase timing, and the source camera transform.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
from pathlib import Path
import random
from typing import Any, Sequence

import numpy as np
from numpy.typing import NDArray
import torch
from torch import Tensor
from torch.nn import functional as F

from badminton_analysis.ml.expert_phase_baseline import (
    ExpertCorrection,
    ExpertPhaseModel,
    MotionSample,
    _criterion_components_for_spec,
    _retarget_root_with_contacts,
)
from badminton_analysis.ml.kinematic_retargeting import (
    COCO_PARENTS,
    implicit_pelvis,
    parent_offsets,
    retarget_expert_canonical_2d_fk,
    stable_parent_lengths,
)
from badminton_analysis.ml.models.expert_motion_diffusion import (
    ExpertMotionDenoiser,
    ExpertMotionGANDiscriminator,
    ExpertMotionGANGenerator,
    linear_diffusion_schedule,
)
from badminton_analysis.ml.skeleton_normalization import (
    CANONICAL_PHASE_INDICES,
    phase_align_sequence,
    restore_phase_timing,
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


@dataclass(frozen=True)
class ExpertMotionBundle:
    skill: str
    method: str
    network: ExpertMotionDenoiser | ExpertMotionGANGenerator
    diffusion_steps: int
    state_mean: NDArray[np.float32]
    state_scale: NDArray[np.float32]
    morphology_mean: NDArray[np.float32]
    morphology_scale: NDArray[np.float32]
    stance_mean: NDArray[np.float32]
    stance_scale: NDArray[np.float32]
    canonical_lengths: NDArray[np.float32]
    expert_states: NDArray[np.float32]
    expert_files: NDArray[np.str_]
    expert_subject_ids: NDArray[np.str_]
    training_manifest_sha256: str
    expert_wrist_velocity_limit: float
    conditioning_policy: str = "selective"
    stance_joint_ids: tuple[int, ...] = tuple(int(value) for value in STANCE_JOINTS)


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


def assert_expert_only_training_data(
    samples: Sequence[MotionSample], expert_root: str | Path
) -> None:
    """Fail closed if a learner archive reaches a training command."""
    root = Path(expert_root).resolve()
    if root.name.lower() != "experts":
        raise ValueError("expert training root must be a directory named 'experts'")
    if not samples:
        raise ValueError("expert-only training requires at least two archives")
    forbidden = {"beginner", "beginners", "student", "students", "初學者"}
    for sample in samples:
        resolved = sample.path.resolve()
        if not resolved.is_relative_to(root):
            raise ValueError(f"training archive is outside expert root: {sample.path}")
        lowered_parts = {part.lower() for part in resolved.parts}
        if lowered_parts & forbidden:
            raise ValueError(f"student archive is forbidden in training: {sample.path}")
        if not sample.subject_id:
            raise ValueError(f"expert identity is missing: {sample.path}")


def training_manifest(samples: Sequence[MotionSample]) -> str:
    digest = hashlib.sha256()
    for sample in sorted(samples, key=lambda value: str(value.path)):
        digest.update(sample.path.name.encode("utf-8"))
        digest.update(sample.subject_id.encode("utf-8"))
        digest.update(hashlib.sha256(sample.path.read_bytes()).digest())
    return digest.hexdigest()


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


def subject_disjoint_split(
    samples: Sequence[MotionSample], *, seed: int = 19, validation_fraction: float = 0.2
) -> tuple[list[MotionSample], list[MotionSample]]:
    subjects = sorted({sample.subject_id for sample in samples})
    if len(subjects) < 2:
        raise ValueError("subject-disjoint validation requires at least two experts")
    rng = random.Random(seed)
    rng.shuffle(subjects)
    count = max(1, min(len(subjects) - 1, round(len(subjects) * validation_fraction)))
    held_out = set(subjects[:count])
    train = [sample for sample in samples if sample.subject_id not in held_out]
    validation = [sample for sample in samples if sample.subject_id in held_out]
    return train, validation


def _stats(values: NDArray[np.floating], minimum: float) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
    array = np.asarray(values, dtype=np.float64)
    mean = array.mean(axis=tuple(range(array.ndim - 1)))
    scale = array.std(axis=tuple(range(array.ndim - 1)))
    return mean.astype(np.float32), np.maximum(scale, minimum).astype(np.float32)


def prepare_training_arrays(
    samples: Sequence[MotionSample],
    *,
    conditioning_policy: str = "selective",
    canonical_phase_indices: NDArray[np.integer] = CANONICAL_PHASE_INDICES,
) -> tuple[dict[str, NDArray[np.float32]], dict[str, NDArray[np.float32]]]:
    encoded = [
        motion_features(
            sample,
            conditioning_policy=conditioning_policy,
            canonical_phase_indices=canonical_phase_indices,
        )
        for sample in samples
    ]
    state = np.stack([item.state for item in encoded])
    morphology = np.stack([item.morphology for item in encoded])
    stance = np.stack([item.stance for item in encoded])
    handedness = np.stack([item.handedness for item in encoded])
    state_mean, state_scale = _stats(state, 0.05)
    morphology_mean, morphology_scale = _stats(morphology, 0.025)
    stance_mean, stance_scale = _stats(stance, 0.05)
    arrays = {
        "state": ((state - state_mean) / state_scale).astype(np.float32),
        "morphology": ((morphology - morphology_mean) / morphology_scale).astype(np.float32),
        "stance": ((stance - stance_mean) / stance_scale).astype(np.float32),
        "handedness": handedness.astype(np.float32),
    }
    metadata = {
        "state_mean": state_mean,
        "state_scale": state_scale,
        "morphology_mean": morphology_mean,
        "morphology_scale": morphology_scale,
        "stance_mean": stance_mean,
        "stance_scale": stance_scale,
        "canonical_lengths": np.median(
            np.stack(
                [item.morphology * item.body_scale for item in encoded]
            ),
            axis=0,
        ).astype(np.float32),
    }
    return arrays, metadata


def transform_features(
    features: MotionFeatures, metadata: dict[str, NDArray[np.float32]]
) -> dict[str, NDArray[np.float32]]:
    return {
        "state": (features.state - metadata["state_mean"]) / metadata["state_scale"],
        "morphology": (features.morphology - metadata["morphology_mean"])
        / metadata["morphology_scale"],
        "stance": (features.stance - metadata["stance_mean"])
        / metadata["stance_scale"],
        "handedness": features.handedness,
    }


def _device(value: str) -> torch.device:
    if value == "auto":
        value = "mps" if torch.backends.mps.is_available() else "cpu"
    if value == "mps" and not torch.backends.mps.is_available():
        raise ValueError("MPS was requested but is unavailable")
    return torch.device(value)


def _batch(
    arrays: dict[str, NDArray[np.float32]], indices: NDArray[np.integer], device: torch.device
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    tensors = tuple(
        torch.as_tensor(arrays[key][indices], device=device)
        for key in ("state", "morphology", "stance", "handedness")
    )
    return tensors  # type: ignore[return-value]


def train_diffusion(
    arrays: dict[str, NDArray[np.float32]],
    *,
    steps: int,
    batch_size: int,
    learning_rate: float,
    diffusion_steps: int,
    device: str = "auto",
    seed: int = 19,
    model_dim: int = 96,
    layers: int = 4,
) -> tuple[ExpertMotionDenoiser, list[float]]:
    torch.manual_seed(seed)
    np_rng = np.random.default_rng(seed)
    target_device = _device(device)
    network = ExpertMotionDenoiser(
        morphology_dim=int(arrays["morphology"].shape[1]),
        stance_dim=int(arrays["stance"].shape[1]),
        model_dim=model_dim,
        layers=layers,
    ).to(target_device)
    optimizer = torch.optim.AdamW(network.parameters(), lr=learning_rate, weight_decay=1e-4)
    schedule = linear_diffusion_schedule(diffusion_steps, device=target_device)
    history: list[float] = []
    network.train()
    for step in range(steps):
        indices = np_rng.integers(0, len(arrays["state"]), size=batch_size)
        clean, morphology, stance, handedness = _batch(arrays, indices, target_device)
        # Synthetic morphology is derived only from expert proportions and is
        # deliberately independent of the target rotations.
        morphology = morphology + 0.08 * torch.randn_like(morphology)
        time = torch.randint(diffusion_steps, (batch_size,), device=target_device)
        noise = torch.randn_like(clean)
        alpha = schedule["sqrt_alpha_bar"][time, None, None]
        sigma = schedule["sqrt_one_minus_alpha_bar"][time, None, None]
        noisy = alpha * clean + sigma * noise
        predicted_noise = network(noisy, time, morphology, stance, handedness)
        denoising = F.mse_loss(predicted_noise, noise)
        predicted_clean = (noisy - sigma * predicted_noise) / alpha.clamp_min(1e-4)
        velocity = F.smooth_l1_loss(
            predicted_clean[:, 1:] - predicted_clean[:, :-1],
            clean[:, 1:] - clean[:, :-1],
        )
        loss = denoising + 0.05 * velocity
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(network.parameters(), 1.0)
        optimizer.step()
        if step % max(steps // 50, 1) == 0 or step == steps - 1:
            history.append(float(loss.detach().cpu()))
    return network.eval(), history


def sample_diffusion(
    network: ExpertMotionDenoiser,
    morphology: Tensor,
    stance: Tensor,
    handedness: Tensor,
    *,
    candidates: int,
    diffusion_steps: int,
    sampling_steps: int = 25,
    seed: int = 19,
) -> Tensor:
    device = next(network.parameters()).device
    generator = torch.Generator(device="cpu").manual_seed(seed)
    state = torch.randn(
        candidates, FRAMES, STATE_DIM, generator=generator, device="cpu"
    ).to(device)
    morphology = morphology.to(device).expand(candidates, -1)
    stance = stance.to(device).expand(candidates, -1)
    handedness = handedness.to(device).expand(candidates, -1)
    schedule = linear_diffusion_schedule(diffusion_steps, device=device)
    indices = torch.linspace(diffusion_steps - 1, 0, sampling_steps, device=device)
    indices = torch.unique_consecutive(indices.round().long())
    with torch.no_grad():
        for position, time_value in enumerate(indices):
            time = torch.full((candidates,), int(time_value), device=device, dtype=torch.long)
            noise = network(state, time, morphology, stance, handedness)
            alpha_bar = schedule["alpha_bar"][time_value]
            clean = (state - torch.sqrt(1.0 - alpha_bar) * noise) / torch.sqrt(alpha_bar)
            clean = clean.clamp(-7.0, 7.0)
            if position == len(indices) - 1:
                state = clean
            else:
                previous = indices[position + 1]
                previous_alpha_bar = schedule["alpha_bar"][previous]
                state = (
                    torch.sqrt(previous_alpha_bar) * clean
                    + torch.sqrt(1.0 - previous_alpha_bar) * noise
                )
    return state


def train_gan(
    arrays: dict[str, NDArray[np.float32]],
    *,
    steps: int,
    batch_size: int,
    learning_rate: float,
    device: str = "auto",
    seed: int = 19,
    model_dim: int = 96,
    layers: int = 3,
) -> tuple[ExpertMotionGANGenerator, list[float]]:
    """Train the explicit hinge-GAN fallback using expert states only."""
    torch.manual_seed(seed)
    np_rng = np.random.default_rng(seed)
    target_device = _device(device)
    morphology_dim = int(arrays["morphology"].shape[1])
    stance_dim = int(arrays["stance"].shape[1])
    generator = ExpertMotionGANGenerator(
        morphology_dim=morphology_dim,
        stance_dim=stance_dim,
        model_dim=model_dim,
        layers=layers,
    ).to(target_device)
    discriminator = ExpertMotionGANDiscriminator(
        morphology_dim=morphology_dim,
        stance_dim=stance_dim,
        model_dim=model_dim,
        layers=layers,
    ).to(target_device)
    generator_optimizer = torch.optim.Adam(generator.parameters(), lr=learning_rate, betas=(0.0, 0.9))
    discriminator_optimizer = torch.optim.Adam(
        discriminator.parameters(), lr=learning_rate, betas=(0.0, 0.9)
    )
    history: list[float] = []
    for step in range(steps):
        indices = np_rng.integers(0, len(arrays["state"]), size=batch_size)
        real, morphology, stance, handedness = _batch(arrays, indices, target_device)
        morphology = morphology + 0.08 * torch.randn_like(morphology)
        latent = torch.randn(batch_size, generator.latent_dim, device=target_device)
        fake = generator(latent, morphology, stance, handedness).detach()
        real_logit = discriminator(real, morphology, stance, handedness)
        fake_logit = discriminator(fake, morphology, stance, handedness)
        discriminator_loss = F.relu(1.0 - real_logit).mean() + F.relu(1.0 + fake_logit).mean()
        discriminator_optimizer.zero_grad(set_to_none=True)
        discriminator_loss.backward()
        discriminator_optimizer.step()

        if step % 2 == 0:
            latent = torch.randn(batch_size, generator.latent_dim, device=target_device)
            fake = generator(latent, morphology, stance, handedness)
            adversarial = -discriminator(fake, morphology, stance, handedness).mean()
            # A small expert moment loss stabilizes tiny-data training without
            # pairing a generated sample to a particular expert trajectory.
            moment = F.smooth_l1_loss(fake.mean(dim=0), real.mean(dim=0))
            fake_velocity = fake[:, 1:] - fake[:, :-1]
            real_velocity = real[:, 1:] - real[:, :-1]
            fake_acceleration = fake_velocity[:, 1:] - fake_velocity[:, :-1]
            real_acceleration = real_velocity[:, 1:] - real_velocity[:, :-1]
            velocity_moment = F.smooth_l1_loss(
                fake_velocity.abs().mean(dim=(0, 1)),
                real_velocity.abs().mean(dim=(0, 1)),
            )
            acceleration_moment = F.smooth_l1_loss(
                fake_acceleration.abs().mean(dim=(0, 1)),
                real_acceleration.abs().mean(dim=(0, 1)),
            )
            generator_loss = (
                adversarial
                + 0.1 * moment
                + 0.15 * velocity_moment
                + 0.25 * acceleration_moment
            )
            generator_optimizer.zero_grad(set_to_none=True)
            generator_loss.backward()
            torch.nn.utils.clip_grad_norm_(generator.parameters(), 1.0)
            generator_optimizer.step()
        if step % max(steps // 50, 1) == 0 or step == steps - 1:
            history.append(float(discriminator_loss.detach().cpu()))
    return generator.eval(), history


def sample_gan(
    network: ExpertMotionGANGenerator,
    morphology: Tensor,
    stance: Tensor,
    handedness: Tensor,
    *,
    candidates: int,
    seed: int = 19,
) -> Tensor:
    device = next(network.parameters()).device
    cpu_generator = torch.Generator(device="cpu").manual_seed(seed)
    latent = torch.randn(
        candidates, network.latent_dim, generator=cpu_generator, device="cpu"
    ).to(device)
    with torch.no_grad():
        return network(
            latent,
            morphology.to(device).expand(candidates, -1),
            stance.to(device).expand(candidates, -1),
            handedness.to(device).expand(candidates, -1),
        )


def validation_report(
    network: ExpertMotionDenoiser | ExpertMotionGANGenerator,
    method: str,
    validation_samples: Sequence[MotionSample],
    metadata: dict[str, NDArray[np.float32]],
    *,
    diffusion_steps: int,
    candidates: int = 8,
    seed: int = 19,
    conditioning_policy: str = "selective",
) -> dict[str, Any]:
    rows = []
    for index, sample in enumerate(validation_samples):
        transformed = transform_features(
            motion_features(sample, conditioning_policy=conditioning_policy),
            metadata,
        )
        morphology = torch.as_tensor(transformed["morphology"])[None]
        stance = torch.as_tensor(transformed["stance"])[None]
        handedness = torch.as_tensor(transformed["handedness"])[None]
        if method == "diffusion":
            assert isinstance(network, ExpertMotionDenoiser)
            generated = sample_diffusion(
                network,
                morphology,
                stance,
                handedness,
                candidates=candidates,
                diffusion_steps=diffusion_steps,
                seed=seed + index,
            )
        else:
            assert isinstance(network, ExpertMotionGANGenerator)
            generated = sample_gan(
                network,
                morphology,
                stance,
                handedness,
                candidates=candidates,
                seed=seed + index,
            )
        generated_array = generated.detach().cpu().numpy()
        target = transformed["state"]
        candidate_mse = np.mean(np.square(generated_array - target[None]), axis=(1, 2))
        baseline_mse = float(np.mean(np.square(target)))
        rows.append(
            {
                "file": sample.path.name,
                "subject_id": sample.subject_id,
                "best_of_k_mse": float(candidate_mse.min()),
                "training_mean_mse": baseline_mse,
                "relative_to_training_mean": float(
                    candidate_mse.min() / max(baseline_mse, 1e-8)
                ),
            }
        )
    relative = [row["relative_to_training_mean"] for row in rows]
    return {
        "method": method,
        "student_data_used": False,
        "held_out_expert_samples": len(rows),
        "median_relative_to_training_mean": float(np.median(relative)),
        "accepted": bool(np.isfinite(relative).all() and np.median(relative) <= 1.5),
        "rows": rows,
    }


def build_bundle(
    *,
    skill: str,
    method: str,
    network: ExpertMotionDenoiser | ExpertMotionGANGenerator,
    diffusion_steps: int,
    arrays: dict[str, NDArray[np.float32]],
    metadata: dict[str, NDArray[np.float32]],
    samples: Sequence[MotionSample],
    conditioning_policy: str = "selective",
) -> ExpertMotionBundle:
    stance_joint_ids = tuple(
        int(value) for value in conditioning_stance_joints(conditioning_policy)
    )
    expected_stance_dimension = 2 * len(stance_joint_ids) or 1
    if int(network.stance_dim) != expected_stance_dimension:
        raise ValueError(
            "network stance dimension does not match conditioning policy: "
            f"network={network.stance_dim}, policy={conditioning_policy}, "
            f"expected={expected_stance_dimension}"
        )
    return ExpertMotionBundle(
        skill=skill,
        method=method,
        network=network,
        diffusion_steps=diffusion_steps,
        state_mean=metadata["state_mean"],
        state_scale=metadata["state_scale"],
        morphology_mean=metadata["morphology_mean"],
        morphology_scale=metadata["morphology_scale"],
        stance_mean=metadata["stance_mean"],
        stance_scale=metadata["stance_scale"],
        canonical_lengths=metadata["canonical_lengths"],
        expert_states=arrays["state"],
        expert_files=np.asarray([sample.path.name for sample in samples]),
        expert_subject_ids=np.asarray([sample.subject_id for sample in samples]),
        training_manifest_sha256=training_manifest(samples),
        expert_wrist_velocity_limit=expert_wrist_velocity_limit(samples),
        conditioning_policy=conditioning_policy,
        stance_joint_ids=stance_joint_ids,
    )


def save_expert_motion_bundle(bundle: ExpertMotionBundle, path: str | Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "format_version": 3,
            "skill": bundle.skill,
            "method": bundle.method,
            "diffusion_steps": bundle.diffusion_steps,
            "network_config": bundle.network.config(),
            "state_dict": bundle.network.state_dict(),
            "state_mean": torch.from_numpy(bundle.state_mean),
            "state_scale": torch.from_numpy(bundle.state_scale),
            "morphology_mean": torch.from_numpy(bundle.morphology_mean),
            "morphology_scale": torch.from_numpy(bundle.morphology_scale),
            "stance_mean": torch.from_numpy(bundle.stance_mean),
            "stance_scale": torch.from_numpy(bundle.stance_scale),
            "canonical_lengths": torch.from_numpy(bundle.canonical_lengths),
            "expert_states": torch.from_numpy(bundle.expert_states),
            "expert_files": bundle.expert_files.tolist(),
            "expert_subject_ids": bundle.expert_subject_ids.tolist(),
            "training_manifest_sha256": bundle.training_manifest_sha256,
            "expert_wrist_velocity_limit": bundle.expert_wrist_velocity_limit,
            "conditioning_policy": bundle.conditioning_policy,
            "stance_joint_ids": list(bundle.stance_joint_ids),
            "student_data_used": False,
        },
        destination,
    )


def load_expert_motion_bundle(
    path: str | Path, *, device: str = "auto"
) -> ExpertMotionBundle:
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if int(checkpoint["format_version"]) not in (1, 2, 3):
        raise ValueError("unsupported expert motion model format")
    if checkpoint.get("student_data_used", True):
        raise ValueError("checkpoint provenance does not certify expert-only training")
    method = str(checkpoint["method"])
    config = checkpoint["network_config"]
    conditioning_policy = str(checkpoint.get("conditioning_policy", "selective"))
    expected_joint_ids = tuple(
        int(value) for value in conditioning_stance_joints(conditioning_policy)
    )
    stance_joint_ids = tuple(
        int(value)
        for value in checkpoint.get("stance_joint_ids", STANCE_JOINTS.tolist())
    )
    if stance_joint_ids != expected_joint_ids:
        raise ValueError(
            "checkpoint stance joints do not match conditioning policy: "
            f"policy={conditioning_policy}, joints={stance_joint_ids}"
        )
    expected_stance_dimension = 2 * len(stance_joint_ids) or 1
    if int(config["stance_dim"]) != expected_stance_dimension:
        raise ValueError(
            "checkpoint network stance dimension does not match conditioning policy"
        )
    if method == "diffusion":
        network: ExpertMotionDenoiser | ExpertMotionGANGenerator = ExpertMotionDenoiser(**config)
    elif method == "gan":
        network = ExpertMotionGANGenerator(**config)
    else:
        raise ValueError(f"unsupported generator method: {method}")
    network.load_state_dict(checkpoint["state_dict"])
    network.to(_device(device)).eval()

    def array(key: str) -> NDArray[np.float32]:
        return checkpoint[key].detach().cpu().numpy().astype(np.float32)

    return ExpertMotionBundle(
        skill=str(checkpoint["skill"]),
        method=method,
        network=network,
        diffusion_steps=int(checkpoint["diffusion_steps"]),
        state_mean=array("state_mean"),
        state_scale=array("state_scale"),
        morphology_mean=array("morphology_mean"),
        morphology_scale=array("morphology_scale"),
        stance_mean=array("stance_mean"),
        stance_scale=array("stance_scale"),
        canonical_lengths=array("canonical_lengths"),
        expert_states=array("expert_states"),
        expert_files=np.asarray(checkpoint["expert_files"]),
        expert_subject_ids=np.asarray(checkpoint["expert_subject_ids"]),
        training_manifest_sha256=str(checkpoint["training_manifest_sha256"]),
        expert_wrist_velocity_limit=float(
            checkpoint.get("expert_wrist_velocity_limit", float("inf"))
        ),
        conditioning_policy=conditioning_policy,
        stance_joint_ids=stance_joint_ids,
    )


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


def _bundle_metadata(bundle: ExpertMotionBundle) -> dict[str, NDArray[np.float32]]:
    return {
        "state_mean": bundle.state_mean,
        "state_scale": bundle.state_scale,
        "morphology_mean": bundle.morphology_mean,
        "morphology_scale": bundle.morphology_scale,
        "stance_mean": bundle.stance_mean,
        "stance_scale": bundle.stance_scale,
    }


def _sample_bundle(
    bundle: ExpertMotionBundle,
    transformed: dict[str, NDArray[np.float32]],
    *,
    candidates: int,
    seed: int,
) -> NDArray[np.float32]:
    morphology = torch.as_tensor(transformed["morphology"])[None]
    stance = torch.as_tensor(transformed["stance"])[None]
    handedness = torch.as_tensor(transformed["handedness"])[None]
    if bundle.method == "diffusion":
        assert isinstance(bundle.network, ExpertMotionDenoiser)
        result = sample_diffusion(
            bundle.network,
            morphology,
            stance,
            handedness,
            candidates=candidates,
            diffusion_steps=bundle.diffusion_steps,
            seed=seed,
        )
    else:
        assert isinstance(bundle.network, ExpertMotionGANGenerator)
        result = sample_gan(
            bundle.network,
            morphology,
            stance,
            handedness,
            candidates=candidates,
            seed=seed,
        )
    return result.detach().cpu().numpy().astype(np.float32)


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


def correct_student_motion_generated(
    bundle: ExpertMotionBundle,
    sample: MotionSample,
    *,
    candidates: int = 16,
    seed: int = 19,
) -> ExpertCorrection:
    """Generate expert motion, then map it through the student's exact FK."""
    if sample.skill != bundle.skill:
        raise ValueError(f"student skill {sample.skill!r} does not match {bundle.skill!r}")
    if candidates < 1:
        raise ValueError("candidates must be positive")
    features = motion_features(
        sample,
        conditioning_policy=bundle.conditioning_policy,
        stance_joint_ids=bundle.stance_joint_ids,
    )
    transformed = transform_features(features, _bundle_metadata(bundle))
    generated = _sample_bundle(bundle, transformed, candidates=candidates, seed=seed)
    generated = project_to_expert_motion_subspace(generated, bundle.expert_states)

    # Candidate choice is based on expert likelihood and temporal regularity,
    # never on the learner's dynamic errors.
    manifold = np.mean(
        np.square(generated[:, None] - bundle.expert_states[None]), axis=(2, 3)
    ).min(axis=1)
    acceleration = np.diff(generated[:, :, :DIRECTION_DIM], n=2, axis=1)
    smoothness = np.mean(np.square(acceleration), axis=(1, 2))
    objective = manifold + 0.025 * smoothness
    selected = int(np.argmin(objective))
    standardized_state = generated[selected]
    state = standardized_state * bundle.state_scale + bundle.state_mean
    directions = _unit(state[:, :DIRECTION_DIM].reshape(FRAMES, JOINTS, 2))
    normalized_root = state[:, DIRECTION_DIM : DIRECTION_DIM + ROOT_DIM]
    contacts = np.clip(state[:, -CONTACT_DIM:], 0.0, 1.0).astype(np.float32)
    directions, normalized_root, contacts = smooth_generated_motion_state(
        directions, normalized_root, contacts
    )
    generated_pose = _fk_from_directions(directions, bundle.canonical_lengths)

    aligned_pose, aligned_confidence, aligned_root, _ = _aligned(sample)
    corrected = retarget_expert_canonical_2d_fk(
        aligned_pose,
        generated_pose,
        aligned_confidence,
        np.ones_like(aligned_confidence),
        root_trajectory=np.zeros((FRAMES, 2), dtype=np.float32),
    )
    root_delta = normalized_root * features.body_scale
    corrected_root = aligned_root[:1] + root_delta - root_delta[:1]
    canonical_root = normalized_root * float(np.median(bundle.canonical_lengths))
    corrected_root = _retarget_root_with_contacts(
        corrected, corrected_root, contacts, generated_pose
    )

    distances = np.mean(
        np.square(bundle.expert_states - standardized_state[None]), axis=(1, 2)
    )
    nearest = np.argsort(distances, kind="stable")[: min(3, len(distances))]
    weights = np.exp(-distances[nearest] / max(float(np.median(distances[nearest])), 1e-4))
    weights /= weights.sum()
    correction = ExpertCorrection(
        student=sample,
        aligned_student_pose=aligned_pose,
        aligned_student_root=aligned_root,
        aligned_corrected_pose=corrected,
        aligned_corrected_root=corrected_root.astype(np.float32),
        corrected_pose=restore_phase_timing(corrected, sample.phase_indices),
        corrected_root=restore_phase_timing(corrected_root, sample.phase_indices),
        aligned_corrected_contacts=contacts,
        corrected_contacts=restore_phase_timing(contacts, sample.phase_indices),
        expert_prototype_pose=generated_pose,
        expert_prototype_root=canonical_root.astype(np.float32),
        reference_indices=nearest.astype(np.int64),
        reference_weights=weights.astype(np.float32),
        reference_distances=np.sqrt(distances[nearest]).astype(np.float32),
    )
    if bundle.skill == "serve" and np.isfinite(
        bundle.expert_wrist_velocity_limit
    ):
        correction = limit_correction_wrist_velocity(
            correction, bundle.expert_wrist_velocity_limit
        )
    return correction


def score_generated_correction(
    score_model: ExpertPhaseModel,
    correction: ExpertCorrection,
    *,
    generator_method: str,
    training_manifest_sha256: str,
) -> dict[str, Any]:
    """Continuous score against subject-held-out expert variability.

    At the criterion's expert p90 distance the awarded fraction is 0.5.  This
    removes the legacy full-score plateau while retaining expert-only robust
    scales as uncertainty intervals.
    """
    spec = score_model.spec
    confidence = np.clip(
        phase_align_sequence(
            correction.student.confidence, correction.student.phase_indices
        ),
        0.0,
        1.0,
    ).astype(np.float32)
    components = _criterion_components_for_spec(
        spec,
        correction.aligned_student_pose,
        correction.aligned_student_root,
        correction.aligned_corrected_pose,
        correction.aligned_corrected_root,
        confidence,
    )

    def fraction(distance: float, reference: float) -> float:
        return float(np.exp(-np.log(2.0) * np.square(distance / reference)))

    criteria = []
    for index, (rule, component) in enumerate(
        zip(spec.rules, components, strict=True)
    ):
        distance = float(component["combined_distance"])
        tolerance = float(score_model.criterion_tolerances[index])
        robust_scale = float(score_model.criterion_scales[index])
        reference = max(tolerance, robust_scale, 0.025)
        lower_distance = max(0.0, distance - robust_scale)
        upper_distance = distance + robust_scale
        ratio = fraction(distance, reference)
        criteria.append(
            {
                "name_zh_tw": rule.name_zh_tw,
                "rule_reference": rule.id,
                "score": rule.maximum * ratio,
                "score_lower": rule.maximum * fraction(upper_distance, reference),
                "score_upper": rule.maximum * fraction(lower_distance, reference),
                "maximum": rule.maximum,
                "expert_p90_distance": tolerance,
                "expert_robust_scale": robust_scale,
                "distance_to_expert_p90_ratio": distance / reference,
                **component,
            }
        )
    references = [
        {
            "file": str(score_model.expert_files[index]),
            "subject_id": str(score_model.expert_subject_ids[index]),
            "identity_level": str(score_model.expert_identity_levels[index]),
            "alignment_contract": str(score_model.expert_alignment_contracts[index]),
            "weight": float(weight),
            "stance_distance": float(distance),
            "usage": "nearest_real_expert_audit_only",
        }
        for index, weight, distance in zip(
            correction.reference_indices,
            correction.reference_weights,
            correction.reference_distances,
            strict=True,
        )
    ]
    return {
        "filename": correction.student.video_name,
        "skill": score_model.skill,
        "handedness": correction.student.handedness,
        "score_method": "continuous_generated_expert_distribution_v1",
        "correction_policy": "expert_only_generated_full_body_fk_projection",
        "generator_method": generator_method,
        "training_manifest_sha256": training_manifest_sha256,
        "student_data_used_for_training": False,
        "total_score": float(sum(item["score"] for item in criteria)),
        "total_score_lower": float(sum(item["score_lower"] for item in criteria)),
        "total_score_upper": float(sum(item["score_upper"] for item in criteria)),
        "criteria": criteria,
        "references": references,
    }
