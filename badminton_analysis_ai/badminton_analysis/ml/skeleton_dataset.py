from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset

from badminton_analysis.ml.skeleton_normalization import phase_align_sequence
from badminton_analysis.models.types import Skill


_LIFT_DIRECTION_CORRUPTION_PROBABILITY = 0.5


def discover_sequence_files(root: str | Path, label: str | None = None) -> list[Path]:
    directory = Path(root)
    if label is not None:
        directory = directory / label
    return sorted(directory.glob("*.npz"))


def load_sequence(path: str | Path) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}


def corrupt_skeleton(
    skeleton: Tensor,
    confidence: Tensor,
    *,
    generator: torch.Generator | None = None,
    noise_std: float = 0.06,
    mask_probability: float = 0.08,
) -> tuple[Tensor, Tensor]:
    """Apply pose and timing corruptions to a clean expert sequence."""
    corrupted = skeleton.clone()
    visible = confidence.clone()
    noise = torch.randn(
        corrupted.shape,
        dtype=corrupted.dtype,
        device=corrupted.device,
        generator=generator,
    )
    corrupted += noise * noise_std

    random_mask = torch.rand(
        visible.shape,
        dtype=visible.dtype,
        device=visible.device,
        generator=generator,
    ) < mask_probability
    visible = visible.masked_fill(random_mask, 0.0)
    corrupted = corrupted.masked_fill(random_mask.unsqueeze(-1), 0.0)

    shift = int(torch.randint(-2, 3, (1,), generator=generator).item())
    if shift > 0:
        corrupted[shift:] = corrupted[:-shift].clone()
        corrupted[:shift] = corrupted[shift]
        visible[shift:] = visible[:-shift].clone()
        visible[:shift] = visible[shift]
    elif shift < 0:
        amount = -shift
        corrupted[:-amount] = corrupted[amount:].clone()
        corrupted[-amount:] = corrupted[-amount - 1]
        visible[:-amount] = visible[amount:].clone()
        visible[-amount:] = visible[-amount - 1]

    # Small coherent limb perturbations are harder than independent joint noise.
    limb_joints = torch.tensor((7, 8, 9, 10, 13, 14, 15, 16), device=skeleton.device)
    limb_offset = torch.randn(
        (len(limb_joints), 3),
        dtype=corrupted.dtype,
        device=corrupted.device,
        generator=generator,
    ) * (noise_std * 0.75)
    corrupted[:, limb_joints] += limb_offset.unsqueeze(0)
    return corrupted, visible


def corrupt_lift_lunge_direction(skeleton: Tensor) -> Tensor:
    """Reverse the dominant-leg horizontal step while retaining its timing."""
    if skeleton.ndim != 3 or skeleton.shape[-2:] != (17, 3):
        raise ValueError("skeleton must have shape (T, 17, 3)")
    corrupted = skeleton.clone()
    dominant_hip = skeleton[:, 12]
    for joint in (14, 16):
        relative = skeleton[:, joint] - dominant_hip
        displacement = relative - relative[:1]
        reversed_relative = relative.clone()
        reversed_relative[:, (0, 2)] = (
            relative[:1, (0, 2)] - displacement[:, (0, 2)]
        )
        reversed_joint = dominant_hip + reversed_relative
        start = len(skeleton) // 4
        full = len(skeleton) // 2
        blend = torch.zeros(
            len(skeleton), dtype=skeleton.dtype, device=skeleton.device
        )
        blend[start:full] = torch.linspace(
            0.0,
            1.0,
            full - start,
            dtype=skeleton.dtype,
            device=skeleton.device,
        )
        blend[full:] = 1.0
        corrupted[:, joint] = (
            (1.0 - blend[:, None]) * skeleton[:, joint]
            + blend[:, None] * reversed_joint
        )
    return corrupted


class SkeletonSequenceDataset(Dataset[dict[str, Tensor]]):
    def __init__(
        self,
        files: list[Path],
        *,
        augment: bool = True,
        deterministic: bool = False,
    ) -> None:
        if not files:
            raise ValueError("skeleton dataset contains no files")
        self.files = files
        self.augment = augment
        self.deterministic = deterministic

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        sample = load_sequence(self.files[index])
        target = torch.as_tensor(sample["skeleton_3d"], dtype=torch.float32)
        confidence = torch.as_tensor(sample["confidence"], dtype=torch.float32)
        input_skeleton = target.clone()
        input_confidence = confidence.clone()
        if self.augment:
            generator = None
            if self.deterministic:
                generator = torch.Generator().manual_seed(1729 + index)
            noise_std = 0.03 + 0.12 * float(
                torch.rand((1,), generator=generator).item()
            )
            mask_probability = 0.04 + 0.12 * float(
                torch.rand((1,), generator=generator).item()
            )
            input_skeleton, input_confidence = corrupt_skeleton(
                input_skeleton,
                input_confidence,
                generator=generator,
                noise_std=noise_std,
                mask_probability=mask_probability,
            )
        features = torch.cat((input_skeleton, input_confidence.unsqueeze(-1)), dim=-1)
        return {
            "features": features,
            "target": target,
            "confidence": confidence,
        }


class SkeletonCorrectionPairDataset(Dataset[dict[str, Tensor]]):
    """Phase-aligned expert reconstruction or student-to-expert training pairs."""

    def __init__(
        self,
        files: list[Path],
        *,
        targets: Mapping[str, tuple[np.ndarray, np.ndarray]] | None = None,
        reference_conditioned: bool = False,
        augment: bool = False,
        deterministic: bool = False,
        skill: Skill | str | None = None,
    ) -> None:
        if not files:
            raise ValueError("skeleton pair dataset contains no files")
        self.files = files
        self.targets = targets
        self.reference_conditioned = reference_conditioned
        self.augment = augment
        self.deterministic = deterministic
        self.skill = (
            Skill.convert_to_enum(skill) if isinstance(skill, str) else skill
        )

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        path = self.files[index]
        sample = load_sequence(path)
        phases = sample["phase_indices"].astype(np.int64)
        source_array = phase_align_sequence(sample["skeleton_3d"], phases)
        source_confidence_array = np.clip(
            phase_align_sequence(sample["confidence"], phases), 0.0, 1.0
        )
        source = torch.as_tensor(source_array, dtype=torch.float32)
        source_confidence = torch.as_tensor(
            source_confidence_array, dtype=torch.float32
        )

        if self.targets is None:
            target = source.clone()
            target_confidence = source_confidence.clone()
        else:
            if path.name not in self.targets:
                raise KeyError(f"missing correction target for {path.name}")
            target_array, target_confidence_array = self.targets[path.name]
            target = torch.as_tensor(target_array, dtype=torch.float32)
            target_confidence = torch.as_tensor(
                target_confidence_array, dtype=torch.float32
            )

        input_skeleton = source.clone()
        input_confidence = source_confidence.clone()
        if self.augment:
            generator = None
            if self.deterministic:
                generator = torch.Generator().manual_seed(2718 + index)
            if self.skill == Skill.LIFT and float(
                torch.rand((1,), generator=generator).item()
            ) < _LIFT_DIRECTION_CORRUPTION_PROBABILITY:
                input_skeleton = corrupt_lift_lunge_direction(input_skeleton)
            noise_std = 0.02 + 0.06 * float(
                torch.rand((1,), generator=generator).item()
            )
            mask_probability = 0.02 + 0.08 * float(
                torch.rand((1,), generator=generator).item()
            )
            input_skeleton, input_confidence = corrupt_skeleton(
                input_skeleton,
                input_confidence,
                generator=generator,
                noise_std=noise_std,
                mask_probability=mask_probability,
            )
        feature_parts = [input_skeleton]
        if self.reference_conditioned:
            feature_parts.append(target)
        feature_parts.append(input_confidence.unsqueeze(-1))
        features = torch.cat(feature_parts, dim=-1)
        return {
            "features": features,
            "source": source,
            "target": target,
            "confidence": torch.minimum(
                source_confidence, target_confidence
            ),
        }
