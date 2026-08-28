"""Conditional full-body expert motion generators.

The diffusion model is the primary model.  A compact Wasserstein GAN is kept
as an explicit fallback for very small expert banks where the diffusion
validation criterion is not met.  Both models consume only static morphology,
preparation stance and handedness; a learner's moving pose is never an input.
"""
from __future__ import annotations

import math

import torch
from torch import Tensor, nn


def _phase_features(frames: int) -> Tensor:
    phase = torch.linspace(0.0, 1.0, frames)
    segment = torch.clamp((phase * 4.0).floor().long(), max=3)
    return torch.cat(
        (
            torch.sin(math.pi * phase)[:, None],
            torch.cos(math.pi * phase)[:, None],
            torch.nn.functional.one_hot(segment, num_classes=4).float(),
        ),
        dim=-1,
    )


def timestep_embedding(timesteps: Tensor, dimension: int) -> Tensor:
    """Standard sinusoidal diffusion-step embedding."""
    half = dimension // 2
    frequency = torch.exp(
        -math.log(10_000.0)
        * torch.arange(half, device=timesteps.device, dtype=torch.float32)
        / max(half - 1, 1)
    )
    angles = timesteps.float()[:, None] * frequency[None]
    embedding = torch.cat((torch.sin(angles), torch.cos(angles)), dim=-1)
    if dimension % 2:
        embedding = torch.nn.functional.pad(embedding, (0, 1))
    return embedding










def linear_diffusion_schedule(
    steps: int, *, device: torch.device | str | None = None
) -> dict[str, Tensor]:
    beta = torch.linspace(1e-4, 0.02, steps, device=device)
    alpha = 1.0 - beta
    alpha_bar = torch.cumprod(alpha, dim=0)
    return {
        "beta": beta,
        "alpha": alpha,
        "alpha_bar": alpha_bar,
        "sqrt_alpha_bar": torch.sqrt(alpha_bar),
        "sqrt_one_minus_alpha_bar": torch.sqrt(1.0 - alpha_bar),
    }
