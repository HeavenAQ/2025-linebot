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


class ConditionEncoder(nn.Module):
    """Keep morphology, stance and handedness in separate control paths."""

    def __init__(
        self,
        morphology_dim: int,
        stance_dim: int,
        model_dim: int,
    ) -> None:
        super().__init__()
        self.morphology = nn.Sequential(
            nn.Linear(morphology_dim, model_dim), nn.SiLU(), nn.Linear(model_dim, model_dim)
        )
        self.stance = nn.Sequential(
            nn.Linear(stance_dim, model_dim), nn.SiLU(), nn.Linear(model_dim, model_dim)
        )
        self.handedness = nn.Sequential(
            nn.Linear(2, model_dim), nn.SiLU(), nn.Linear(model_dim, model_dim)
        )
        self.norm = nn.LayerNorm(model_dim)

    def forward(self, morphology: Tensor, stance: Tensor, handedness: Tensor) -> Tensor:
        return self.norm(
            self.morphology(morphology)
            + self.stance(stance)
            + self.handedness(handedness)
        )


class ExpertMotionDenoiser(nn.Module):
    """Temporal denoiser for canonical full-body expert motion states."""

    def __init__(
        self,
        *,
        state_dim: int = 38,
        morphology_dim: int = 17,
        stance_dim: int = 16,
        frames: int = 64,
        model_dim: int = 96,
        heads: int = 4,
        layers: int = 4,
        dropout: float = 0.05,
    ) -> None:
        super().__init__()
        self.state_dim = state_dim
        self.morphology_dim = morphology_dim
        self.stance_dim = stance_dim
        self.frames = frames
        self.model_dim = model_dim
        self.heads = heads
        self.layers = layers
        self.dropout = dropout
        self.state_projection = nn.Linear(state_dim + 6, model_dim)
        self.condition = ConditionEncoder(morphology_dim, stance_dim, model_dim)
        self.diffusion_step = nn.Sequential(
            nn.Linear(model_dim, model_dim * 2),
            nn.SiLU(),
            nn.Linear(model_dim * 2, model_dim),
        )
        self.time_embedding = nn.Parameter(torch.zeros(1, frames, model_dim))
        layer = nn.TransformerEncoderLayer(
            model_dim,
            heads,
            dim_feedforward=model_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.temporal = nn.TransformerEncoder(layer, layers)
        self.output = nn.Sequential(nn.LayerNorm(model_dim), nn.Linear(model_dim, state_dim))
        self.register_buffer("phase_features", _phase_features(frames), persistent=False)
        nn.init.normal_(self.time_embedding, std=0.02)

    def config(self) -> dict[str, int | float]:
        return {
            "state_dim": self.state_dim,
            "morphology_dim": self.morphology_dim,
            "stance_dim": self.stance_dim,
            "frames": self.frames,
            "model_dim": self.model_dim,
            "heads": self.heads,
            "layers": self.layers,
            "dropout": self.dropout,
        }

    def forward(
        self,
        noisy_state: Tensor,
        diffusion_step: Tensor,
        morphology: Tensor,
        stance: Tensor,
        handedness: Tensor,
    ) -> Tensor:
        batch, frames, state_dim = noisy_state.shape
        if frames != self.frames or state_dim != self.state_dim:
            raise ValueError(
                f"state must have shape (B, {self.frames}, {self.state_dim})"
            )
        phase = self.phase_features[None].expand(batch, -1, -1)
        hidden = self.state_projection(torch.cat((noisy_state, phase), dim=-1))
        condition = self.condition(morphology, stance, handedness)
        step = self.diffusion_step(timestep_embedding(diffusion_step, self.model_dim))
        hidden = hidden + self.time_embedding + condition[:, None] + step[:, None]
        return self.output(self.temporal(hidden))


class ExpertMotionGANGenerator(nn.Module):
    """Conditional sequence generator used only when diffusion fails validation."""

    def __init__(
        self,
        *,
        state_dim: int = 38,
        morphology_dim: int = 17,
        stance_dim: int = 16,
        frames: int = 64,
        latent_dim: int = 64,
        model_dim: int = 96,
        heads: int = 4,
        layers: int = 3,
        dropout: float = 0.05,
    ) -> None:
        super().__init__()
        self.state_dim = state_dim
        self.morphology_dim = morphology_dim
        self.stance_dim = stance_dim
        self.frames = frames
        self.latent_dim = latent_dim
        self.model_dim = model_dim
        self.heads = heads
        self.layers = layers
        self.dropout = dropout
        self.condition = ConditionEncoder(morphology_dim, stance_dim, model_dim)
        self.latent = nn.Linear(latent_dim, model_dim)
        self.time_embedding = nn.Parameter(torch.zeros(1, frames, model_dim))
        self.register_buffer("phase_features", _phase_features(frames), persistent=False)
        self.phase_projection = nn.Linear(6, model_dim)
        layer = nn.TransformerEncoderLayer(
            model_dim,
            heads,
            dim_feedforward=model_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.temporal = nn.TransformerEncoder(layer, layers)
        self.output = nn.Sequential(nn.LayerNorm(model_dim), nn.Linear(model_dim, state_dim))
        nn.init.normal_(self.time_embedding, std=0.02)

    def config(self) -> dict[str, int | float]:
        return {
            "state_dim": self.state_dim,
            "morphology_dim": self.morphology_dim,
            "stance_dim": self.stance_dim,
            "frames": self.frames,
            "latent_dim": self.latent_dim,
            "model_dim": self.model_dim,
            "heads": self.heads,
            "layers": self.layers,
            "dropout": self.dropout,
        }

    def forward(
        self,
        latent: Tensor,
        morphology: Tensor,
        stance: Tensor,
        handedness: Tensor,
    ) -> Tensor:
        condition = self.condition(morphology, stance, handedness)
        hidden = (
            self.latent(latent)[:, None]
            + condition[:, None]
            + self.time_embedding
            + self.phase_projection(self.phase_features)[None]
        )
        return self.output(self.temporal(hidden))


class ExpertMotionGANDiscriminator(nn.Module):
    def __init__(
        self,
        *,
        state_dim: int = 38,
        morphology_dim: int = 17,
        stance_dim: int = 16,
        frames: int = 64,
        model_dim: int = 96,
        heads: int = 4,
        layers: int = 3,
    ) -> None:
        super().__init__()
        self.frames = frames
        self.state_dim = state_dim
        self.condition = ConditionEncoder(morphology_dim, stance_dim, model_dim)
        self.state_projection = nn.Linear(state_dim + 6, model_dim)
        self.register_buffer("phase_features", _phase_features(frames), persistent=False)
        self.time_embedding = nn.Parameter(torch.zeros(1, frames, model_dim))
        layer = nn.TransformerEncoderLayer(
            model_dim,
            heads,
            dim_feedforward=model_dim * 4,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.temporal = nn.TransformerEncoder(layer, layers)
        self.head = nn.Sequential(nn.LayerNorm(model_dim), nn.Linear(model_dim, 1))

    def forward(
        self,
        state: Tensor,
        morphology: Tensor,
        stance: Tensor,
        handedness: Tensor,
    ) -> Tensor:
        batch = len(state)
        phase = self.phase_features[None].expand(batch, -1, -1)
        hidden = self.state_projection(torch.cat((state, phase), dim=-1))
        hidden = hidden + self.time_embedding + self.condition(
            morphology, stance, handedness
        )[:, None]
        return self.head(self.temporal(hidden).mean(dim=1)).squeeze(-1)


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
