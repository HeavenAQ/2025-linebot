"""Phase-aware diffusion with learned joint reliability isolation."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from badminton_analysis.ml.models.expert_motion_diffusion import (
    _phase_features,
    timestep_embedding,
)


class PhaseJointReliabilityEncoder(nn.Module):
    """Infer trustworthy joint-phase tokens before motion conditioning.

    A diagnostic transformer may compare all raw joint tokens to estimate
    reliability.  The motion-fusion transformer only receives gated local
    tokens, preventing rejected pose values from leaking through contextual
    features computed before the gate.
    """

    def __init__(
        self,
        *,
        morphology_dim: int = 17,
        joints: int = 17,
        phases: int = 4,
        model_dim: int = 96,
        heads: int = 4,
        diagnostic_layers: int = 2,
        fusion_layers: int = 2,
        dropout: float = 0.05,
        use_reliability_gate: bool = True,
    ) -> None:
        super().__init__()
        self.morphology_dim = morphology_dim
        self.joints = joints
        self.phases = phases
        self.model_dim = model_dim
        self.heads = heads
        self.diagnostic_layers = diagnostic_layers
        self.fusion_layers = fusion_layers
        self.dropout = dropout
        self.use_reliability_gate = use_reliability_gate
        self.local_projection = nn.Sequential(
            nn.Linear(4, model_dim),
            nn.SiLU(),
            nn.Linear(model_dim, model_dim),
        )
        self.morphology_projection = nn.Linear(morphology_dim, joints)
        self.handedness_projection = nn.Sequential(
            nn.Linear(2, model_dim), nn.SiLU(), nn.Linear(model_dim, model_dim)
        )
        self.joint_embedding = nn.Parameter(torch.zeros(1, 1, joints, model_dim))
        self.phase_embedding = nn.Parameter(torch.zeros(1, phases, 1, model_dim))
        self.missing_token = nn.Parameter(torch.zeros(1, 1, 1, model_dim))

        diagnostic_layer = nn.TransformerEncoderLayer(
            model_dim,
            heads,
            dim_feedforward=model_dim * 3,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.diagnostic = nn.TransformerEncoder(
            diagnostic_layer, diagnostic_layers
        )
        self.reliability_head = nn.Sequential(
            nn.LayerNorm(model_dim),
            nn.Linear(model_dim, model_dim // 2),
            nn.SiLU(),
            nn.Linear(model_dim // 2, 1),
        )
        fusion_layer = nn.TransformerEncoderLayer(
            model_dim,
            heads,
            dim_feedforward=model_dim * 3,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.fusion = nn.TransformerEncoder(fusion_layer, fusion_layers)
        self.phase_norm = nn.LayerNorm(model_dim)
        nn.init.normal_(self.joint_embedding, std=0.02)
        nn.init.normal_(self.phase_embedding, std=0.02)
        nn.init.normal_(self.missing_token, std=0.02)

    def forward(
        self,
        directions: Tensor,
        confidence: Tensor,
        morphology: Tensor,
        handedness: Tensor,
    ) -> tuple[Tensor, Tensor]:
        batch = len(directions)
        expected_directions = (batch, self.phases, self.joints, 2)
        expected_confidence = (batch, self.phases, self.joints)
        if tuple(directions.shape) != expected_directions:
            raise ValueError(
                f"directions must have shape {expected_directions}, got "
                f"{tuple(directions.shape)}"
            )
        if tuple(confidence.shape) != expected_confidence:
            raise ValueError(
                f"confidence must have shape {expected_confidence}, got "
                f"{tuple(confidence.shape)}"
            )
        if tuple(morphology.shape) != (batch, self.morphology_dim):
            raise ValueError("morphology has the wrong shape")
        if tuple(handedness.shape) != (batch, 2):
            raise ValueError("handedness has the wrong shape")

        joint_morphology = self.morphology_projection(morphology)
        joint_morphology = joint_morphology[:, None, :, None].expand(
            -1, self.phases, -1, -1
        )
        local_values = torch.cat(
            (directions, confidence[..., None], joint_morphology), dim=-1
        )
        local = self.local_projection(local_values)
        position = self.joint_embedding + self.phase_embedding
        handedness_token = self.handedness_projection(handedness)[:, None, None]
        raw_tokens = local + position + handedness_token
        diagnostic = self.diagnostic(raw_tokens.flatten(1, 2)).reshape(
            batch, self.phases, self.joints, self.model_dim
        )
        reliability_logits = self.reliability_head(diagnostic).squeeze(-1)
        predicted_reliability = torch.sigmoid(reliability_logits)[..., None]
        reliability = (
            predicted_reliability
            if self.use_reliability_gate
            else torch.ones_like(predicted_reliability)
        )

        gated = reliability * local + (1.0 - reliability) * self.missing_token
        gated = gated + position + handedness_token
        fused = self.fusion(gated.flatten(1, 2)).reshape(
            batch, self.phases, self.joints, self.model_dim
        )
        weights = confidence.clamp(0.0, 1.0)[..., None] * reliability
        phase_tokens = (fused * weights).sum(dim=2) / weights.sum(dim=2).clamp_min(
            1e-4
        )
        return self.phase_norm(phase_tokens), reliability_logits


class ErrorIsolatedMotionDenoiser(nn.Module):
    """Diffuse expert motion while isolating unreliable conditioning joints."""

    def __init__(
        self,
        *,
        state_dim: int = 38,
        morphology_dim: int = 17,
        joints: int = 17,
        condition_phases: int = 4,
        frames: int = 64,
        model_dim: int = 96,
        heads: int = 4,
        layers: int = 4,
        diagnostic_layers: int = 2,
        fusion_layers: int = 2,
        dropout: float = 0.05,
        use_reliability_gate: bool = True,
    ) -> None:
        super().__init__()
        self.state_dim = state_dim
        self.morphology_dim = morphology_dim
        self.joints = joints
        self.condition_phases = condition_phases
        self.frames = frames
        self.model_dim = model_dim
        self.heads = heads
        self.layers = layers
        self.diagnostic_layers = diagnostic_layers
        self.fusion_layers = fusion_layers
        self.dropout = dropout
        self.use_reliability_gate = use_reliability_gate
        self.condition = PhaseJointReliabilityEncoder(
            morphology_dim=morphology_dim,
            joints=joints,
            phases=condition_phases,
            model_dim=model_dim,
            heads=heads,
            diagnostic_layers=diagnostic_layers,
            fusion_layers=fusion_layers,
            dropout=dropout,
            use_reliability_gate=use_reliability_gate,
        )
        self.state_projection = nn.Linear(state_dim + 6, model_dim)
        self.diffusion_step = nn.Sequential(
            nn.Linear(model_dim, model_dim * 2),
            nn.SiLU(),
            nn.Linear(model_dim * 2, model_dim),
        )
        self.time_embedding = nn.Parameter(torch.zeros(1, frames, model_dim))
        self.cross_attention = nn.MultiheadAttention(
            model_dim, heads, dropout=dropout, batch_first=True
        )
        self.cross_norm = nn.LayerNorm(model_dim)
        temporal_layer = nn.TransformerEncoderLayer(
            model_dim,
            heads,
            dim_feedforward=model_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.temporal = nn.TransformerEncoder(temporal_layer, layers)
        self.output = nn.Sequential(
            nn.LayerNorm(model_dim), nn.Linear(model_dim, state_dim)
        )
        self.register_buffer(
            "phase_features", _phase_features(frames), persistent=False
        )
        nn.init.normal_(self.time_embedding, std=0.02)

    def config(self) -> dict[str, int | float]:
        return {
            "state_dim": self.state_dim,
            "morphology_dim": self.morphology_dim,
            "joints": self.joints,
            "condition_phases": self.condition_phases,
            "frames": self.frames,
            "model_dim": self.model_dim,
            "heads": self.heads,
            "layers": self.layers,
            "diagnostic_layers": self.diagnostic_layers,
            "fusion_layers": self.fusion_layers,
            "dropout": self.dropout,
            "use_reliability_gate": self.use_reliability_gate,
        }

    def encode_condition(
        self,
        directions: Tensor,
        confidence: Tensor,
        morphology: Tensor,
        handedness: Tensor,
    ) -> tuple[Tensor, Tensor]:
        return self.condition(directions, confidence, morphology, handedness)

    def denoise(
        self,
        noisy_state: Tensor,
        diffusion_step: Tensor,
        phase_tokens: Tensor,
    ) -> Tensor:
        batch, frames, state_dim = noisy_state.shape
        if frames != self.frames or state_dim != self.state_dim:
            raise ValueError(
                f"state must have shape (B, {self.frames}, {self.state_dim})"
            )
        phase = self.phase_features[None].expand(batch, -1, -1)
        hidden = self.state_projection(torch.cat((noisy_state, phase), dim=-1))
        step = self.diffusion_step(
            timestep_embedding(diffusion_step, self.model_dim)
        )
        hidden = hidden + self.time_embedding + step[:, None]
        conditioned, _ = self.cross_attention(hidden, phase_tokens, phase_tokens)
        hidden = self.cross_norm(hidden + conditioned)
        return self.output(self.temporal(hidden))

    def forward(
        self,
        noisy_state: Tensor,
        diffusion_step: Tensor,
        directions: Tensor,
        confidence: Tensor,
        morphology: Tensor,
        handedness: Tensor,
    ) -> tuple[Tensor, Tensor]:
        phase_tokens, reliability_logits = self.encode_condition(
            directions, confidence, morphology, handedness
        )
        return (
            self.denoise(noisy_state, diffusion_step, phase_tokens),
            reliability_logits,
        )
