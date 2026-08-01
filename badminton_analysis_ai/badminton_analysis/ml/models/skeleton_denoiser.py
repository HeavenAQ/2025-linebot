from __future__ import annotations

import torch
from torch import Tensor, nn


class SkeletonDenoiser(nn.Module):
    """Separable temporal/spatial Transformer for 17-joint pose correction."""

    def __init__(
        self,
        *,
        num_joints: int = 17,
        max_frames: int = 64,
        input_features: int = 4,
        model_dim: int = 128,
        num_heads: int = 4,
        temporal_layers: int = 3,
        spatial_layers: int = 2,
        dropout: float = 0.1,
        max_correction: float | None = None,
    ) -> None:
        super().__init__()
        self.num_joints = num_joints
        self.max_frames = max_frames
        self.input_features = input_features
        self.model_dim = model_dim
        self.num_heads = num_heads
        self.temporal_layers = temporal_layers
        self.spatial_layers = spatial_layers
        self.dropout = dropout
        self.max_correction = max_correction

        self.input_projection = nn.Linear(input_features, model_dim)
        self.joint_embedding = nn.Parameter(torch.zeros(1, 1, num_joints, model_dim))
        self.time_embedding = nn.Parameter(torch.zeros(1, max_frames, 1, model_dim))
        temporal_layer = nn.TransformerEncoderLayer(
            model_dim,
            num_heads,
            dim_feedforward=model_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        spatial_layer = nn.TransformerEncoderLayer(
            model_dim,
            num_heads,
            dim_feedforward=model_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.temporal_encoder = nn.TransformerEncoder(temporal_layer, temporal_layers)
        self.spatial_encoder = nn.TransformerEncoder(spatial_layer, spatial_layers)
        self.output_norm = nn.LayerNorm(model_dim)
        self.correction_head = nn.Linear(model_dim, 3)
        self.register_buffer(
            "expert_reference_skeletons", None, persistent=False
        )
        self.register_buffer(
            "expert_reference_confidence", None, persistent=False
        )

        nn.init.normal_(self.joint_embedding, std=0.02)
        nn.init.normal_(self.time_embedding, std=0.02)
        nn.init.zeros_(self.correction_head.weight)
        nn.init.zeros_(self.correction_head.bias)

    def set_expert_reference_bank(
        self, skeletons: Tensor, confidence: Tensor
    ) -> None:
        if skeletons.ndim != 4 or skeletons.shape[-1] != 3:
            raise ValueError("expert skeletons must have shape (N, T, J, 3)")
        if confidence.shape != skeletons.shape[:3]:
            raise ValueError("expert confidence must have shape (N, T, J)")
        self.expert_reference_skeletons = skeletons
        self.expert_reference_confidence = confidence

    def config(self) -> dict[str, int | float | None]:
        return {
            "num_joints": self.num_joints,
            "max_frames": self.max_frames,
            "input_features": self.input_features,
            "model_dim": self.model_dim,
            "num_heads": self.num_heads,
            "temporal_layers": self.temporal_layers,
            "spatial_layers": self.spatial_layers,
            "dropout": self.dropout,
            "max_correction": self.max_correction,
        }

    def forward(self, features: Tensor) -> Tensor:
        if features.ndim != 4:
            raise ValueError("features must have shape (B, T, J, F)")
        batch, frames, joints, _ = features.shape
        if joints != self.num_joints or frames > self.max_frames:
            raise ValueError(
                f"expected at most {self.max_frames} frames and {self.num_joints} joints"
            )
        hidden = self.input_projection(features)
        hidden = hidden + self.joint_embedding[:, :, :joints]
        hidden = hidden + self.time_embedding[:, :frames]

        temporal = hidden.permute(0, 2, 1, 3).reshape(batch * joints, frames, -1)
        temporal = self.temporal_encoder(temporal)
        hidden = temporal.reshape(batch, joints, frames, -1).permute(0, 2, 1, 3)

        spatial = hidden.reshape(batch * frames, joints, -1)
        spatial = self.spatial_encoder(spatial)
        hidden = spatial.reshape(batch, frames, joints, -1)
        correction: Tensor = self.correction_head(self.output_norm(hidden))
        if self.max_correction is not None:
            magnitude = correction.norm(dim=-1, keepdim=True).clamp_min(1e-8)
            correction = correction * (self.max_correction / magnitude).clamp(max=1.0)
        output: Tensor = features[..., :3] + correction
        return output
