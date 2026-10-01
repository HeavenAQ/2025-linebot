"""Frozen inference-only checkpoint graph; no trainer or research imports."""

import math
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

EDGES = (
    (0, 1),
    (0, 2),
    (1, 3),
    (2, 4),
    (5, 6),
    (5, 7),
    (7, 9),
    (6, 8),
    (8, 10),
    (5, 11),
    (6, 12),
    (11, 12),
    (11, 13),
    (13, 15),
    (12, 14),
    (14, 16),
)


def normalized_graph_pose(pose):
    p = np.asarray(pose, dtype=np.float32)
    if p.shape != (64, 17, 2) or not np.isfinite(p).all():
        raise ValueError("Expected finite COCO17 sequence")
    pelvis = p[:, [11, 12]].mean(1)
    scale = np.median(np.linalg.norm(p[:5, [5, 6]].mean(1) - pelvis[:5], axis=1))
    if scale < 1e-6:
        raise ValueError("Degenerate preparation torso")
    # Feature coordinates only; does not change either video overlay.
    return (p - pelvis[:, None]) / scale


def checkpoint_index_map(phase_indices, anchor_groups):
    phases = np.asarray(phase_indices)
    if (
        phases.shape != (5,)
        or not np.issubdtype(phases.dtype, np.integer)
        or phases[0] != 0
        or phases[-1] != 63
        or np.any(np.diff(phases) <= 0)
    ):
        raise ValueError("Expected five increasing cached anchors spanning 64 frames")
    windows, rule_indices, anchor_indices = [], [], []
    for rule, anchors in enumerate(anchor_groups):
        if not anchors:
            raise ValueError("Each checkpoint needs specified anchors")
        for anchor in anchors:
            if anchor not in range(5):
                raise ValueError("Invalid checkpoint anchor")
            windows.append(np.clip(phases[anchor] + np.arange(-2, 3), 0, 63))
            rule_indices.append(rule)
            anchor_indices.append(anchor)
    return np.asarray(windows), np.asarray(rule_indices), np.asarray(anchor_indices)


def checkpoint_windows(pose, confidence, indices):
    p = normalized_graph_pose(pose)
    c = np.asarray(confidence, np.float32)
    if c.shape != (64, 17) or not np.isfinite(c).all():
        raise ValueError("Expected finite confidence")
    indices = np.asarray(indices)
    if (
        indices.ndim != 2
        or indices.shape[1] != 5
        or not np.issubdtype(indices.dtype, np.integer)
        or np.any((indices < 0) | (indices > 63))
    ):
        raise ValueError("Explicit in-range five-frame checkpoint indices required")
    return p[indices], np.clip(c[indices], 0, 1)


def interpolate_vector_direction_length(left, right, amount):
    """Shortest-arc 2D direction interpolation with linearly varying length."""
    a, b = np.asarray(left, float), np.asarray(right, float)
    la, lb = np.linalg.norm(a, axis=-1), np.linalg.norm(b, axis=-1)
    ta, tb = np.arctan2(a[..., 1], a[..., 0]), np.arctan2(b[..., 1], b[..., 0])
    ta = np.where(la > 1e-8, ta, tb)
    tb = np.where(lb > 1e-8, tb, ta)
    delta = np.arctan2(np.sin(tb - ta), np.cos(tb - ta))
    theta = ta + amount * delta
    length = (1 - amount) * la + amount * lb
    return np.stack((np.cos(theta), np.sin(theta)), axis=-1) * length[..., None]


def kinematic_temporal_interpolation(left, right, amount):
    """Interpolate only a reference's own geometry, with no observed-pose fit."""
    a, b = np.asarray(left, float), np.asarray(right, float)
    if a.shape != b.shape or a.shape[-2:] != (17, 2) or amount.shape != a.shape[:-2]:
        raise ValueError("Expected matching COCO17 endpoint poses and fractions")
    pelvis_a, pelvis_b = a[..., [11, 12], :].mean(-2), b[..., [11, 12], :].mean(-2)
    shoulders_a, shoulders_b = a[..., [5, 6], :].mean(-2), b[..., [5, 6], :].mean(-2)
    pelvis = (1 - amount[..., None]) * pelvis_a + amount[..., None] * pelvis_b
    vector = lambda x, y: interpolate_vector_direction_length(x, y, amount)
    shoulders = pelvis + vector(shoulders_a - pelvis_a, shoulders_b - pelvis_b)
    hip_axis = vector(a[..., 12, :] - a[..., 11, :], b[..., 12, :] - b[..., 11, :])
    shoulder_axis = vector(a[..., 6, :] - a[..., 5, :], b[..., 6, :] - b[..., 5, :])
    out = np.zeros_like(a)
    out[..., 11, :], out[..., 12, :] = pelvis - hip_axis / 2, pelvis + hip_axis / 2
    out[..., 5, :], out[..., 6, :] = (
        shoulders - shoulder_axis / 2,
        shoulders + shoulder_axis / 2,
    )
    out[..., 0, :] = shoulders + vector(
        a[..., 0, :] - shoulders_a, b[..., 0, :] - shoulders_b
    )
    for parent, child in (
        (0, 1),
        (0, 2),
        (1, 3),
        (2, 4),
        (5, 7),
        (7, 9),
        (6, 8),
        (8, 10),
        (11, 13),
        (13, 15),
        (12, 14),
        (14, 16),
    ):
        out[..., child, :] = out[..., parent, :] + vector(
            a[..., child, :] - a[..., parent, :], b[..., child, :] - b[..., parent, :]
        )
    return out


def checkpoint_reference_at_source_frames(
    pose, native_phases, source_phases, target_source_frames, interpolation="cartesian"
):
    """Sample the corrected pose at exactly the observed checkpoint timestamps.

    This changes temporal sampling only. The existing clip-scale/body-relative
    feature normalization remains unchanged; no spatial registration is fitted.
    Each query uses adjacent native samples, not a full-interval descriptor.
    """
    native = np.asarray(native_phases, float)
    source = np.asarray(source_phases, float)
    target = np.asarray(target_source_frames, float)
    if (
        native.shape != (5,)
        or source.shape != (5,)
        or not np.isfinite(native).all()
        or not np.isfinite(source).all()
        or native[0] != 0
        or native[-1] != 63
        or np.any(np.diff(native) <= 0)
        or np.any(np.diff(source) <= 0)
    ):
        raise ValueError("Expected five increasing native/source anchors")
    if target.ndim != 2 or target.shape[1] != 5 or not np.isfinite(target).all():
        raise ValueError("Expected explicit five-frame source checkpoint windows")
    if (
        np.any(target < source[0])
        or np.any(target > source[-1])
        or np.any(np.diff(target, axis=1) < 0)
    ):
        raise ValueError("Target times must be ordered and inside the source interval")
    query = np.interp(target, source, native)
    before = np.floor(query).astype(int)
    after = np.minimum(before + 1, 63)
    alpha = (query - before)[..., None, None]
    normalized = normalized_graph_pose(pose)
    if interpolation == "cartesian":
        values = (1 - alpha) * normalized[before] + alpha * normalized[after]
    elif interpolation == "kinematic":
        values = kinematic_temporal_interpolation(
            normalized[before], normalized[after], query - before
        )
    else:
        raise ValueError("Expected cartesian or kinematic temporal interpolation")
    return values.astype(np.float32), query


def graph_inputs(coordinates, confidence, visible=None):
    """No masked target leaks through velocity or confidence channels."""
    if coordinates.ndim != 4 or coordinates.shape[1:] != (5, 17, 2):
        raise ValueError("Expected B,5,17,2 checkpoint coordinates")
    if confidence.shape != coordinates.shape[:-1]:
        raise ValueError("Confidence shape mismatch")
    visible = torch.ones_like(confidence) if visible is None else visible
    if visible.shape != confidence.shape:
        raise ValueError("Visibility shape mismatch")
    positions = coordinates * visible[..., None]
    valid_pair = visible[:, 1:] * visible[:, :-1]
    velocity = torch.zeros_like(positions)
    velocity[:, 1:] = (positions[:, 1:] - positions[:, :-1]) * valid_pair[..., None]
    values = torch.cat(
        [positions, velocity, (confidence * visible)[..., None], visible[..., None]],
        dim=-1,
    )
    return values.contiguous()


class GraphTemporalBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        adjacency = torch.zeros(17, 17)
        for a, b in EDGES:
            adjacency[a, b] = adjacency[b, a] = 1
        adjacency /= adjacency.sum(1, keepdim=True).clamp_min(1)
        self.register_buffer("adjacency", adjacency)
        self.self_map = nn.Linear(channels, channels)
        self.neighbor_map = nn.Linear(channels, channels, bias=False)
        self.temporal = nn.Linear(3 * channels, channels)
        self.norm = nn.LayerNorm(channels)

    def forward(self, x):
        # Spatial aggregation with an explicit shared adjacency matrix.
        neighbors = torch.matmul(self.adjacency, x)
        spatial = F.silu(self.self_map(x) + self.neighbor_map(neighbors))
        # Explicit three-tap temporal convolution in channel-last layout avoids
        # this environment's MPS Conv2d-backward noncontiguous-view failure.
        padded = F.pad(spatial, (0, 0, 0, 0, 1, 1))
        taps = torch.cat([padded[:, :-2], padded[:, 1:-1], padded[:, 2:]], dim=-1)
        return F.silu(x + self.norm(self.temporal(taps)))


class CheckpointGraphEncoder(nn.Module):
    def __init__(self, channels=32, layers=3, input_channels=6):
        super().__init__()
        self.config = {
            "channels": channels,
            "layers": layers,
            "input_channels": input_channels,
        }
        self.input = nn.Linear(input_channels, channels)
        self.blocks = nn.Sequential(
            *(GraphTemporalBlock(channels) for _ in range(layers))
        )
        self.decoder = nn.Linear(channels, 2)

    def encode(self, x):
        return self.blocks(F.silu(self.input(x)))

    def forward(self, x):
        return self.decoder(self.encode(x))


class CheckpointMetricGraph(nn.Module):
    def __init__(self, measured_joints, active_rules=(0, 2, 3, 4)):
        super().__init__()
        if (
            len(measured_joints) != 6
            or not active_rules
            or any(k not in range(6) for k in active_rules)
        ):
            raise ValueError("Invalid checkpoint contract")
        self.config = dict(
            measured_joints=measured_joints, active_rules=list(active_rules)
        )
        self.active_rules = tuple(active_rules)
        weights = torch.zeros(6, 17)
        for k, joints in enumerate(measured_joints):
            if not joints or any(j not in range(17) for j in joints):
                raise ValueError("Invalid measured joints")
            weights[k, joints] = 1 / len(joints)
        self.register_buffer("pool_weights", weights)
        active = torch.zeros(6, dtype=torch.bool)
        active[list(active_rules)] = True
        self.register_buffer("active", active)
        self.encoder = CheckpointGraphEncoder(input_channels=12)
        self.encoder.decoder = nn.Identity()
        self.projection = nn.Linear(32, 6 * 16)
        self.raw_intercept = nn.Parameter(torch.full((6,), math.log(math.expm1(2.0))))
        self.raw_scale = nn.Parameter(torch.full((6,), math.log(math.expm1(8.0))))

    def encode(self, pose, confidence, rule_ids):
        inputs = graph_inputs(pose, confidence)
        identity = (
            F.one_hot(rule_ids, 6).to(inputs.dtype)[:, None, None].expand(-1, 5, 17, -1)
        )
        features = self.encoder.encode(torch.cat((inputs, identity), dim=-1))
        pooled = (features * self.pool_weights[rule_ids, None, :, None]).sum(2).mean(1)
        projected = self.projection(pooled).reshape(-1, 6, 16)
        return F.normalize(
            projected[torch.arange(len(pose), device=pose.device), rule_ids],
            dim=-1,
            eps=1e-6,
        )

    def distance(self, observed, confidence, corrected, rule_ids):
        if (
            observed.shape != corrected.shape
            or rule_ids.shape != observed.shape[:1]
            or rule_ids.dtype != torch.long
            or torch.any((rule_ids < 0) | (rule_ids >= 6))
        ):
            raise ValueError("Matching checkpoint windows and IDs required")
        if not self.active[rule_ids].all():
            raise ValueError(
                "Unsupported head must use the explicitly declared baseline, not this model"
            )
        both = self.encode(
            torch.cat((observed, corrected)),
            torch.cat((confidence, confidence)),
            torch.cat((rule_ids, rule_ids)),
        )
        left, right = both.chunk(2)
        return (left - right).square().sum(-1)

    def forward(self, observed, confidence, corrected, rule_ids):
        return F.softplus(self.raw_intercept[rule_ids]) - F.softplus(
            self.raw_scale[rule_ids]
        ) * self.distance(observed, confidence, corrected, rule_ids)


def infer(model, p, c, q, ids):
    selected = np.isin(ids, model.active_rules)
    device = next(model.parameters()).device
    tensors = [
        torch.as_tensor(a[:, selected].reshape(-1, *a.shape[2:]), device=device)
        for a in (p, c, q)
    ]
    rid = torch.as_tensor(
        np.tile(ids[selected], len(p)), device=device, dtype=torch.long
    )
    with torch.no_grad():
        values = torch.sigmoid(model(*tensors, rid)).cpu().numpy().reshape(len(p), -1)
    output = np.full(p.shape[:2], np.nan)
    output[:, selected] = values
    return output


def aggregate(values, ids, active):
    output = np.full((len(values), 6), np.nan)
    for k in active:
        output[:, k] = values[:, ids == k].mean(1)
    return output
