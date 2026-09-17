"""Observed-only contact-anchored checkpoint transfer from the fixed expert template."""

import numpy as np
from scipy.ndimage import median_filter

TRIPLES = (
    (11, 5, 7),
    (12, 6, 8),
    (5, 7, 9),
    (6, 8, 10),
    (5, 11, 13),
    (6, 12, 14),
    (11, 13, 15),
    (12, 14, 16),
)


def motion_features(pose, confidence):
    p, c = np.asarray(pose, float), np.asarray(confidence, float)
    if p.ndim != 3 or p.shape[1:] != (17, 2) or c.shape != p.shape[:-1]:
        raise ValueError("Matching interval COCO17 coordinates and confidence required")
    if (
        not np.isfinite(p).all()
        or not np.isfinite(c).all()
        or np.any((c < 0) | (c > 1))
    ):
        raise ValueError("Finite input and bounded confidence required")
    values, weights = [], []
    for a, b, d in TRIPLES:
        u, v = p[:, a] - p[:, b], p[:, d] - p[:, b]
        denominator = np.linalg.norm(u, axis=1) * np.linalg.norm(v, axis=1)
        valid = denominator > 1e-8
        cos = np.divide((u * v).sum(1), denominator, out=np.zeros(len(p)), where=valid)
        sin = np.divide(
            abs(u[:, 0] * v[:, 1] - u[:, 1] * v[:, 0]),
            denominator,
            out=np.zeros(len(p)),
            where=valid,
        )
        values.extend((np.clip(cos, -1, 1), np.clip(sin, 0, 1)))
        weight = np.min(c[:, [a, b, d]], axis=1) * valid
        weights.extend((weight, weight))
    vectors = (
        p[:, [5, 6]].mean(1) - p[:, [11, 12]].mean(1),
        p[:, 6] - p[:, 5],
        p[:, 12] - p[:, 11],
    )
    joints = ([5, 6, 11, 12], [5, 6], [11, 12])
    for vector, ids in zip(vectors, joints):
        norm = np.linalg.norm(vector, axis=1)
        unit = np.divide(
            vector, norm[:, None], out=np.zeros_like(vector), where=norm[:, None] > 1e-8
        )
        values.extend((unit[:, 0], unit[:, 1]))
        weight = np.min(c[:, ids], axis=1) * (norm > 1e-8)
        weights.extend((weight, weight))
    return np.array(values).T, np.array(weights).T


def descriptors(values, radius=2):
    x = np.asarray(values, float)
    if x.ndim != 2 or len(x) < 2 or not np.isfinite(x).all():
        raise ValueError("Finite interval feature matrix required")
    padded = np.pad(x, ((radius, radius), (0, 0)), mode="edge")
    return np.stack([padded[i : i + 2 * radius + 1].ravel() for i in range(len(x))])


def local_cost(a, b, ca, cb):
    weights = np.minimum(ca[:, None], cb[None])
    denominator = weights.sum(-1)
    return np.sqrt(
        np.divide(
            ((a[:, None] - b[None]) ** 2 * weights).sum(-1),
            denominator,
            out=np.full(denominator.shape, np.inf),
            where=denominator > 1e-8,
        )
    )


def observed_features(pixels, confidence):
    """Translation/scale invariant, without rotating away shoulder/hip motion."""
    points = np.asarray(pixels, dtype=float).copy()
    confidence = np.asarray(confidence, dtype=float)
    if (
        points.ndim != 3
        or points.shape[1:] != (17, 2)
        or confidence.shape != points.shape[:2]
    ):
        raise ValueError("Expected full source-clock COCO17 poses and confidence")
    clock = np.arange(len(points))
    for joint in range(17):
        if joint not in range(5, 13):
            # Alignment uses shoulders, elbows, wrists and hips only. An
            # occluded ear/knee must not reject otherwise observable motion.
            points[:, joint] = np.nan_to_num(points[:, joint])
            continue
        valid = np.isfinite(points[:, joint]).all(1) & (confidence[:, joint] >= 0.25)
        if valid.sum() < 3:
            raise ValueError(f"Insufficient observed evidence for joint {joint}")
        for axis in range(2):
            points[:, joint, axis] = np.interp(
                clock, clock[valid], points[valid, joint, axis]
            )
    # Three-frame median rejects isolated detector spikes, not global movement.
    points = median_filter(points, size=(3, 1, 1), mode="nearest")
    root = points[:, [11, 12]].mean(1)
    torso = points[:, [5, 6]].mean(1) - root
    scale = float(np.median(np.linalg.norm(torso, axis=1)))
    if scale < 1:
        raise ValueError("Degenerate observed torso scale")
    angles, angle_confidence = motion_features(points, np.clip(confidence, 0, 1))
    # Both shoulders/elbows, spine direction, shoulder axis and hip axis.
    columns = [*range(8), *range(16, 22)]
    angles = angles[:, columns]
    angle_confidence = angle_confidence[:, columns]
    relative = ((points[:, 5:13] - root[:, None]) / scale).reshape(len(points), -1)
    relative_confidence = np.repeat(confidence[:, 5:13], 2, axis=1)
    return angles, angle_confidence, relative, relative_confidence


def observed_cost(reference, target):
    a, ca, pa, cpa = reference
    b, cb, pb, cpb = target
    angular = local_cost(
        descriptors(a), descriptors(b), descriptors(ca), descriptors(cb)
    )
    positional = local_cost(
        descriptors(pa), descriptors(pb), descriptors(cpa), descriptors(cpb)
    )
    return 0.6 * angular + 0.4 * positional


def outward_path(cost, expected_ratio=1.0, minimum_target_length=2):
    """Symmetric2 DTW, fixed contact origin and free outer endpoint.

    A small non-diagonal penalty limits holds. A weak physical-time prior breaks
    ties in long static preparation poses; it cannot replace pose matching.
    """
    raw = np.asarray(cost, float)
    n, m = raw.shape
    expected = np.arange(n)[:, None] * expected_ratio
    prior = 0.025 * np.abs(np.arange(m)[None] - expected) / 15.0
    local = raw + prior
    dp = np.full((n + 1, m + 1), np.inf)
    parent = np.full((n + 1, m + 1), -1, np.int8)
    dp[0, 0] = 0
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            options = (
                dp[i - 1, j - 1] + 2 * local[i - 1, j - 1],
                dp[i - 1, j] + local[i - 1, j - 1] + 0.04,
                dp[i, j - 1] + local[i - 1, j - 1] + 0.04,
            )
            step = int(np.argmin(options))
            dp[i, j], parent[i, j] = options[step], step
    eligible = np.arange(max(2, minimum_target_length), m + 1)
    if not len(eligible):
        raise ValueError("Insufficient target frames for an open-end alignment")
    endpoint = int(eligible[np.argmin(dp[n, eligible] / (n + eligible))])
    i, j = n, endpoint
    path = []
    while i or j:
        path.append((i - 1, j - 1))
        step = parent[i, j]
        if step in (0, 1):
            i -= 1
        if step in (0, 2):
            j -= 1
    return np.asarray(path[::-1], int), float(dp[n, endpoint] / (n + endpoint))


def align_contacts(cost, reference_contact, target_contact):
    n, m = cost.shape
    if not 0 < reference_contact < n - 2 or not 0 < target_contact < m - 2:
        raise ValueError("Both contact anchors must have pre/post-contact evidence")
    before = cost[: reference_contact + 1, : target_contact + 1][::-1, ::-1]
    ratio = min(1.0, target_contact / reference_contact)
    pre, pre_cost = outward_path(before, ratio, max(3, int(min(before.shape) * 0.7)))
    # The short reference ends during follow-through. Ignore long target rest tails.
    post_limit = min(m, target_contact + 3 * (n - reference_contact))
    after = cost[reference_contact:, target_contact:post_limit]
    post, post_cost = outward_path(after, 1.0, max(3, int(min(after.shape) * 0.5)))
    pre = np.array([reference_contact, target_contact]) - pre
    post = post + [reference_contact, target_contact]
    path = np.concatenate((pre[::-1], post[1:]))
    mapping = np.asarray(
        [int(np.rint(np.median(path[path[:, 0] == i, 1]))) for i in range(n)]
    )
    mapping[reference_contact] = target_contact
    if np.any(np.diff(mapping) < 0):
        raise ValueError("Nonmonotone contact mapping")
    return mapping, path, (pre_cost + post_cost) / 2


def transfer_intervals(annotations, mapping):
    output = {}
    for key, original in annotations.items():
        value = {
            field: int(mapping[original[field]]) for field in ("start", "end", "anchor")
        }
        value.update(
            reviewed=False,
            note="依張宸愷1.mp4人工區間，以偵測骨架時序對齊提出；待人工確認。",
        )
        output[key] = value
    return output
