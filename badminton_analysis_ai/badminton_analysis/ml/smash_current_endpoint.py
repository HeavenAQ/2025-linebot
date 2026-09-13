"""Frozen interval ShapeDTW and best-endpoint scoring primitives."""

import numpy as np

JOINTS = (
    (6, 8, 10, 11, 12),
    (5, 6, 11, 12),
    (5, 6, 7, 8, 9),
    (0, 6, 8),
    (6, 8, 10, 12),
    (5, 6, 10, 11, 12),
)
METHODS = ("euclidean", "shape_dtw")


def joint_angles(pose, handedness="right"):
    """Two unsigned included angles in units of pi, with explicit invalidity."""
    p = np.asarray(pose, dtype=float)
    if p.shape[-2:] != (17, 2) or not np.isfinite(p).all():
        raise ValueError("Finite COCO-17 2D poses required")
    if handedness not in ("right", "left"):
        raise ValueError("Explicit handedness required")
    shoulder, elbow, wrist, hip = (
        (6, 8, 10, 12) if handedness == "right" else (5, 7, 9, 11)
    )
    values = []
    for a, b, c in ((hip, shoulder, elbow), (shoulder, elbow, wrist)):
        u, v = p[..., a, :] - p[..., b, :], p[..., c, :] - p[..., b, :]
        lu, lv = np.linalg.norm(u, axis=-1), np.linalg.norm(v, axis=-1)
        if np.any(lu < 1e-8) or np.any(lv < 1e-8):
            raise ValueError("Degenerate joint cannot receive an angle score")
        cosine = np.sum(u * v, axis=-1) / (lu * lv)
        values.append(np.arccos(np.clip(cosine, -1, 1)) / np.pi)
    return np.stack(values, axis=-1)


def score_distance(cost, calibration):
    tolerance, scale = calibration
    return np.exp(-np.maximum(0, np.asarray(cost) - tolerance) / scale)


def unit(x):
    norm = np.linalg.norm(x, axis=-1, keepdims=True)
    if not np.isfinite(x).all() or np.any(norm <= 1e-8):
        raise ValueError("Finite nondegenerate direction required")
    return x / norm


def descriptors(windows, initial):
    p, initial = np.asarray(windows, float), np.asarray(initial, float)
    if (
        p.shape != (6, 17, 17, 2)
        or initial.shape != (17, 2)
        or not np.isfinite(p).all()
        or not np.isfinite(initial).all()
    ):
        raise ValueError("Six finite17-frame windows and a fixed initial pose required")
    spine0 = initial[[5, 6]].mean(0) - initial[[11, 12]].mean(0)
    scale = np.linalg.norm(spine0)
    up = unit(spine0)
    basis = np.stack(([up[1], -up[0]], up), axis=-1)
    shoulder0 = (initial[6] - initial[5]) @ basis / scale
    hip0 = (initial[12] - initial[11]) @ basis / scale
    result = []
    for k, x in enumerate(p):
        vector = lambda a, b: (x[:, a] - x[:, b]) @ basis / scale
        spine = x[:, [5, 6]].mean(1) - x[:, [11, 12]].mean(1)
        if k == 0:
            feature = np.c_[
                (x[:, 10] - x[:, [11, 12]].mean(1)) @ basis / scale, vector(8, 6)
            ]
        elif k == 1:
            feature = np.c_[
                vector(6, 5) - shoulder0, vector(12, 11) - hip0, unit(spine) @ basis
            ]
        elif k == 2:
            feature = np.c_[vector(7, 5), vector(8, 6), vector(9, 5)]
        elif k == 3:
            feature = np.c_[unit(x[:, 8] - x[:, 6]) @ basis, vector(8, 0)]
        elif k == 4:
            feature = joint_angles(x)
        else:
            feature = np.c_[
                unit(spine) @ basis, vector(6, 5) - shoulder0, vector(10, 6)
            ]
        result.append(feature)
    return result


def reliability(confidence, initial_confidence):
    c, initial = np.asarray(confidence, float), np.asarray(initial_confidence, float)
    if (
        c.shape != (6, 17, 17)
        or initial.shape != (17,)
        or not np.isfinite(c).all()
        or not np.isfinite(initial).all()
        or np.any((c < 0) | (c > 1))
        or np.any((initial < 0) | (initial > 1))
    ):
        raise ValueError("Finite bounded checkpoint confidence required")
    context = initial[[5, 6, 11, 12]].min()
    return np.asarray(
        [
            c[k, 0, list(JOINTS[k])].min()
            if k == 4
            else min(float(np.median(c[k][:, JOINTS[k]].min(1))), float(context))
            for k in range(6)
        ]
    )


def local_features(points, initial, checkpoint):
    """Reuse the frozen rubric feature function on any number of frames."""
    output = []
    for start in range(0, len(points), 17):
        part = points[start : start + 17]
        padded = np.pad(part, ((0, 17 - len(part)), (0, 0), (0, 0)), mode="edge")
        output.append(
            descriptors(np.repeat(padded[None], 6, axis=0), initial)[checkpoint][
                : len(part)
            ]
        )
    return np.concatenate(output)


def normalized_shape_dtw(left, right, radius=2, band=2):
    a, b = np.asarray(left, float), np.asarray(right, float)
    if (
        a.shape != b.shape
        or a.ndim != 2
        or not a.shape[0]
        or not a.shape[1]
        or not np.isfinite(a).all()
        or not np.isfinite(b).all()
    ):
        raise ValueError("Matching nonempty finite temporal feature matrices required")
    if (
        not isinstance(radius, int)
        or radius < 0
        or not isinstance(band, int)
        or band < 0
    ):
        raise ValueError("Nonnegative integer radius/band required")

    def descriptors(x):
        padded = np.pad(x, ((radius, radius), (0, 0)), mode="edge")
        return np.stack([padded[i : i + 2 * radius + 1].ravel() for i in range(len(x))])

    da, db = descriptors(a), descriptors(b)
    local = np.linalg.norm(da[:, None] - db[None], axis=-1) / np.sqrt(2 * radius + 1)
    raw = np.linalg.norm(a[:, None] - b[None], axis=-1)
    n, m = local.shape
    dp = np.full((n + 1, m + 1), np.inf)
    dp[0, 0] = 0
    parents = {}
    for i in range(1, n + 1):
        for j in range(max(1, i - band), min(m, i + band) + 1):
            choices = [
                (dp[i - 1, j - 1] + 2 * local[i - 1, j - 1], i - 1, j - 1, 2),
                (dp[i - 1, j] + local[i - 1, j - 1], i - 1, j, 1),
                (dp[i, j - 1] + local[i - 1, j - 1], i, j - 1, 1),
            ]
            cost, pi, pj, weight = min(choices, key=lambda item: item[0])
            dp[i, j] = cost
            parents[i, j] = (pi, pj, weight)
    if not np.isfinite(dp[n, m]):
        raise ValueError("No feasible local path")
    i, j = n, m
    path = []
    weights = []
    while i or j:
        pi, pj, weight = parents[i, j]
        path.append((i - 1, j - 1))
        weights.append(weight)
        i, j = pi, pj
    path = np.array(path[::-1], int)
    weights = np.array(weights[::-1], int)
    assert weights.sum() == n + m
    cost = float(np.sum(raw[path[:, 0], path[:, 1]] * weights) / (n + m))
    return dict(
        cost=cost, descriptor_cost=float(dp[n, m] / (n + m)), path=path, weights=weights
    )


def interval_frames(interval, criterion):
    start, anchor, end = [interval[k] for k in ("start", "anchor", "end")]
    if (
        any(not isinstance(x, (int, np.integer)) for x in (start, anchor, end))
        or not 0 <= start <= anchor <= end
    ):
        raise ValueError("Ordered integer source interval required")
    if criterion == 4:
        return np.full(17, anchor, int)
    frames = np.rint(np.linspace(start, end, 17)).astype(int)
    if anchor not in frames:
        index = 1 + np.argmin(abs(frames[1:-1] - anchor))
        frames[index] = anchor
        frames.sort()
    return frames


def local_cost(left, right, criterion, method):
    if method not in METHODS or left.shape != right.shape or len(left) != 17:
        raise ValueError("Matching17-frame features and registered metric required")
    if criterion == 4:
        return float(np.sqrt(np.mean((left[0] - right[0]) ** 2)))
    if method == "euclidean":
        return float(np.sqrt(np.mean((left - right) ** 2)))
    return normalized_shape_dtw(left, right, radius=2, band=2)["cost"] / np.sqrt(
        left.shape[1]
    )


def generated_shoulder_cut(
    corrected, frames, acceleration_frame, forward_sign, shoulder=6
):
    """Search every available frame strictly after the existing acceleration anchor.

    Front means signed image-horizontal shoulder displacement from pelvis,
    not absolute screen position. Direction must be supplied from the video;
    handedness alone does not determine viewing direction. Do not smooth or
    extrapolate the shoulder curve. Earliest exact maximum resolves ties.
    """
    q, frames = np.asarray(corrected, float), np.asarray(frames)
    if (
        q.ndim != 3
        or q.shape[1:] != (17, 2)
        or frames.shape != (len(q),)
        or not np.issubdtype(frames.dtype, np.integer)
        or len(q) < 2
        or not np.isfinite(q).all()
        or np.any(np.diff(frames) != 1)
    ):
        raise ValueError(
            "Finite generated poses on a contiguous integer source clock required"
        )
    if forward_sign not in (-1, 1) or shoulder not in (5, 6):
        raise ValueError("Explicit viewing direction and hitting shoulder required")
    if not frames[0] <= acceleration_frame < frames[-1]:
        raise ValueError(
            "Acceleration anchor must be inside generated coverage with a later frame"
        )
    position = forward_sign * (q[:, shoulder, 0] - q[:, [11, 12], 0].mean(1))
    candidates = np.flatnonzero(frames > acceleration_frame)
    selected = int(candidates[np.argmax(position[candidates])])
    return dict(
        start=int(frames[0]),
        end=int(frames[selected]),
        rule="generated_shoulder_forwardmost_after_acceleration_v1",
        acceleration_source_frame=int(acceleration_frame),
        searched_through_source_frame=int(frames[-1]),
        forward_sign=int(forward_sign),
        shoulder_joint=shoulder,
        shoulder_forward_displacement_px=float(position[selected]),
        coordinates="shoulder_x_minus_pelvis_x",
        ties="earliest",
        selected_last_available_frame=bool(selected == len(q) - 1),
    )


def select_best_endpoint(frames, costs, scores, valid):
    frames, costs, scores, valid = map(np.asarray, (frames, costs, scores, valid))
    if frames.ndim != 1 or any(x.shape != frames.shape for x in (costs, scores, valid)):
        raise ValueError("One matching vector per candidate field required")
    if not np.issubdtype(frames.dtype, np.integer) or np.any(np.diff(frames) <= 0):
        raise ValueError("Ordered distinct source frames required")
    eligible = np.flatnonzero(valid & np.isfinite(costs) & np.isfinite(scores))
    if not len(eligible):
        raise ValueError("No reliable endpoint candidate")
    # Actual confidence-adjusted credit, not a manually chosen favourable frame.
    return min(eligible, key=lambda i: (-scores[i], costs[i], frames[i]))
