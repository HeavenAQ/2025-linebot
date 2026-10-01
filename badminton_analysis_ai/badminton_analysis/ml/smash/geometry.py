"""Frozen checkpoint-local geometry ported from the accepted 2026-09-13 scorer."""

import numpy as np
from scipy.ndimage import median_filter


def legacy_rotation_trace(poses, confidence, count=17):
    """Invariant to shared per-frame image translation, roll, and uniform scale.

    Ratios and relative axis angles preserve differential torso changes while
    removing shared lean. They cannot identify every kind of out-of-plane turn.
    Missing/foreshortened endpoints are unknown, not zero rotation.
    """
    p, c = np.asarray(poses, float), np.asarray(confidence, float)
    if p.ndim != 3 or p.shape[1:] != (17, 2) or c.shape != p.shape[:2] or len(p) < 3:
        raise ValueError("At least three COCO17 frames with confidence required")
    axes = [p[:, b] - p[:, a] for a, b in ((5, 6), (11, 12), (15, 16))]
    lengths = [np.linalg.norm(v, axis=1) for v in axes]
    scale = np.linalg.norm(p[:, [5, 6]].mean(1) - p[:, [11, 12]].mean(1), axis=1)
    output = np.full((count, 4), np.nan)
    certainty, reasons = [], []
    for k, joints in enumerate(((5, 6, 11, 12), (11, 12, 15, 16))):
        quality = c[:, joints].min(1)
        valid = (
            np.isfinite(p[:, joints]).all(axis=(1, 2))
            & np.isfinite(quality)
            & (quality > 0.05)
            & (scale > 1e-6)
            & (lengths[k] > 0.05 * scale)
            & (lengths[k + 1] > 0.05 * scale)
        )
        indices = np.flatnonzero(valid)
        if (
            not valid[0]
            or not valid[-1]
            or valid.mean() < 0.8
            or np.max(np.diff(indices), initial=0) > 4
        ):
            certainty.extend([0.0, 0.0])
            reasons.append("Missing/foreshortened relative axis")
            continue
        u, v = axes[k][valid], axes[k + 1][valid]
        angle = np.unwrap(
            np.arctan2(u[:, 0] * v[:, 1] - u[:, 1] * v[:, 0], (u * v).sum(1))
        )
        if np.any(np.abs(np.diff(angle)) / np.diff(indices) > np.pi / 3):
            certainty.extend([0.0, 0.0])
            reasons.append("Abrupt relative-axis flip")
            continue
        ratio = np.log(lengths[k][valid] / lengths[k + 1][valid])
        query = np.linspace(0, len(p) - 1, count)
        output[:, 2 * k] = np.interp(query, indices, angle) / np.pi
        output[:, 2 * k + 1] = np.interp(query, indices, ratio)
        certainty.extend([float(np.median(quality[valid]))] * 2)
        reasons.append(None)
    return output - output[0], np.array(certainty), reasons


def angle_trace(poses, confidence, handedness="right"):
    p, c = np.asarray(poses, float), np.asarray(confidence, float)
    if p.ndim != 3 or p.shape[1:] != (17, 2) or c.shape != p.shape[:2] or len(p) < 5:
        raise ValueError("At least five COCO17 frames and matching confidence required")
    if handedness not in ("left", "right"):
        raise ValueError("Explicit handedness required")
    hip, vertex, other = (12, 16, 15) if handedness == "right" else (11, 15, 16)
    u, v = p[:, hip] - p[:, vertex], p[:, other] - p[:, vertex]
    scale = np.linalg.norm(p[0, [5, 6]].mean(0) - p[0, [11, 12]].mean(0))
    if not np.isfinite(scale) or scale <= 1e-6:
        raise ValueError("Degenerate initial spine")
    quality = c[:, [hip, vertex, other]].min(1)
    valid = (
        np.isfinite(u).all(1)
        & np.isfinite(v).all(1)
        & np.isfinite(quality)
        & (quality >= 0.5)
        & (np.linalg.norm(u, axis=1) > 0.05 * scale)
        & (np.linalg.norm(v, axis=1) > 0.05 * scale)
    )
    indices = np.flatnonzero(valid)
    if (
        not valid[0]
        or not valid[-1]
        or valid.mean() < 0.8
        or np.max(np.diff(indices), initial=0) > 4
    ):
        raise ValueError("Angle has unreliable/degenerate endpoints or a long gap")
    a, b = u[valid], v[valid]
    theta = np.rad2deg(
        np.arctan2(np.abs(a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0]), (a * b).sum(1))
    )
    filled = np.interp(np.arange(len(p)), indices, theta)
    smooth = median_filter(filled, size=5, mode="nearest")
    return smooth, {
        "valid_fraction": float(valid.mean()),
        "interpolated_frames": np.flatnonzero(~valid).tolist(),
        "minimum_valid_confidence": float(quality[valid].min()),
    }


def features(theta):
    a = np.asarray(theta, float)
    if a.ndim != 1 or len(a) < 2 or not np.isfinite(a).all():
        raise ValueError("Finite chronological angle trace required")
    rise = a - np.minimum.accumulate(a)
    return {
        "initial_degrees": float(a[0]),
        "final_degrees": float(a[-1]),
        "net_change_degrees": float(a[-1] - a[0]),
        "excursion_degrees": float(np.ptp(a)),
        "ordered_opening_degrees": float(rise.max()),
        "ordered_closing_degrees": float((np.maximum.accumulate(a) - a).max()),
        "opening_peak_offset": int(rise.argmax()),
    }


def axis_angles(poses, confidence, pair):
    p, c = np.asarray(poses, float), np.asarray(confidence, float)
    left, right = pair
    axis = p[:, right] - p[:, left]
    scale = np.linalg.norm(p[0, [5, 6]].mean(0) - p[0, [11, 12]].mean(0))
    quality = c[:, [left, right]].min(1)
    valid = (
        np.isfinite(axis).all(1)
        & np.isfinite(quality)
        & (quality >= 0.5)
        & (np.linalg.norm(axis, axis=1) > 0.05 * scale)
    )
    ids = np.flatnonzero(valid)
    if (
        scale <= 1e-6
        or not valid[0]
        or not valid[-1]
        or valid.mean() < 0.8
        or np.max(np.diff(ids), initial=0) > 4
    ):
        raise ValueError("Unreliable or foreshortened shoulder/hip axis")
    a = np.rad2deg(np.unwrap(np.arctan2(axis[valid, 1], axis[valid, 0])))
    if np.any(np.abs(np.diff(a)) / np.diff(ids) > 60):
        raise ValueError("Abrupt axis flip")
    return median_filter(np.interp(np.arange(len(p)), ids, a), size=5, mode="nearest")


def combined_features(poses, confidence, handedness="right"):
    ankle, quality = angle_trace(poses, confidence, handedness)
    shoulder = axis_angles(poses, confidence, (5, 6))
    hip = axis_angles(poses, confidence, (11, 12))
    # Anatomical left->right image axes: canonical right-handed expert motion
    # decreases the shoulder angle, increasing hip-minus-shoulder orientation.
    # Mirror canonicalization for a left-handed input reverses the angle sign.
    sign = 1 if handedness == "right" else -1
    shoulder_turn = -sign * (shoulder - shoulder[0])
    relative_turn = sign * ((hip - shoulder) - (hip[0] - shoulder[0]))
    traces = np.column_stack((ankle - ankle[0], shoulder_turn, relative_turn))
    amount = np.array(
        [features(traces[:, i])["ordered_opening_degrees"] for i in range(3)]
    )
    return amount, traces, quality


def rotation_credit(amount, lower, full, hard_presence=True):
    amount, lower, full = (np.asarray(x, float) for x in (amount, lower, full))
    if (
        any(x.shape != (3,) for x in (amount, lower, full))
        or not np.isfinite([amount, lower, full]).all()
    ):
        raise ValueError("Three finite angular features and references required")
    if np.any(amount < 0) or np.any(lower <= 0) or np.any(full < lower):
        raise ValueError("Nonnegative motion and positive ordered references required")
    present = bool(np.all(amount >= lower - 1e-8))
    credit = float(np.clip(amount / full, 0, 1).min())
    return (credit if present or not hard_presence else 0.0), present


def preparation_height(poses, confidence, handedness="right"):
    """Median wrist height along the torso axis: hips=0, shoulders=1.

    This is a body-keypoint proxy, not a measurement of the racket head or the
    anatomical waist. Low-confidence frames are not interpreted as bad motion.
    """
    p, c = np.asarray(poses, float), np.asarray(confidence, float)
    if p.ndim != 3 or p.shape[1:] != (17, 2) or c.shape != p.shape[:2] or not len(p):
        raise ValueError("Nonempty matching COCO17 preparation poses required")
    if handedness not in ("right", "left"):
        raise ValueError("Explicit handedness required")
    wrist = 10 if handedness == "right" else 9
    pelvis = p[:, [11, 12]].mean(1)
    spine = p[:, [5, 6]].mean(1) - pelvis
    length2 = np.square(spine).sum(1)
    joints = [5, 6, 11, 12, wrist]
    valid = (
        np.isfinite(p[:, joints]).all((1, 2))
        & np.isfinite(c[:, joints]).all(1)
        & (c[:, joints].min(1) >= 0.5)
        & (length2 > 1e-8)
    )
    if valid.mean() < 0.8:
        raise ValueError("Insufficient reliable preparation wrist/torso evidence")
    height = (
        np.sum((p[valid, wrist] - pelvis[valid]) * spine[valid], axis=1)
        / length2[valid]
    )
    return {
        "height": float(np.median(height)),
        "valid_frames": int(valid.sum()),
        "frames": len(p),
        "heights": height.tolist(),
    }


def cap_preparation(original, height, full_credit_height):
    """Keep existing credit below expert support; zero at shoulder height."""
    if not np.isfinite([original, height, full_credit_height]).all():
        raise ValueError("Finite score and heights required")
    if not 0 <= original <= 5 or not 0 < full_credit_height < 1:
        raise ValueError(
            "Score must be 0..5; upper waist tolerance must be below shoulders"
        )
    fraction = float(np.clip((1 - height) / (1 - full_credit_height), 0, 1) ** 2)
    return min(float(original), 5 * fraction), fraction


def measure_balance_height(
    poses, confidence, initial, start, end, contact, handedness="right", fps=30.0
):
    p, c = np.asarray(poses, float), np.asarray(confidence, float)
    if p.ndim != 3 or p.shape[1:] != (17, 2) or c.shape != p.shape[:2]:
        raise ValueError("Dense COCO17 source poses and confidences required")
    if handedness not in ("left", "right") or fps != 30.0:
        raise ValueError("Explicit handedness and verified 30fps required")
    if not (0 <= initial <= end < contact < len(p) and 0 <= start <= end):
        raise ValueError("Source-frame balancing interval must precede contact")
    result = dict(
        initial_frame=int(initial),
        start=int(start),
        end=int(end),
        contact=int(contact),
        handedness=handedness,
        width=5,
    )
    reference_joints = [5, 6, 11, 12]
    if (
        not np.isfinite(p[initial, reference_joints]).all()
        or not np.isfinite(c[initial, reference_joints]).all()
        or c[initial, reference_joints].min() < 0.5
    ):
        return dict(
            result, available=False, reason="Unreliable initial torso reference"
        )
    spine = p[initial, [5, 6]].mean(0) - p[initial, [11, 12]].mean(0)
    length = float(np.linalg.norm(spine))
    if length <= 1e-6:
        return dict(
            result, available=False, reason="Degenerate initial torso reference"
        )
    up = spine / length
    ds, dw, os, ow = (6, 10, 5, 9) if handedness == "right" else (5, 9, 6, 10)
    x, conf = p[start : end + 1], c[start : end + 1]
    joints = [ds, dw, os, ow]
    valid = (
        np.isfinite(x[:, joints]).all(axis=(1, 2))
        & np.isfinite(conf[:, joints]).all(axis=1)
        & (conf[:, joints] >= 0.5).all(axis=1)
    )
    gap = (x[:, ds] - x[:, dw]) @ up / length
    other_gap = (x[:, os] - x[:, ow]) @ up / length
    # Do not grade ordinary preparation before the supporting hand is raised.
    active = valid & (other_gap <= 0.1)
    result.update(
        valid_fraction=float(valid.mean()),
        active_frames=int(active.sum()),
        scale_pixels=length,
        gap_trace=[float(v) if ok else None for v, ok in zip(gap, valid)],
        active_trace=active.tolist(),
    )
    if valid.mean() < 0.8:
        return dict(
            result, available=False, reason="Less than 80% reliable balancing evidence"
        )
    runs = [
        (float(gap[i : i + 5].min()), i)
        for i in range(len(x) - 4)
        if active[i : i + 5].all()
    ]
    if not runs:
        return dict(
            result, available=False, reason="No five-frame raised-support-hand window"
        )
    worst, index = max(runs, key=lambda item: (item[0], -item[1]))
    return dict(
        result,
        available=True,
        sustained_gap=max(0.0, worst),
        event_start=int(start + index),
        event_end=int(start + index + 4),
    )


def cap_balance(previous, sustained_gap, expert_tolerance):
    """Allow expert asymmetry; cap at zero for a racket hand a torso below shoulder."""
    if not np.isfinite([previous, sustained_gap, expert_tolerance]).all():
        raise ValueError("Finite score and height evidence required")
    if not 0 <= previous <= 5 or sustained_gap < 0 or not 0 <= expert_tolerance < 1:
        raise ValueError("Invalid score, gap, or expert tolerance")
    fraction = float(np.clip((1 - sustained_gap) / (1 - expert_tolerance), 0, 1) ** 2)
    return min(float(previous), 5 * fraction), fraction
