"""Smash checkpoint rules: how each checkpoint is measured on the learner's clip."""

from dataclasses import dataclass
from pathlib import Path
import json

from scipy.ndimage import median_filter
import numpy as np


MAXIMA = np.array([5, 20, 5, 20, 30, 20.0])


LEGACY_MAXIMA = np.array([10, 10, 20, 20, 20, 20.0])


CRITERIA = (
    "preparation",
    "body_rotation",
    "arm_balance",
    "elbow_forward",
    "wrist_flick",
    "follow_through",
)


@dataclass(frozen=True)
class CurrentSmashCalibration:
    rotation_lower: tuple[float, ...]
    rotation_full: tuple[float, ...]
    preparation_height: float
    balance_gap: float
    shoulder_span_reference: float
    endpoint_map: tuple[float, float]
    legacy_rotation_reference: float

    @classmethod
    def load(cls, path: Path):
        raw = json.loads(path.read_text())
        if raw["version"] != "smash_local_20260913":
            raise ValueError("Unsupported smash calibration contract")
        return cls(**raw["calibration"])


def compose_points(prior, *, rotation=None, endpoint=None, span_gate=None):
    points = np.asarray(prior, float).copy()
    if (
        points.shape != (6,)
        or not np.isfinite(points).all()
        or np.any(points < 0)
        or np.any(points > LEGACY_MAXIMA + 1e-8)
    ):
        raise ValueError("Six bounded historical checkpoint scores required")
    for value, maximum in ((rotation, 1), (endpoint, 20), (span_gate, 1)):
        if value is not None and (not np.isfinite(value) or not 0 <= value <= maximum):
            raise ValueError("Invalid bounded checkpoint credit")
    if endpoint is not None:
        points[5] = max(points[5], endpoint)
    if span_gate is not None:
        points[5] *= span_gate
    points *= MAXIMA / LEGACY_MAXIMA
    if rotation is not None:
        points[1] = 20 * rotation
    return points


def score_source_evidence(
    *,
    poses,
    confidence,
    corrected_pixels,
    generated_frames,
    intervals,
    contact,
    historical_points,
    calibration,
    handedness="right",
    fps=30.0,
    forward_sign=1,
    rescore_interval=True,
    endpoint_acceleration=None,
):
    """Grade source poses against the frozen projected/EMA correction."""
    p, c, q = (
        np.asarray(x, dtype=float) for x in (poses, confidence, corrected_pixels)
    )
    frames = np.asarray(generated_frames)
    if (
        p.ndim != 3
        or p.shape[1:] != (17, 2)
        or c.shape != p.shape[:2]
        or q.shape != (len(frames), 17, 2)
        or not len(frames)
        or not np.issubdtype(frames.dtype, np.integer)
        or np.any(np.diff(frames) != 1)
        or not 0 <= frames[0] < contact < frames[-1] < len(p)
    ):
        raise ValueError("Explicit contiguous source-clock correction required")
    if handedness != "right" or fps != 30.0:
        raise ValueError(
            "This frozen full scorer requires right-canonical 30fps evidence"
        )
    if set(intervals) != set(CRITERIA):
        raise ValueError("All six source-clock checkpoint intervals required")
    for value in intervals.values():
        if not 0 <= value["start"] <= value["anchor"] <= value["end"] < len(p):
            raise ValueError("Checkpoint interval is outside source coverage")
    initial = int(frames[0])
    evidence = {}
    prior = np.asarray(historical_points, float).copy()
    compose_points(prior)  # Validate before changing a head.
    balance_anchor = int(intervals["arm_balance"]["anchor"])
    rotation = None
    try:
        if not initial + 2 <= balance_anchor < contact:
            raise ValueError("Unassessed pre-contact rotation window")
        amounts, _, quality = combined_features(
            p[initial : balance_anchor + 1], c[initial : balance_anchor + 1], handedness
        )
        rotation, present = rotation_credit(
            amounts, calibration.rotation_lower, calibration.rotation_full
        )
        evidence["body_rotation"] = dict(
            available=True,
            amounts=amounts.tolist(),
            credit=rotation,
            present=present,
            **quality,
        )
    except ValueError as error:
        evidence["body_rotation"] = dict(available=False, reason=str(error))

    if rescore_interval:
        try:
            selected = interval_frames(intervals["follow_through"], 5)
            if selected.min() < frames[0] or selected.max() > frames[-1]:
                raise ValueError("Marked follow-through lacks generated coverage")
            observed = local_features(p[selected], p[initial], 5)
            target = local_features(q[selected - initial], q[0], 5)
            cost = local_cost(observed, target, 5, "shape_dtw")
            certainty = reliability(
                np.repeat(c[selected][None], 6, axis=0), c[initial]
            )[5]
            prior[5] = 20 * (
                certainty * score_distance(cost, calibration.endpoint_map)
                + (1 - certainty) * 0.5
            )
            evidence["follow_interval"] = dict(available=True, cost=float(cost))
        except ValueError as error:
            evidence["follow_interval"] = dict(available=False, reason=str(error))

    endpoint, span_gate = None, None
    selected_endpoint = int(intervals["follow_through"]["anchor"])
    try:
        cut = generated_shoulder_cut(
            q,
            frames,
            contact if endpoint_acceleration is None else endpoint_acceleration,
            forward_sign,
        )
        reference = cut["end"]
        queries = np.arange(reference, len(p))
        target = local_features(q[[reference - initial]], q[0], 5)[0]
        values = local_features(p[queries], p[initial], 5)
        costs = np.sqrt(np.mean((values - target) ** 2, axis=1))
        certainty = np.array(
            [
                reliability(np.repeat(c[[i] * 17][None], 6, axis=0), c[initial])[5]
                for i in queries
            ]
        )
        scores = 20 * (
            certainty * score_distance(costs, calibration.endpoint_map)
            + (1 - certainty) * 0.5
        )
        best = select_best_endpoint(queries, costs, scores, certainty > 0.05)
        selected_endpoint, endpoint = int(queries[best]), float(scores[best])
        evidence["follow_through"] = dict(
            available=True,
            reference_frame=reference,
            selected_frame=selected_endpoint,
            candidate_score=endpoint,
            candidate_cost=float(costs[best]),
            search_end=len(p) - 1,
        )
        span_joints = [5, 6]
        if (
            not np.isfinite(p[[initial, selected_endpoint]][:, span_joints]).all()
            or not np.isfinite(c[[initial, selected_endpoint]][:, span_joints]).all()
            or c[[initial, selected_endpoint]][:, span_joints].min() < 0.5
        ):
            raise ValueError("Unreliable initial/final shoulder span")
        first_span = np.linalg.norm(p[initial, 6] - p[initial, 5])
        final_span = np.linalg.norm(p[selected_endpoint, 6] - p[selected_endpoint, 5])
        if first_span <= 1e-6 or final_span <= 1e-6:
            raise ValueError("Degenerate initial shoulder span")
        change = float(abs(final_span - first_span) / first_span)
        span_gate = float(
            np.clip(change / calibration.shoulder_span_reference, 0, 1) ** 2
        )
        evidence["follow_through"].update(span_change=change, multiplier=span_gate)
    except ValueError as error:
        evidence.setdefault("follow_through", {}).update(
            available=False, reason=str(error)
        )

    points = compose_points(
        prior, rotation=rotation, endpoint=endpoint, span_gate=span_gate
    )
    if rotation is None and span_gate is not None:
        # The accepted local chain retained this earlier directional/span rule when the new pre-contact window was unassessed.
        try:
            if not initial + 2 <= balance_anchor <= frames[-1]:
                raise ValueError("Legacy rotation lacks generated coverage")
            observed, certainty, _ = legacy_rotation_trace(
                p[initial : balance_anchor + 1], c[initial : balance_anchor + 1]
            )
            target, _, _ = legacy_rotation_trace(
                q[: balance_anchor - initial + 1],
                np.ones_like(c[initial : balance_anchor + 1]),
            )
            if (
                not np.isfinite([observed[-1, 0], target[-1, 0]]).all()
                or target[-1, 0] <= 1e-6
            ):
                raise ValueError("Legacy reference lacks directional rotation")
            credit = float(
                np.clip(
                    observed[-1, 0]
                    / target[-1, 0]
                    / calibration.legacy_rotation_reference,
                    0,
                    1,
                )
            )
            points[1] = (
                20 * (certainty[0] * credit + (1 - certainty[0]) * 0.5) * span_gate
            )
            evidence["body_rotation"]["fallback"] = "legacy_directional_span"
        except ValueError as error:
            evidence["body_rotation"]["fallback_reason"] = str(error)
    interval = intervals["preparation"]
    try:
        if interval["end"] >= contact:
            raise ValueError("Preparation must precede contact")
        measurement = preparation_height(
            p[interval["start"] : interval["end"] + 1],
            c[interval["start"] : interval["end"] + 1],
            handedness,
        )
        points[0], fraction = cap_preparation(
            points[0], measurement["height"], calibration.preparation_height
        )
        evidence["preparation"] = dict(available=True, fraction=fraction, **measurement)
    except ValueError as error:
        evidence["preparation"] = dict(available=False, reason=str(error))
    try:
        interval = intervals["arm_balance"]
        measurement = measure_balance_height(
            p, c, initial, interval["start"], interval["end"], contact, handedness, fps
        )
    except ValueError as error:
        measurement = dict(available=False, reason=str(error))
    if measurement["available"]:
        points[2], fraction = cap_balance(
            points[2], measurement["sustained_gap"], calibration.balance_gap
        )
        measurement.update(fraction=fraction, tolerance=calibration.balance_gap)
    evidence["arm_balance"] = measurement
    return dict(
        points=points.tolist(),
        total=float(points.sum()),
        evidence=evidence,
        selected_endpoint=selected_endpoint,
        initial_frame=initial,
        balance_anchor=balance_anchor,
        maxima=MAXIMA.tolist(),
    )


def legacy_rotation_trace(poses, confidence, count=17):
    """Invariant to shared per-frame image translation, roll, and uniform scale."""
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
    # Anatomical left->right image axes: canonical right-handed expert motion decreases the shoulder angle.
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
    """Median wrist height along the torso axis: hips=0, shoulders=1."""
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


JOINTS = (
    (6, 8, 10, 11, 12),
    (5, 6, 11, 12),
    (5, 6, 7, 8, 9),
    (0, 6, 8),
    (6, 8, 10, 12),
    (5, 6, 11, 12),
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
            # The ending is judged on the trunk and the shoulder line turning forward.
            feature = np.c_[unit(spine) @ basis, vector(6, 5) - shoulder0]
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
            (
                c[k, 0, list(JOINTS[k])].min()
                if k == 4
                else min(float(np.median(c[k][:, JOINTS[k]].min(1))), float(context))
            )
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
    """Search every available frame strictly after the existing acceleration anchor."""
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
