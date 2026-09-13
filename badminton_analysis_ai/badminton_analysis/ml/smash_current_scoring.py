"""Accepted local smash score composition on explicit source-clock evidence.

No cohort names, human ratings, fitting, or research imports enter inference.
The caller supplies the historical heads and the exact displayed correction.
"""

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np

from badminton_analysis.ml.smash_current_geometry import (
    cap_balance,
    cap_preparation,
    combined_features,
    measure_balance_height,
    preparation_height,
    rotation_credit,
    legacy_rotation_trace,
)
from badminton_analysis.ml.smash_current_endpoint import (
    generated_shoulder_cut,
    interval_frames,
    local_cost,
    local_features,
    reliability,
    score_distance,
    select_best_endpoint,
)

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
    """Grade source poses against the frozen projected/EMA correction.

    The registered reference convention is image-right-forward. Callers must
    not choose its sign by maximizing a score. Left-canonicalization and view
    validation belong to the caller, before this frozen right-canonical path.
    Missing evidence retains the previous checkpoint and reports why.
    """
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
        # The accepted local chain retained this earlier directional/span rule
        # when the new pre-contact window was unassessed (e.g. a late anchor).
        # Recompute it from evidence; never insert a cached per-person score.
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
