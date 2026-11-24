"""What the coach is told and shown for a smash."""

from typing import Any

from badminton_analysis.ml.coaching_feedback import criterion_evidence_frames
from badminton_analysis.prompts import prompt


CRITERIA = (
    "preparation",
    "body_rotation",
    "arm_balance",
    "elbow_forward",
    "wrist_flick",
    "follow_through",
)
ROTATION_REVIEW_PHASES = {
    "body_rotation": ("setup", "準備時可能轉身過多"),
    "follow_through": ("finish", "擊球後可能轉身過多"),
}


def rotation_review_reporting(measurement: dict, criteria: list[dict]) -> dict:
    """Expose observable rotation flags even on analyses with coaching disabled."""
    rows = {row["rule_reference"]: row for row in criteria}
    diagnostics = {}
    for criterion, (phase, note) in ROTATION_REVIEW_PHASES.items():
        result = measurement[phase]
        prefix = f"smash_excess_rotation_{phase}"
        assessed = result["status"] != "unassessable"
        diagnostics[f"{prefix}_assessed"] = float(assessed)
        if not assessed:
            continue
        flagged = result["status"] == "flagged"
        diagnostics[f"{prefix}_flagged"] = float(flagged)
        if flagged:
            start, end = result["event_source_interval"]
            diagnostics[f"{prefix}_start_frame"] = start
            diagnostics[f"{prefix}_end_frame"] = end
            row = rows[criterion]
            row["grade_description"] = f"{row['name_zh_tw']}（{note}）"
    return diagnostics


def build_checkpoint_evidence(
    *,
    intervals: dict[str, Any],
    window_start: int,
    window_end: int,
    initial_frame: int,
    balance_anchor: int,
    selected_endpoint: int,
    balance_measurement: dict[str, Any],
    rotation_measurement: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if not 0 <= window_start <= window_end or set(intervals) != set(CRITERIA):
        raise ValueError("ordered source window and six scoring intervals required")
    result = {}
    for reference in CRITERIA:
        interval = intervals[reference]
        start, end = int(interval["start"]), int(interval["end"])
        if reference == "body_rotation":
            start, end = initial_frame, balance_anchor
        elif reference == "follow_through":
            start, end = initial_frame, selected_endpoint
        if start > end:
            raise ValueError(f"unordered scoring interval: {reference}")
        requested = {start, end}
        rotation_frames = []
        if rotation_measurement and reference in ROTATION_REVIEW_PHASES:
            rotation_phase, _ = ROTATION_REVIEW_PHASES[reference]
            rotation = rotation_measurement[rotation_phase]
            if rotation["status"] == "flagged":
                first, last = rotation["event_source_interval"]
                rotation_frames = list(range(first, last + 1))
                requested.update(rotation_frames)
        if reference != "follow_through":
            requested.update((round((start + end) / 2), int(interval["anchor"])))
        if reference == "arm_balance" and balance_measurement.get("available"):
            event_start = int(balance_measurement["event_start"])
            event_end = int(balance_measurement["event_end"])
            if (
                not start <= event_start <= event_end <= end
                or event_end - event_start != 4
            ):
                raise ValueError("balance evidence must be the scored five-frame event")
            requested.update(range(event_start, event_end + 1))
        # Include actual visible interval boundaries if only part is rendered.
        visible_start, visible_end = max(start, window_start), min(end, window_end)
        if visible_start > visible_end:
            raise ValueError(f"no visible scored evidence for {reference}")
        requested.update((visible_start, visible_end))
        available = sorted(
            frame
            for frame in requested
            if window_start <= frame <= window_end and start <= frame <= end
        )
        result[reference] = {
            "source_interval": [start, end],
            "output_frame_indices": [frame - window_start for frame in available],
            "source_frame_indices": available,
            "source_window_start": window_start,
            "full_interval_visible": window_start <= start <= end <= window_end,
            "missing_requested_source_frames": sorted(requested - set(available)),
            "sampling": "scored boundaries, midpoint/anchor, and exact sustained balance event",
        }
        if rotation_frames:
            result[reference]["possible_excess_rotation"] = {
                "phase": rotation_phase,
                "event_source_interval": rotation["event_source_interval"],
                "all_event_frames_visible": set(rotation_frames).issubset(available),
            }
        if reference == "arm_balance":
            result[reference]["measurement"] = {
                key: balance_measurement[key]
                for key in (
                    "available",
                    "reason",
                    "handedness",
                    "event_start",
                    "event_end",
                    "sustained_gap",
                    "tolerance",
                    "fraction",
                    "valid_fraction",
                )
                if key in balance_measurement
            }
        if reference == "follow_through":
            # Initial pose is a scoring/coaching reference.
            contact = int(intervals["wrist_flick"]["anchor"])
            result[reference]["replay_source_interval"] = [
                max(visible_start, min(contact, visible_end)),
                visible_end,
            ]
    return result


# -- The interface every skill's coaching module provides --------------------

OWNS_CHECKPOINT_EVIDENCE = True


def coaching_instructions():
    """What this skill adds to the coach's system instructions."""
    return prompt("smash/coach_system")


def score_warning():
    """What the coach is told about where the grade comes from."""
    return prompt("smash/score_warning")


def describe_scoring(diagnostics, status, method):
    """(score status, method text) the coach is told the grade came from."""
    if diagnostics.get("scorer") == "smash_local_checkpoint_graph_geometry_v20260913":
        return "frozen_checkpoint_calibration", prompt("smash/score_method")
    return status, method


def attach_grade_context(correction_grade, score):
    """Copy this skill's measurements into the grade the coach reads."""
    return None


def feedback_phase(frame_index, anchor_1, anchor_2):
    """Which coaching phase a normalized frame belongs to."""
    if frame_index < anchor_1:
        return "preparation"
    if frame_index < anchor_2:
        return "rotation"
    if frame_index <= anchor_2:
        return "contact"
    return "follow_through"


def comparison_frames(rule, samples, anchors):
    """The frames the coach compares for one criterion."""
    return criterion_evidence_frames(rule, samples, anchors)
