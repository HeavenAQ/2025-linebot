"""What the coach is told and shown for a serve."""

from __future__ import annotations

from typing import Any

from badminton_analysis.ml.coaching_feedback import criterion_evidence_frames
from badminton_analysis.prompts import prompt


# The serve coaching prompt names these measurements, so GPT has to receive them.
_SERVE_TRANSFER_MEASUREMENTS = ("pelvis_loading_shift",)


# The stance is judged against the learner's own corrected skeleton.
_SERVE_CORRECTION_STANCE_MEASUREMENTS = (
    "correction_learner_stance_retention",
    "correction_corrected_stance_retention",
    "correction_stance_retention_shortfall",
    "correction_stance_allowance",
    "correction_learner_transfer_lead_frames",
    "correction_corrected_transfer_lead_frames",
    "correction_transfer_lead_excess_frames",
    "correction_transfer_lead_allowance_frames",
)


# The racket elbow at contact, against the learner's own corrected skeleton.
_SERVE_CORRECTION_ELBOW_MEASUREMENTS = (
    "correction_learner_elbow_at_contact_degrees",
    "correction_corrected_elbow_at_contact_degrees",
    "correction_elbow_at_contact_shortfall_degrees",
    "correction_elbow_allowance_degrees",
    "correction_learner_shoulder_stance_angle_degrees",
    "correction_corrected_shoulder_stance_angle_degrees",
    "correction_shoulder_turn_shortfall_degrees",
    "correction_shoulder_turn_allowance_degrees",
)


def _attach_serve_transfer_measurements(
    correction_grade: dict[str, Any], score: dict[str, Any]
) -> None:
    measured = next(
        (
            item
            for item in score.get("criteria", [])
            if str(item.get("rule_reference")) == "weight_transfer"
        ),
        None,
    )
    if measured is None:
        return
    criterion = next(
        item
        for item in correction_grade["criteria"]
        if item["rule_reference"] == "weight_transfer"
    )
    for cue in _SERVE_TRANSFER_MEASUREMENTS:
        for prefix in ("source_", "expert_lower_", "standardized_shortfall_"):
            if prefix + cue in measured:
                criterion[prefix + cue] = float(measured[prefix + cue])
    for key in _SERVE_CORRECTION_STANCE_MEASUREMENTS:
        if key in measured:
            criterion[key] = float(measured[key])
    wrist = next(
        (
            item
            for item in score.get("criteria", [])
            if str(item.get("rule_reference")) == "wrist_flick"
        ),
        None,
    )
    if wrist is not None:
        wrist_criterion = next(
            item
            for item in correction_grade["criteria"]
            if item["rule_reference"] == "wrist_flick"
        )
        for key in _SERVE_CORRECTION_ELBOW_MEASUREMENTS:
            if key in wrist:
                wrist_criterion[key] = float(wrist[key])


# -- The interface every skill's coaching module provides --------------------

OWNS_CHECKPOINT_EVIDENCE = False


def coaching_instructions():
    """What this skill adds to the coach's system instructions."""
    return prompt("serve/coach_system")


def score_warning():
    """What the coach is told about where the grade comes from."""
    return prompt("coach/score_warning")


def describe_scoring(diagnostics, status, method):
    """(score status, method text) the coach is told the grade came from."""
    return status, method + prompt("serve/score_method")


def attach_grade_context(correction_grade, score):
    """Copy this skill's measurements into the grade the coach reads."""
    _attach_serve_transfer_measurements(correction_grade, score)


def feedback_phase(frame_index, anchor_1, anchor_2):
    """Which coaching phase a normalized frame belongs to."""
    if frame_index <= anchor_1:
        return "preparation"
    if frame_index < anchor_2:
        return "weight_transfer"
    if frame_index <= anchor_2:
        return "contact"
    return "follow_through"


def comparison_frames(rule, samples, anchors):
    """The frames the coach compares for one criterion."""
    # Weight transfer is judged across the whole swing, start to finish.
    if rule.id == "weight_transfer":
        return [anchors[0], anchors[-1]]
    return criterion_evidence_frames(rule, samples, anchors)
