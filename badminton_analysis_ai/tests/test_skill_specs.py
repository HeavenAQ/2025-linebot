from __future__ import annotations

import re
import pytest
from pydantic import ValidationError

from badminton_analysis.ml.coaching_feedback import (
    SkillFeedbackAnalysis,
    phase_for_frame,
)
from badminton_analysis.ml.skill_specs import (
    SUPPORTED_CORRECTION_SKILLS,
    get_skill_spec,
    motion_completion_bounds,
)
from badminton_analysis.models.types import Skill

PHASE_INDICES = (0, 20, 39, 51, 63)

EXPECTED_CRITERIA = {
    Skill.SERVE: (
        "雙手平舉",
        "將重心放至持拍腳",
        "重心轉移至非持拍腳",
        "髖關節前旋",
        "持拍手手腕發力",
        "肩膀旋轉朝前",
    ),
    Skill.SMASH: (
        "球拍舉至腰部預備",
        "轉身",
        "雙手手肘平衡",
        "手肘往前轉至前方",
        "手腕發力",
        "慣用手肩膀往前轉",
    ),
}


def _feedback_payload(skill: Skill) -> dict[str, object]:
    spec = get_skill_spec(skill)
    rule = spec.rules[0]
    return {
        "skill": spec.slug,
        "language": "zh-TW",
        "overall_feedback": "整體動作順序正確，但第一個技術階段仍需要調整。",
        "problems": [
            {
                "priority": "中",
                "title": rule.name_zh_tw,
                "feedback": "請依照專家動作調整這個階段的身體位置與動作節奏。",
                "evidence": "目前畫面中的關節位置與專家化修正骨架仍有明顯差距。",
                "frame_index": 0,
                "phase": rule.phase,
                "joint_ids": [rule.measured_joints[0]],
                "rule_reference": rule.id,
                "confidence": 0.8,
            }
        ],
    }


def test_each_supported_skill_has_an_independent_complete_contract() -> None:
    assert set(SUPPORTED_CORRECTION_SKILLS) == set(EXPECTED_CRITERIA)
    for skill, expected_names in EXPECTED_CRITERIA.items():
        spec = get_skill_spec(skill)
        assert tuple(rule.name_zh_tw for rule in spec.rules) == expected_names
        assert sum(rule.maximum for rule in spec.rules) == pytest.approx(100.0)
        assert sum(detail.maximum for detail in spec.details) == pytest.approx(100.0)
        assert len(spec.joint_weights) == 17


def test_scoring_windows_scale_with_motion_completion() -> None:
    serve = get_skill_spec(Skill.SERVE)
    wrist = next(
        detail
        for detail, rule in zip(serve.details, serve.rules, strict=True)
        if rule.id == "wrist_flick"
    )
    follow_through = next(
        detail
        for detail, rule in zip(serve.details, serve.rules, strict=True)
        if rule.id == "shoulder_rotation"
    )

    assert wrist.bounds(64) == (36, 56)
    assert wrist.bounds(128) == (72, 112)
    assert follow_through.bounds(64) == (48, 64)
    assert follow_through.bounds(128) == (96, 128)
    assert motion_completion_bounds(80, 0.875, 1.0) == (70, 80)


def test_rules_retain_qualitative_grader_instructions() -> None:
    expected_movements = {
        Skill.SERVE: ("雙手平舉", "持拍腳", "非持拍腳", "髖關節", "手腕", "肩膀"),
        Skill.SMASH: (
            "球拍舉至腰部",
            "轉身",
            "手肘保持平衡",
            "手肘往前",
            "手腕發力",
            "肩膀往前",
        ),
    }
    for skill, movements in expected_movements.items():
        calculations = tuple(
            rule.calculation_zh_tw for rule in get_skill_spec(skill).rules
        )
        assert all(
            expected in calculation
            for expected, calculation in zip(movements, calculations, strict=True)
        )
        # Do not invent numeric angle thresholds in coaching prose. Words such
        # as 高度/程度 describe the actual smash height rule, not degrees.
        assert all(
            not re.search(r"\d+(?:\.\d+)?\s*度", calculation)
            for calculation in calculations
        )


@pytest.mark.parametrize("skill", SUPPORTED_CORRECTION_SKILLS)
def test_feedback_schema_accepts_each_skill_contract(skill: Skill) -> None:
    analysis = SkillFeedbackAnalysis.model_validate(_feedback_payload(skill))
    assert analysis.skill == str(skill)


def test_feedback_schema_rejects_a_criterion_from_another_skill() -> None:
    payload = _feedback_payload(Skill.SMASH)
    problem = payload["problems"][0]  # type: ignore[index]
    problem["title"] = "雙手平舉"  # type: ignore[index]

    with pytest.raises(ValidationError, match="criterion title"):
        SkillFeedbackAnalysis.model_validate(payload)


def test_serve_weight_transfer_uses_the_windowed_criterion_metric() -> None:
    serve = get_skill_spec(Skill.SERVE)
    metrics = {
        rule.id: detail.metric
        for detail, rule in zip(serve.details, serve.rules, strict=True)
    }

    assert metrics["weight_transfer"] == "window_distance"
    assert metrics["shoulder_rotation"] == "serve_follow_through_cross_body"


def test_serve_follow_through_checks_forearm_near_opposite_neck() -> None:
    follow_through = get_skill_spec(Skill.SERVE).rule("shoulder_rotation")

    assert "前臂" in follow_through.calculation_zh_tw
    assert "對側頸部" in follow_through.calculation_zh_tw
    assert follow_through.coaching_joints == (6, 8, 10)


@pytest.mark.parametrize("skill", SUPPORTED_CORRECTION_SKILLS)
def test_each_rule_anchor_has_its_declared_display_phase(skill: Skill) -> None:
    spec = get_skill_spec(skill)
    for rule in spec.rules:
        for anchor_index in rule.allowed_anchor_indices:
            frame_index = PHASE_INDICES[anchor_index]
            assert phase_for_frame(frame_index, PHASE_INDICES, spec) == (
                rule.display_phase or rule.phase
            )


def test_serve_contact_and_preparation_rule_anchors_match_extraction_events() -> None:
    spec = get_skill_spec(Skill.SERVE)
    # Serve extraction aligns the dominant wrist's maximum acceleration to
    # anchor 2, and the grader measures the burst around it. The checkpoint has
    # to be drawn at that event, not at the deceleration after it.
    assert spec.rule("racket_foot_weight").allowed_anchor_indices == (1,)
    assert spec.rule("racket_foot_weight").phase == "preparation"
    assert spec.rule("racket_foot_weight").display_phase is None
    assert spec.rule("weight_transfer").phase == "weight_transfer"
    assert spec.rule("weight_transfer").display_phase == "contact"
    assert spec.rule("wrist_flick").allowed_anchor_indices == (2,)
    assert spec.rule("wrist_flick").phase == "contact"
    assert spec.rule("wrist_flick").display_phase is None
    assert "最大手腕加速度" in spec.checkpoint_roles_zh_tw[2]
    assert "隨揮" in spec.checkpoint_roles_zh_tw[3]
