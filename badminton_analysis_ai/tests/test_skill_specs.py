from __future__ import annotations

import numpy as np
import pytest
from pydantic import ValidationError

from badminton_analysis.ml.clear_feedback import (
    DEFAULT_PHASE_INDICES,
    SkillFeedbackAnalysis,
    phase_for_frame,
)
from badminton_analysis.ml.infer_skeleton_corrector import phase_grading_details
from badminton_analysis.ml.skeleton_scoring import ScoreCalibration
from badminton_analysis.ml.skill_specs import (
    SUPPORTED_CORRECTION_SKILLS,
    get_skill_spec,
    validate_checkpoint_spec,
)
from badminton_analysis.models.types import Skill


EXPECTED_CRITERIA = {
    Skill.SERVE: (
        "雙手平舉",
        "將重心放至持拍腳",
        "重心轉移至非持拍腳",
        "髖關節前旋",
        "持拍手手腕發力",
        "肩膀旋轉朝前",
    ),
    Skill.LIFT: (
        "球拍置於身前放鬆預備",
        "持拍腳跨步並放鬆引拍",
        "弓步穩定並以前臂手腕擊球",
        "順勢隨揮並回復平衡",
    ),
    Skill.CLEAR: (
        "球拍舉至腰部預備",
        "轉身",
        "雙手手肘平衡",
        "手肘往前轉至前方",
        "手腕發力",
        "慣用手肩膀往前轉",
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
        assert spec.dataset_root.name == spec.slug
        assert spec.slug in spec.model_path.name


def test_rules_retain_qualitative_grader_instructions() -> None:
    expected_movements = {
        Skill.SERVE: ("雙手平舉", "持拍腳", "非持拍腳", "髖關節", "手腕", "肩膀"),
        Skill.LIFT: ("球拍置於身前", "持拍腳跨步", "弓步", "順勢隨揮"),
        Skill.CLEAR: (
            "球拍舉至腰部",
            "轉身",
            "手肘保持平衡",
            "手肘往前",
            "手腕發力",
            "肩膀往前",
        ),
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
        calculations = tuple(rule.calculation_zh_tw for rule in get_skill_spec(skill).rules)
        assert all(
            expected in calculation
            for expected, calculation in zip(movements, calculations, strict=True)
        )
        assert all("度" not in calculation for calculation in calculations)


@pytest.mark.parametrize("skill", SUPPORTED_CORRECTION_SKILLS)
def test_feedback_schema_accepts_each_skill_contract(skill: Skill) -> None:
    analysis = SkillFeedbackAnalysis.model_validate(_feedback_payload(skill))
    assert analysis.skill == str(skill)


def test_feedback_schema_rejects_a_criterion_from_another_skill() -> None:
    payload = _feedback_payload(Skill.LIFT)
    problem = payload["problems"][0]  # type: ignore[index]
    problem["title"] = "雙手平舉"  # type: ignore[index]

    with pytest.raises(ValidationError, match="criterion title"):
        SkillFeedbackAnalysis.model_validate(payload)


def test_checkpoint_metadata_enforces_skill_separation() -> None:
    lift = get_skill_spec(Skill.LIFT)
    validate_checkpoint_spec(
        {
            "skill": "lift",
            "joint_weights": list(lift.joint_weights),
            "transition_weight": lift.transition_weight,
            "transition_joints": list(lift.transition_joints),
            "transition_lean_joints": list(lift.transition_lean_joints),
            "transition_direction_joint": lift.transition_direction_joint,
        },
        lift,
    )
    with pytest.raises(ValueError, match="checkpoint skill"):
        validate_checkpoint_spec(
            {"skill": "serve", "joint_weights": list(lift.joint_weights)}, lift
        )
    with pytest.raises(ValueError, match="does not contain joint weights"):
        validate_checkpoint_spec({"skill": "lift"}, lift)
    assert lift.transition_direction_joint == 16


def test_serve_contract_requires_full_body_transition_metadata() -> None:
    serve = get_skill_spec(Skill.SERVE)
    checkpoint = {
        "skill": "serve",
        "joint_weights": list(serve.joint_weights),
        "transition_weight": serve.transition_weight,
        "transition_joints": list(serve.transition_joints),
        "transition_lean_joints": list(serve.transition_lean_joints),
    }

    validate_checkpoint_spec(checkpoint, serve)
    weight_transfer_detail = next(
        detail
        for detail in serve.details
        if detail.name_zh_tw == "重心轉移至非持拍腳"
    )
    assert weight_transfer_detail.metric == "full_transition"
    assert serve.transition_joints == (11, 12, 13, 14, 15, 16)
    assert serve.transition_lean_joints == (5, 6, 11, 12)

    checkpoint["transition_lean_joints"] = [11, 12, 15, 16]
    with pytest.raises(ValueError, match="transition scoring"):
        validate_checkpoint_spec(checkpoint, serve)


def test_serve_follow_through_checks_forearm_near_opposite_neck() -> None:
    follow_through = get_skill_spec(Skill.SERVE).rule("shoulder_rotation")

    assert "前臂" in follow_through.calculation_zh_tw
    assert "對側頸部" in follow_through.calculation_zh_tw
    assert follow_through.coaching_joints == (6, 8, 10)


@pytest.mark.parametrize("skill", SUPPORTED_CORRECTION_SKILLS)
def test_each_rule_anchor_has_the_rule_phase(skill: Skill) -> None:
    spec = get_skill_spec(skill)
    for rule in spec.rules:
        for anchor_index in rule.allowed_anchor_indices:
            frame_index = DEFAULT_PHASE_INDICES[anchor_index]
            assert (
                phase_for_frame(frame_index, DEFAULT_PHASE_INDICES, spec)
                == rule.phase
            )


@pytest.mark.parametrize("skill", SUPPORTED_CORRECTION_SKILLS)
def test_skill_detail_grades_reconcile_to_the_total(skill: Skill) -> None:
    spec = get_skill_spec(skill)
    original = np.zeros((64, 17, 3), dtype=np.float32)
    corrected = original.copy()
    corrected[:, 10, 0] = np.linspace(0.0, 0.2, 64)
    confidence = np.ones((64, 17), dtype=np.float32)
    details = phase_grading_details(
        original,
        corrected,
        confidence,
        ScoreCalibration(distance_offset=0.0, alpha=8.0),
        total_grade=47.5,
        spec=spec,
    )

    assert len(details) == len(spec.rules)
    assert [detail[0] for detail in details] == [
        rule.name_zh_tw for rule in spec.rules
    ]
    assert sum(detail[2] for detail in details) == pytest.approx(47.5)
