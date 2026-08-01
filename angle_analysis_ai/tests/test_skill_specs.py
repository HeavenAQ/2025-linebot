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
        "手腕放置腰部放鬆預備",
        "手腕往後引拍",
        "手腕往前壓",
        "手腕放鬆回到預備姿勢",
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
        {"skill": "lift", "joint_weights": list(lift.joint_weights)}, lift
    )
    with pytest.raises(ValueError, match="checkpoint skill"):
        validate_checkpoint_spec(
            {"skill": "serve", "joint_weights": list(lift.joint_weights)}, lift
        )
    with pytest.raises(ValueError, match="does not contain joint weights"):
        validate_checkpoint_spec({"skill": "lift"}, lift)


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
