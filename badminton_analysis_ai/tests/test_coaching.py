from __future__ import annotations

from badminton_analysis.ml.skill_specs import get_skill_spec
from badminton_analysis.models.types import Skill

from service.coaching import CoachingGenerator


def test_fallback_coaching_uses_lowest_normalized_criterion() -> None:
    spec = get_skill_spec(Skill.LIFT)
    correction_grade = {
        "criteria": [
            {
                "name_zh_tw": rule.name_zh_tw,
                "rule_reference": rule.id,
                "score": score,
                "maximum": rule.maximum,
            }
            for rule, score in zip(spec.rules, (8.0, 20.0, 7.0, 24.0), strict=True)
        ]
    }

    analysis = CoachingGenerator._fallback_analysis(spec, correction_grade)

    assert analysis["skill"] == "lift"
    assert analysis["language"] == "zh-TW"
    assert len(analysis["problems"]) == 1
    problem = analysis["problems"][0]
    assert problem["rule_reference"] == "forward_extension"
    assert problem["joint_ids"] == [8, 10]
    assert "7.0/35.0" in problem["evidence"]
