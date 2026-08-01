from service.pipeline import _correction_grade_context
from badminton_analysis.ml.skill_specs import get_skill_spec


def test_serve_gpt_context_includes_full_body_transition_evidence() -> None:
    spec = get_skill_spec("serve")
    diagnostics = {
        "correction_distance": 0.8,
        "position_distance": 0.4,
        "angle_distance": 0.1,
        "velocity_distance": 0.1,
        "bone_length_distance": 0.0,
        "support_transition_distance": 0.3,
        "torso_lean_transition_distance": 0.2,
        "transition_distance": 0.265,
    }
    criteria = [
        (rule.name_zh_tw, 0.1 + index * 0.01, rule.maximum * 0.5)
        for index, rule in enumerate(spec.rules)
    ]

    context = _correction_grade_context(
        {"total_grade": 45.0}, diagnostics, spec, criteria
    )

    assert context["distance_components"]["support_transition_distance"] == 0.3
    assert context["distance_components"]["torso_lean_transition_distance"] == 0.2
    assert context["distance_components"]["transition_distance"] == 0.265
    assert "軀幹前傾" in context["score_method_zh_tw"]
