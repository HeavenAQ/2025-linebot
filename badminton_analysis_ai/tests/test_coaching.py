from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from badminton_analysis.ml.clear_feedback import (
    RawSkillFeedbackAnalysis,
    SampledFrame,
    phase_for_frame,
)
from badminton_analysis.ml.skill_specs import get_skill_spec
from badminton_analysis.models.types import Skill

import service.coaching as coaching_module
from service.coaching import CoachingGenerator


PHASES = (0, 20, 39, 51, 63)


def _sample(frame_index: int, spec) -> SampledFrame:
    return SampledFrame(
        frame_index=frame_index,
        source_frame_index=frame_index,
        timestamp_seconds=frame_index / 20,
        phase=phase_for_frame(frame_index, PHASES, spec),
        checkpoint_role_zh_tw="測試關鍵幀",
        image_path=Path(f"frame-{frame_index}.jpg"),
        data_url="data:image/jpeg;base64,test",
    )


def _correction_grade(spec, scores: tuple[float, ...]) -> dict:
    return {
        "total_score": sum(scores),
        "criteria": [
            {
                "name_zh_tw": rule.name_zh_tw,
                "rule_reference": rule.id,
                "score": score,
                "maximum": rule.maximum,
            }
            for rule, score in zip(spec.rules, scores, strict=True)
        ],
    }


def test_fallback_coaching_uses_lowest_normalized_criterion() -> None:
    spec = get_skill_spec(Skill.LIFT)
    correction_grade = _correction_grade(spec, (8.0, 20.0, 7.0, 24.0))

    analysis = CoachingGenerator._fallback_analysis(spec, correction_grade)

    assert analysis["skill"] == "lift"
    assert analysis["language"] == "zh-TW"
    assert len(analysis["problems"]) == 1
    problem = analysis["problems"][0]
    assert problem["rule_reference"] == "forward_extension"
    assert problem["joint_ids"] == [8, 10]
    assert problem["feedback"] == spec.rule("forward_extension").calculation_zh_tw
    assert "度" not in problem["feedback"]
    assert "7.0/35.0" in problem["evidence"]


def test_normalize_analysis_accepts_exact_criterion_title_as_rule_reference() -> None:
    spec = get_skill_spec(Skill.CLEAR)
    correction_grade = _correction_grade(
        spec, tuple(rule.maximum for rule in spec.rules)
    )
    rule = spec.rules[0]
    analysis = {
        "skill": spec.slug,
        "language": "zh-TW",
        "overall_feedback": "請依照專家動作調整準備姿勢與擊球節奏。",
        "problems": [
            {
                "priority": "高",
                "title": rule.name_zh_tw,
                "feedback": "請將球拍穩定舉至腰部位置完成準備。",
                "evidence": "準備階段的手臂位置與專家示範仍有明顯差距。",
                "frame_index": 7,
                "phase": "preparation",
                "joint_ids": [0],
                "rule_reference": rule.name_zh_tw,
                "confidence": 0.9,
            }
        ],
    }

    normalized = CoachingGenerator._normalize_analysis(
        analysis,
        spec=spec,
        correction_grade=correction_grade,
        phase_indices=PHASES,
        samples=[_sample(0, spec)],
    )

    problem = normalized["problems"][0]
    assert problem["rule_reference"] == rule.id
    assert problem["frame_index"] == 0
    assert problem["joint_ids"] == list(rule.coaching_joints)
    assert problem["timestamp_seconds"] == 0


def test_generate_falls_back_when_llm_rule_is_not_in_skill_spec(
    monkeypatch, tmp_path: Path
) -> None:
    spec = get_skill_spec(Skill.CLEAR)
    scores = (0.0,) + tuple(rule.maximum for rule in spec.rules[1:])
    correction_grade = _correction_grade(spec, scores)
    parsed = RawSkillFeedbackAnalysis.model_validate(
        {
            "skill": spec.slug,
            "language": "zh-TW",
            "overall_feedback": "請優先修正準備姿勢並依照專家節奏完成擊球。",
            "problems": [
                {
                    "priority": "高",
                    "title": "不存在的技術標準",
                    "feedback": "請依照畫面中的專家姿勢調整手臂位置。",
                    "evidence": "目前動作與專家示範的關鍵姿勢有明顯差距。",
                    "frame_index": 0,
                    "phase": "preparation",
                    "joint_ids": [6],
                    "rule_reference": "不存在的技術標準",
                    "confidence": 0.8,
                }
            ],
        }
    )
    response = SimpleNamespace(output_parsed=parsed, id="invalid-response")
    client = SimpleNamespace(
        responses=SimpleNamespace(parse=lambda **_: response)
    )
    samples = [_sample(0, spec)]
    monkeypatch.setattr(coaching_module, "sample_video_frames", lambda *_, **__: samples)
    monkeypatch.setattr(coaching_module, "prompt_context", lambda *_, **__: {})
    monkeypatch.setattr(coaching_module, "build_response_input", lambda *_, **__: [])
    generator = CoachingGenerator.__new__(CoachingGenerator)
    generator.client = client
    generator.model = "test-model"
    generator.max_attempts = 1

    payload = generator.generate(
        video_path=tmp_path / "input.mp4",
        working_dir=tmp_path,
        filename="student.mp4",
        handedness="right",
        phase_indices=PHASES,
        spec=spec,
        correction_grade=correction_grade,
    )

    assert payload["source"] == "deterministic_fallback"
    assert payload["response_id"] == ""
    assert payload["fallback_error"] == "KeyError"
    assert payload["analysis"]["problems"][0]["rule_reference"] == spec.rules[0].id
