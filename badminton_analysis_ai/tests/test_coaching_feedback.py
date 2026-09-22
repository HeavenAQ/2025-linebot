from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest
from pydantic import ValidationError

from badminton_analysis.ml.coaching_feedback import (
    RawSkillFeedbackAnalysis,
    SkillFeedbackAnalysis,
    coaching_target_joint_ids,
    feedback_frame_indices,
    handedness_note_zh_tw,
    phase_for_frame,
    prompt_context,
    build_response_input,
    sample_video_frames,
    system_instructions,
)
from badminton_analysis.ml.skill_specs import get_skill_spec
from badminton_analysis.models.types import Skill

PHASE_INDICES = (0, 20, 39, 51, 63)


def _write_test_video(path: Path, frame_count: int = 64) -> None:
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter.fourcc(*"mp4v"), 30.0, (48, 64))
    assert writer.isOpened()
    try:
        for frame_index in range(frame_count):
            frame = np.full((64, 48, 3), frame_index, dtype=np.uint8)
            writer.write(frame)
    finally:
        writer.release()


def test_phase_for_frame_uses_smash_feedback_windows() -> None:
    spec = get_skill_spec(Skill.SMASH)

    assert phase_for_frame(0, PHASE_INDICES, spec) == "preparation"
    assert phase_for_frame(19, PHASE_INDICES, spec) == "preparation"
    assert phase_for_frame(20, PHASE_INDICES, spec) == "rotation"
    assert phase_for_frame(38, PHASE_INDICES, spec) == "rotation"
    assert phase_for_frame(39, PHASE_INDICES, spec) == "contact"
    assert phase_for_frame(40, PHASE_INDICES, spec) == "follow_through"


def test_serve_uses_maximum_wrist_acceleration_as_contact_anchor() -> None:
    phases = (0, 29, 59, 61, 63)
    spec = get_skill_spec(Skill.SERVE)

    assert phase_for_frame(29, phases, spec) == "preparation"
    assert phase_for_frame(30, phases, spec) == "weight_transfer"
    assert phase_for_frame(58, phases, spec) == "weight_transfer"
    assert phase_for_frame(59, phases, spec) == "contact"
    assert phase_for_frame(60, phases, spec) == "follow_through"
    assert phase_for_frame(61, phases, spec) == "follow_through"


def test_sample_video_frames_includes_exact_grading_checkpoints(tmp_path: Path) -> None:
    video_path = tmp_path / "stroke.mp4"
    _write_test_video(video_path)

    samples = sample_video_frames(
        video_path,
        tmp_path / "frames",
        phase_indices=PHASE_INDICES,
        spec=get_skill_spec(Skill.SMASH),
    )

    assert tuple(sample.frame_index for sample in samples) == feedback_frame_indices(
        PHASE_INDICES
    )
    assert set(PHASE_INDICES).issubset(
        {sample.frame_index for sample in samples}
    )
    assert all(sample.image_path.exists() for sample in samples)
    assert samples[5].timestamp_seconds == pytest.approx(39 / 30)
    assert samples[5].data_url.startswith("data:image/jpeg;base64,")


def test_sample_video_frames_uses_source_frame_provenance(tmp_path: Path) -> None:
    video_path = tmp_path / "source.mp4"
    _write_test_video(video_path, frame_count=128)
    source_mapping = tuple(index * 2 for index in range(64))

    samples = sample_video_frames(
        video_path,
        tmp_path / "source_frames",
        phase_indices=PHASE_INDICES,
        source_frame_indices=source_mapping,
        spec=get_skill_spec(Skill.SMASH),
    )

    contact = next(sample for sample in samples if sample.frame_index == 39)
    assert contact.source_frame_index == 78
    assert contact.timestamp_seconds == pytest.approx(78 / 30)
    assert contact.manifest()["source_frame_index"] == 78


def test_feedback_schema_rejects_unknown_frame_or_joint() -> None:
    payload = {
        "skill": "smash",
        "language": "zh-TW",
        "overall_feedback": "擊球階段的慣用手動作仍需要調整。",
        "problems": [
            {
                "priority": "高",
                "title": "手肘往前轉至前方",
                "feedback": "擊球時請讓慣用側手肘更明確地往前轉動。",
                "evidence": "擊球畫面中的慣用側手肘仍停留在肩膀旁邊。",
                "frame_index": 31,
                "phase": "contact",
                "joint_ids": [99],
                "rule_reference": "elbow_forward",
                "confidence": 0.9,
            }
        ],
    }

    with pytest.raises(ValidationError):
        SkillFeedbackAnalysis.model_validate(payload)


def test_raw_feedback_schema_defers_skill_rule_normalization() -> None:
    payload = {
        "skill": "smash",
        "language": "zh-TW",
        "overall_feedback": "擊球階段的慣用手動作仍需要調整。",
        "problems": [
            {
                "priority": "高",
                "title": "模型暫定標題",
                "feedback": "擊球時請讓慣用側手肘更明確地往前轉動。",
                "evidence": "擊球畫面中的慣用側手肘仍停留在肩膀旁邊。",
                "frame_index": 31,
                "phase": "rotation",
                "joint_ids": [8],
                "rule_reference": "elbow_forward",
                "confidence": 0.9,
            }
        ],
    }

    parsed = RawSkillFeedbackAnalysis.model_validate(payload)

    assert parsed.problems[0].rule_reference == "elbow_forward"
    with pytest.raises(ValidationError):
        SkillFeedbackAnalysis.model_validate(payload)


def test_follow_through_coaching_targets_only_dominant_shoulder() -> None:
    spec = get_skill_spec(Skill.SMASH)

    assert coaching_target_joint_ids("follow_through", spec) == [6]
    assert coaching_target_joint_ids("arm_balance", spec) == [7, 8]
    assert coaching_target_joint_ids("preparation", spec) == [6, 8, 10]


def test_handedness_note_uses_physical_side() -> None:
    assert "左手持拍" in handedness_note_zh_tw("left")
    assert "身體左側" in handedness_note_zh_tw("left")
    assert "右手持拍" in handedness_note_zh_tw("right")
    assert "身體右側" in handedness_note_zh_tw("right")


def test_serve_prompt_compares_first_and_last_full_body_frames() -> None:
    spec = get_skill_spec(Skill.SERVE)
    context = prompt_context(
        {"filename": "serve.mp4", "handedness": "right"},
        (),
        phase_indices=PHASE_INDICES,
        correction_grade={"total_score": 45.0},
        spec=spec,
    )

    assert context["criterion_comparison_frames"]["重心轉移至非持拍腳"] == [
        PHASE_INDICES[0],
        PHASE_INDICES[-1],
    ]
    prompt = build_response_input(context, (), spec)[0]["content"][0]["text"]
    assert context["maximum_problem_count"] == 3
    assert "最多回報3項不同標準" in prompt
    assert "只有feedback_candidate_criteria為空時，problems才可為空陣列" in prompt
    assert "不得只檢查其中一項" in prompt
    assert "不可只回報最低分的一項" in prompt
    assert "下肢支撐轉換" in prompt
    assert "雙肩相對雙髖是否向前傾" in prompt


def test_serve_weight_transfer_accepts_upper_and_lower_body_circle_targets() -> None:
    payload = {
        "skill": "serve",
        "language": "zh-TW",
        "overall_feedback": "發球的完整重心轉移仍需要同時調整下肢支撐與軀幹前傾。",
        "problems": [
            {
                "priority": "高",
                "title": "重心轉移至非持拍腳",
                "feedback": "由預備到隨揮時，請讓下肢完成支撐轉換並讓軀幹自然向前傾。",
                "evidence": "第一與最後畫面的腳部支撐和雙肩相對雙髖位置仍與專家動作不同。",
                "frame_index": PHASE_INDICES[2],
                "phase": "weight_transfer",
                "joint_ids": [5, 6, 11, 12, 15, 16],
                "rule_reference": "weight_transfer",
                "confidence": 0.9,
            }
        ],
    }

    analysis = SkillFeedbackAnalysis.model_validate(payload)

    assert analysis.problems[0].joint_ids == [5, 6, 11, 12, 15, 16]


def test_smash_display_anchors_match_semantic_phases() -> None:
    spec = get_skill_spec(Skill.SMASH)
    phases = (0, 16, 32, 48, 63)

    for rule in spec.rules:
        for anchor_index in rule.allowed_anchor_indices:
            frame = phases[anchor_index]
            assert phase_for_frame(frame, phases, spec) == rule.phase


def test_serve_instructions_name_the_two_transfer_error_modes() -> None:
    serve = system_instructions(get_skill_spec(Skill.SERVE))
    smash = system_instructions(get_skill_spec(Skill.SMASH))

    assert "correction_stance_retention_shortfall" in serve
    assert "source_pelvis_loading_shift" in serve
    assert "量測為0時不得宣稱該錯誤" in serve
    assert "stance_retention" not in smash
