from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

from badminton_analysis.ml.clear_feedback import (
    CLEAR_RULES,
    ClearFeedbackAnalysis,
    DEFAULT_PHASE_INDICES,
    RawSkillFeedbackAnalysis,
    coaching_target_joint_ids,
    feedback_frame_indices,
    handedness_note_zh_tw,
    load_correction_grade_context,
    load_feedback_display_score,
    load_feedback_problems,
    phase_for_frame,
    prompt_context,
    build_response_input,
    sample_video_frames,
)
from badminton_analysis.ml.skill_specs import get_skill_spec
from badminton_analysis.models.types import Skill


def _write_test_video(path: Path, frame_count: int = 64) -> None:
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter.fourcc(*"mp4v"), 30.0, (48, 64)
    )
    assert writer.isOpened()
    try:
        for frame_index in range(frame_count):
            frame = np.full((64, 48, 3), frame_index, dtype=np.uint8)
            writer.write(frame)
    finally:
        writer.release()


def test_phase_for_frame_uses_clear_feedback_windows() -> None:
    assert phase_for_frame(0) == "preparation"
    assert phase_for_frame(19) == "preparation"
    assert phase_for_frame(20) == "rotation"
    assert phase_for_frame(38) == "rotation"
    assert phase_for_frame(39) == "contact"
    assert phase_for_frame(40) == "follow_through"


@pytest.mark.parametrize("skill", (Skill.SERVE, Skill.LIFT))
def test_short_final_phase_keeps_last_anchor_in_follow_through(
    skill: Skill,
) -> None:
    phases = (0, 29, 59, 61, 63)
    spec = get_skill_spec(skill)

    assert phase_for_frame(61, phases, spec) == "contact"
    assert phase_for_frame(63, phases, spec) == "follow_through"


def test_sample_video_frames_includes_exact_grading_checkpoints(tmp_path: Path) -> None:
    video_path = tmp_path / "stroke.mp4"
    _write_test_video(video_path)

    samples = sample_video_frames(video_path, tmp_path / "frames")

    assert tuple(sample.frame_index for sample in samples) == feedback_frame_indices(
        DEFAULT_PHASE_INDICES
    )
    assert set(DEFAULT_PHASE_INDICES).issubset(
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
        source_frame_indices=source_mapping,
    )

    contact = next(sample for sample in samples if sample.frame_index == 39)
    assert contact.source_frame_index == 78
    assert contact.timestamp_seconds == pytest.approx(78 / 30)
    assert contact.manifest()["source_frame_index"] == 78


def test_clear_rule_names_match_coaching_contract() -> None:
    assert tuple(rule["name_zh_tw"] for rule in CLEAR_RULES) == (
        "球拍舉至腰部預備",
        "轉身",
        "雙手手肘平衡",
        "手肘往前轉至前方",
        "手腕發力",
        "慣用手肩膀往前轉",
    )


def test_feedback_schema_rejects_unknown_frame_or_joint() -> None:
    payload = {
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
        ClearFeedbackAnalysis.model_validate(payload)


def test_raw_feedback_schema_defers_skill_rule_normalization() -> None:
    payload = {
        "skill": "clear",
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
        ClearFeedbackAnalysis.model_validate(payload)


def test_load_feedback_problems_validates_renderer_contract(tmp_path: Path) -> None:
    feedback_path = tmp_path / "feedback.json"
    problem = {
        "frame_index": 39,
        "joint_ids": [8, 10],
        "title": "手腕發力",
        "feedback": "擊球時請讓慣用側手腕更完整地向前發力。",
        "phase": "contact",
    }
    feedback_path.write_text(
        json.dumps({"analysis": {"problems": [problem]}}), encoding="utf-8"
    )

    assert load_feedback_problems(feedback_path) == [problem]


def test_load_feedback_display_score(tmp_path: Path) -> None:
    feedback_path = tmp_path / "feedback.json"
    feedback_path.write_text(
        json.dumps({"correction_total_score": 45.0}), encoding="utf-8"
    )

    assert load_feedback_display_score(feedback_path) == pytest.approx(45.0)


def test_correction_grade_context_uses_distance_scores(tmp_path: Path) -> None:
    grading_path = tmp_path / "grading.csv"
    row: dict[str, str | float] = {
        "filename": "student.mp4",
        "label": "beginners",
        "total_grade": 14.6,
        "correction_distance": 0.959,
        "position_distance": 0.840,
        "angle_distance": 0.134,
        "velocity_distance": 0.104,
        "bone_length_distance": 0.0,
    }
    for index, (grade, distance) in enumerate(
        (
            (1.60, 0.88),
            (1.69, 0.86),
            (0.39, 1.67),
            (0.49, 1.59),
            (1.53, 1.16),
            (8.90, 0.50),
        ),
        start=1,
    ):
        row[f"detail_{index}_grade"] = grade
        row[f"detail_{index}_distance"] = distance
    pd.DataFrame([row]).to_csv(grading_path, index=False)

    context = load_correction_grade_context(
        grading_path,
        "student.mp4",
    )

    assert context["total_score"] == pytest.approx(14.6)
    assert context["criteria"][2]["name_zh_tw"] == "雙手手肘平衡"
    assert context["criteria"][2]["score"] == pytest.approx(0.39)


def test_follow_through_coaching_targets_only_dominant_shoulder() -> None:
    assert coaching_target_joint_ids("follow_through") == [6]
    assert coaching_target_joint_ids("arm_balance") == [7, 8]
    assert coaching_target_joint_ids("preparation") == [6, 8, 10]


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
        phase_indices=DEFAULT_PHASE_INDICES,
        correction_grade={"total_score": 45.0},
        spec=spec,
    )

    assert context["criterion_comparison_frames"]["重心轉移至非持拍腳"] == [
        DEFAULT_PHASE_INDICES[0],
        DEFAULT_PHASE_INDICES[-1],
    ]
    prompt = build_response_input(context, (), spec)[0]["content"][0]["text"]
    assert "下肢支撐轉換" in prompt
    assert "雙肩相對雙髖是否向前傾" in prompt
