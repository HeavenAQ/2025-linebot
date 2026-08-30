from __future__ import annotations

import base64
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Sequence

import cv2
import pandas as pd
from numpy.typing import NDArray
from pydantic import BaseModel, Field, field_validator, model_validator

from badminton_analysis.ml.skill_specs import (
    CANONICAL_JOINTS_ZH_TW,
    SkillCorrectionSpec,
    get_skill_spec,
)
from badminton_analysis.models.types import Skill

CanonicalJointId = Literal[0, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
FeedbackPhase = str
RuleReference = str
CriterionName = str

DEFAULT_PHASE_INDICES = (0, 20, 39, 51, 63)

CANONICAL_JOINTS = CANONICAL_JOINTS_ZH_TW

CLEAR_RULES = tuple(
    rule.as_prompt_dict() for rule in get_skill_spec(Skill.CLEAR).rules
)

RULE_CONTRACTS: dict[str, dict[str, Any]] = {
    rule.id: {
        "title": rule.name_zh_tw,
        "phase": rule.phase,
        "joint_ids": set(rule.measured_joints),
    }
    for rule in get_skill_spec(Skill.CLEAR).rules
}

COACHING_TARGET_JOINTS: dict[str, tuple[int, ...]] = {
    rule.id: rule.coaching_joints for rule in get_skill_spec(Skill.CLEAR).rules
}


def maximum_feedback_problem_count(total_score: float) -> int:
    if total_score < 60.0:
        return 3
    if total_score < 90.0:
        return 2
    return 1


def minimum_feedback_problem_count(total_score: float) -> int:
    """Require useful breadth once the visual review finds a problem.

    A low total can contain several large weighted deficits.  Accepting a
    single, low-value preparation cue in that case hides the movement fault
    that matters most.  An empty problem list remains valid when the images
    genuinely contradict the diagnostic score.
    """
    if total_score < 60.0:
        return 2
    if total_score < 90.0:
        return 1
    return 0


def _contains_chinese(value: str) -> str:
    if not any("\u4e00" <= character <= "\u9fff" for character in value):
        raise ValueError("feedback must be written in Traditional Chinese")
    return value


class FeedbackProblem(BaseModel):
    priority: Literal["高", "中", "低"]
    title: CriterionName
    feedback: str = Field(min_length=10, max_length=180)
    evidence: str = Field(min_length=10, max_length=240)
    frame_index: int = Field(ge=0, le=63)
    phase: FeedbackPhase
    joint_ids: list[CanonicalJointId] = Field(min_length=1, max_length=6)
    rule_reference: RuleReference
    confidence: float = Field(ge=0.0, le=1.0)

    _feedback_in_chinese = field_validator("feedback")(_contains_chinese)
    _evidence_in_chinese = field_validator("evidence")(_contains_chinese)


class RawSkillFeedbackAnalysis(BaseModel):
    """API response shape before repository-owned rule fields are normalized."""

    skill: str
    language: Literal["zh-TW"]
    overall_feedback: str = Field(min_length=10, max_length=320)
    problems: list[FeedbackProblem] = Field(max_length=3)

    _overall_in_chinese = field_validator("overall_feedback")(_contains_chinese)


class SkillFeedbackAnalysis(BaseModel):
    skill: str
    language: Literal["zh-TW"]
    overall_feedback: str = Field(min_length=10, max_length=320)
    problems: list[FeedbackProblem] = Field(max_length=3)

    _overall_in_chinese = field_validator("overall_feedback")(_contains_chinese)

    @model_validator(mode="after")
    def follows_skill_rule_contract(self) -> "SkillFeedbackAnalysis":
        spec = get_skill_spec(self.skill)
        for problem in self.problems:
            try:
                rule = spec.rule(problem.rule_reference)
            except KeyError as exc:
                raise ValueError(str(exc)) from exc
            if problem.title != rule.name_zh_tw:
                raise ValueError("criterion title does not match rule_reference")
            if problem.phase != rule.phase:
                raise ValueError("criterion phase does not match rule_reference")
            if not set(problem.joint_ids).issubset(rule.measured_joints):
                raise ValueError("joint IDs are not measured by the selected criterion")
        return self


class ClearFeedbackAnalysis(SkillFeedbackAnalysis):
    skill: Literal["clear"] = "clear"


@dataclass(frozen=True)
class SampledFrame:
    frame_index: int
    source_frame_index: int
    timestamp_seconds: float
    phase: FeedbackPhase
    checkpoint_role_zh_tw: str
    image_path: Path
    data_url: str

    def manifest(self) -> dict[str, str | int | float]:
        return {
            "frame_index": self.frame_index,
            "source_frame_index": self.source_frame_index,
            "timestamp_seconds": self.timestamp_seconds,
            "phase": self.phase,
            "checkpoint_role_zh_tw": self.checkpoint_role_zh_tw,
            "image_path": str(self.image_path),
        }


def _validated_phase_indices(phase_indices: Sequence[int]) -> tuple[int, ...]:
    values = tuple(int(value) for value in phase_indices)
    if len(values) != 5 or any(first > second for first, second in zip(values, values[1:])):
        raise ValueError("phase_indices must contain five ordered frame indices")
    if values[0] < 0 or values[-1] > 63:
        raise ValueError("phase indices must be inside the 64-frame sequence")
    return values






def feedback_frame_indices(phase_indices: Sequence[int]) -> tuple[int, ...]:
    start, rotation, contact, follow, end = _validated_phase_indices(phase_indices)
    candidates = (
        start,
        (start + rotation) // 2,
        rotation,
        (rotation + contact) // 2,
        max(rotation, contact - 3),
        contact,
        min(follow, contact + 3),
        (contact + follow) // 2,
        follow,
        (follow + end) // 2,
        end,
    )
    return tuple(sorted(set(candidates)))


def phase_for_frame(
    frame_index: int,
    phase_indices: Sequence[int] = DEFAULT_PHASE_INDICES,
    spec: SkillCorrectionSpec | None = None,
) -> FeedbackPhase:
    resolved_spec = spec or get_skill_spec(Skill.CLEAR)
    _, anchor_1, anchor_2, anchor_3, _ = _validated_phase_indices(phase_indices)
    if resolved_spec.skill == Skill.LIFT:
        if frame_index < anchor_1:
            return "preparation"
        if frame_index < anchor_3:
            return "backswing"
        if frame_index <= anchor_3:
            return "contact"
        return "follow_through"
    if resolved_spec.skill == Skill.SERVE:
        if frame_index <= anchor_1:
            return "preparation"
        if frame_index < anchor_2:
            return "weight_transfer"
        if frame_index <= anchor_2:
            return "contact"
        return "follow_through"
    if frame_index < anchor_1:
        return "preparation"
    if frame_index < anchor_2:
        return "rotation"
    if frame_index <= anchor_2:
        return "contact"
    return "follow_through"


def checkpoint_role(
    frame_index: int,
    phase_indices: Sequence[int],
    spec: SkillCorrectionSpec | None = None,
) -> str:
    resolved_spec = spec or get_skill_spec(Skill.CLEAR)
    anchors = _validated_phase_indices(phase_indices)
    roles = dict(zip(anchors, resolved_spec.checkpoint_roles_zh_tw, strict=True))
    return roles.get(frame_index, "關鍵幀之間的動作過渡畫面")




def load_feedback_problems(
    path: Path, spec: SkillCorrectionSpec | None = None
) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    analysis_payload = payload.get("analysis", {})
    if spec is not None:
        validated_analysis = SkillFeedbackAnalysis.model_validate(analysis_payload)
        if validated_analysis.skill != spec.slug:
            raise ValueError(
                f"feedback skill is {validated_analysis.skill}, but dataset skill is "
                f"{spec.slug}"
            )
    problems = analysis_payload.get("problems")
    if not isinstance(problems, list):
        raise ValueError(f"feedback file has invalid analysis problems: {path}")
    validated: list[dict[str, Any]] = []
    for problem in problems:
        if not isinstance(problem, dict):
            raise ValueError("each feedback problem must be an object")
        frame_index = int(problem.get("frame_index", -1))
        joint_ids = problem.get("joint_ids")
        if not 0 <= frame_index < 64:
            raise ValueError(f"invalid feedback frame index: {frame_index}")
        if not isinstance(joint_ids, list) or not joint_ids:
            raise ValueError("feedback problem must include joint_ids")
        if any(not 0 <= int(joint_id) < 17 for joint_id in joint_ids):
            raise ValueError(f"invalid feedback joint IDs: {joint_ids}")
        validated.append(problem)
    return validated


def load_feedback_display_score(path: Path) -> float | None:
    payload = json.loads(path.read_text(encoding="utf-8"))
    value = payload.get("correction_total_score")
    if value is None:
        return None
    score = float(value)
    if not math.isfinite(score) or not 0.0 <= score <= 100.0:
        raise ValueError(f"invalid feedback correction score: {value}")
    return score


def coaching_target_joint_ids(
    rule_reference: str, spec: SkillCorrectionSpec | None = None
) -> list[int]:
    resolved_spec = spec or get_skill_spec(Skill.CLEAR)
    return list(resolved_spec.rule(rule_reference).coaching_joints)


def handedness_note_zh_tw(handedness: str | None) -> str:
    normalized = str(handedness).lower()
    if normalized == "left":
        side = "左側"
        hand = "左手"
    elif normalized == "right":
        side = "右側"
        hand = "右手"
    else:
        return "關節編號採慣用側正規化；請依提供的handedness判斷實際身體側。"
    return (
        f"關節編號採慣用側正規化。此學生為{hand}持拍，"
        f"因此慣用側關節對應身體{side}。"
    )


def load_correction_grade_context(
    grading_results_path: Path,
    filename: str,
    spec: SkillCorrectionSpec | None = None,
) -> dict[str, Any]:
    resolved_spec = spec or get_skill_spec(Skill.CLEAR)
    grading = pd.read_csv(grading_results_path)
    rows = grading[grading["filename"] == filename]
    if "label" in grading.columns:
        student_rows = rows[rows["label"] == "beginners"]
        if not student_rows.empty:
            rows = student_rows
    if rows.empty:
        raise ValueError(
            f"no correction-distance grade found for {filename} in "
            f"{grading_results_path}"
        )
    row = rows.iloc[0]
    criteria: list[dict[str, Any]] = []
    for index, rule in enumerate(resolved_spec.rules, start=1):
        criteria.append(
            {
                "name_zh_tw": rule.name_zh_tw,
                "rule_reference": rule.id,
                "score": float(row[f"detail_{index}_grade"]),
                "maximum": rule.maximum,
                "correction_distance": float(
                    row[f"detail_{index}_distance"]
                ),
            }
        )
    component_names = (
        "position_distance",
        "angle_distance",
        "velocity_distance",
        "bone_length_distance",
        "support_transition_distance",
        "torso_lean_transition_distance",
        "transition_distance",
    )
    return {
        "score_method_zh_tw": (
            "學生原始骨架與專家化修正骨架之加權差距，經專家與學生群組分布校準；"
            "發球重心轉移另比較完整下肢支撐軌跡與軀幹前傾變化"
        ),
        "total_score": float(row["total_grade"]),
        "correction_distance": float(row["correction_distance"]),
        "distance_components": {
            name: float(row[name])
            for name in component_names
            if name in row.index and not pd.isna(row[name])
        },
        "criteria": criteria,
    }


def _encode_jpeg(frame: NDArray[Any], quality: int) -> tuple[bytes, str]:
    success, buffer = cv2.imencode(
        ".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality]
    )
    if not success:
        raise RuntimeError("could not encode sampled video frame")
    encoded = bytes(buffer)
    data_url = "data:image/jpeg;base64," + base64.b64encode(encoded).decode("ascii")
    return encoded, data_url


def sample_video_frames(
    video_path: Path,
    output_dir: Path,
    *,
    phase_indices: Sequence[int] = DEFAULT_PHASE_INDICES,
    source_frame_indices: Sequence[int] | None = None,
    spec: SkillCorrectionSpec | None = None,
    frame_indices: Sequence[int] | None = None,
    max_width: int = 640,
    jpeg_quality: int = 85,
) -> list[SampledFrame]:
    phases = _validated_phase_indices(phase_indices)
    resolved_spec = spec or get_skill_spec(Skill.CLEAR)
    source_mapping = (
        tuple(range(64))
        if source_frame_indices is None
        else tuple(int(value) for value in source_frame_indices)
    )
    if len(source_mapping) != 64:
        raise ValueError("source_frame_indices must contain 64 values")
    selected_frames = (
        feedback_frame_indices(phases) if frame_indices is None else frame_indices
    )
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise ValueError(f"could not open video: {video_path}")
    try:
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        if frame_count <= 0:
            raise ValueError(f"video has no frames: {video_path}")
        if fps <= 0:
            fps = 30.0
        output_dir.mkdir(parents=True, exist_ok=True)
        for old_frame in output_dir.glob("frame_*.jpg"):
            old_frame.unlink()
        samples: list[SampledFrame] = []
        for frame_index in selected_frames:
            source_frame_index = source_mapping[int(frame_index)]
            if not 0 <= source_frame_index < frame_count:
                raise ValueError(
                    f"requested source frame {source_frame_index}, but video has "
                    f"{frame_count} frames"
                )
            capture.set(cv2.CAP_PROP_POS_FRAMES, source_frame_index)
            success, frame = capture.read()
            if not success or frame is None:
                raise ValueError(
                    f"could not read source frame {source_frame_index} from {video_path}"
                )
            height, width = frame.shape[:2]
            if width > max_width:
                resized_height = max(1, round(height * max_width / width))
                frame = cv2.resize(
                    frame, (max_width, resized_height), interpolation=cv2.INTER_AREA
                )
            encoded, data_url = _encode_jpeg(frame, jpeg_quality)
            image_path = output_dir / f"frame_{frame_index:02d}.jpg"
            image_path.write_bytes(encoded)
            samples.append(
                SampledFrame(
                    frame_index=frame_index,
                    source_frame_index=source_frame_index,
                    timestamp_seconds=source_frame_index / fps,
                    phase=phase_for_frame(frame_index, phases, resolved_spec),
                    checkpoint_role_zh_tw=checkpoint_role(
                        frame_index, phases, resolved_spec
                    ),
                    image_path=image_path,
                    data_url=data_url,
                )
            )
        return samples
    finally:
        capture.release()


def prompt_context(
    advice: dict[str, Any],
    samples: Sequence[SampledFrame],
    *,
    phase_indices: Sequence[int],
    correction_grade: dict[str, Any],
    spec: SkillCorrectionSpec | None = None,
) -> dict[str, Any]:
    resolved_spec = spec or get_skill_spec(Skill.CLEAR)
    anchors = _validated_phase_indices(phase_indices)
    keypoints = sorted(
        advice.get("keypoints", []),
        key=lambda item: float(item.get("score", 100.0)),
    )
    priority_criteria = sorted(
        correction_grade.get("criteria", []),
        key=lambda item: (
            float(item.get("score", 0.0))
            - max(float(item.get("maximum", 1.0)), 1e-6),
            float(item.get("score", 0.0))
            / max(float(item.get("maximum", 1.0)), 1e-6),
            -float(item.get("correction_distance", 0.0)),
        ),
    )
    total_grade = float(correction_grade.get("total_score", 0.0))
    maximum_problem_count = maximum_feedback_problem_count(total_grade)
    minimum_problem_count = minimum_feedback_problem_count(total_grade)
    required_priority_criteria = [
        str(item["rule_reference"])
        for item in priority_criteria[:1]
        if float(item.get("maximum", 0.0)) - float(item.get("score", 0.0))
        >= 10.0
    ]
    feedback_candidate_criteria = [
        str(item["rule_reference"])
        for item in priority_criteria
        if float(item.get("score", 0.0))
        / max(float(item.get("maximum", 1.0)), 1e-6)
        < 0.8
    ]
    return {
        "required_output_language": "繁體中文（臺灣，zh-TW）",
        "skill": resolved_spec.slug,
        "skill_name_zh_tw": resolved_spec.name_zh_tw,
        "student": {
            "filename": advice.get("filename"),
            "handedness": advice.get("handedness"),
            "diagnostic_total_grade": advice.get("total_grade"),
            "score_status": advice.get("score_status"),
        },
        "score_warning_zh_tw": (
            "總分與各項分數來自學生原始骨架和專家化修正骨架之差距；"
            "目前只是群組校準的診斷分數，並非人工驗證的事實。每一項回饋仍須先由影像確認。"
        ),
        "overlay_legend_zh_tw": {
            "cyan": "偵測到的學生骨架",
            "green": "模型預測的專家化修正骨架",
        },
        "canonical_joint_ids_zh_tw": CANONICAL_JOINTS,
        "handedness_note_zh_tw": handedness_note_zh_tw(
            advice.get("handedness")
        ),
        "technical_criteria": [
            rule.as_prompt_dict() for rule in resolved_spec.rules
        ],
        "maximum_problem_count": maximum_problem_count,
        "minimum_problem_count_when_nonempty": minimum_problem_count,
        "required_priority_criteria_when_nonempty": required_priority_criteria,
        "feedback_candidate_criteria": feedback_candidate_criteria,
        "criterion_priority_supporting_only": priority_criteria,
        "correction_distance_grade": correction_grade,
        "criterion_allowed_frames": {
            rule.name_zh_tw: [anchors[index] for index in rule.allowed_anchor_indices]
            for rule in resolved_spec.rules
        },
        "criterion_comparison_frames": {
            rule.name_zh_tw: (
                [anchors[0], anchors[-1]]
                if resolved_spec.skill == Skill.SERVE
                and rule.id == "weight_transfer"
                else [anchors[index] for index in rule.allowed_anchor_indices]
            )
            for rule in resolved_spec.rules
        },
        "criterion_coaching_target_joint_ids": {
            rule.name_zh_tw: list(rule.coaching_joints)
            for rule in resolved_spec.rules
        },
        "model_priority_corrections_supporting_only": advice.get(
            "priority_corrections", []
        ),
        "lowest_keypoint_scores_supporting_only": keypoints[:8],
        "available_frames": [sample.manifest() for sample in samples],
    }


def build_response_input(
    context: dict[str, Any],
    samples: Sequence[SampledFrame],
    spec: SkillCorrectionSpec | None = None,
) -> list[dict[str, Any]]:
    resolved_spec = spec or get_skill_spec(str(context.get("skill", "clear")))
    criterion_count = len(resolved_spec.rules)
    maximum_problem_count = int(context.get("maximum_problem_count", 1))
    minimum_problem_count = int(
        context.get("minimum_problem_count_when_nonempty", 0)
    )
    required_priority_criteria = list(
        context.get("required_priority_criteria_when_nonempty", [])
    )
    feedback_candidate_criteria = list(
        context.get("feedback_candidate_criteria", [])
    )
    content: list[dict[str, Any]] = [
        {
            "type": "input_text",
            "text": (
                f"請依照提供的{criterion_count}項{resolved_spec.name_zh_tw}技術標準"
                "逐項分析這組依時間排序的動作畫面，不得只檢查其中一項。"
                f"skill欄位必須填寫{resolved_spec.slug}。最多回報"
                f"{maximum_problem_count}項不同標準的問題，title必須逐字使用標準名稱。"
                f"若確認至少一項問題，必須回報至少{minimum_problem_count}項不同標準；"
                f"並且必須包含重大加權缺失{required_priority_criteria}，逐項由影像驗證。"
                f"只能從未達標準的{feedback_candidate_criteria}選擇問題；"
                "已達八成的標準不得列為問題。"
                "只有影像清楚支持時才可列為問題；若動作符合全部標準，problems必須為空陣列，"
                "不得為了配合診斷分數而勉強產生建議。"
                "不得自行新增其他技術標準。請只使用available_frames中的frame_index，"
                "而且每項標準只能選criterion_allowed_frames指定的原始評分關鍵幀。"
                "判斷發球的重心轉移時，必須同時比較criterion_comparison_frames的"
                "第一與最後畫面，檢查下肢支撐轉換，以及雙肩相對雙髖是否向前傾；"
                "回報問題時仍使用criterion_allowed_frames指定的停格畫面。"
                "請只圈選criterion_coaching_target_joint_ids指定的教練提示目標。所有"
                "overall_feedback、"
                "feedback與evidence必須使用臺灣繁體中文，禁止英文句子與簡體中文。"
                "顯示分數只能使用correction_distance_grade，不得另算總分。"
                "骨架修正與分數只能作為輔助，必須先由影像"
                "確認問題。\n\n分析資料：\n"
                + json.dumps(context, ensure_ascii=False)
            ),
        }
    ]
    for sample in samples:
        content.extend(
            (
                {
                    "type": "input_text",
                    "text": (
                        f"畫面{sample.frame_index}；階段={sample.phase}；"
                        f"原始影片畫面={sample.source_frame_index}；"
                        f"影片時間={sample.timestamp_seconds:.3f}秒；"
                        f"用途={sample.checkpoint_role_zh_tw}"
                    ),
                },
                {
                    "type": "input_image",
                    "image_url": sample.data_url,
                    "detail": "high",
                },
            )
        )
    return [{"role": "user", "content": content}]


def validate_analysis_frames(
    analysis: SkillFeedbackAnalysis,
    samples: Sequence[SampledFrame],
    phase_indices: Sequence[int],
    spec: SkillCorrectionSpec | None = None,
) -> None:
    resolved_spec = spec or get_skill_spec(analysis.skill)
    if analysis.skill != resolved_spec.slug:
        raise ValueError(
            f"feedback skill {analysis.skill} does not match {resolved_spec.slug}"
        )
    anchors = _validated_phase_indices(phase_indices)
    allowed_by_rule = {
        rule.id: {anchors[index] for index in rule.allowed_anchor_indices}
        for rule in resolved_spec.rules
    }
    available = {sample.frame_index: sample for sample in samples}
    for problem in analysis.problems:
        sample = available.get(problem.frame_index)
        if sample is None:
            raise ValueError(
                f"feedback frame {problem.frame_index} was not supplied to the model"
            )
        # The problem phase names the semantic criterion, while the sampled
        # frame names the visual segment. They usually agree, but serve weight
        # transfer is evaluated across preparation/follow-through and shown at
        # the contact anchor. Frame membership below is the authoritative
        # re-indexing contract; requiring identical labels rejects that valid
        # cross-frame criterion after GPT selects it.
        if problem.frame_index not in allowed_by_rule[problem.rule_reference]:
            raise ValueError(
                f"feedback frame {problem.frame_index} is not an original grading "
                f"checkpoint for {problem.rule_reference}"
            )


def system_instructions(spec: SkillCorrectionSpec) -> str:
    return f"""你是專業羽球教練，正在分析{spec.description_zh_tw}。
你必須嚴格依照提供的{len(spec.rules)}項{spec.name_zh_tw}技術標準，不得新增、改寫或混用其他技術標準。
影像是主要證據；青色與綠色骨架差異及其分數只能作為輔助假設。診斷分數可能低估正確動作，只回報影像清楚支持的問題；若沒有問題就回傳空的problems。
所有給使用者看的文字必須使用臺灣繁體中文（zh-TW），不得使用英文句子或簡體中文。
每項建議必須簡短明確，能在兩秒的影片暫停畫面中閱讀。關節編號必須使用提供的慣用側正規化對照。"""


SYSTEM_INSTRUCTIONS = system_instructions(get_skill_spec(Skill.CLEAR))
