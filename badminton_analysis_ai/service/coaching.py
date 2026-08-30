from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Any

from openai import OpenAI

from badminton_analysis.ml.clear_feedback import (
    RawSkillFeedbackAnalysis,
    SampledFrame,
    SkillFeedbackAnalysis,
    build_response_input,
    coaching_target_joint_ids,
    maximum_feedback_problem_count,
    prompt_context,
    sample_video_frames,
    system_instructions,
    validate_analysis_frames,
)
from badminton_analysis.ml.skill_specs import SkillCorrectionSpec

LOGGER = logging.getLogger("badminton-analysis.coaching")


def _criterion_priority(item: dict[str, Any]) -> tuple[float, float]:
    """Rank rubric evidence by missing weighted points, then by ratio.

    The rubric maxima encode the relative importance of each checkpoint. A
    zero on a five-point preparation detail must not hide a 25-point deficit
    in weight transfer merely because its normalized ratio is slightly lower.
    """
    score = float(item.get("score", 0.0))
    maximum = max(float(item.get("maximum", 1.0)), 1e-6)
    return (score - maximum, score / maximum)


def _normalized_to_output_frame_indices(
    normalized_frame_count: int,
    output_frame_count: int,
) -> tuple[int, ...]:
    """Map canonical motion indices onto the rendered analysis-only clip.

    The correction model always reasons in a normalized timeline while the
    coaching video contains only the inclusive source analysis range.  Keep
    the canonical indices as the feedback identities, but read their images
    from the corresponding output-local frames.
    """
    if normalized_frame_count < 2:
        raise ValueError("normalized motion must contain at least two frames")
    if output_frame_count < 1:
        raise ValueError("coaching output must contain at least one frame")
    denominator = normalized_frame_count - 1
    last_output = output_frame_count - 1
    return tuple(
        int(round(index * last_output / denominator))
        for index in range(normalized_frame_count)
    )


class CoachingGenerator:
    def __init__(self, model: str = "gpt-5.6-terra") -> None:
        self.client = OpenAI()
        self.model = model
        self.max_attempts = max(1, int(os.getenv("OPENAI_COACHING_ATTEMPTS", "2")))

    @staticmethod
    def _is_good_performance(correction_grade: dict[str, Any]) -> bool:
        total_threshold = float(
            os.getenv("COACHING_NO_SUGGESTION_MIN_SCORE", "90")
        )
        criterion_threshold = float(
            os.getenv("COACHING_NO_SUGGESTION_MIN_CRITERION_RATIO", "0.8")
        )
        if float(correction_grade["total_score"]) < total_threshold:
            return False
        criteria = correction_grade.get("criteria", [])
        return bool(criteria) and all(
            float(criterion["score"])
            / max(float(criterion["maximum"]), 1e-6)
            >= criterion_threshold
            for criterion in criteria
        )

    def _good_performance_payload(
        self,
        spec: SkillCorrectionSpec,
        correction_grade: dict[str, Any],
    ) -> dict[str, Any]:
        return {
            "model": self.model,
            "response_id": "",
            "source": "score_gate",
            "attempts": 0,
            "fallback_error": "",
            "latency_llm_inference_seconds": 0.0,
            "context": {
                "total_grade": float(correction_grade["total_score"]),
                "score_status": correction_grade.get(
                    "score_status", "diagnostic_group_calibrated"
                ),
            },
            "analysis": {
                "skill": spec.slug,
                "language": "zh-TW",
                "overall_feedback": (
                    f"本次{spec.name_zh_tw}動作表現良好，未發現需要修正的技術問題。"
                ),
                "problems": [],
            },
        }

    @staticmethod
    def _fallback_analysis(
        spec: SkillCorrectionSpec,
        correction_grade: dict[str, Any],
        problem_count: int = 1,
    ) -> dict[str, Any]:
        criteria = sorted(
            correction_grade["criteria"],
            key=_criterion_priority,
        )[:problem_count]
        rules = [spec.rule(str(item["rule_reference"])) for item in criteria]
        return {
            "skill": spec.slug,
            "language": "zh-TW",
            "overall_feedback": (
                f"本次{spec.name_zh_tw}應依序改善"
                + "、".join(f"「{rule.name_zh_tw}」" for rule in rules)
                + "。"
            ),
            "problems": [
                {
                    "priority": "高" if index == 0 else "中",
                    "title": rule.name_zh_tw,
                    "feedback": rule.calculation_zh_tw,
                    "evidence": (
                        f"此項得分為{float(criterion['score']):.1f}/"
                        f"{float(criterion['maximum']):.1f}分，"
                        "學生原始骨架與修正骨架在這個技術階段的差距最明顯。"
                    ),
                    "frame_index": 0,
                    "phase": rule.phase,
                    "joint_ids": list(rule.coaching_joints),
                    "rule_reference": rule.id,
                    "confidence": 0.8,
                }
                for index, (criterion, rule) in enumerate(
                    zip(criteria, rules, strict=True)
                )
            ],
        }

    @staticmethod
    def _normalize_analysis(
        analysis: dict[str, Any],
        *,
        spec: SkillCorrectionSpec,
        correction_grade: dict[str, Any],
        phase_indices: tuple[int, ...],
        samples: list[SampledFrame],
    ) -> dict[str, Any]:
        criteria = {
            str(item["rule_reference"]): item
            for item in correction_grade["criteria"]
        }
        anchors = tuple(int(value) for value in phase_indices)
        for problem in analysis["problems"]:
            reference = str(problem["rule_reference"])
            try:
                matched_rule = spec.rule(reference)
            except KeyError:
                fallback_rule = next(
                    (candidate for candidate in spec.rules if candidate.name_zh_tw == reference),
                    None,
                )
                if fallback_rule is None:
                    raise
                matched_rule = fallback_rule
            rule = matched_rule
            problem["rule_reference"] = rule.id
            problem["title"] = rule.name_zh_tw
            problem["phase"] = rule.phase
            allowed = [anchors[index] for index in rule.allowed_anchor_indices]
            problem["frame_index"] = min(
                allowed, key=lambda frame: abs(frame - int(problem["frame_index"]))
            )
            problem["joint_ids"] = coaching_target_joint_ids(rule.id, spec)
            criterion = criteria[rule.id]
            problem["criterion_score"] = float(criterion["score"])
            problem["criterion_maximum"] = float(criterion["maximum"])
        maximum_problem_count = maximum_feedback_problem_count(
            float(correction_grade["total_score"])
        )
        references = [
            str(problem["rule_reference"]) for problem in analysis["problems"]
        ]
        if len(references) > maximum_problem_count:
            raise ValueError(
                f"feedback must contain at most {maximum_problem_count} problems"
            )
        if len(set(references)) != len(references):
            raise ValueError("feedback problems must use distinct criteria")
        validated = SkillFeedbackAnalysis.model_validate(analysis)
        validate_analysis_frames(validated, samples, anchors, spec)
        timestamps = {sample.frame_index: sample.timestamp_seconds for sample in samples}
        for problem in analysis["problems"]:
            problem["timestamp_seconds"] = timestamps[problem["frame_index"]]
        return analysis

    def generate(
        self,
        *,
        video_path: Path,
        working_dir: Path,
        filename: str,
        handedness: str,
        phase_indices: tuple[int, ...],
        normalized_sequence_length: int,
        output_frame_count: int,
        spec: SkillCorrectionSpec,
        correction_grade: dict[str, Any],
    ) -> dict[str, Any]:
        if self._is_good_performance(correction_grade):
            payload = self._good_performance_payload(spec, correction_grade)
            (working_dir / "feedback.json").write_text(
                json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            return payload

        samples = sample_video_frames(
            video_path,
            working_dir / "coaching_frames",
            phase_indices=phase_indices,
            source_frame_indices=_normalized_to_output_frame_indices(
                normalized_sequence_length,
                output_frame_count,
            ),
            spec=spec,
        )
        advice = {
            "filename": filename,
            "handedness": handedness,
            "total_grade": correction_grade["total_score"],
            "score_status": correction_grade.get(
                "score_status", "diagnostic_group_calibrated"
            ),
            "priority_corrections": [],
            "keypoints": [],
        }
        context = prompt_context(
            advice,
            samples,
            phase_indices=phase_indices,
            correction_grade=correction_grade,
            spec=spec,
        )
        request_input = build_response_input(context, samples, spec)
        response = None
        analysis = None
        last_error: Exception | None = None
        llm_started = time.perf_counter()
        for attempt in range(1, self.max_attempts + 1):
            try:
                response = self.client.responses.parse(
                    model=self.model,
                    instructions=system_instructions(spec),
                    input=request_input,  # type: ignore[arg-type]
                    text_format=RawSkillFeedbackAnalysis,
                    reasoning={"effort": "medium"},
                    max_output_tokens=2200,
                    store=False,
                )
                parsed = response.output_parsed
                if parsed is None:
                    raise ValueError("OpenAI response did not contain parsed coaching")
                analysis = self._normalize_analysis(
                    parsed.model_dump(),
                    spec=spec,
                    correction_grade=correction_grade,
                    phase_indices=tuple(int(value) for value in phase_indices),
                    samples=samples,
                )
                break
            except Exception as exc:  # OpenAI and schema errors share this boundary.
                last_error = exc
                LOGGER.warning(
                    "coaching attempt failed skill=%s attempt=%d/%d error=%s",
                    spec.slug,
                    attempt,
                    self.max_attempts,
                    type(exc).__name__,
                )
        llm_finished = time.perf_counter()
        if analysis is None:
            analysis = self._normalize_analysis(
                self._fallback_analysis(
                    spec,
                    correction_grade,
                    problem_count=1,
                ),
                spec=spec,
                correction_grade=correction_grade,
                phase_indices=tuple(int(value) for value in phase_indices),
                samples=samples,
            )
            response_id = ""
            source = "deterministic_fallback"
        else:
            response_id = response.id if response is not None else ""
            source = "openai"
        payload = {
            "model": self.model,
            "response_id": response_id,
            "source": source,
            "attempts": attempt,
            "fallback_error": (
                type(last_error).__name__
                if source == "deterministic_fallback" and last_error
                else ""
            ),
            "latency_llm_inference_seconds": llm_finished - llm_started,
            "context": context,
            "analysis": analysis,
        }
        (working_dir / "feedback.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return payload
