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
    SkillFeedbackAnalysis,
    build_response_input,
    coaching_target_joint_ids,
    prompt_context,
    sample_video_frames,
    system_instructions,
    validate_analysis_frames,
)
from badminton_analysis.ml.skill_specs import SkillCorrectionSpec

LOGGER = logging.getLogger("badminton-analysis.coaching")


class CoachingGenerator:
    def __init__(self, model: str = "gpt-5.6-terra") -> None:
        self.client = OpenAI()
        self.model = model
        self.max_attempts = max(1, int(os.getenv("OPENAI_COACHING_ATTEMPTS", "2")))

    @staticmethod
    def _fallback_analysis(
        spec: SkillCorrectionSpec,
        correction_grade: dict[str, Any],
    ) -> dict[str, Any]:
        criterion = min(
            correction_grade["criteria"],
            key=lambda item: float(item["score"]) / max(float(item["maximum"]), 1e-6),
        )
        rule = spec.rule(str(criterion["rule_reference"]))
        score = float(criterion["score"])
        maximum = float(criterion["maximum"])
        return {
            "skill": spec.slug,
            "language": "zh-TW",
            "overall_feedback": (
                f"本次{spec.name_zh_tw}應優先改善「{rule.name_zh_tw}」，"
                "請依照專家動作的關鍵姿勢與節奏逐步修正。"
            ),
            "problems": [
                {
                    "priority": "高",
                    "title": rule.name_zh_tw,
                    "feedback": (
                        f"請在「{rule.name_zh_tw}」階段對照專家姿勢，"
                        "調整慣用側關節的位置、角度與動作速度。"
                    ),
                    "evidence": (
                        f"此項得分為{score:.1f}/{maximum:.1f}分，"
                        "學生原始骨架與修正骨架在這個技術階段的差距最明顯。"
                    ),
                    "frame_index": 0,
                    "phase": rule.phase,
                    "joint_ids": list(rule.coaching_joints),
                    "rule_reference": rule.id,
                    "confidence": 0.8,
                }
            ],
        }

    def generate(
        self,
        *,
        video_path: Path,
        working_dir: Path,
        filename: str,
        handedness: str,
        phase_indices: tuple[int, ...],
        spec: SkillCorrectionSpec,
        correction_grade: dict[str, Any],
    ) -> dict[str, Any]:
        samples = sample_video_frames(
            video_path,
            working_dir / "coaching_frames",
            phase_indices=phase_indices,
            source_frame_indices=tuple(range(64)),
            spec=spec,
        )
        advice = {
            "filename": filename,
            "handedness": handedness,
            "total_grade": correction_grade["total_score"],
            "score_status": "diagnostic_group_calibrated",
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
        parsed = None
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
        if parsed is None:
            analysis = self._fallback_analysis(spec, correction_grade)
            response_id = ""
            source = "deterministic_fallback"
        else:
            analysis = parsed.model_dump()
            response_id = response.id if response is not None else ""
            source = "openai"
        criteria = {
            item["name_zh_tw"]: item
            for item in correction_grade["criteria"]
        }
        anchors = tuple(int(value) for value in phase_indices)
        for problem in analysis["problems"]:
            rule = spec.rule(problem["rule_reference"])
            problem["title"] = rule.name_zh_tw
            problem["phase"] = rule.phase
            allowed = [anchors[index] for index in rule.allowed_anchor_indices]
            problem["frame_index"] = min(
                allowed, key=lambda frame: abs(frame - int(problem["frame_index"]))
            )
            problem["joint_ids"] = coaching_target_joint_ids(rule.id, spec)
            criterion = criteria[rule.name_zh_tw]
            problem["criterion_score"] = float(criterion["score"])
            problem["criterion_maximum"] = float(criterion["maximum"])
        validated = SkillFeedbackAnalysis.model_validate(analysis)
        validate_analysis_frames(validated, samples, anchors, spec)
        timestamps = {sample.frame_index: sample.timestamp_seconds for sample in samples}
        for problem in analysis["problems"]:
            problem["timestamp_seconds"] = timestamps[problem["frame_index"]]
        payload = {
            "model": self.model,
            "response_id": response_id,
            "source": source,
            "attempts": self.max_attempts if parsed is None else attempt,
            "fallback_error": type(last_error).__name__ if parsed is None and last_error else "",
            "latency_llm_inference_seconds": llm_finished - llm_started,
            "context": context,
            "analysis": analysis,
        }
        (working_dir / "feedback.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return payload
