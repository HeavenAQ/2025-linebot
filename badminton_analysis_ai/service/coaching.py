from __future__ import annotations

import json
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


class CoachingGenerator:
    def __init__(self, model: str = "gpt-5.6-terra") -> None:
        self.client = OpenAI()
        self.model = model

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
        llm_started = time.perf_counter()
        response = self.client.responses.parse(
            model=self.model,
            instructions=system_instructions(spec),
            input=build_response_input(context, samples, spec),  # type: ignore[arg-type]
            text_format=RawSkillFeedbackAnalysis,
            reasoning={"effort": "medium"},
            max_output_tokens=2200,
            store=False,
        )
        llm_finished = time.perf_counter()
        parsed = response.output_parsed
        if parsed is None:
            raise ValueError("OpenAI response did not contain parsed coaching")
        analysis = parsed.model_dump()
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
            "response_id": response.id,
            "latency_llm_inference_seconds": llm_finished - llm_started,
            "context": context,
            "analysis": analysis,
        }
        (working_dir / "feedback.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return payload
