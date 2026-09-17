import importlib.util
import sys
from pathlib import Path

import pytest

_SPEC = importlib.util.spec_from_file_location(
    "gpt_validation_batch", Path(__file__).resolve().parents[1] / "scripts" / "gpt_validation_batch.py"
)
batch = importlib.util.module_from_spec(_SPEC)
# dataclasses resolve their module through sys.modules.
sys.modules[_SPEC.name] = batch
_SPEC.loader.exec_module(batch)

PREFIX = "gpt-validation/beginners-2026-09"


def _video(skill: str, name: str) -> "batch.InputVideo":
    return batch.parse_input(f"{PREFIX}/inputs/{skill}/{name}", PREFIX)


def test_parse_input_keeps_originals_and_drops_mirrors_and_strays() -> None:
    video = _video("smash", "CG11-較佳.mp4")
    assert video is not None
    assert (video.skill, video.source_file) == ("smash", "CG11-較佳.mp4")
    assert _video("smash", "CG11-較佳_left.mp4") is None
    assert _video("clear", "EG1.mp4") is None
    assert _video("serve", "notes.txt") is None
    assert batch.parse_input(f"{PREFIX}/renders/serve-x/feedback.mp4", PREFIX) is None
    assert batch.parse_input(f"{PREFIX}/inputs/serve/nested/CG01.mp4", PREFIX) is None


def test_item_ids_are_stable_and_skill_scoped() -> None:
    assert batch.item_id_for("serve", "CG01.mp4") == batch.item_id_for("serve", "CG01.mp4")
    assert batch.item_id_for("serve", "CG01.mp4").startswith("serve-")
    assert batch.item_id_for("serve", "CG01.mp4") != batch.item_id_for("serve", "CG02.mp4")


def test_display_codes_are_neutral_and_numbered_per_skill() -> None:
    videos = [_video("serve", "CG01.mp4"), _video("serve", "CG02-較佳.mp4"), _video("smash", "CG1.mp4")]
    codes = batch.display_codes(videos)
    assert sorted(code for code in codes.values() if code.startswith("SV-")) == ["SV-001", "SV-002"]
    assert codes[videos[2].item_id] == "SM-001"
    assert all("CG" not in code and "較佳" not in code for code in codes.values())


def test_shards_partition_every_video_exactly_once() -> None:
    videos = [_video("serve", f"CG{n:02d}.mp4") for n in range(1, 23)]
    shards = [batch.shard(videos, index, 4) for index in range(4)]
    assert sorted(video.item_id for part in shards for video in part) == sorted(v.item_id for v in videos)
    with pytest.raises(ValueError):
        batch.shard(videos, 4, 4)


def test_should_process_resumes_and_retries_failures_within_budget() -> None:
    assert batch.should_process(None, force=False, max_attempts=2)
    assert not batch.should_process({"status": "ready"}, force=False, max_attempts=2)
    assert batch.should_process({"status": "ready"}, force=True, max_attempts=2)
    assert batch.should_process({"status": "failed", "attempts": 1}, force=False, max_attempts=2)
    assert not batch.should_process({"status": "failed", "attempts": 2}, force=False, max_attempts=2)
    assert batch.should_process({"status": "processing", "attempts": 1}, force=False, max_attempts=2)


RULES = [("arms_raised", "雙手舉起", 5.0), ("wrist_flick", "手腕發力", 30.0)]


def _item(**overrides):
    arguments = dict(
        video=_video("serve", "CG01.mp4"),
        batch_id="beginners-2026-09",
        display_code="SV-001",
        rules=RULES,
        grade={"total_grade": 21.0, "grading_details": [{"description": "雙手舉起", "grade": 5.0}, {"description": "手腕發力", "grade": 16.0}]},
        handedness="right",
        coaching_source="openai",
        coaching_model="gpt-5.6-terra",
        overall_feedback="整體不錯。",
        problems=[
            {"rule_reference": "wrist_flick", "title": "手腕發力", "feedback": "擊球時手腕加速。"},
            {"rule_reference": "wrist_flick", "title": "手腕發力", "feedback": "再補充一次。"},
        ],
        media={"detected_overlay": "a", "feedback_video": "b", "skeleton_overlay": "c"},
        attempts=1,
    )
    arguments.update(overrides)
    return batch.build_item(**arguments)


def test_build_item_matches_the_design_schema() -> None:
    item = _item()
    assert item["eligible"] is True
    assert item["status"] == "ready"
    assert item["criteria"] == [
        {"id": "arms_raised", "name_zh": "雙手舉起", "grade": 5.0, "maximum": 5.0},
        {"id": "wrist_flick", "name_zh": "手腕發力", "grade": 16.0, "maximum": 30.0},
    ]
    assert [cue["index"] for cue in item["gpt_cues"]] == [1, 2]
    assert item["gpt_flagged_criteria"] == ["wrist_flick"]
    assert item["source_file"] == "CG01.mp4" and item["display_code"] == "SV-001"


def test_only_real_gpt_output_is_eligible() -> None:
    assert _item(coaching_source="deterministic_fallback")["eligible"] is False
    assert _item(coaching_source="score_gate", problems=[])["eligible"] is False


def test_build_item_rejects_mismatched_rubric_or_unknown_cue_criterion() -> None:
    with pytest.raises(ValueError):
        _item(rules=RULES[:1])
    with pytest.raises(ValueError):
        _item(problems=[{"rule_reference": "made_up", "title": "", "feedback": ""}])
