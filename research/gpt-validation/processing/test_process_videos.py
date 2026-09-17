import csv
import importlib.util
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, HERE / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    # dataclasses resolve their module through sys.modules.
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


process = _load("process_videos")
manifest = _load("build_manifest")


def _video(skill: str, source_file: str, workbook_id: str = "X") -> "process.InputVideo":
    return process.InputVideo(skill=skill, workbook_id=workbook_id, source_file=source_file, source_path=f"dir/{source_file}")


def test_workbook_ids_match_files_by_prefix_and_number() -> None:
    files = ["EG1.mp4", "EG1_left.mp4", "EG20-較佳.mp4", "EG20-較佳_left.mp4", "EG49原CG46.mp4", "CG04_left.mp4", "notes.txt"]
    assert manifest.match_file("EG01", files) == "EG1.mp4"
    assert manifest.match_file("EG20", files) == "EG20-較佳.mp4"
    assert manifest.match_file("EG49", files) == "EG49原CG46.mp4"
    # A mirrored copy is used only when it is the student's sole recording.
    assert manifest.match_file("CG04", files) == "CG04_left.mp4"
    with pytest.raises(ValueError):
        manifest.match_file("EG99", files)
    with pytest.raises(ValueError):
        manifest.match_file("EG1", files + ["EG01.mp4"])


def test_committed_manifest_covers_the_100_workbook_videos() -> None:
    videos = process.read_manifest(HERE / "manifest.csv")
    assert len(videos) == 100
    assert sum(video.skill == "smash" for video in videos) == 50
    assert sum(video.skill == "serve" for video in videos) == 50
    assert len({(video.skill, video.workbook_id) for video in videos}) == 100


def test_read_manifest_rejects_duplicates_and_unknown_skills(tmp_path: Path) -> None:
    path = tmp_path / "manifest.csv"
    fields = ["skill", "workbook_id", "source_file", "source_path"]

    def write(rows):
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)

    row = {"skill": "serve", "workbook_id": "CG01", "source_file": "CG01.mp4", "source_path": "s/CG01.mp4"}
    write([row, row])
    with pytest.raises(ValueError):
        process.read_manifest(path)
    write([{**row, "skill": "clear"}])
    with pytest.raises(ValueError):
        process.read_manifest(path)


def test_item_ids_are_stable_and_skill_scoped() -> None:
    assert process.item_id_for("serve", "CG01.mp4") == process.item_id_for("serve", "CG01.mp4")
    assert process.item_id_for("serve", "CG01.mp4").startswith("serve-")
    assert process.item_id_for("serve", "CG01.mp4") != process.item_id_for("serve", "CG02.mp4")


def test_display_codes_are_neutral_and_numbered_per_skill() -> None:
    videos = [_video("serve", "CG01.mp4"), _video("serve", "CG02-較佳.mp4"), _video("smash", "CG1.mp4")]
    codes = process.display_codes(videos)
    assert sorted(code for code in codes.values() if code.startswith("SV-")) == ["SV-001", "SV-002"]
    assert codes[videos[2].item_id] == "SM-001"
    assert all("CG" not in code and "較佳" not in code for code in codes.values())


def test_recorded_handedness_follows_the_recording() -> None:
    assert process.recorded_handedness(_video("serve", "CG04_left.mp4")) == "left"
    assert process.recorded_handedness(_video("serve", "CG01.mp4")) == "right"


def test_should_process_resumes_and_retries_failures_within_budget() -> None:
    assert process.should_process(None, force=False, max_attempts=2)
    assert not process.should_process({"status": "ready"}, force=False, max_attempts=2)
    assert process.should_process({"status": "ready"}, force=True, max_attempts=2)
    assert process.should_process({"status": "failed", "attempts": 1}, force=False, max_attempts=2)
    assert not process.should_process({"status": "failed", "attempts": 2}, force=False, max_attempts=2)
    assert process.should_process({"status": "processing", "attempts": 1}, force=False, max_attempts=2)


RULES = [("arms_raised", "雙手舉起", 5.0), ("wrist_flick", "手腕發力", 30.0)]


def _item(**overrides):
    arguments = dict(
        video=_video("serve", "CG01.mp4", "CG01"),
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
    return process.build_item(**arguments)


def test_build_item_matches_the_design_schema() -> None:
    item = _item()
    assert item["eligible"] is True
    assert item["status"] == "ready"
    assert item["workbook_id"] == "CG01"
    assert item["criteria"] == [
        {"id": "arms_raised", "name_zh": "雙手舉起", "grade": 5.0, "maximum": 5.0},
        {"id": "wrist_flick", "name_zh": "手腕發力", "grade": 16.0, "maximum": 30.0},
    ]
    assert [cue["index"] for cue in item["gpt_cues"]] == [1, 2]
    assert item["gpt_flagged_criteria"] == ["wrist_flick"]


def test_only_real_gpt_output_is_eligible() -> None:
    assert _item(coaching_source="deterministic_fallback")["eligible"] is False
    assert _item(coaching_source="score_gate", problems=[])["eligible"] is False


def test_build_item_rejects_mismatched_rubric_or_unknown_cue_criterion() -> None:
    with pytest.raises(ValueError):
        _item(rules=RULES[:1])
    with pytest.raises(ValueError):
        _item(problems=[{"rule_reference": "made_up", "title": "", "feedback": ""}])
