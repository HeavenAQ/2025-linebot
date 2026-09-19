import cv2
import numpy as np
import pytest

from badminton_analysis.ml.coaching_feedback import (
    sample_video_frames,
    prompt_context,
    system_instructions,
)
from badminton_analysis.ml.skill_specs import get_skill_spec
from badminton_analysis.ml.smash_coaching_evidence import build_checkpoint_evidence
from api.coaching import CoachingGenerator
from api.coaching_timeline import coaching_video_frame


def eg27_evidence():
    bounds = [
        (51, 81, 57),
        (84, 100, 95),
        (94, 112, 112),
        (114, 121, 119),
        (119, 123, 119),
        (126, 128, 128),
    ]
    intervals = {
        rule.id: dict(start=a, end=b, anchor=c)
        for rule, (a, b, c) in zip(get_skill_spec("smash").rules, bounds)
    }
    return build_checkpoint_evidence(
        intervals=intervals,
        window_start=60,
        window_end=130,
        initial_frame=60,
        balance_anchor=112,
        selected_endpoint=130,
        balance_measurement=dict(
            available=True,
            event_start=103,
            event_end=107,
            handedness="right",
            sustained_gap=0.568,
        ),
    )


@pytest.fixture
def samples(tmp_path):
    video = tmp_path / "clip.avi"
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"MJPG"), 30, (32, 32))
    assert writer.isOpened()
    for frame in range(71):
        writer.write(np.full((32, 32, 3), frame * 3, dtype=np.uint8))
    writer.release()
    return sample_video_frames(
        video,
        tmp_path / "frames",
        phase_indices=(0, 27, 53, 59, 63),
        source_frame_indices=[round(i * 70 / 63) for i in range(64)],
        spec=get_skill_spec("smash"),
        checkpoint_evidence=eg27_evidence(),
    )


def test_exact_early_balance_frames_are_supplied(samples):
    owned = [s for s in samples if "arm_balance" in s.criterion_ids]
    assert set(range(43, 48)) <= {s.source_frame_index for s in owned}
    assert all(s.timestamp_seconds == s.source_frame_index / 30 for s in owned)
    assert all(s.frame_index >= 64 for s in owned)
    # JPEG images really came from the requested frames, not nearby anchors.
    for sample in owned:
        assert (
            abs(
                cv2.imread(str(sample.image_path)).mean()
                - 3 * sample.source_frame_index
            )
            < 3
        )
    assert not eg27_evidence()["preparation"]["full_interval_visible"]
    assert 51 in eg27_evidence()["preparation"]["missing_requested_source_frames"]


def grade():
    return dict(
        total_score=96.5,
        checkpoint_evidence=eg27_evidence(),
        criteria=[
            dict(
                rule_reference=r.id,
                maximum=r.maximum,
                score=1.5 if r.id == "arm_balance" else r.maximum,
            )
            for r in get_skill_spec("smash").rules
        ],
    )


def test_follow_through_replay_does_not_include_initial_scoring_reference():
    evidence = eg27_evidence()["follow_through"]
    assert evidence["source_interval"] == [60, 130]
    assert evidence["replay_source_interval"] == [119, 130]
    assert evidence["source_frame_indices"] == [60, 130]


def analysis(frame):
    return dict(
        skill="smash",
        language="zh-TW",
        overall_feedback="請留意蓄力初段持拍手的位置。",
        problems=[
            dict(
                priority="高",
                title="雙手手肘平衡",
                feedback="非慣用手抬起後，持拍手也要及時抬起。",
                evidence="連續數幀可見慣用手仍然低於自身肩部。",
                frame_index=frame,
                phase="rotation",
                joint_ids=[8, 10],
                rule_reference="arm_balance",
                confidence=0.9,
            )
        ],
    )


def test_valid_evidence_survives_without_anchor_snapping(samples):
    sample = next(s for s in samples if s.source_frame_index == 45 and s.criterion_ids)
    result = CoachingGenerator._normalize_analysis(
        analysis(sample.frame_index),
        spec=get_skill_spec("smash"),
        correction_grade=grade(),
        phase_indices=(0, 27, 53, 59, 63),
        samples=samples,
    )
    problem = result["problems"][0]
    assert problem["video_frame_index"] == 45
    assert problem["timestamp_seconds"] == 1.5
    assert problem["frame_index"] == round(45 * 63 / 70)
    assert coaching_video_frame(problem, 64, 71) == 45
    with pytest.raises(ValueError, match="scored evidence"):
        CoachingGenerator._normalize_analysis(
            analysis(27),
            spec=get_skill_spec("smash"),
            correction_grade=grade(),
            phase_indices=(0, 27, 53, 59, 63),
            samples=samples,
        )


def test_rubric_and_prompt_agree(samples):
    spec = get_skill_spec("smash")
    assert [r.maximum for r in spec.rules] == [5, 20, 5, 20, 30, 20]
    context = prompt_context(
        {},
        samples,
        phase_indices=(0, 27, 53, 59, 63),
        correction_grade=grade(),
        spec=spec,
    )
    assert all(i >= 64 for i in context["criterion_allowed_frames"]["雙手手肘平衡"])
    assert "5／20／5／20／30／20" in system_instructions(spec)
    assert "checkpoint_evidence" not in system_instructions(get_skill_spec("serve"))


def test_out_of_window_pause_is_rejected():
    with pytest.raises(ValueError, match="outside"):
        coaching_video_frame(dict(frame_index=40, video_frame_index=71), 64, 71)
