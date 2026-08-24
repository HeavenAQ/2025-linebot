import pytest

from service.pipeline import (
    _correction_grade_context,
    _qualitative_phase_results,
    expert_phase_results,
)
from badminton_analysis.ml.skill_specs import get_skill_spec


def test_serve_gpt_context_includes_full_body_transition_evidence() -> None:
    spec = get_skill_spec("serve")
    diagnostics = {
        "correction_distance": 0.8,
        "position_distance": 0.4,
        "angle_distance": 0.1,
        "velocity_distance": 0.1,
        "bone_length_distance": 0.0,
        "support_transition_distance": 0.3,
        "torso_lean_transition_distance": 0.2,
        "transition_distance": 0.265,
    }
    criteria = [
        (rule.name_zh_tw, 0.1 + index * 0.01, rule.maximum * 0.5)
        for index, rule in enumerate(spec.rules)
    ]

    context = _correction_grade_context(
        {"total_grade": 45.0}, diagnostics, spec, criteria
    )

    assert context["distance_components"]["support_transition_distance"] == 0.3
    assert context["distance_components"]["torso_lean_transition_distance"] == 0.2
    assert context["distance_components"]["transition_distance"] == 0.265
    assert "軀幹前傾" in context["score_method_zh_tw"]


def test_generated_expert_gpt_context_describes_expert_only_score() -> None:
    spec = get_skill_spec("smash")
    diagnostics = {
        "correction_distance": 0.25,
        "position_distance": 0.2,
        "angle_distance": 0.1,
        "scorer": "continuous_generated_expert_distribution_v1",
    }
    criteria = [
        (rule.name_zh_tw, 0.1, rule.maximum * 0.8) for rule in spec.rules
    ]

    context = _correction_grade_context(
        {"total_grade": 80.0}, diagnostics, spec, criteria
    )

    assert context["score_status"] == "expert_only_generated_distribution"
    assert "歐氏距離" in context["score_method_zh_tw"]
    assert "專家動作分布" in context["score_method_zh_tw"]


def test_playback_timeline_uses_ordered_qualitative_skill_rules() -> None:
    spec = get_skill_spec("smash")

    timeline = _qualitative_phase_results(spec, sequence_length=64, fps=30.0)

    assert [marker.id for marker in timeline] == [rule.id for rule in spec.rules]
    assert [marker.label for marker in timeline] == [
        "球拍舉至腰部預備",
        "轉身",
        "雙手手肘平衡",
        "手肘往前轉至前方",
        "手腕發力",
        "慣用手肩膀往前轉",
    ]
    assert [marker.normalized_frame for marker in timeline] == sorted(
        marker.normalized_frame for marker in timeline
    )


def test_lift_playback_timeline_has_four_qualitative_checkpoints() -> None:
    spec = get_skill_spec("lift")
    phases = (0, 15, 29, 41, 63)

    timeline = _qualitative_phase_results(
        spec,
        phase_indices=phases,
        sequence_length=64,
        fps=30.0,
    )

    assert [marker.label for marker in timeline] == [
        "球拍置於身前放鬆預備",
        "持拍腳跨步並放鬆引拍",
        "弓步穩定並以前臂手腕擊球",
        "順勢隨揮並回復平衡",
    ]
    assert [marker.normalized_frame for marker in timeline] == [0, 29, 41, 63]


def test_lift_playback_timeline_uses_lunge_and_follow_through_standard() -> None:
    spec = get_skill_spec("lift")

    assert [rule.name_zh_tw for rule in spec.rules] == [
        "球拍置於身前放鬆預備",
        "持拍腳跨步並放鬆引拍",
        "弓步穩定並以前臂手腕擊球",
        "順勢隨揮並回復平衡",
    ]
    assert spec.transition_joints == (11, 12, 13, 14, 15, 16)
    assert spec.transition_weight > 0.0


def test_expert_timeline_reuses_student_rule_anchors() -> None:
    spec = get_skill_spec("smash")
    phases = (0, 12, 30, 47, 63)
    phase_seconds = (1.0, 1.4, 2.0, 2.6, 3.1)

    student = _qualitative_phase_results(
        spec, phase_indices=phases, sequence_length=64, fps=30.0
    )
    expert = expert_phase_results(
        spec,
        phase_indices=phases,
        phase_seconds=phase_seconds,
        sequence_length=64,
    )

    # Marker i must be the same criterion on both sides, otherwise playback
    # would align a checkpoint against the wrong moment of the stroke.
    assert [marker.id for marker in expert] == [marker.id for marker in student]
    assert [marker.normalized_frame for marker in expert] == [
        marker.normalized_frame for marker in student
    ]
    assert expert[0].timestamp_seconds == 1.0
    assert expert[-1].timestamp_seconds == 3.1


def test_serve_expert_timeline_follows_scoring_order_not_stroke_order() -> None:
    # Serve grades 髖關節前旋 (keyframe 4) before 持拍手手腕發力 (keyframe 3), so
    # the timeline is deliberately not chronological. Playback pairs marker for
    # marker and sorts by position, so the contract is that each criterion keeps
    # its own keyframe — not that the list runs forwards.
    spec = get_skill_spec("serve")
    phases = (0, 12, 30, 47, 63)

    expert = expert_phase_results(
        spec,
        phase_indices=phases,
        phase_seconds=(1.0, 1.3, 1.9, 2.5, 3.0),
        sequence_length=64,
    )

    by_id = {marker.id: marker for marker in expert}
    # The wrist flick marks the strike itself -- anchor 2, where serve
    # extraction puts maximum wrist acceleration -- not the anchor after it.
    assert by_id["wrist_flick"].timestamp_seconds == 1.9
    assert by_id["weight_transfer"].timestamp_seconds == 1.9
    assert by_id["hip_rotation"].timestamp_seconds == 3.0
    assert by_id["shoulder_rotation"].timestamp_seconds == 3.0
    assert [marker.timestamp_seconds for marker in expert] != sorted(
        marker.timestamp_seconds for marker in expert
    )

    # Sorted by position — the order playback interpolates through — time only
    # ever moves forwards.
    ordered = sorted(expert, key=lambda marker: marker.normalized_position)
    assert [marker.timestamp_seconds for marker in ordered] == sorted(
        marker.timestamp_seconds for marker in ordered
    )


def test_expert_timeline_timestamps_track_the_experts_own_tempo() -> None:
    spec = get_skill_spec("clear")
    phases = (0, 16, 32, 48, 63)

    # An expert who reaches impact early (1.2s into a 1.0-3.0s motion) must
    # report that, not the midpoint a uniform stretch would assume.
    expert = expert_phase_results(
        spec,
        phase_indices=phases,
        phase_seconds=(1.0, 1.1, 1.2, 1.7, 3.0),
        sequence_length=64,
    )

    impact = [marker for marker in expert if marker.normalized_frame == 32]
    assert impact and all(marker.timestamp_seconds == 1.2 for marker in impact)


def test_expert_timeline_rejects_mismatched_phase_timestamps() -> None:
    spec = get_skill_spec("clear")

    with pytest.raises(ValueError):
        expert_phase_results(
            spec,
            phase_indices=(0, 16, 32, 48, 63),
            phase_seconds=(1.0, 2.0),
            sequence_length=64,
        )


def test_analysis_runs_pose_on_the_batched_tensorrt_path() -> None:
    """The service must extract poses in batches, not frame by frame.

    Only the batched path reaches the cached TensorRT engine; `process_frames`
    runs RF-DETR in PyTorch a frame at a time, which is roughly an order of
    magnitude slower on a GPU that is billed by the second. The two are
    interchangeable at the call site, so nothing else would notice the swap --
    hence this check on the source itself.
    """
    import ast
    import pathlib

    source = pathlib.Path(__file__).resolve().parents[1] / "service" / "pipeline.py"
    tree = ast.parse(source.read_text())
    called = {
        node.func.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    }

    assert "process_frames_batched" in called
    assert "process_frames" not in called
