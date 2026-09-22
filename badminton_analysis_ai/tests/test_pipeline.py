import re
import pytest
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import api.pipeline as pipeline_module
from badminton_analysis.ml.expert_reference_bank import SkillSupport
from api.pipeline import (
    SkeletonAnalysisPipeline,
    _attach_serve_transfer_measurements,
    _correction_grade_context,
    _rule_anchor_frames,
    _source_qualitative_phase_results,
    expert_phase_results,
)
from badminton_analysis.ml.coaching_feedback import system_instructions
from badminton_analysis.ml.skill_specs import get_skill_spec
from badminton_analysis.models.types import Handedness, Skill


def test_serve_and_smash_backends_enable_ankle_spine_projection(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    created: list[tuple[Skill, dict[str, object]]] = []

    class Backend:
        def __init__(
            self,
            model_root: Path,
            skill: Skill,
            **kwargs: object,
        ) -> None:
            del model_root
            created.append((skill, kwargs))

    monkeypatch.setattr(pipeline_module, "PoseDetector", lambda: object())
    monkeypatch.setattr(pipeline_module, "CoachingGenerator", lambda model: object())
    monkeypatch.setattr(pipeline_module, "ExpertMotionGeneratorBackend", Backend)
    reference_bank = tmp_path / "expert-reference-bank.npz"
    reference_bank.touch()
    sentinel_bank = object()
    monkeypatch.setattr(
        pipeline_module, "ExpertReferenceBank", lambda path: sentinel_bank
    )

    pipeline = SkeletonAnalysisPipeline(
        tmp_path / "models",
        expert_reference_bank=reference_bank,
    )

    assert pipeline.expert_bank is sentinel_bank
    assert set(pipeline.loaded_skills) == {Skill.SERVE, Skill.SMASH}
    assert [skill for skill, _ in created] == [Skill.SERVE, Skill.SMASH]
    for skill, contract in created:
        assert contract == {
            "device": "auto",
            "candidates": 8,
            "seed": 19,
            "align_ankle_spine_view": True,
            "current_smash": skill == Skill.SMASH,
        }


def test_pipeline_refuses_to_start_without_temporal_skill_support_bank(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(pipeline_module, "PoseDetector", lambda: object())
    monkeypatch.setattr(pipeline_module, "CoachingGenerator", lambda model: object())

    with pytest.raises(FileNotFoundError, match="temporal skill validation"):
        SkeletonAnalysisPipeline(
            tmp_path / "models",
            expert_reference_bank=tmp_path / "missing.npz",
        )


def test_skill_mismatch_stops_before_generation_and_rendering(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    tracking: dict[str, object] = {}

    class Processor:
        def process_frames_batched(self, _):
            return tracking

    class Backend:
        def __init__(self, skill: Skill, pose: object) -> None:
            self.spec = get_skill_spec(skill)
            self.pose = pose
            self.infer_called = False

        def prepare(self, *_):
            return SimpleNamespace(pose=self.pose), (0, 1, 2), object()

        def infer(self, *_args, **_kwargs):
            self.infer_called = True
            raise AssertionError("mismatched motion reached diffusion inference")

    class Bank:
        def temporal_skill_support(
            self, requested_pose, alternative_pose, *, requested_skill
        ):
            assert requested_pose is prepared_serve_pose
            assert alternative_pose is prepared_smash_pose
            assert requested_skill == "serve"
            return SkillSupport(
                requested_skill="serve",
                alternative_skill="smash",
                requested_distance=0.4,
                alternative_distance=0.2,
                alternative_advantage=0.2,
                rejection_margin=0.0675,
            )

    prepared_serve_pose = object()
    prepared_smash_pose = object()
    serve_backend = Backend(Skill.SERVE, prepared_serve_pose)
    smash_backend = Backend(Skill.SMASH, prepared_smash_pose)
    pipeline = SkeletonAnalysisPipeline.__new__(SkeletonAnalysisPipeline)
    pipeline.backends = {
        Skill.SERVE: serve_backend,
        Skill.SMASH: smash_backend,
    }
    pipeline.expert_bank = Bank()
    pipeline.pose_detector = object()
    pipeline.pose_batcher = SimpleNamespace(
        request_detector=lambda: pipeline.pose_detector
    )
    pipeline.lock = threading.Lock()
    pipeline.request_lock = threading.Lock()
    pipeline.coaching = SimpleNamespace(
        generate=lambda **_: (_ for _ in ()).throw(
            AssertionError("mismatched motion reached coaching")
        )
    )

    monkeypatch.setattr(pipeline_module, "VideoProcessor", lambda *_: Processor())
    monkeypatch.setattr(
        pipeline_module, "_resolve_handedness", lambda *_: Handedness.RIGHT
    )
    monkeypatch.setattr(pipeline_module, "_populate_dominant_motion", lambda *_: None)
    monkeypatch.setattr(
        pipeline_module,
        "render_correction_video",
        lambda **_: (_ for _ in ()).throw(
            AssertionError("mismatched motion reached rendering")
        ),
    )

    with pytest.raises(
        pipeline_module.SkillMismatchError,
        match="requested serve conflicts with smash",
    ):
        pipeline.analyze(
            video_path=tmp_path / "wrong.mp4",
            output_path=tmp_path / "feedback.mp4",
            skeleton_overlay_path=tmp_path / "overlay.mp4",
            filename="not-used-for-gating.mp4",
            skill=Skill.SERVE,
            requested_handedness="right",
        )

    assert not serve_backend.infer_called
    assert not smash_backend.infer_called


def test_concurrent_analyses_never_overlap_from_pose_to_generation(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # Interleaving requests on the GPU changed what each one detected, so the
    # stretch from pose extraction to generation admits one request at a time.
    inside = 0
    peak = 0
    guard = threading.Lock()

    class Stop(Exception):
        pass

    class Processor:
        def process_frames_batched(self, _):
            nonlocal inside, peak
            with guard:
                inside += 1
                peak = max(peak, inside)
            time.sleep(0.05)
            return {}

    class Backend:
        spec = get_skill_spec(Skill.SERVE)

        def prepare(self, *_):
            return SimpleNamespace(pose=object()), (0, 1, 2), object()

        def infer(self, *_args, **_kwargs):
            nonlocal inside
            time.sleep(0.02)
            with guard:
                inside -= 1
            raise Stop()

    pipeline = SkeletonAnalysisPipeline.__new__(SkeletonAnalysisPipeline)
    pipeline.backends = {Skill.SERVE: Backend(), Skill.SMASH: Backend()}
    pipeline.expert_bank = SimpleNamespace(temporal_skill_support=lambda *_a, **_k: None)
    pipeline.pose_detector = object()
    pipeline.pose_batcher = SimpleNamespace(request_detector=lambda: pipeline.pose_detector)
    pipeline.lock = threading.Lock()
    pipeline.request_lock = threading.Lock()
    monkeypatch.setattr(pipeline_module, "VideoProcessor", lambda *_: Processor())
    monkeypatch.setattr(pipeline_module, "_resolve_handedness", lambda *_: Handedness.RIGHT)
    monkeypatch.setattr(pipeline_module, "_populate_dominant_motion", lambda *_: None)
    monkeypatch.setattr(pipeline_module, "source_fps", lambda *_: 30.0)

    def analyse(index: int) -> None:
        with pytest.raises(Stop):
            pipeline.analyze(
                video_path=tmp_path / f"{index}.mp4",
                output_path=tmp_path / f"{index}-feedback.mp4",
                skeleton_overlay_path=tmp_path / f"{index}-overlay.mp4",
                filename=f"{index}.mp4",
                skill=Skill.SERVE,
                requested_handedness="right",
            )

    workers = [threading.Thread(target=analyse, args=(index,)) for index in range(4)]
    for worker in workers:
        worker.start()
    for worker in workers:
        worker.join(timeout=5)
    assert peak == 1


def test_serve_gpt_context_reports_backend_distance_components() -> None:
    spec = get_skill_spec("serve")
    diagnostics = {
        "correction_distance": 0.8,
        "position_distance": 0.4,
        "angle_distance": 0.1,
        "scorer": "expert_only_identity_distribution_v6",
    }
    criteria = [
        (rule.name_zh_tw, 0.1 + index * 0.01, rule.maximum * 0.5)
        for index, rule in enumerate(spec.rules)
    ]

    context = _correction_grade_context(
        {"total_grade": 45.0}, diagnostics, spec, criteria
    )

    assert context["distance_components"] == {
        "position_distance": 0.4,
        "angle_distance": 0.1,
    }
    # Serve is graded against expert-only distributions; the prompt must not
    # claim learner-group calibration or describe another skill.
    assert context["score_status"] == "expert_only_generated_distribution"
    assert "專家動作分布" in context["score_method_zh_tw"]
    assert "最大加速度" in context["score_method_zh_tw"]
    assert "學生群組" not in context["score_method_zh_tw"]
    assert "挑球" not in context["score_method_zh_tw"]


def test_serve_gpt_receives_every_measurement_its_instructions_name() -> None:
    spec = get_skill_spec("serve")
    diagnostics = {"correction_distance": 0.8, "scorer": "expert_only_identity_distribution_v6"}
    criteria = [(rule.name_zh_tw, 0.1, rule.maximum * 0.5) for rule in spec.rules]
    context = _correction_grade_context({"total_grade": 45.0}, diagnostics, spec, criteria)
    measured = {"rule_reference": "weight_transfer"}
    measured["source_pelvis_loading_shift"] = 0.20
    measured["expert_lower_pelvis_loading_shift"] = 0.29
    measured["standardized_shortfall_pelvis_loading_shift"] = 1.0
    measured["correction_learner_stance_retention"] = 0.76
    measured["correction_corrected_stance_retention"] = 1.0
    measured["correction_stance_retention_shortfall"] = 0.24
    measured["correction_stance_allowance"] = 0.19
    measured["correction_learner_transfer_lead_frames"] = 8.0
    measured["correction_corrected_transfer_lead_frames"] = -3.0
    measured["correction_transfer_lead_excess_frames"] = 11.0
    measured["correction_transfer_lead_allowance_frames"] = 4.6

    wrist = {
        "rule_reference": "wrist_flick",
        "correction_learner_elbow_at_contact_degrees": 133.0,
        "correction_corrected_elbow_at_contact_degrees": 173.0,
        "correction_elbow_at_contact_shortfall_degrees": 40.0,
        "correction_elbow_allowance_degrees": 14.1,
    }

    _attach_serve_transfer_measurements(context, {"criteria": [measured, wrist]})

    # An instruction to judge by a measurement GPT never receives is worse
    # than none: it is told the error cannot be claimed without the number.
    named = set(
        re.findall(r"(?:source|expert_lower|standardized_shortfall|correction)_[a-z_]+", system_instructions(spec))
    )
    sent = set()
    for item in context["criteria"]:
        if item["rule_reference"] in {"weight_transfer", "wrist_flick"}:
            sent |= set(item)
    assert named, "the serve instructions name the measurements they rely on"
    assert named <= sent, f"named in the prompt but not sent: {sorted(named - sent)}"


def test_generated_expert_gpt_context_describes_expert_only_score() -> None:
    spec = get_skill_spec("smash")
    diagnostics = {
        "correction_distance": 0.25,
        "position_distance": 0.2,
        "angle_distance": 0.1,
        "scorer": "continuous_generated_expert_distribution_v1",
    }
    criteria = [(rule.name_zh_tw, 0.1, rule.maximum * 0.8) for rule in spec.rules]

    context = _correction_grade_context(
        {"total_grade": 80.0}, diagnostics, spec, criteria
    )

    assert context["score_status"] == "expert_only_generated_distribution"
    assert "歐氏距離" in context["score_method_zh_tw"]
    assert "專家動作分布" in context["score_method_zh_tw"]


def test_source_playback_timeline_uses_analysis_clip_clock() -> None:
    spec = get_skill_spec("serve")
    source_phases = [12, 18, 24, 31, 43]

    timeline = _source_qualitative_phase_results(
        spec,
        phase_indices=(0, 16, 32, 48, 63),
        source_phase_frames=source_phases,
        normalized_sequence_length=64,
        source_sequence_length=58,
        analysis_window_start_frame=12,
        analysis_window_end_frame=43,
        fps=30.0,
    )

    assert timeline[0].timestamp_seconds == pytest.approx(0.0)
    assert timeline[-1].timestamp_seconds == pytest.approx(31 / 30)
    assert timeline[-1].normalized_position == pytest.approx(1.0)
    assert timeline[-1].normalized_frame == 63
    assert all(marker.end_seconds > marker.start_seconds for marker in timeline)


@pytest.mark.parametrize("skill", ["serve", "smash"])
def test_expert_replay_ranges_are_movements_not_single_anchor_frames(skill):
    from api.pipeline import expert_phase_results

    markers = expert_phase_results(
        get_skill_spec(skill),
        phase_indices=(0, 1, 2, 3, 4),
        phase_seconds=(1.0, 1.5, 2.0, 2.5, 3.0),
        sequence_length=5,
    )
    for marker in markers:
        assert 1 <= marker.start_seconds < marker.end_seconds <= 3
        assert marker.start_seconds <= marker.timestamp_seconds <= marker.end_seconds


def test_follow_through_replay_uses_local_action_not_full_scoring_evidence():
    evidence = {
        "follow_through": {
            "source_interval": [12, 43],
            "replay_source_interval": [31, 43],
        }
    }
    timeline = _source_qualitative_phase_results(
        get_skill_spec("smash"),
        phase_indices=(0, 16, 32, 48, 63),
        source_phase_frames=[12, 18, 24, 31, 43],
        normalized_sequence_length=64,
        source_sequence_length=58,
        analysis_window_start_frame=12,
        analysis_window_end_frame=43,
        fps=30.0,
        checkpoint_evidence=evidence,
    )
    ending = next(marker for marker in timeline if marker.id == "follow_through")
    assert ending.start_seconds == pytest.approx(19 / 30)
    assert ending.end_seconds == pytest.approx(31 / 30)
    assert evidence["follow_through"]["source_interval"] == [12, 43]


def test_expert_timeline_reuses_student_rule_anchors() -> None:
    spec = get_skill_spec("smash")
    phases = (0, 12, 30, 47, 63)
    phase_seconds = (1.0, 1.4, 2.0, 2.6, 3.1)

    student_frames = _rule_anchor_frames(spec, phases, 63)
    expert = expert_phase_results(
        spec,
        phase_indices=phases,
        phase_seconds=phase_seconds,
        sequence_length=64,
    )

    # Marker i must be the same criterion on both sides, otherwise playback
    # would align a checkpoint against the wrong moment of the stroke.
    assert [marker.id for marker in expert] == [rule.id for rule in spec.rules]
    assert [marker.normalized_frame for marker in expert] == student_frames
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
    spec = get_skill_spec("smash")
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
    spec = get_skill_spec("smash")

    with pytest.raises(ValueError):
        expert_phase_results(
            spec,
            phase_indices=(0, 16, 32, 48, 63),
            phase_seconds=(1.0, 2.0),
            sequence_length=64,
        )


def test_pose_dump_records_skill_and_handedness_names_and_backend(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import numpy as np

    import api.storage

    captured: dict[str, object] = {}

    class Storage:
        def __init__(self, project: str, bucket: str) -> None:
            pass

        def upload_file(self, local: Path, object_path: str, **_: object) -> None:
            with np.load(local) as archive:
                captured.update({key: archive[key] for key in archive.files})
            captured["object_path"] = object_path

    monkeypatch.setattr(api.storage, "ObjectStorage", Storage)
    pipeline_module._dump_pose_arrays(
        "dumps",
        "clip.mp4",
        Skill.SERVE,
        Handedness.RIGHT,
        {},
        np.zeros((2, 17, 2)),
        np.ones((2, 17)),
        np.zeros((2, 2)),
        (0, 1),
        (0, 1, 1, 1, 1),
        (0, 1),
        "torch",
    )

    assert str(captured["skill"]) == "serve"
    assert str(captured["handedness"]) == "right"
    assert str(captured["pose_backend"]) == "torch"
    assert Skill.convert_to_enum(str(captured["skill"])) == Skill.SERVE
