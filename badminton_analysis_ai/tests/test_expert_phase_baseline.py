from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

import badminton_analysis.ml.expert_motion_backend as expert_motion_backend
from badminton_analysis.ml.expert_motion_backend import (
    apply_score_conditioned_correction,
)
from badminton_analysis.ml.expert_phase_baseline import (
    _aggregate_qualitative_checkpoint_ratios,
    _criterion_components_for_spec,
    _retarget_root_with_contacts,
    _aligned,
    _serve_checklist_aggregation,
    _serve_semantic_evidence,
    _serve_expert_envelope_components,
    _serve_hip_rotation_components,
    _serve_qualitative_pose_evidence,
    _serve_weight_transfer_components,
    _serve_wrist_action_components,
    ankle_spine_view_rotation,
    apply_constrained_hierarchical_pose_placement,
    apply_fixed_hierarchical_pose_placement,
    correct_student_motion,
    discover_motion_samples,
    load_expert_phase_model,
    load_motion_sample,
    project_pose_to_student_view,
    shift_expert_body_chain_to_student_hip,
    shift_expert_body_chain_to_student_knee,
    save_expert_phase_model,
    score_expert_correction,
    train_expert_phase_model,
)
from badminton_analysis.ml.skill_specs import get_skill_spec


PHASES = np.asarray((0, 16, 32, 48, 63), dtype=np.int64)


def test_serve_checkpoint_aggregation_is_a_soft_conjunction() -> None:
    one_missing = _aggregate_qualitative_checkpoint_ratios(
        np.asarray((1.0, 1.0, 1.0, 1.0, 1.0, 0.0))
    )
    uniformly_good = _aggregate_qualitative_checkpoint_ratios(
        np.full(6, 0.8, dtype=np.float64)
    )

    assert one_missing < 0.8
    assert uniformly_good == pytest.approx(0.8)


def test_serve_checkpoint_aggregation_rejects_invalid_input() -> None:
    with pytest.raises(ValueError, match="non-empty vector"):
        _aggregate_qualitative_checkpoint_ratios(np.asarray([]))
    with pytest.raises(ValueError, match="power must be positive"):
        _aggregate_qualitative_checkpoint_ratios(
            np.ones(6, dtype=np.float64), power=0.0
        )


def _aggregation_criteria(
    ratios: tuple[float, ...], *, preparation_distance: float
) -> list[dict[str, float | str]]:
    rule_ids = (
        "arms_raised",
        "racket_foot_weight",
        "weight_transfer",
        "hip_rotation",
        "wrist_flick",
        "shoulder_rotation",
    )
    return [
        {
            "rule_reference": rule_id,
            "score": ratio,
            "maximum": 1.0,
            "expert_tolerance": 0.19,
            "expert_robust_scale": 0.10,
            "generated_target_distance": (
                preparation_distance if index == 0 else 0.0
            ),
        }
        for index, (rule_id, ratio) in enumerate(zip(rule_ids, ratios))
    ]


def test_serve_aggregation_rescues_only_supported_isolated_preparation_error() -> None:
    isolated = _serve_checklist_aggregation(
        _aggregation_criteria(
            (0.02, 1.0, 1.0, 1.0, 1.0, 1.0),
            preparation_distance=0.40,
        )
    )
    incomplete = _serve_checklist_aggregation(
        _aggregation_criteria(
            (0.02, 1.0, 0.70, 1.0, 1.0, 1.0),
            preparation_distance=0.40,
        )
    )
    unsupported = _serve_checklist_aggregation(
        _aggregation_criteria(
            (0.02, 1.0, 1.0, 1.0, 1.0, 1.0),
            preparation_distance=0.50,
        )
    )

    assert isolated[0] == pytest.approx(100.0 * 5.02 / 6.0)
    assert isolated[1] == "expert_supported_isolated_preparation_additive_v4"
    assert isolated[-1] is True
    assert incomplete[1] == "soft_conjunctive_qualitative_checkpoints_v4"
    assert incomplete[-1] is False
    assert unsupported[1] == "soft_conjunctive_qualitative_checkpoints_v4"
    assert unsupported[-1] is False


def test_ankle_spine_rotation_projects_corrected_pose_into_student_view() -> None:
    student = _pose()
    angle = np.deg2rad(18.0)
    forward = np.asarray(
        ((np.cos(angle), -np.sin(angle)), (np.sin(angle), np.cos(angle))),
        dtype=np.float32,
    )
    corrected = student @ forward.T

    rotation = ankle_spine_view_rotation(
        student, corrected, start=0, end=24
    )
    projected = project_pose_to_student_view(student, corrected, rotation)

    assert np.linalg.det(rotation) == pytest.approx(1.0, abs=1e-6)
    np.testing.assert_allclose(projected, student, atol=1e-5)
    np.testing.assert_allclose(
        np.linalg.norm(projected[:, 10] - projected[:, 8], axis=-1),
        np.linalg.norm(corrected[:, 10] - corrected[:, 8], axis=-1),
        atol=1e-6,
    )


def test_knee_then_hip_shift_uses_hierarchical_body_chains() -> None:
    corrected = _pose()
    student = corrected.copy()
    student[:, 11:15] += np.asarray((0.3, -0.2), dtype=np.float32)

    knee_shifted = shift_expert_body_chain_to_student_knee(
        student, corrected, start=0, end=24
    )

    np.testing.assert_allclose(
        knee_shifted[:, :15] - corrected[:, :15],
        np.broadcast_to(
            np.asarray((0.3, -0.2), dtype=np.float32), (64, 15, 2)
        ),
        atol=1e-6,
    )
    np.testing.assert_allclose(knee_shifted[:, 15:], corrected[:, 15:])

    student[:, 11:13] += np.asarray((0.1, 0.05), dtype=np.float32)
    shifted = shift_expert_body_chain_to_student_hip(
        student, knee_shifted, start=0, end=24
    )
    np.testing.assert_allclose(
        shifted[:, :13] - knee_shifted[:, :13],
        np.broadcast_to(
            np.asarray((0.1, 0.05), dtype=np.float32), (64, 13, 2)
        ),
        atol=1e-6,
    )
    np.testing.assert_allclose(shifted[:, 13:], knee_shifted[:, 13:])
    np.testing.assert_allclose(
        shifted[:, 10] - shifted[:, 6],
        corrected[:, 10] - corrected[:, 6],
        atol=1e-6,
    )


def test_fixed_pose_placement_preserves_every_joint_velocity() -> None:
    corrected = _pose()
    student = corrected.copy()
    student[:, 15:] += np.asarray((0.2, 0.1), dtype=np.float32)
    student[:, 13:15] += np.asarray((0.1, -0.05), dtype=np.float32)

    placed = apply_fixed_hierarchical_pose_placement(
        student, corrected, start=0, end=24
    )

    np.testing.assert_allclose(
        np.diff(placed, axis=0), np.diff(corrected, axis=0), atol=1e-6
    )


def test_constrained_pose_placement_tracks_smooth_root_without_deforming_arms() -> None:
    corrected = _pose()
    student = corrected.copy()
    root_motion = np.column_stack(
        (
            0.18 * np.linspace(0.0, 1.0, len(student)),
            0.04 * np.sin(np.linspace(0.0, np.pi, len(student))),
        )
    ).astype(np.float32)
    student += root_motion[:, None]
    # Deliberately request implausible knee/hip residuals. The placement may
    # follow them only as far as the adjoining generated leg lengths allow.
    student[:, :15, 0] += np.linspace(0.0, 0.5, len(student))[:, None]
    student[:, :13, 1] -= np.linspace(0.0, 0.4, len(student))[:, None]

    placed = apply_constrained_hierarchical_pose_placement(
        student, corrected, start=0, end=24
    )

    # Every upper-body joint receives the same per-frame rigid shift, so arm
    # vectors and therefore the generated arm correction are unchanged.
    np.testing.assert_allclose(
        placed[:, 10] - placed[:, 6],
        corrected[:, 10] - corrected[:, 6],
        atol=1e-6,
    )
    for moving, fixed in ((13, 15), (14, 16), (11, 13), (12, 14)):
        original_length = np.linalg.norm(
            corrected[:, moving] - corrected[:, fixed], axis=-1
        )
        placed_length = np.linalg.norm(placed[:, moving] - placed[:, fixed], axis=-1)
        ratio = placed_length / original_length
        assert float(np.min(ratio)) >= 0.85 - 1e-5
        assert float(np.max(ratio)) <= 1.15 + 1e-5


def _pose(offset: float = 0.0) -> np.ndarray:
    timeline = np.linspace(0.0, 1.0, 64, dtype=np.float32)
    pose = np.zeros((64, 17, 2), dtype=np.float32)
    base = np.asarray(
        [
            (0.0, -1.2), (-0.1, -1.25), (0.1, -1.25),
            (-0.2, -1.2), (0.2, -1.2), (-0.35, -0.8),
            (0.35, -0.8), (-0.55, -0.35), (0.55, -0.35),
            (-0.65, 0.1), (0.65, 0.1), (-0.25, 0.0), (0.25, 0.0),
            (-0.25, 0.65), (0.25, 0.65), (-0.25, 1.3), (0.25, 1.3),
        ],
        dtype=np.float32,
    )
    pose[:] = base
    pose[:, 8, 0] += 0.4 * timeline + offset
    pose[:, 10, 0] += 0.8 * timeline + offset
    pose[:, 10, 1] -= 0.3 * np.sin(np.pi * timeline)
    return pose


def _write_archive(
    path: Path,
    *,
    schema: str,
    skill: str = "serve",
    offset: float = 0.0,
    subject_id: str | None = None,
    root_motion: bool = False,
) -> None:
    pose = _pose(offset)
    common = {
        "confidence": np.ones((64, 17), dtype=np.float32),
        "phase_indices": PHASES,
        "handedness": np.asarray("right"),
        "skill": np.asarray(skill),
        "video_name": np.asarray(f"{path.stem}.mp4"),
    }
    if schema == "current":
        root = np.zeros((64, 2), dtype=np.float32)
        if root_motion:
            root[:, 0] = np.linspace(0.0, 0.5, 64)
        values = {
            **common,
            "skeleton": pose,
            "root_trajectory": root,
            "phase_source": np.asarray("acceleration_ending_range_v4"),
        }
    elif schema == "legacy":
        values = {**common, "skeleton_2d": pose}
    else:
        raise ValueError(schema)
    if subject_id is not None:
        values["subject_id"] = np.asarray(subject_id)
    np.savez_compressed(path, **values)


def test_loader_adapts_current_and_legacy_archives(tmp_path: Path) -> None:
    current = tmp_path / "current.npz"
    legacy = tmp_path / "legacy.npz"
    _write_archive(current, schema="current", skill="smash", subject_id="coach-a")
    _write_archive(legacy, schema="legacy")

    current_sample = load_motion_sample(current)
    legacy_sample = load_motion_sample(legacy)

    assert current_sample.pose.shape == (64, 17, 2)
    assert current_sample.alignment_contract == "overhead_asymmetric_ending_range_v4"
    assert current_sample.identity_level == "subject"
    assert legacy_sample.root.shape == (64, 2)
    assert legacy_sample.alignment_contract == "serve_detector_proxy_anchors_v1"
    assert legacy_sample.identity_level == "archive_fallback"


def test_expert_only_model_corrects_scores_and_round_trips(tmp_path: Path) -> None:
    expert_dir = tmp_path / "experts"
    student_dir = tmp_path / "beginners"
    expert_dir.mkdir()
    student_dir.mkdir()
    for index, offset in enumerate((-0.05, 0.0, 0.05)):
        _write_archive(
            expert_dir / f"expert-{index}.npz",
            schema="current",
            skill="smash",
            offset=offset,
            subject_id=f"coach-{index}",
        )
    _write_archive(
        student_dir / "student.npz",
        schema="current",
        skill="smash",
        offset=-0.25,
        subject_id="student-a",
    )

    experts = discover_motion_samples(expert_dir, expected_skill="smash")
    students = discover_motion_samples(student_dir, expected_skill="smash")
    model, report = train_expert_phase_model(experts, skill="smash", top_k=2)
    correction = correct_student_motion(model, students[0])
    grade = score_expert_correction(model, correction)
    custom_timing_grade = score_expert_correction(
        model,
        correction,
        canonical_phase_indices=np.asarray((0, 20, 42, 55, 63)),
    )

    assert report["expert_samples"] == 3
    assert len(report["held_out_expert_folds"]) == 3
    assert correction.corrected_pose.shape == (64, 17, 2)
    assert len(grade["criteria"]) == len(model.spec.rules)
    assert 0.0 <= grade["total_score"] <= 100.0
    assert 0.0 <= custom_timing_grade["total_score"] <= 100.0
    assert {row["file"] for row in grade["references"]} <= {
        path.name for path in expert_dir.glob("*.npz")
    }

    model_path = tmp_path / "model.npz"
    save_expert_phase_model(model, model_path)
    restored = load_expert_phase_model(model_path)
    np.testing.assert_allclose(restored.expert_pose, model.expert_pose)
    np.testing.assert_allclose(
        restored.expert_foot_contacts,
        model.expert_foot_contacts,
    )
    np.testing.assert_array_equal(
        restored.expert_alignment_contracts,
        model.expert_alignment_contracts,
    )
    assert restored.criterion_metric_version == model.criterion_metric_version


def test_serve_weight_transfer_measures_body_over_feet_not_camera_motion() -> None:
    source = _pose()
    target = source.copy()
    progress = np.clip((np.arange(64, dtype=np.float32) - 36.0) / 20.0, 0.0, 1.0)
    target[:, (5, 6, 11, 12), 0] -= 0.55 * progress[:, None]
    target[:, (13, 14), 0] -= 0.25 * progress[:, None]
    confidence = np.ones((64, 17), dtype=np.float32)
    root = np.zeros((64, 2), dtype=np.float32)

    identical = _serve_weight_transfer_components(
        target, root, target, root, confidence
    )
    missing = _serve_weight_transfer_components(
        source, root, target, root, confidence
    )
    moving_camera = root.copy()
    moving_camera[:, 0] = np.linspace(0.0, 2.0, 64)
    camera_invariant = _serve_weight_transfer_components(
        source, moving_camera, target, moving_camera, confidence
    )

    assert identical["combined_distance"] == pytest.approx(0.0)
    assert missing["combined_distance"] > 0.05
    assert missing["dominant_chain_trajectory_distance"] > 0.05
    assert missing["dominant_chain_change_distance"] > 0.02
    assert camera_invariant["combined_distance"] == pytest.approx(
        missing["combined_distance"]
    )

    angle = np.deg2rad(37.0)
    rotation = np.asarray(
        ((np.cos(angle), -np.sin(angle)), (np.sin(angle), np.cos(angle))),
        dtype=np.float32,
    )
    rotated = _serve_weight_transfer_components(
        source @ rotation.T,
        root,
        target @ rotation.T,
        root,
        confidence,
    )
    assert rotated["combined_distance"] == pytest.approx(
        missing["combined_distance"]
    )


def test_serve_qualitative_evidence_requires_two_raised_arms_and_real_stance() -> None:
    expert_like = np.zeros((64, 17, 2), dtype=np.float32)
    expert_like[:, 5] = (-0.4, 1.5)
    expert_like[:, 6] = (0.4, 1.5)
    expert_like[:, 7] = (-0.7, 1.2)
    expert_like[:, 8] = (0.7, 1.2)
    expert_like[:, 9] = (-0.9, 1.1)
    expert_like[:, 10] = (0.9, 1.1)
    expert_like[:, 11] = (-0.2, 0.0)
    expert_like[:, 12] = (0.2, 0.0)
    expert_like[:, 15] = (-0.5, -1.5)
    expert_like[:, 16] = (0.5, -1.5)
    weak = expert_like.copy()
    weak[:, (7, 8, 9, 10), 1] = 0.2
    weak[:, 15, 0] = -0.1
    weak[:, 16, 0] = 0.1

    expert_evidence = _serve_qualitative_pose_evidence(expert_like)
    weak_evidence = _serve_qualitative_pose_evidence(weak)

    assert weak_evidence["simultaneous_arm_elevation"] < (
        expert_evidence["simultaneous_arm_elevation"] * 0.25
    )
    assert weak_evidence["preparation_stance_width"] < (
        expert_evidence["preparation_stance_width"] * 0.25
    )


def test_serve_weight_transfer_rejects_uncoupled_pelvis_translation() -> None:
    drift = _pose()
    progress = np.clip(
        (np.arange(64, dtype=np.float32) - 20.0) / 36.0, 0.0, 1.0
    )
    # Translate the body over stationary feet without rotating the hip line.
    # Pelvis displacement alone must not be treated as a completed transfer.
    drift[:, :15, 0] += 0.65 * progress[:, None]

    evidence = _serve_qualitative_pose_evidence(drift)

    assert evidence["pelvis_loading_shift"] > 0.30
    assert evidence["dominant_chain_excursion"] > 0.02
    assert evidence["hip_rotation_excursion"] == pytest.approx(0.0)
    assert evidence["coordinated_hip_rotation"] == pytest.approx(0.0)


def test_serve_weight_transfer_accepts_root_as_camera_robust_support() -> None:
    pose = _pose()
    root = np.zeros((len(pose), 2), dtype=np.float32)
    confidence = np.ones((len(pose), 17), dtype=np.float32)
    evidence, names, weights = _serve_semantic_evidence(
        "weight_transfer", pose, root, confidence
    )
    # Both dominant-chain cues and root transfer meet expert support, while
    # the pelvis-over-ankle support transition is one scale short. Root motion
    # remains an alternate support cue for camera robustness; the stricter
    # all-cues distance is retained separately for rubric attribution.
    lower = evidence.copy()
    lower[2] += 1.0
    envelope = {
        "weight_transfer": {
            "feature_names": names,
            "lower_envelope": lower,
            "feature_scale": np.ones_like(lower),
            "feature_weights": weights,
        }
    }

    components = _serve_expert_envelope_components(
        "weight_transfer", pose, root, confidence, envelope
    )

    assert components["standardized_shortfall_pelvis_loading_shift"] == pytest.approx(
        1.0
    )
    assert components["combined_distance"] == pytest.approx(0.0)
    assert components["semantic_cue_aggregation"] == (
        "dominant_chain_with_pelvis_or_root_support"
    )


def test_serve_hip_rotation_is_coupled_to_dominant_chain_transfer() -> None:
    target = _pose()
    progress = np.clip((np.arange(64, dtype=np.float32) - 36.0) / 20.0, 0.0, 1.0)
    for frame, amount in enumerate(progress):
        hip_centre = 0.5 * (target[frame, 11] + target[frame, 12])
        angle = 0.55 * float(amount)
        rotation = np.asarray(
            (
                (np.cos(angle), -np.sin(angle)),
                (np.sin(angle), np.cos(angle)),
            ),
            dtype=np.float32,
        )
        target[frame, (11, 12)] = hip_centre + (
            target[frame, (11, 12)] - hip_centre
        ) @ rotation.T
        target[frame, 6, 0] -= 0.35 * amount
        target[frame, 14, 0] -= 0.20 * amount
    uncoupled = target.copy()
    # Preserve the same dominant-side chain transfer but remove its paired
    # pelvis rotation.
    uncoupled[:, (11, 12)] = _pose()[:, (11, 12)]
    confidence = np.ones((64, 17), dtype=np.float32)

    identical = _serve_hip_rotation_components(target, target, confidence)
    missing = _serve_hip_rotation_components(uncoupled, target, confidence)

    assert identical["combined_distance"] == pytest.approx(0.0)
    assert missing["combined_distance"] > 0.05
    assert missing["target_transfer_rotation_correlation"] > 0.5
    assert missing["transfer_rotation_coupling_distance"] > 0.2


def test_serve_hip_pattern_requires_contraction_and_orientation_groups() -> None:
    pose = _pose()
    confidence = np.ones((64, 17), dtype=np.float32)
    root = np.zeros((64, 2), dtype=np.float32)
    evidence, names, weights = _serve_semantic_evidence(
        "hip_rotation", pose, root, confidence
    )
    assert names == (
        "projected_hip_contraction",
        "projected_shoulder_contraction",
        "projected_hip_axis_rotation",
        "projected_torso_twist",
    )
    lower = evidence + 1.0
    lower[-1] = evidence[-1]
    components = _serve_expert_envelope_components(
        "hip_rotation",
        pose,
        root,
        confidence,
        {
            "hip_rotation": {
                "lower_envelope": lower,
                "feature_scale": np.ones(4, dtype=np.float64),
                "feature_weights": weights,
                "subject_values": lower[None],
                "subject_ids": np.asarray(("expert-a",)),
            }
        },
    )
    assert components["combined_distance"] == pytest.approx(1.0)
    assert components["strict_required_cue_distance"] == pytest.approx(
        np.sqrt(3.0 / 4.0)
    )
    assert components["semantic_cue_aggregation"] == (
        "contraction_and_orientation_camera_robust"
    )
    assert components["matched_expert_subject"] == (
        "subject_balanced_lower_envelope"
    )


def test_serve_wrist_action_uses_contact_window_speed_and_acceleration() -> None:
    source = _pose()
    target = source.copy()
    impulse = np.zeros(64, dtype=np.float32)
    impulse[42:48] = np.asarray((0.0, 0.15, 0.55, 1.0, 0.45, 0.0))
    target[:, 10, 0] += impulse
    target[:, 10, 1] -= 0.35 * impulse
    confidence = np.ones((64, 17), dtype=np.float32)

    identical = _serve_wrist_action_components(
        target, target, confidence, start=36, end=56
    )
    missing = _serve_wrist_action_components(
        source, target, confidence, start=36, end=56
    )

    assert identical["combined_distance"] == pytest.approx(0.0)
    assert missing["combined_distance"] > 0.03
    assert missing["source_wrist_speed_p90"] < missing["target_wrist_speed_p90"]
    assert missing["source_wrist_acceleration_p90"] < missing[
        "target_wrist_acceleration_p90"
    ]


def test_serve_expert_envelope_uses_acceleration_event_window() -> None:
    baseline = _pose()
    event = baseline.copy()
    late = baseline.copy()
    impulse = np.asarray((0.0, 0.2, 0.7, 1.2, 0.6, 0.0), dtype=np.float32)
    event[27:33, 10, 0] += impulse
    late[45:51, 10, 0] += impulse
    confidence = np.ones((64, 17), dtype=np.float32)
    root = np.zeros((64, 2), dtype=np.float32)

    event_evidence = _serve_semantic_evidence(
        "wrist_flick", event, root, confidence
    )[0]
    late_evidence = _serve_semantic_evidence(
        "wrist_flick", late, root, confidence
    )[0]

    assert event_evidence[0] > late_evidence[0]
    assert len(event_evidence) == 2


def test_serve_v6_score_is_independent_of_generated_motion_style(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expert_dir = tmp_path / "experts"
    expert_dir.mkdir()
    for index, offset in enumerate((-0.05, 0.0, 0.05)):
        _write_archive(
            expert_dir / f"expert-{index}.npz",
            schema="current",
            skill="serve",
            offset=offset,
            subject_id=f"coach-{index}",
            root_motion=True,
        )
    student_path = tmp_path / "student.npz"
    _write_archive(
        student_path,
        schema="current",
        skill="serve",
        subject_id="student-a",
        root_motion=True,
    )
    model, _ = train_expert_phase_model(
        discover_motion_samples(expert_dir, expected_skill="serve"),
        skill="serve",
        top_k=1,
    )
    student = load_motion_sample(student_path)
    correction = correct_student_motion(model, student)
    distorted = replace(
        correction,
        aligned_student_pose=np.zeros_like(correction.aligned_student_pose),
    )
    expected = score_expert_correction(model, correction)
    grade = score_expert_correction(model, distorted)
    wrist = next(
        item for item in grade["criteria"] if item["rule_reference"] == "wrist_flick"
    )
    hip = next(
        item for item in grade["criteria"] if item["rule_reference"] == "hip_rotation"
    )
    shoulder = next(
        item
        for item in grade["criteria"]
        if item["rule_reference"] == "shoulder_rotation"
    )

    assert model.criterion_metric_version == "serve_expert_distribution_v6"
    assert grade["score_method"] == "expert_only_identity_distribution_v6"
    assert grade["total_score"] == pytest.approx(expected["total_score"])
    assert wrist["selected_expert_evidence"] == (
        "expert_only_identity_distribution"
    )
    assert wrist["generated_target_distance"] != pytest.approx(
        expected["criteria"][4]["generated_target_distance"]
    )
    for item in (hip, shoulder):
        assert 0.0 <= item["serve_motion_completeness_gate"] <= 1.0
        assert item["selected_camera_evidence_ratio"] == pytest.approx(
            item["raw_checkpoint_ratio"]
        )
        assert item["selected_camera_evidence_ratio"] >= item[
            "strict_required_cue_ratio"
        ]

    assert sum(item["score"] for item in grade["criteria"]) == pytest.approx(
        grade["total_score"]
    )
    assert [item["maximum"] for item in grade["criteria"]] == [
        5.0,
        5.0,
        30.0,
        10.0,
        30.0,
        20.0,
    ]
    assert sum(
        item["checklist_score_contribution"] for item in grade["criteria"]
    ) == pytest.approx(grade["checklist_total_score"])
    assert all(
        item["checklist_maximum"] == pytest.approx(100.0 / 6.0)
        for item in grade["criteria"]
    )

    full_credit = {
        **expected,
        "criteria": [
            {**item, "score": item["maximum"]}
            for item in expected["criteria"]
        ],
    }
    preserved = apply_score_conditioned_correction(
        correction,
        full_credit,
        get_skill_spec("serve"),
        canonical_phase_indices=PHASES,
    )
    np.testing.assert_allclose(
        preserved.corrected_pose, correction.student.pose, atol=0.0
    )
    np.testing.assert_allclose(
        preserved.corrected_root, correction.student.root, atol=0.0
    )

    projected_width_arguments: list[tuple[tuple[int, int], ...]] = []
    original_projector = expert_motion_backend.project_stable_bone_lengths

    def record_projection_widths(*args: object, **kwargs: object) -> np.ndarray:
        projected_width_arguments.append(
            kwargs.get("expert_length_bones", ())  # type: ignore[arg-type]
        )
        return original_projector(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(
        expert_motion_backend,
        "project_stable_bone_lengths",
        record_projection_widths,
    )
    deficient = {
        **expected,
        "criteria": [
            {**item, "score": 0.0, "raw_checkpoint_ratio": 0.0}
            for item in expected["criteria"]
        ],
    }
    apply_score_conditioned_correction(
        correction,
        deficient,
        get_skill_spec("serve"),
        canonical_phase_indices=PHASES,
    )
    assert projected_width_arguments == [
        expert_motion_backend.TORSO_WIDTH_BONES,
        expert_motion_backend.TORSO_WIDTH_BONES,
    ]


def test_serve_completion_score_is_not_controlled_by_one_endpoint_frame() -> None:
    target = _pose()
    confidence = np.ones((64, 17), dtype=np.float32)
    root = np.zeros((64, 2), dtype=np.float32)
    one_frame_error = target.copy()
    one_frame_error[-1, (6, 8, 10), 0] += 1.0
    sustained_error = target.copy()
    sustained_error[-8:, (6, 8, 10), 0] += 1.0
    spec = get_skill_spec("serve")

    one_frame = _criterion_components_for_spec(
        spec,
        one_frame_error,
        root,
        target,
        root,
        confidence,
    )[-1]
    sustained = _criterion_components_for_spec(
        spec,
        sustained_error,
        root,
        target,
        root,
        confidence,
    )[-1]

    assert one_frame["completion_start_fraction"] == pytest.approx(0.875)
    assert one_frame["combined_distance"] < sustained["combined_distance"] * 0.35


def test_rootless_serve_bank_is_rejected_for_full_body_correction(
    tmp_path: Path,
) -> None:
    expert_dir = tmp_path / "experts"
    expert_dir.mkdir()
    for index, offset in enumerate((-0.05, 0.0, 0.05)):
        _write_archive(
            expert_dir / f"expert-{index}.npz",
            schema="legacy",
            skill="serve",
            offset=offset,
            subject_id=f"coach-{index}",
        )
    with pytest.raises(ValueError, match="global root motion"):
        train_expert_phase_model(
            discover_motion_samples(expert_dir, expected_skill="serve"),
            skill="serve",
            top_k=1,
        )


def test_contact_retarget_preserves_expert_support_foot_world_path() -> None:
    reference_pose = _pose()
    pose = reference_pose.copy()
    pose[:, 15, 0] -= 0.2
    root = np.zeros((64, 2), dtype=np.float32)
    root[:, 0] = np.linspace(0.0, 0.6, 64)
    contacts = np.zeros((64, 2), dtype=np.float32)
    contacts[8:32, 0] = 1.0

    corrected_root = _retarget_root_with_contacts(
        pose,
        root,
        contacts,
        reference_pose,
    )
    corrected_world_ankle = pose[:, 15] + corrected_root
    reference_world_ankle = reference_pose[:, 15] + root

    np.testing.assert_allclose(
        corrected_world_ankle[8:32],
        reference_world_ankle[8:32],
        atol=1e-6,
    )
