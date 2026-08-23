from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from badminton_analysis.ml.expert_phase_baseline import (
    _retarget_root_with_contacts,
    _aligned,
    _serve_semantic_evidence,
    _serve_hip_rotation_components,
    _serve_weight_transfer_components,
    _serve_wrist_action_components,
    correct_student_motion,
    discover_motion_samples,
    load_expert_phase_model,
    load_motion_sample,
    save_expert_phase_model,
    score_expert_correction,
    train_expert_phase_model,
)


PHASES = np.asarray((0, 16, 32, 48, 63), dtype=np.int64)


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
    assert missing["combined_distance"] > 0.25
    assert missing["source_weight_transfer"] < missing["target_weight_transfer"]
    assert camera_invariant["combined_distance"] == pytest.approx(
        missing["combined_distance"]
    )


def test_serve_hip_rotation_uses_projected_pelvis_and_torso_change() -> None:
    source = _pose()
    target = source.copy()
    progress = np.clip((np.arange(64, dtype=np.float32) - 36.0) / 20.0, 0.0, 1.0)
    for frame, amount in enumerate(progress):
        hip_centre = 0.5 * (target[frame, 11] + target[frame, 12])
        shoulder_centre = 0.5 * (target[frame, 5] + target[frame, 6])
        target[frame, (11, 12)] = hip_centre + (
            target[frame, (11, 12)] - hip_centre
        ) * (1.0 - 0.65 * amount)
        target[frame, (5, 6)] = shoulder_centre + (
            target[frame, (5, 6)] - shoulder_centre
        ) * (1.0 - 0.35 * amount)
        target[frame, 6, 1] += 0.30 * amount
    confidence = np.ones((64, 17), dtype=np.float32)

    identical = _serve_hip_rotation_components(target, target, confidence)
    missing = _serve_hip_rotation_components(source, target, confidence)

    assert identical["combined_distance"] == pytest.approx(0.0)
    assert missing["combined_distance"] > 0.2
    assert missing["source_projected_hip_contraction"] > missing[
        "target_projected_hip_contraction"
    ]


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
    assert event_evidence[1] > late_evidence[1]
    assert event_evidence[2] > late_evidence[2]


def test_serve_envelope_scores_in_its_own_canonical_time_basis(
    tmp_path: Path,
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
    grade = score_expert_correction(model, distorted)
    aligned_pose, aligned_confidence, aligned_root = _aligned(student)
    expected = _serve_semantic_evidence(
        "wrist_flick", aligned_pose, aligned_root, aligned_confidence
    )[0]
    wrist = next(
        item for item in grade["criteria"] if item["rule_reference"] == "wrist_flick"
    )

    assert model.criterion_metric_version == (
        "serve_subject_balanced_expert_envelope_v3"
    )
    assert wrist["source_wrist_event_speed_mean"] == pytest.approx(expected[0])


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
