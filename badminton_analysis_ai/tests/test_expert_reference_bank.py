from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from badminton_analysis.ml.expert_reference_bank import (
    ExpertReferenceBank,
    skill_temporal_descriptor,
)

BANK = Path("models/expert_reference_bank.npz")
pytestmark = pytest.mark.skipif(not BANK.exists(), reason="expert bank is not built")


@pytest.fixture(scope="module")
def bank() -> ExpertReferenceBank:
    return ExpertReferenceBank(BANK)


def test_bank_holds_the_clips_the_checkpoints_were_trained_on(bank: ExpertReferenceBank) -> None:
    assert len(bank) == 26
    assert int((bank.skill == "serve").sum()) == 14
    assert int((bank.skill == "smash").sum()) == 12
    # Withdrawn skills have no diffusion checkpoint, so no experts to show.
    assert set(np.unique(bank.skill)) == {"serve", "smash"}


def test_an_expert_is_its_own_closest_match(bank: ExpertReferenceBank) -> None:
    for index in (0, len(bank) // 2, len(bank) - 1):
        reference = bank.select(
            bank.skeletons[index],
            skill=str(bank.skill[index]),
            handedness=str(bank.handedness[index]),
        )
        assert reference is not None
        assert reference.subject_id == str(bank.subject_id[index])
        assert reference.similarity == pytest.approx(1.0, abs=1e-5)
        assert reference.distance == pytest.approx(0.0, abs=1e-4)


def test_both_metrics_agree_on_an_exact_match(bank: ExpertReferenceBank) -> None:
    pose = bank.skeletons[3]
    cosine = bank.select(pose, skill="serve", handedness=str(bank.handedness[3]), metric="cosine")
    euclidean = bank.select(pose, skill="serve", handedness=str(bank.handedness[3]), metric="euclidean")
    assert cosine is not None and euclidean is not None
    assert cosine.subject_id == euclidean.subject_id


# A learner is never shown a demonstration of a different stroke.
def test_selection_never_crosses_skills(bank: ExpertReferenceBank) -> None:
    serve_pose = bank.skeletons[int(np.flatnonzero(bank.skill == "serve")[0])]
    reference = bank.select(serve_pose, skill="smash", handedness="right")
    assert reference is not None
    assert reference.skill == "smash"


def test_left_handed_learners_get_a_left_handed_expert_when_one_exists(
    bank: ExpertReferenceBank,
) -> None:
    left = np.flatnonzero((bank.skill == "serve") & (bank.handedness == "left"))
    assert len(left), "serve bank should contain left-handed experts"
    reference = bank.select(bank.skeletons[int(left[0])], skill="serve", handedness="left")
    assert reference is not None
    assert reference.handedness == "left"


# Smash has only right-handed experts; a left-handed learner still gets a
# demonstration rather than an empty panel.
def test_falls_back_across_handedness_rather_than_showing_nothing(
    bank: ExpertReferenceBank,
) -> None:
    smash = int(np.flatnonzero(bank.skill == "smash")[0])
    reference = bank.select(bank.skeletons[smash], skill="smash", handedness="left")
    assert reference is not None
    assert reference.skill == "smash"


def test_unknown_skill_yields_no_reference(bank: ExpertReferenceBank) -> None:
    assert bank.select(bank.skeletons[0], skill="clear", handedness="right") is None


def test_temporal_gate_artifact_is_expert_only_identity_held_out(
    bank: ExpertReferenceBank,
) -> None:
    assert bank.skill_support_fit_policy == "expert_only_leave_one_identity_out"
    assert bank.skill_support_expert_count == 103
    assert not bank.skill_support_student_data_used
    assert len(bank.skill_support_features) == 103
    assert int((bank.skill_support_skill == "serve").sum()) == 53
    assert int((bank.skill_support_skill == "smash").sum()) == 50
    assert bank.skill_rejection_margin > 0.0


def test_temporal_descriptor_is_invariant_to_translation_scale_and_rotation(
    bank: ExpertReferenceBank,
) -> None:
    pose = bank.skeletons[0].astype(np.float64)
    expected = skill_temporal_descriptor(pose)
    transformed = pose * 2.7 + np.asarray((42.0, -17.0))
    angle = np.deg2rad(31.0)
    rotation = np.asarray(
        ((np.cos(angle), -np.sin(angle)), (np.sin(angle), np.cos(angle)))
    )
    transformed = transformed @ rotation.T
    assert skill_temporal_descriptor(transformed) == pytest.approx(
        expected, abs=2e-5
    )


def test_playback_window_comes_from_the_source_video(bank: ExpertReferenceBank) -> None:
    reference = bank.select(bank.skeletons[0], skill="serve", handedness=str(bank.handedness[0]))
    assert reference is not None
    seconds = reference.phase_seconds()
    assert len(seconds) == 5
    assert seconds == tuple(sorted(seconds))
    assert reference.motion_start_seconds == pytest.approx(seconds[0])
    assert reference.motion_end_seconds > seconds[-1]
    assert reference.video_object_path.startswith(f"experts/v3/{reference.skill}/videos/")


def test_a_malformed_pose_is_refused(bank: ExpertReferenceBank) -> None:
    with pytest.raises(ValueError):
        bank.select(np.zeros((64, 17, 3), dtype=np.float32), skill="serve", handedness="right")
