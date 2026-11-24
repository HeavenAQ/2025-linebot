import json

import numpy as np
import pytest

from badminton_analysis.ml.smash.checkpoints import excess_rotation_evidence


def observed_pose():
    pose = np.zeros((60, 17, 2))
    pose[:, 5] = [-0.3, -1]
    pose[:, 6] = [0.3, -1]
    pose[:, 11] = [-0.2, 0]
    pose[:, 12] = [0.2, 0]
    pose[:, 15] = [-0.4, 1.5]
    pose[:, 16] = [0.4, 1.5]
    return pose, np.ones((len(pose), 17))


def reverse(pose, first, last, pairs=((5, 6), (15, 16))):
    for left, right in pairs:
        original = pose[first - 5, [left, right]].copy()
        for frame in range(first - 4, first):
            fraction = (frame - first + 5) / 5
            pose[frame, [left, right]] = (
                1 - fraction
            ) * original + fraction * original[::-1]
        pose[first : last + 1, [left, right]] = pose[first : last + 1, [right, left]]


def measure(pose, confidence):
    return excess_rotation_evidence(pose, confidence, 0, 25, 30)


def test_possible_excess_turn_reports_exact_observed_phase_and_frames():
    pose, confidence = observed_pose()
    reverse(pose, 36, 45)
    result = measure(pose, confidence)
    assert result["setup"]["status"] == "clear"
    finish = result["finish"]
    assert finish["status"] == "flagged"
    assert finish["source_interval"] == [30, 44]
    assert finish["baseline_source_interval"] == [18, 29]
    assert finish["event_source_interval"] == [35, 37]
    assert finish["event_reversal_spans"] == pytest.approx([0.6, 0.8])
    assert finish["minimum_event_confidence"] == 1
    json.dumps(result, allow_nan=False)


def test_setup_reports_precontact_turn_independently():
    pose, confidence = observed_pose()
    reverse(pose, 15, 18)
    result = measure(pose, confidence)
    assert result["setup"]["status"] == "flagged"
    assert result["setup"]["event_source_interval"] == [14, 16]
    assert result["finish"]["status"] == "clear"


@pytest.mark.parametrize("pairs", [((5, 6),), ((15, 16),)])
def test_a_step_or_shoulder_turn_alone_is_clear(pairs):
    pose, confidence = observed_pose()
    reverse(pose, 36, 45, pairs)
    assert measure(pose, confidence)["finish"]["status"] == "clear"


@pytest.mark.parametrize("length", [1, 2])
def test_transient_reversal_does_not_become_sustained(length):
    pose, confidence = observed_pose()
    for left, right in ((5, 6), (15, 16)):
        pose[36 : 36 + length, [left, right]] = pose[36 : 36 + length, [right, left]]
    assert measure(pose, confidence)["finish"]["status"] == "clear"


def test_late_recovery_turn_is_outside_the_half_second_window():
    pose, confidence = observed_pose()
    reverse(pose, 49, 59)
    assert measure(pose, confidence)["finish"]["status"] == "clear"


@pytest.mark.parametrize("transform", ["mirror", "roll", "scale", "translation"])
def test_review_follows_anatomical_baseline_in_the_same_camera_canvas(transform):
    pose, confidence = observed_pose()
    reverse(pose, 36, 45)
    if transform == "mirror":
        pose[..., 0] *= -1
    elif transform == "roll":
        angle = np.deg2rad(35)
        matrix = np.array(
            [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
        )
        pose = pose @ matrix.T
    elif transform == "scale":
        pose *= 100
    else:
        pose += [410, 275]
    result = measure(pose, confidence)
    assert result["finish"]["status"] == "flagged"
    assert result["finish"]["event_reversal_spans"] == pytest.approx([0.6, 0.8])


@pytest.mark.parametrize("region", ["baseline", "finish"])
def test_unreliable_observations_cannot_supply_a_review_event(region):
    pose, confidence = observed_pose()
    reverse(pose, 36, 45)
    first, last = (18, 29) if region == "baseline" else (30, 44)
    confidence[first : last + 1, 16] = 0.1
    result = measure(pose, confidence)["finish"]
    assert result["status"] == "unassessable"
    assert "event_source_interval" not in result


def test_unstable_or_foreshortened_baseline_cannot_establish_a_direction():
    pose, confidence = observed_pose()
    for left, right in ((5, 6), (15, 16)):
        pose[18:24, [left, right]] = pose[18:24, [right, left]]
    result = measure(pose, confidence)["finish"]
    assert result["status"] == "unassessable"


def test_missing_frame_breaks_the_sustained_event():
    pose, confidence = observed_pose()
    reverse(pose, 36, 37)
    pose[36, 5] = np.nan
    result = measure(pose, confidence)
    assert result["finish"]["status"] == "clear"
    json.dumps(result, allow_nan=False)


def test_padding_and_repeated_neutral_frames_do_not_create_an_event():
    pose, confidence = observed_pose()
    pose[30:] = pose[29]
    assert measure(pose, confidence)["finish"]["status"] == "clear"
    pose[35:] = 0
    confidence[35:] = 0
    result = measure(pose, confidence)["finish"]
    assert result["status"] == "unassessable"
    assert "event_source_interval" not in result


def test_truncated_evidence_only_flags_an_event_that_was_fully_observed():
    pose, confidence = observed_pose()
    reverse(pose, 36, 45)
    assert measure(pose[:38], confidence[:38])["finish"]["status"] == "flagged"
    assert measure(pose[:37], confidence[:37])["finish"]["status"] == "unassessable"


def test_instantaneous_bilateral_tracking_swap_is_unassessable():
    pose, confidence = observed_pose()
    for left, right in ((5, 6), (15, 16)):
        pose[36:, [left, right]] = pose[36:, [right, left]]
    assert measure(pose, confidence)["finish"]["status"] == "unassessable"


def test_long_tracking_gap_across_the_turn_is_unassessable():
    pose, confidence = observed_pose()
    reverse(pose, 36, 45)
    confidence[31:35] = 0.1
    assert measure(pose, confidence)["finish"]["status"] == "unassessable"


def test_two_missing_frames_can_be_bracketed_by_a_smooth_observed_turn():
    pose, confidence = observed_pose()
    reverse(pose, 36, 45)
    confidence[32:34] = 0.1
    assert measure(pose, confidence)["finish"]["status"] == "flagged"


def test_short_phases_do_not_report_out_of_bounds_evidence():
    pose, confidence = observed_pose()
    result = excess_rotation_evidence(pose[:10], confidence[:10], 0, 5, 8)
    for phase in ("setup", "finish"):
        assert result[phase]["status"] == "unassessable"
        if "source_interval" in result[phase]:
            first, last = result[phase]["source_interval"]
            assert 0 <= first <= last < 10


def test_anatomical_canonicalization_preserves_the_same_physical_review():
    from badminton_analysis.ml.smash.scorer import canonical_source

    pose, confidence = observed_pose()
    reverse(pose, 36, 45)
    original = pose.copy()
    mirrored, mirrored_confidence = canonical_source(pose, confidence, "left")
    result = measure(mirrored, mirrored_confidence)
    assert result["finish"]["status"] == "flagged"
    assert result["finish"]["event_source_interval"] == [35, 37]
    np.testing.assert_array_equal(pose, original)


def test_transition_starts_at_the_last_reliable_baseline_observation():
    pose, confidence = observed_pose()
    reverse(pose, 36, 45)
    confidence[28:33] = 0.1
    result = measure(pose, confidence)["finish"]
    assert result["status"] == "unassessable"
    assert "event_source_interval" not in result


@pytest.mark.parametrize("phase", ["setup", "finish"])
def test_three_missing_observations_can_hide_a_complete_event(phase):
    pose, confidence = observed_pose()
    if phase == "setup":
        pose = np.repeat(pose[:1], 120, axis=0)
        confidence = np.ones((len(pose), 17))
        confidence[40:43] = 0.1
        result = excess_rotation_evidence(pose, confidence, 0, 80, 90)
    else:
        confidence[35:38] = 0.1
        result = measure(pose, confidence)
    assert result[phase]["status"] == "unassessable"
    assert "event_source_interval" not in result[phase]
