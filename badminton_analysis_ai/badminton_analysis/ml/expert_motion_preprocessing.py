from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
from numpy.typing import NDArray

from badminton_analysis.ml.expert_phase_baseline import MotionSample
from badminton_analysis.ml.skeleton_normalization import (
    estimate_foot_contacts,
    interpolate_pose_sequence,
    tracking_body_arrays,
    normalize_skeleton_motion,
    refine_delayed_overhead_contact_phase_indices,
    resample_detected_phase_indices,
    resample_sequence,
)
from badminton_analysis.ml.skeleton_scoring import (
    TORSO_WIDTH_BONES,
    project_stable_bone_lengths,
)
from badminton_analysis.models.types import Handedness, Skill, TrackingData
from badminton_analysis.services.video_analyzer import VideoAnalyzer


def _serve_hip_minimum_start(
    skeleton_2d: NDArray[np.floating],
    *,
    detected_start: int,
    acceleration: int,
    handedness: Handedness = Handedness.RIGHT,
    latest_start: int | None = None,
) -> int:
    """Anchor serve start at minimum canonical pelvis x before acceleration.

    Left-handed/mirrored clips are canonicalized by reflecting image x later
    in preprocessing.  Apply that same reflection while selecting the start;
    otherwise ``argmin`` chooses the opposite end of a left-handed motion.
    """
    coordinates = np.asarray(skeleton_2d, dtype=np.float64)
    if coordinates.ndim != 3 or coordinates.shape[1:] != (17, 2):
        raise ValueError("skeleton_2d must have shape (T, 17, 2)")
    # A raw global minimum can occur during the swing itself (for example
    # because the pelvis continues travelling after wrist acceleration). Such
    # a start collapses preparation to one or two frames and then stretches
    # them across the normalized sequence. Reserve at least the final quarter
    # of the detected start-to-acceleration interval for preparation motion,
    # while retaining a four-frame floor for short clips.
    detected_span = acceleration - detected_start
    required_preparation = max(4, int(np.ceil(0.25 * detected_span)))
    percentage_cutoff = acceleration - required_preparation
    search_cutoff = (
        percentage_cutoff
        if latest_start is None
        else min(percentage_cutoff, latest_start)
    )
    search_end = min(len(coordinates), search_cutoff + 1)
    if not 0 <= detected_start < search_end:
        return detected_start

    hips = coordinates[:, (11, 12), 0]
    valid = np.isfinite(hips)
    counts = valid.sum(axis=1)
    pelvis_x = np.divide(
        np.where(valid, hips, 0.0).sum(axis=1),
        counts,
        out=np.full(len(coordinates), np.nan, dtype=np.float64),
        where=counts > 0,
    )
    observed = np.flatnonzero(np.isfinite(pelvis_x))
    if not len(observed):
        return detected_start
    pelvis_x = np.interp(np.arange(len(pelvis_x)), observed, pelvis_x[observed])
    kernel = np.ones(5, dtype=np.float64) / 5.0
    smoothed = np.convolve(np.pad(pelvis_x, (2, 2), mode="edge"), kernel, mode="valid")
    canonical_x = -smoothed if handedness == Handedness.LEFT else smoothed
    return detected_start + int(
        np.argmin(canonical_x[detected_start:search_end])
    )


def _serve_motion_onset_interval(
    skeleton_2d: NDArray[np.floating],
    *,
    detected_start: int,
    acceleration: int,
    handedness: Handedness = Handedness.RIGHT,
) -> tuple[int, int]:
    """Find stable preparation immediately preceding the main wrist episode.

    The legacy serve parser used a fixed 30-frame pre-impact window. For a
    slowly prepared swing that can begin after hand raising and weight transfer
    have already started. Locate the coherent pre-acceleration episode in
    percentage space, scan backward to its last stable interval, and retain a
    small proportional preparation context before that onset.
    """
    coordinates = np.asarray(skeleton_2d, dtype=np.float64)
    if coordinates.ndim != 3 or coordinates.shape[1:] != (17, 2):
        raise ValueError("skeleton_2d must have shape (T, 17, 2)")
    if not 2 <= acceleration < len(coordinates):
        return detected_start, detected_start
    elbow, wrist = (
        (7, 9) if handedness == Handedness.LEFT else (8, 10)
    )
    relative_wrist = coordinates[:, wrist] - coordinates[:, elbow]
    trajectory = np.column_stack(
        [
            np.convolve(
                np.pad(relative_wrist[:, axis], (2, 2), mode="edge"),
                np.ones(5, dtype=np.float64) / 5.0,
                mode="valid",
            )
            for axis in range(2)
        ]
    )
    speed = np.linalg.norm(np.diff(trajectory, axis=0), axis=-1)
    pre_acceleration = speed[:acceleration]
    if not len(pre_acceleration) or not np.any(np.isfinite(pre_acceleration)):
        return detected_start, detected_start
    finite = np.where(
        np.isfinite(pre_acceleration), pre_acceleration, 0.0
    )
    kernel_width = min(7, len(finite))
    if kernel_width % 2 == 0:
        kernel_width -= 1
    kernel_width = max(kernel_width, 1)
    kernel = np.ones(kernel_width, dtype=np.float64) / kernel_width
    padding = kernel_width // 2
    coherent_speed = np.convolve(
        np.pad(finite, (padding, padding), mode="edge"),
        kernel,
        mode="valid",
    )
    episode = int(np.argmax(coherent_speed))
    peak_speed = float(coherent_speed[episode])
    lower_half = finite[finite <= np.quantile(finite, 0.5)]
    baseline = float(np.median(lower_half)) if len(lower_half) else 0.0
    noise_scale = (
        1.4826 * float(np.median(np.abs(lower_half - baseline)))
        if len(lower_half)
        else 0.0
    )
    threshold = max(0.15 * peak_speed, baseline + 3.0 * noise_scale, 1e-6)
    stable_frames = max(3, min(7, int(np.ceil(0.05 * acceleration))))
    onset = 0
    for end in range(episode, stable_frames - 1, -1):
        stable = coherent_speed[end - stable_frames : end]
        if float(np.mean(stable < threshold)) >= 0.8:
            onset = end
            break
    context = max(3, int(np.ceil(0.10 * max(acceleration - onset, 1))))
    search_start = min(detected_start, max(0, onset - context))
    return int(search_start), int(max(search_start, onset))


def _serve_preparation_was_truncated(
    *,
    detected_start: int,
    detected_peak: int,
    raw_onset_start: int,
    interpolated_onset_start: int,
) -> bool:
    """Require both observed and gap-filled evidence of a clipped start."""
    required_interpolated_extension = max(
        8,
        int(
            np.ceil(
                0.25 * max(detected_peak - interpolated_onset_start, 1)
            )
        ),
    )
    return (
        detected_start - raw_onset_start >= 4
        and detected_start - interpolated_onset_start
        > required_interpolated_extension
    )


def _serve_shoulder_completion_phases(
    detected_phases: Sequence[int],
    skeleton_2d: NDArray[np.floating],
    handedness: Handedness,
    *,
    motion_skeleton_2d: NDArray[np.floating] | None = None,
) -> tuple[int, int, int, int, int]:
    """End serve at maximum shoulder angle after maximum acceleration.

    Raw detections retain the established acceleration and shoulder-angle
    semantics.  An interpolated copy may be supplied only for recovering a
    coherent preparation onset when the legacy fixed window starts too late.
    This prevents short detector gaps from hiding preparation without moving
    the original contact/completion landmarks on otherwise complete clips.
    """
    phases = tuple(int(value) for value in detected_phases)
    if len(phases) != 5 or any(b <= a for a, b in zip(phases, phases[1:])):
        raise ValueError("serve phases must contain five increasing frames")
    coordinates = np.asarray(skeleton_2d, dtype=np.float64)
    if coordinates.ndim != 3 or coordinates.shape[1:] != (17, 2):
        raise ValueError("skeleton_2d must have shape (T, 17, 2)")
    motion_coordinates = (
        coordinates
        if motion_skeleton_2d is None
        else np.asarray(motion_skeleton_2d, dtype=np.float64)
    )
    if motion_coordinates.shape != coordinates.shape:
        raise ValueError("motion_skeleton_2d must match skeleton_2d")
    (
        start,
        detected_preparation,
        detected_peak,
        detected_follow_through,
        detected_end,
    ) = phases
    if handedness == Handedness.LEFT:
        hip, shoulder, elbow, wrist, opposite_shoulder = 11, 5, 7, 9, 6
    else:
        hip, shoulder, elbow, wrist, opposite_shoulder = 12, 6, 8, 10, 5

    kernel = np.ones(5, dtype=np.float64) / 5.0
    motion_relative_wrist = (
        motion_coordinates[:, wrist] - motion_coordinates[:, shoulder]
    )
    motion_smoothed_wrist = np.column_stack(
        [
            np.convolve(
                np.pad(motion_relative_wrist[:, axis], (2, 2), mode="edge"),
                kernel,
                mode="valid",
            )
            for axis in range(2)
        ]
    )
    forward_axes = (
        motion_coordinates[:, opposite_shoulder]
        - motion_coordinates[:, shoulder]
    )
    valid_axes = np.all(np.isfinite(forward_axes), axis=1)
    forward_axis = (
        np.median(forward_axes[valid_axes], axis=0)
        if np.any(valid_axes)
        else np.asarray((1.0, 0.0), dtype=np.float64)
    )
    if float(np.linalg.norm(forward_axis)) <= 1e-8:
        forward_axis = np.asarray((1.0, 0.0), dtype=np.float64)

    # Search the complete observed clip. The legacy acceleration window can
    # end during the backswing; using it as a hard upper bound excluded the
    # actual across-body forward swing (CG43 ended at 96 while the forward
    # acceleration occurred at 101). The body-derived axis makes the later
    # recovery in the opposite direction ineligible.
    # The legacy preparation anchor can itself occur after the true forward
    # event (CG07: legacy preparation 82, forward acceleration 77). Start from
    # the detected analysis onset; the signed across-body projection already
    # rejects the opposite backswing.
    directional_search_start = start + 2
    directional_search_stop = len(motion_smoothed_wrist) - 2
    if directional_search_stop <= directional_search_start:
        raise ValueError("serve directional acceleration range is too short")
    provisional_acceleration = directional_search_start + int(
        VideoAnalyzer._directional_acceleration_peak(
            motion_smoothed_wrist[
                directional_search_start : directional_search_stop + 1
            ],
            forward_axis=forward_axis,
        )
    )
    provisional_acceleration = int(
        np.clip(
            provisional_acceleration,
            directional_search_start,
            directional_search_stop,
        )
    )
    raw_onset_start, _ = _serve_motion_onset_interval(
        coordinates,
        detected_start=start,
        acceleration=provisional_acceleration,
        handedness=handedness,
    )
    onset_start, onset_end = _serve_motion_onset_interval(
        motion_coordinates,
        detected_start=start,
        acceleration=provisional_acceleration,
        handedness=handedness,
    )
    # Never fall back to acceleration magnitude here: a fast backswing can
    # have the largest magnitude while pointing away from the demonstrated
    # across-body serve direction.
    acceleration = provisional_acceleration
    onset_start, onset_end = _serve_motion_onset_interval(
        motion_coordinates,
        detected_start=start,
        acceleration=acceleration,
        handedness=handedness,
    )
    start = _serve_hip_minimum_start(
        coordinates,
        detected_start=onset_start,
        acceleration=acceleration,
        handedness=handedness,
        latest_start=onset_end,
    )

    incoming = coordinates[:, hip] - coordinates[:, shoulder]
    outgoing = coordinates[:, elbow] - coordinates[:, shoulder]
    denominator = np.linalg.norm(incoming, axis=-1) * np.linalg.norm(
        outgoing, axis=-1
    )
    cosine = np.divide(
        np.sum(incoming * outgoing, axis=-1),
        denominator,
        out=np.ones_like(denominator),
        where=denominator > 1e-8,
    )
    shoulder_angle = np.arccos(np.clip(cosine, -1.0, 1.0))
    shoulder_angle = np.convolve(
        np.pad(shoulder_angle, (2, 2), mode="edge"),
        kernel,
        mode="valid",
    )
    completion_start = acceleration + 2
    completion_end = len(coordinates) - 1
    if completion_start > completion_end:
        raise ValueError("serve shoulder-completion range is too short")
    completion = completion_start + int(
        np.nanargmax(shoulder_angle[completion_start : completion_end + 1])
    )
    preparation = start + max(
        1, int(round(0.45 * (acceleration - start)))
    )
    preparation = min(preparation, acceleration - 1)
    follow_through = (acceleration + completion) // 2
    return start, preparation, acceleration, follow_through, completion


def prepare_expert_motion_sample(
    tracking: TrackingData,
    handedness: Handedness,
    skill: Skill,
    filename: str,
    *,
    target_frames: int = 64,
) -> tuple[MotionSample, tuple[int, int, int], NDArray[np.int64]]:
    """Apply the same 2D extraction contract used by the frozen generator."""
    if skill not in {Skill.SERVE, Skill.SMASH}:
        raise ValueError("expert-motion generation currently supports serve and smash")
    body_2d = tracking.get("body_landmarks_2d")
    if not body_2d or len(body_2d) < 5:
        raise ValueError("at least five aligned 2D poses are required")
    full_skeleton, full_confidence = tracking_body_arrays(tracking)
    motion_skeleton, _ = interpolate_pose_sequence(
        full_skeleton, full_confidence
    )
    phases = VideoAnalyzer.find_analysis_phases(
        skill=skill,
        hand_positions=tracking.get("hand_positions"),
        elbow_positions=tracking.get("elbow_positions"),
    )
    phase_source = "acceleration_wrist_velocity_stop_v6"
    if skill == Skill.SERVE:
        phases = _serve_shoulder_completion_phases(
            phases,
            full_skeleton,
            handedness,
            motion_skeleton_2d=motion_skeleton,
        )
        phase_source = "across_body_directional_wrist_acceleration_v14"
    if any(second <= first for first, second in zip(phases, phases[1:])):
        raise ValueError("analysis phases must be strictly increasing")
    start, peak, end = int(phases[0]), int(phases[2]), int(phases[-1])
    if start < 0 or end >= len(full_skeleton) or end - start < 4:
        raise ValueError(f"invalid analysis window: {(start, peak, end)}")

    normalized = normalize_skeleton_motion(
        full_skeleton[start : end + 1],
        full_confidence[start : end + 1],
        handedness,
    )
    pose = resample_sequence(normalized.skeleton, target_frames)
    confidence = np.clip(
        resample_sequence(normalized.confidence, target_frames), 0.0, 1.0
    )
    pose = project_stable_bone_lengths(
        pose,
        pose,
        confidence,
        expert_length_bones=TORSO_WIDTH_BONES,
    )
    root = resample_sequence(normalized.root_trajectory, target_frames)
    contacts = estimate_foot_contacts(pose, root, confidence)
    phase_indices = resample_detected_phase_indices(phases, target_frames)
    if skill == Skill.SMASH:
        refined = refine_delayed_overhead_contact_phase_indices(
            pose, phase_indices
        )
        if not np.array_equal(refined, phase_indices):
            phase_indices = refined
            phase_source = "acceleration_wrist_velocity_stop_delayed_contact_v7"

    source_indices = np.rint(
        np.linspace(start, end, target_frames)
    ).astype(np.int64)
    sample = MotionSample(
        path=Path(filename),
        pose=pose.astype(np.float32),
        confidence=confidence.astype(np.float32),
        root=root.astype(np.float32),
        foot_contacts=contacts.astype(np.float32),
        phase_indices=phase_indices,
        handedness=str(handedness),
        skill=str(skill),
        video_name=filename,
        subject_id="inference",
        phase_source=phase_source,
        alignment_contract=(
            "serve_across_body_directional_wrist_acceleration_v14"
            if skill == Skill.SERVE
            else "overhead_wrist_velocity_stop_v6"
        ),
        identity_level="inference_only",
    )
    return sample, (start, peak, end), source_indices
