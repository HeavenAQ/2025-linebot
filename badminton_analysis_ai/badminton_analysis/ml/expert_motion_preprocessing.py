from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
from numpy.typing import NDArray

from badminton_analysis.ml.expert_phase_baseline import MotionSample
from badminton_analysis.ml.skeleton_normalization import (
    estimate_foot_contacts,
    landmark_dicts_to_array,
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


def _serve_shoulder_completion_phases(
    detected_phases: Sequence[int],
    skeleton_2d: NDArray[np.floating],
    handedness: Handedness,
) -> tuple[int, int, int, int, int]:
    """End serve at maximum shoulder angle after maximum acceleration."""
    phases = tuple(int(value) for value in detected_phases)
    if len(phases) != 5 or any(b <= a for a, b in zip(phases, phases[1:])):
        raise ValueError("serve phases must contain five increasing frames")
    coordinates = np.asarray(skeleton_2d, dtype=np.float64)
    if coordinates.ndim != 3 or coordinates.shape[1:] != (17, 2):
        raise ValueError("skeleton_2d must have shape (T, 17, 2)")
    start, preparation, _, detected_follow_through, detected_end = phases
    if handedness == Handedness.LEFT:
        hip, shoulder, elbow, wrist = 11, 5, 7, 9
    else:
        hip, shoulder, elbow, wrist = 12, 6, 8, 10

    kernel = np.ones(5, dtype=np.float64) / 5.0
    relative_wrist = coordinates[:, wrist] - coordinates[:, shoulder]
    smoothed_wrist = np.column_stack(
        [
            np.convolve(
                np.pad(relative_wrist[:, axis], (2, 2), mode="edge"),
                kernel,
                mode="valid",
            )
            for axis in range(2)
        ]
    )
    acceleration_magnitude = np.linalg.norm(
        np.diff(smoothed_wrist, n=2, axis=0), axis=-1
    )
    acceleration_start = max(start + 2, preparation)
    acceleration_stop = min(detected_follow_through, detected_end - 2)
    if acceleration_stop < acceleration_start:
        raise ValueError("serve acceleration range is too short")
    acceleration = acceleration_start + int(
        np.nanargmax(
            acceleration_magnitude[acceleration_start - 1 : acceleration_stop]
        )
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
    if completion_start > detected_end:
        raise ValueError("serve shoulder-completion range is too short")
    completion = completion_start + int(
        np.nanargmax(shoulder_angle[completion_start : detected_end + 1])
    )
    preparation = max(start + 1, min(preparation, acceleration - 1))
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
    full_skeleton, full_confidence = landmark_dicts_to_array(body_2d, 2)
    phases = VideoAnalyzer.find_analysis_phases(
        skill=skill,
        hand_positions=tracking.get("hand_positions"),
        elbow_positions=tracking.get("elbow_positions"),
    )
    phase_source = "acceleration_ending_range_v4"
    if skill == Skill.SERVE:
        phases = _serve_shoulder_completion_phases(
            phases, full_skeleton, handedness
        )
        phase_source = "max_acceleration_shoulder_angle_v1"
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
            phase_source = "acceleration_ending_range_delayed_contact_v5"

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
            "serve_max_acceleration_shoulder_angle_v1"
            if skill == Skill.SERVE
            else "overhead_asymmetric_ending_range_v4"
        ),
        identity_level="inference_only",
    )
    return sample, (start, peak, end), source_indices
