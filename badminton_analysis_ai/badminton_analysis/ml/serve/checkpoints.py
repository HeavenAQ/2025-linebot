"""Serve checkpoint rules: how each checkpoint is measured on the learner's clip."""

from __future__ import annotations

from typing import Any, Sequence

from numpy.typing import NDArray
import numpy as np

from badminton_analysis.ml.motion.view import _EPS
from badminton_analysis.ml.skill_specs import motion_completion_bounds


# Room for an expert the envelope has never seen.
_SERVE_UNSEEN_EXPERT_MARGIN = 0.15


# Experts finish with the hips 10+ degrees over the front foot, well-rated learners 8+; a stacked body line has not transferred.
_SERVE_HIP_LEAN_FULL_DEGREES = 8.0
_SERVE_HIP_LEAN_NONE_DEGREES = 3.0
# Every expert's elbow opens 3+ degrees into the wrist's peak acceleration; an arm that only folds has no flick.
_SERVE_ELBOW_OPENING_FULL_DEGREES = 3.0
_SERVE_ELBOW_OPENING_NONE_DEGREES = 1.0
# After the wrist's peak an expert's shoulder keeps turning while the elbow stays quiet; an elbow outrunning the shoulder is the arm hitting.
_SERVE_ARM_RATIO_FULL = 2.7
_SERVE_ARM_RATIO_NONE = 5.0


def _wrapped_angle(value: float) -> float:
    """Wrap an angle difference to ``[-pi, pi]``."""
    return float((value + np.pi) % (2.0 * np.pi) - np.pi)


def _robust_window_value(
    values: NDArray[np.floating],
    confidence: NDArray[np.floating],
    *,
    start: int,
    end: int,
    minimum_confidence: float = 0.20,
) -> float:
    """Return a confidence-masked median without silently treating misses as zero."""
    selected = np.asarray(values, dtype=np.float64)[start:end]
    observed = np.asarray(confidence, dtype=np.float64)[start:end]
    valid = np.isfinite(selected) & (observed >= minimum_confidence)
    if np.any(valid):
        return float(np.median(selected[valid]))
    finite = selected[np.isfinite(selected)]
    return float(np.median(finite)) if len(finite) else 0.0


def _serve_best_terminal_value(
    values: NDArray[np.floating],
    confidence: NDArray[np.floating],
    *,
    start: int,
    end: int,
) -> float:
    """The best the follow-through gets over the ending frames."""
    window = np.asarray(values, dtype=np.float64)[start:end]
    seen = np.asarray(confidence, dtype=np.float64)[start:end] >= 0.20
    eligible = window[seen & np.isfinite(window)]
    if not eligible.size:
        eligible = window[np.isfinite(window)]
    if not eligible.size:
        return float("nan")
    return float(np.min(eligible))


def _serve_image_hip_lean(
    keypoints: NDArray[np.floating],
    confidence: NDArray[np.floating],
    source_frames: NDArray[np.integer],
    handedness: str,
) -> dict[str, float]:
    """How far the hips lean over the front foot at the finish, in degrees from true image vertical."""
    frames = np.asarray(source_frames, dtype=np.int64)
    values = np.asarray(keypoints, dtype=np.float64)[frames]
    observed = np.asarray(confidence, dtype=np.float64)[frames]
    racket, front = (15, 16) if str(handedness).lower() == "left" else (16, 15)
    preparation_start, preparation_end = motion_completion_bounds(
        len(frames), 0.125, 0.34375
    )
    completion_start, completion_end = motion_completion_bounds(
        len(frames), 0.71875, 1.0
    )
    front_side = float(
        np.sign(
            np.median(
                values[preparation_start:preparation_end, front, 0]
                - values[preparation_start:preparation_end, racket, 0]
            )
        )
    )
    hip_center = 0.5 * (values[:, 11] + values[:, 12])
    ankle_center = 0.5 * (values[:, 15] + values[:, 16])
    # Image y grows downward, so the hips sit above the ankles at positive height.
    lean = np.degrees(
        np.arctan2(
            front_side * (hip_center[:, 0] - ankle_center[:, 0]),
            ankle_center[:, 1] - hip_center[:, 1],
        )
    )
    seen = np.min(observed[:, (11, 12, 15, 16)], axis=1) > 0.2
    finish = np.arange(len(frames))[completion_start:completion_end]
    finish = finish[seen[finish]] if np.any(seen[finish]) else finish
    finish_lean = float(np.median(lean[finish]))
    return {
        "image_hip_lean_degrees": finish_lean,
        "image_hip_lean_full_degrees": _SERVE_HIP_LEAN_FULL_DEGREES,
        "image_hip_lean_factor": float(
            np.clip(
                (finish_lean - _SERVE_HIP_LEAN_NONE_DEGREES)
                / (_SERVE_HIP_LEAN_FULL_DEGREES - _SERVE_HIP_LEAN_NONE_DEGREES),
                0.0,
                1.0,
            )
        ),
    }


def _serve_image_elbow_opening(
    keypoints: NDArray[np.floating],
    window: tuple[int, int, int],
    handedness: str,
    fps: float,
) -> dict[str, float]:
    """The racket elbow around the wrist's peak acceleration: opening into it, and its speed against the shoulder's after it."""
    start, contact, end = (int(v) for v in window)
    hip, shoulder, elbow, wrist = (11, 5, 7, 9) if str(handedness).lower() == "left" else (12, 6, 8, 10)
    values = np.asarray(keypoints, dtype=np.float64)[:, (hip, shoulder, elbow, wrist)]
    padded = np.pad(values, ((2, 2), (0, 0), (0, 0)), mode="edge")
    values = np.mean([padded[i : i + len(values)] for i in range(5)], axis=0)

    def joint_angle(a: int, o: int, b: int) -> NDArray[np.float64]:
        u, v = values[:, a] - values[:, o], values[:, b] - values[:, o]
        cosine = np.sum(u * v, axis=-1) / np.maximum(
            np.linalg.norm(u, axis=-1) * np.linalg.norm(v, axis=-1), _EPS
        )
        return np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))

    elbow_angle, shoulder_angle = joint_angle(1, 2, 3), joint_angle(0, 1, 2)
    acceleration = np.linalg.norm(
        np.diff(values[:, 3] - values[:, 1], n=2, axis=0), axis=-1
    )
    low, high = start + (contact - start) // 2, min(end, contact + (end - contact) // 2)
    high = max(high, low + 1)
    peak = low + int(np.argmax(acceleration[low:high])) + 1
    step = max(1, int(round(fps / 30.0)))
    span = elbow_angle[max(start, peak - 8 * step) : peak + 3 * step + 1]
    opening = float(np.max(span - np.minimum.accumulate(span)))
    after = slice(peak, peak + max(2, int(round(0.2 * fps))) + 1)
    elbow_speed = float(np.median(np.abs(np.diff(elbow_angle[after])))) * fps
    shoulder_speed = float(np.median(np.abs(np.diff(shoulder_angle[after])))) * fps
    # A still shoulder (under 30 deg/s) is not allowed to make any elbow motion look large.
    arm_ratio = elbow_speed / max(shoulder_speed, 30.0)
    opening_factor = np.clip(
        (opening - _SERVE_ELBOW_OPENING_NONE_DEGREES)
        / (_SERVE_ELBOW_OPENING_FULL_DEGREES - _SERVE_ELBOW_OPENING_NONE_DEGREES),
        0.0,
        1.0,
    )
    arm_factor = np.clip(
        (_SERVE_ARM_RATIO_NONE - arm_ratio) / (_SERVE_ARM_RATIO_NONE - _SERVE_ARM_RATIO_FULL),
        0.0,
        1.0,
    )
    return {
        "image_elbow_opening_degrees": opening,
        "image_elbow_opening_full_degrees": _SERVE_ELBOW_OPENING_FULL_DEGREES,
        "image_elbow_speed_after_peak_degrees_per_second": elbow_speed,
        "image_shoulder_speed_after_peak_degrees_per_second": shoulder_speed,
        "image_elbow_to_shoulder_speed_ratio": arm_ratio,
        "image_elbow_to_shoulder_speed_ratio_full": _SERVE_ARM_RATIO_FULL,
        "image_elbow_opening_factor": float(min(opening_factor, arm_factor)),
    }


def _serve_image_elbow_between_shoulders(
    keypoints: NDArray[np.floating],
    confidence: NDArray[np.floating],
    window: tuple[int, int, int],
    handedness: str,
) -> float:
    """The smallest angle the two shoulders subtend at the racket elbow over the finish, on the video's own keypoints."""
    start, _, end = (int(v) for v in window)
    racket, other, elbow = (5, 6, 7) if str(handedness).lower() == "left" else (6, 5, 8)
    values = np.asarray(keypoints, dtype=np.float64)
    observed = np.asarray(confidence, dtype=np.float64)
    end = min(end + 1, len(values))
    finish = slice(start + int(round(0.875 * (end - 1 - start))), end)
    to_racket = values[finish, racket] - values[finish, elbow]
    to_other = values[finish, other] - values[finish, elbow]
    cosine = np.sum(to_racket * to_other, axis=-1) / np.maximum(
        np.linalg.norm(to_racket, axis=-1) * np.linalg.norm(to_other, axis=-1), _EPS
    )
    angle = np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))
    seen = np.min(observed[finish][:, (racket, other, elbow)], axis=1) >= 0.20
    return float(np.min(angle[seen] if np.any(seen) else angle))


def _serve_wrist_descent_onset(pose: NDArray[np.float64], contact: int) -> int:
    """The frame the racket wrist starts coming down, which is where the transfer has to begin."""
    height = _smooth_trajectory(pose[:, 10, 1][:, None])[:, 0]
    onset = int(np.argmax(height[:contact])) if contact > 1 else 0
    return int(min(onset, max(0, contact - 4)))


def _joint_angle_trajectory(
    pose: NDArray[np.floating], first: int, pivot: int, third: int
) -> NDArray[np.float64]:
    values = np.asarray(pose, dtype=np.float64)
    incoming = values[:, first] - values[:, pivot]
    outgoing = values[:, third] - values[:, pivot]
    denominator = np.maximum(
        np.linalg.norm(incoming, axis=-1) * np.linalg.norm(outgoing, axis=-1),
        _EPS,
    )
    cosine = np.sum(incoming * outgoing, axis=-1) / denominator
    return np.arccos(np.clip(cosine, -1.0, 1.0))


def _serve_dominant_chain_angles(
    pose: NDArray[np.floating],
) -> NDArray[np.float64]:
    """Return canonical dominant-side transfer angles in radians."""
    return np.stack(
        (
            _joint_angle_trajectory(pose, 6, 12, 14),
            _joint_angle_trajectory(pose, 12, 14, 16),
            _joint_angle_trajectory(pose, 6, 12, 16),
        ),
        axis=-1,
    )


def _serve_qualitative_pose_evidence(
    pose: NDArray[np.floating],
    root: NDArray[np.floating] | None = None,
) -> dict[str, float]:
    """Return camera-scale-invariant evidence required by serve checkpoints."""
    values = np.asarray(pose, dtype=np.float64)
    preparation_start, preparation_end = motion_completion_bounds(
        len(values), 0.125, 0.34375
    )
    completion_start, completion_end = motion_completion_bounds(
        len(values), 0.71875, 1.0
    )
    # The pelvis has to have arrived by the time the racket accelerates.
    _, contact_frame = motion_completion_bounds(len(values), 0.0, 0.5)
    descent = _serve_wrist_descent_onset(values, contact_frame)
    loading_start, loading_end = min(descent, contact_frame - 1), contact_frame
    baseline_start = max(0, descent - max(2, len(values) // 16))
    baseline_end = max(1, descent)
    hip_center = 0.5 * (values[:, 11] + values[:, 12])
    shoulder_center = 0.5 * (values[:, 5] + values[:, 6])
    torso = np.maximum(np.linalg.norm(shoulder_center - hip_center, axis=-1), _EPS)
    arm_elevation = np.stack(
        [(values[:, joint, 1] - hip_center[:, 1]) / torso for joint in (7, 8, 9, 10)],
        axis=-1,
    )
    # Both elbows and wrists must be raised together.
    simultaneous_elevation = np.minimum(
        np.max(arm_elevation[:, (0, 2)], axis=-1),
        np.max(arm_elevation[:, (1, 3)], axis=-1),
    )
    arms_raised = float(
        np.quantile(simultaneous_elevation[preparation_start:preparation_end], 0.70)
    )
    stance_width = np.linalg.norm(values[:, 16] - values[:, 15], axis=-1) / torso
    preparation_stance = float(
        np.median(stance_width[preparation_start:preparation_end])
    )
    ankle_axis = values[:, 16] - values[:, 15]
    ankle_denominator = np.maximum(np.sum(ankle_axis * ankle_axis, axis=-1), _EPS)
    pelvis_loading = (
        np.sum((hip_center - values[:, 15]) * ankle_axis, axis=-1) / ankle_denominator
    )
    baseline_loading = float(np.median(pelvis_loading[baseline_start:baseline_end]))
    preparation_loading = float(
        np.median(pelvis_loading[preparation_start:preparation_end])
    )
    # The furthest the pelvis gets, not where it sits on one frame.
    loading_shift = float(
        np.max(np.abs(pelvis_loading[loading_start:loading_end] - baseline_loading))
    )
    chain_angles = _serve_dominant_chain_angles(values)
    preparation_chain = np.median(
        chain_angles[preparation_start:preparation_end], axis=0
    )
    completion_chain = np.median(
        chain_angles[completion_start:completion_end], axis=0
    )
    chain_change = float(np.linalg.norm(completion_chain - preparation_chain) / np.pi)
    chain_baseline = np.median(chain_angles[preparation_start:preparation_end], axis=0)
    chain_excursion = _smooth_trajectory(
        np.linalg.norm(chain_angles - chain_baseline[None], axis=1)[:, None]
    )[:, 0]
    transfer_start, transfer_end = motion_completion_bounds(len(values), 0.25, 1.0)
    dominant_chain_excursion = float(
        np.quantile(chain_excursion[transfer_start:transfer_end], 0.80) / np.pi
    )
    hip_vector = values[:, 12] - values[:, 11]
    hip_rotation = _smooth_trajectory(
        np.unwrap(np.arctan2(hip_vector[:, 1], hip_vector[:, 0]))[:, None]
    )[:, 0]
    hip_rotation -= float(np.median(hip_rotation[preparation_start:preparation_end]))
    transfer_rotation = np.abs(hip_rotation[transfer_start:transfer_end])
    transfer_chain = chain_excursion[transfer_start:transfer_end]
    centred_rotation = transfer_rotation - np.mean(transfer_rotation)
    centred_chain = transfer_chain - np.mean(transfer_chain)
    coupling_denominator = float(
        np.linalg.norm(centred_rotation) * np.linalg.norm(centred_chain)
    )
    transfer_rotation_correlation = (
        0.0
        if coupling_denominator <= _EPS
        else float(np.dot(centred_rotation, centred_chain) / coupling_denominator)
    )
    hip_rotation_excursion = float(np.quantile(transfer_rotation, 0.80) / np.pi)
    coordinated_hip_rotation = float(
        hip_rotation_excursion * max(transfer_rotation_correlation, 0.0)
    )
    root_values = (
        np.zeros((len(values), 2), dtype=np.float64)
        if root is None
        else np.asarray(root, dtype=np.float64)
    )
    if root_values.shape != (len(values), 2):
        raise ValueError("serve root evidence must have shape (T, 2)")
    preparation_root = np.median(root_values[preparation_start:preparation_end], axis=0)
    completion_root = np.median(root_values[completion_start:completion_end], axis=0)
    preparation_torso = max(
        float(np.median(torso[preparation_start:preparation_end])), _EPS
    )
    root_transfer = float(
        np.linalg.norm(completion_root - preparation_root) / preparation_torso
    )
    # A learner who brings the ankles together has no base to transfer across.
    stance_retention = float(
        np.min(stance_width[preparation_start:completion_end])
    ) / max(preparation_stance, _EPS)
    return {
        "simultaneous_arm_elevation": arms_raised,
        "preparation_stance_width": preparation_stance,
        "pelvis_loading_shift": loading_shift,
        "dominant_chain_change": chain_change,
        "dominant_chain_excursion": dominant_chain_excursion,
        "hip_rotation_excursion": hip_rotation_excursion,
        "transfer_rotation_correlation": transfer_rotation_correlation,
        "coordinated_hip_rotation": coordinated_hip_rotation,
        "root_transfer_distance": root_transfer,
        "stance_retention": stance_retention,
        # How far the pelvis had already moved toward the front foot, from the preparation stance.
        "pelvis_transfer_before_swing": preparation_loading - baseline_loading,
        "pelvis_loading_at_preparation": preparation_loading,
        "pelvis_loading_before_swing": baseline_loading,
    }


def _serve_projected_rotation_features(
    pose: NDArray[np.floating],
    confidence: NDArray[np.floating],
) -> NDArray[np.float64]:
    """Return monocular 2D proxies for pelvis and torso axial rotation."""
    values = np.asarray(pose, dtype=np.float64)
    observed = np.asarray(confidence, dtype=np.float64)
    hip_vector = values[:, 12] - values[:, 11]
    shoulder_vector = values[:, 6] - values[:, 5]
    hip_width = np.maximum(np.linalg.norm(hip_vector, axis=-1), _EPS)
    shoulder_width = np.maximum(np.linalg.norm(shoulder_vector, axis=-1), _EPS)
    hip_angle = np.unwrap(np.arctan2(hip_vector[:, 1], hip_vector[:, 0]))
    shoulder_angle = np.unwrap(np.arctan2(shoulder_vector[:, 1], shoulder_vector[:, 0]))
    hip_confidence = np.min(observed[:, (11, 12)], axis=1)
    shoulder_confidence = np.min(observed[:, (5, 6)], axis=1)
    torso_confidence = np.minimum(hip_confidence, shoulder_confidence)

    def window(
        value: NDArray[np.floating],
        conf: NDArray[np.floating],
        start: int,
        end: int,
    ) -> float:
        return _robust_window_value(value, conf, start=start, end=end)

    preparation_start, preparation_end = motion_completion_bounds(
        len(values), 0.125, 0.34375
    )
    completion_start, completion_end = motion_completion_bounds(
        len(values), 0.71875, 1.0
    )
    prep_hip_width = window(
        hip_width, hip_confidence, preparation_start, preparation_end
    )
    end_hip_width = window(hip_width, hip_confidence, completion_start, completion_end)
    prep_shoulder_width = window(
        shoulder_width, shoulder_confidence, preparation_start, preparation_end
    )
    end_shoulder_width = window(
        shoulder_width, shoulder_confidence, completion_start, completion_end
    )
    prep_hip_angle = window(
        hip_angle, hip_confidence, preparation_start, preparation_end
    )
    end_hip_angle = window(hip_angle, hip_confidence, completion_start, completion_end)
    torso_twist = np.unwrap(shoulder_angle - hip_angle)
    prep_twist = window(
        torso_twist, torso_confidence, preparation_start, preparation_end
    )
    end_twist = window(torso_twist, torso_confidence, completion_start, completion_end)
    return np.asarray(
        (
            np.log(end_hip_width / max(prep_hip_width, _EPS)),
            np.log(end_shoulder_width / max(prep_shoulder_width, _EPS)),
            _wrapped_angle(end_hip_angle - prep_hip_angle),
            _wrapped_angle(end_twist - prep_twist),
        ),
        dtype=np.float64,
    )


def _smooth_trajectory(values: NDArray[np.floating]) -> NDArray[np.float64]:
    trajectory = np.asarray(values, dtype=np.float64)
    if trajectory.ndim != 2:
        raise ValueError("trajectory smoothing requires shape (T, D)")
    padded = np.pad(trajectory, ((2, 2), (0, 0)), mode="edge")
    kernel = np.asarray((1.0, 2.0, 3.0, 2.0, 1.0), dtype=np.float64) / 9.0
    return np.stack(
        [
            np.convolve(padded[:, axis], kernel, mode="valid")
            for axis in range(trajectory.shape[1])
        ],
        axis=-1,
    )


def _serve_semantic_evidence(
    rule_id: str,
    pose: NDArray[np.floating],
    root: NDArray[np.floating],
    confidence: NDArray[np.floating],
) -> tuple[NDArray[np.float64], tuple[str, ...], NDArray[np.float64]]:
    """Return higher-is-better evidence supported by every training expert."""
    if rule_id == "arms_raised":
        evidence = _serve_qualitative_pose_evidence(pose, root)
        return (
            np.asarray((evidence["simultaneous_arm_elevation"],), dtype=np.float64),
            ("simultaneous_arm_elevation",),
            np.asarray((1.0,), dtype=np.float64),
        )
    if rule_id == "racket_foot_weight":
        values = np.asarray(pose, dtype=np.float64)
        observed = np.asarray(confidence, dtype=np.float64)
        start, end = motion_completion_bounds(len(values), 0.125, 0.34375)
        hip_center = 0.5 * (values[:, 11] + values[:, 12])
        shoulder_center = 0.5 * (values[:, 5] + values[:, 6])
        torso = np.maximum(np.linalg.norm(shoulder_center - hip_center, axis=-1), _EPS)
        ankle_axis = values[:, 16] - values[:, 15]
        ankle_squared = np.maximum(np.sum(ankle_axis * ankle_axis, axis=-1), _EPS)
        # Canonical joint 16 is the racket-side ankle.
        racket_side_loading = (
            np.sum((hip_center - values[:, 15]) * ankle_axis, axis=-1) / ankle_squared
        )
        stance_width = np.linalg.norm(ankle_axis, axis=-1) / torso
        loading_confidence = np.min(observed[:, (11, 12, 15, 16)], axis=1)
        return (
            np.asarray(
                (
                    _robust_window_value(
                        racket_side_loading,
                        loading_confidence,
                        start=start,
                        end=end,
                    ),
                    _robust_window_value(
                        stance_width,
                        loading_confidence,
                        start=start,
                        end=end,
                    ),
                ),
                dtype=np.float64,
            ),
            ("preparation_racket_side_loading", "preparation_stance_width"),
            np.asarray((1.5, 0.75), dtype=np.float64),
        )
    if rule_id == "weight_transfer":
        motion = _serve_qualitative_pose_evidence(pose, root)
        return (
            np.asarray(
                (
                    motion["dominant_chain_excursion"],
                    motion["dominant_chain_change"],
                    motion["pelvis_loading_shift"],
                    motion["root_transfer_distance"],
                    motion["coordinated_hip_rotation"],
                    motion["stance_retention"],
                ),
                dtype=np.float64,
            ),
            (
                "dominant_chain_excursion",
                "dominant_chain_completion_change",
                "pelvis_loading_shift",
                "root_transfer_distance",
                "coordinated_hip_rotation",
                "stance_retention",
            ),
            # The dominant-side joint angles carry the decision: invariant to translation.
            np.asarray((2.5, 1.5, 0.5, 0.0, 0.0, 1.0), dtype=np.float64),
        )
    if rule_id == "hip_rotation":
        rotation = _serve_projected_rotation_features(pose, confidence)
        return (
            np.asarray(
                (
                    -rotation[0],
                    -rotation[1],
                    abs(rotation[2]),
                    abs(rotation[3]),
                ),
                dtype=np.float64,
            ),
            (
                "projected_hip_contraction",
                "projected_shoulder_contraction",
                "projected_hip_axis_rotation",
                "projected_torso_twist",
            ),
            np.ones(4, dtype=np.float64),
        )
    if rule_id == "wrist_flick":
        # Measure a tempo-invariant contact impulse.
        start, end = motion_completion_bounds(len(pose), 0.375, 0.625)
        values = np.asarray(pose, dtype=np.float64)
        shoulder_center = 0.5 * (values[:, 5] + values[:, 6])
        hip_center = 0.5 * (values[:, 11] + values[:, 12])
        torso_scale = max(
            float(np.median(np.linalg.norm(shoulder_center - hip_center, axis=-1))),
            _EPS,
        )
        relative_wrist = _smooth_trajectory(values[:, 10] - values[:, 6])
        acceleration = np.linalg.norm(np.diff(relative_wrist, n=2, axis=0), axis=-1)
        event_displacement = float(
            np.linalg.norm(relative_wrist[end - 1] - relative_wrist[start])
            / torso_scale
        )
        event_acceleration = float(
            np.quantile(acceleration[start : max(start + 1, end - 2)], 0.90)
        )
        baseline_acceleration = max(float(np.quantile(acceleration, 0.50)), 1e-4)
        acceleration_prominence = event_acceleration / baseline_acceleration
        contact_impulse = float(
            np.sqrt(max(event_displacement * acceleration_prominence, 0.0))
        )
        forward_axis = np.median(values[start:end, 5] - values[start:end, 6], axis=0)
        forward_axis /= max(float(np.linalg.norm(forward_axis)), _EPS)
        projected_acceleration = (np.diff(relative_wrist, n=2, axis=0) @ forward_axis)[
            start : max(start + 1, end - 2)
        ]
        directional_acceleration_ratio = float(
            np.sum(np.maximum(projected_acceleration, 0.0))
            / max(float(np.sum(np.abs(projected_acceleration))), _EPS)
        )
        return (
            np.asarray(
                (contact_impulse, directional_acceleration_ratio),
                dtype=np.float64,
            ),
            (
                "tempo_invariant_contact_impulse",
                "forward_acceleration_coherence",
            ),
            np.asarray((1.0, 1.0), dtype=np.float64),
        )
    if rule_id == "shoulder_rotation":
        values = np.asarray(pose, dtype=np.float64)
        observed = np.asarray(confidence, dtype=np.float64)
        shoulder_center = 0.5 * (values[:, 5] + values[:, 6])
        hip_center = 0.5 * (values[:, 11] + values[:, 12])
        preparation_start, preparation_end = motion_completion_bounds(
            len(values), 0.125, 0.34375
        )
        torso = np.linalg.norm(
            shoulder_center[preparation_start:preparation_end]
            - hip_center[preparation_start:preparation_end],
            axis=-1,
        )
        torso_scale = max(float(np.median(torso[np.isfinite(torso)])), _EPS)
        completion_start, completion_end = motion_completion_bounds(
            len(values), 0.875, 1.0
        )
        confidence_mask = np.min(observed[:, (5, 6, 8, 10)], axis=1)
        forearm_offset = (
            0.75 * values[:, 8, 0] + 0.25 * values[:, 10, 0] - shoulder_center[:, 0]
        ) / torso_scale
        elbow_drop = (values[:, 8, 1] - shoulder_center[:, 1]) / torso_scale
        wrist_drop = (values[:, 10, 1] - shoulder_center[:, 1]) / torso_scale
        # The follow-through, read at the racket elbow: the angle it subtends between the two shoulders.
        to_racket_shoulder = values[:, 6] - values[:, 8]
        to_other_shoulder = values[:, 5] - values[:, 8]
        shoulder_elbow_cosine = np.sum(
            to_racket_shoulder * to_other_shoulder, axis=-1
        ) / np.maximum(
            np.linalg.norm(to_racket_shoulder, axis=-1)
            * np.linalg.norm(to_other_shoulder, axis=-1),
            _EPS,
        )
        shoulder_elbow_angle = np.degrees(
            np.arccos(np.clip(shoulder_elbow_cosine, -1.0, 1.0))
        )
        # Shoulders that turn to face forward foreshorten in the picture, and past edge-on they swap sides.
        setup_line = np.median(
            values[preparation_start:preparation_end, 6]
            - values[preparation_start:preparation_end, 5],
            axis=0,
        )
        setup_line /= max(float(np.linalg.norm(setup_line)), _EPS)
        shoulder_width = ((values[:, 6] - values[:, 5]) @ setup_line) / torso_scale
        rotation = _serve_projected_rotation_features(values, observed)
        return (
            np.asarray(
                (
                    -rotation[1],
                    -_robust_window_value(
                        forearm_offset,
                        confidence_mask,
                        start=completion_start,
                        end=completion_end,
                    ),
                    _robust_window_value(
                        elbow_drop,
                        confidence_mask,
                        start=completion_start,
                        end=completion_end,
                    ),
                    _robust_window_value(
                        wrist_drop,
                        confidence_mask,
                        start=completion_start,
                        end=completion_end,
                    ),
                    -_serve_best_terminal_value(
                        shoulder_elbow_angle,
                        confidence_mask,
                        start=completion_start,
                        end=completion_end,
                    ),
                    -_serve_best_terminal_value(
                        shoulder_width,
                        confidence_mask,
                        start=completion_start,
                        end=completion_end,
                    ),
                ),
                dtype=np.float64,
            ),
            (
                "terminal_shoulder_contraction",
                "terminal_cross_body_reach",
                "terminal_elbow_drop",
                "terminal_wrist_drop",
                "terminal_elbow_between_shoulders",
                "terminal_shoulder_foreshortening",
            ),
            # The checkpoint is shoulder-forward rotation.
            np.asarray((1.5, 0.0, 0.0, 0.0, 1.0, 1.0), dtype=np.float64),
        )
    raise KeyError(f"serve rule has no semantic expert evidence: {rule_id}")


# How much more an expert's ankles close than their own corrected skeleton's do, at most.
_SERVE_CORRECTION_STANCE_ALLOWANCE = 0.167 * (1.0 + _SERVE_UNSEEN_EXPERT_MARGIN)


# The weight arrives on the front foot as the racket arm speeds up.
_SERVE_CORRECTION_TRANSFER_LEAD_ALLOWANCE_FRAMES = 4.0 * (
    1.0 + _SERVE_UNSEEN_EXPERT_MARGIN
)


_SERVE_CORRECTION_TRANSFER_LEAD_SCALE_FRAMES = 4.0


# The racket arm is extended at contact.
_SERVE_CORRECTION_ELBOW_ALLOWANCE_DEGREES = 12.3 * (1.0 + _SERVE_UNSEEN_EXPERT_MARGIN)


_SERVE_CORRECTION_ELBOW_SCALE_DEGREES = 10.0


# At maximum acceleration the shoulders have turned against the stance at least as far as the corrected skeleton's: experts turn at most 2.5 degrees less than theirs.
_SERVE_CORRECTION_SHOULDER_TURN_ALLOWANCE_DEGREES = 2.5 * (
    1.0 + _SERVE_UNSEEN_EXPERT_MARGIN
)


_SERVE_CORRECTION_SHOULDER_TURN_SCALE_DEGREES = 5.0


def _angles(
    sequence: NDArray[np.floating], triplets: Sequence[tuple[int, int, int]]
) -> NDArray[np.float64]:
    values = np.asarray(sequence, dtype=np.float64)
    indices = np.asarray(triplets, dtype=np.int64)
    incoming = values[:, indices[:, 0]] - values[:, indices[:, 1]]
    outgoing = values[:, indices[:, 2]] - values[:, indices[:, 1]]
    denominator = np.linalg.norm(incoming, axis=-1) * np.linalg.norm(outgoing, axis=-1)
    cosine = np.divide(
        np.sum(incoming * outgoing, axis=-1),
        denominator,
        out=np.ones_like(denominator),
        where=denominator > _EPS,
    )
    return np.arccos(np.clip(cosine, -1.0, 1.0))


def _serve_weight_transfer_components(
    source_pose: NDArray[np.floating],
    source_root: NDArray[np.floating],
    target_pose: NDArray[np.floating],
    target_root: NDArray[np.floating],
    confidence: NDArray[np.floating],
) -> dict[str, float]:
    del source_root, target_root
    source = _serve_dominant_chain_angles(source_pose)
    target = _serve_dominant_chain_angles(target_pose)
    observed = np.min(np.asarray(confidence)[:, (6, 12, 14, 16)], axis=1)
    preparation_start, preparation_end = motion_completion_bounds(
        len(source), 0.125, 0.34375
    )
    transfer_start, transfer_end = motion_completion_bounds(len(source), 0.25, 1.0)
    completion_start, completion_end = motion_completion_bounds(
        len(source), 0.71875, 1.0
    )
    valid = observed[transfer_start:transfer_end] >= 0.20
    if not np.any(valid):
        valid = np.ones(transfer_end - transfer_start, dtype=bool)
    trajectory_delta = (
        source[transfer_start:transfer_end][valid]
        - target[transfer_start:transfer_end][valid]
    ) / np.pi
    trajectory_distance = float(np.sqrt(np.mean(trajectory_delta**2)))

    def window(
        values: NDArray[np.floating], start: int, end: int
    ) -> NDArray[np.float64]:
        return np.asarray(
            [
                _robust_window_value(values[:, column], observed, start=start, end=end)
                for column in range(values.shape[1])
            ],
            dtype=np.float64,
        )

    source_change = window(source, completion_start, completion_end) - window(
        source, preparation_start, preparation_end
    )
    target_change = window(target, completion_start, completion_end) - window(
        target, preparation_start, preparation_end
    )
    change_distance = float(
        np.sqrt(np.mean(((source_change - target_change) / np.pi) ** 2))
    )
    distance = float(
        np.sqrt((trajectory_distance**2 + 0.75 * change_distance**2) / 1.75)
    )
    return {
        "euclidean_distance": distance,
        "target_angle_distance": trajectory_distance,
        "combined_distance": distance,
        "dominant_chain_trajectory_distance": trajectory_distance,
        "dominant_chain_change_distance": change_distance,
        "source_shoulder_hip_knee_change": float(source_change[0]),
        "target_shoulder_hip_knee_change": float(target_change[0]),
        "source_hip_knee_ankle_change": float(source_change[1]),
        "target_hip_knee_ankle_change": float(target_change[1]),
        "source_shoulder_hip_ankle_change": float(source_change[2]),
        "target_shoulder_hip_ankle_change": float(target_change[2]),
    }


def _serve_transfer_correction_residuals(
    learner_pose: NDArray[np.floating],
    corrected_pose: NDArray[np.floating],
) -> dict[str, float]:
    """How much more the learner's ankles close than their corrected skeleton's."""
    learner = np.asarray(learner_pose, dtype=np.float64)
    corrected = np.asarray(corrected_pose, dtype=np.float64)
    preparation_start, preparation_end = motion_completion_bounds(
        len(learner), 0.125, 0.34375
    )
    _, completion_end = motion_completion_bounds(len(learner), 0.71875, 1.0)

    def retention(pose: NDArray[np.float64]) -> float:
        width = np.linalg.norm(pose[:, 16] - pose[:, 15], axis=-1)
        start = float(np.median(width[preparation_start:preparation_end]))
        return float(np.min(width[preparation_start:completion_end])) / max(start, _EPS)

    learner_retention = retention(learner)
    corrected_retention = retention(corrected)
    _, contact = motion_completion_bounds(len(learner), 0.0, 0.5)
    descent = _serve_wrist_descent_onset(learner, contact)

    def transfer_lead(pose: NDArray[np.float64]) -> float:
        # Frames between the pelvis reaching the front foot and the racket arm speeding up.
        arm = _smooth_trajectory(pose[:, 10] - pose[:, 6])
        speed = np.r_[0.0, np.linalg.norm(np.diff(arm, axis=0), axis=-1)]
        span = slice(descent, min(len(pose), contact + 3))
        swing_start = descent + int(np.argmax(speed[span] >= 0.5 * np.max(speed[span])))
        hip = 0.5 * (pose[:, 11] + pose[:, 12])
        axis = pose[:, 16] - pose[:, 15]
        loading = np.sum((hip - pose[:, 15]) * axis, axis=-1) / np.maximum(
            np.sum(axis * axis, axis=-1), _EPS
        )
        start = float(np.mean(loading[max(0, descent - 4) : max(1, descent)]))
        lowest = float(np.min(loading[descent : contact + 1]))
        arrived = descent + int(
            np.argmax(loading[descent : contact + 1] <= start - 0.9 * (start - lowest))
        )
        return float(swing_start - arrived)

    learner_lead = transfer_lead(learner)
    corrected_lead = transfer_lead(corrected)
    return {
        "correction_learner_transfer_lead_frames": learner_lead,
        "correction_corrected_transfer_lead_frames": corrected_lead,
        "correction_transfer_lead_excess_frames": max(0.0, learner_lead - corrected_lead),
        "correction_learner_stance_retention": learner_retention,
        "correction_corrected_stance_retention": corrected_retention,
        "correction_stance_retention_shortfall": max(
            0.0, corrected_retention - learner_retention
        ),
    }


def _serve_wrist_correction_residuals(
    learner_pose: NDArray[np.floating],
    corrected_pose: NDArray[np.floating],
) -> dict[str, float]:
    """How much more the racket elbow is bent at contact than the correction's."""
    learner = np.asarray(learner_pose, dtype=np.float64)
    corrected = np.asarray(corrected_pose, dtype=np.float64)
    _, contact = motion_completion_bounds(len(learner), 0.0, 0.5)

    def elbow_at_contact(pose: NDArray[np.float64]) -> float:
        upper = pose[:, 6] - pose[:, 8]
        fore = pose[:, 10] - pose[:, 8]
        cosine = np.sum(upper * fore, axis=-1) / np.maximum(
            np.linalg.norm(upper, axis=-1) * np.linalg.norm(fore, axis=-1), _EPS
        )
        angle = np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))
        return float(np.median(angle[max(0, contact - 1) : contact + 2]))

    learner_angle = elbow_at_contact(learner)
    corrected_angle = elbow_at_contact(corrected)
    def shoulder_to_stance(pose: NDArray[np.float64]) -> float:
        # Signed angle from the ankle line to the shoulder line.
        shoulders = pose[:, 6] - pose[:, 5]
        ankles = pose[:, 16] - pose[:, 15]
        cross = ankles[:, 0] * shoulders[:, 1] - ankles[:, 1] * shoulders[:, 0]
        dot = np.sum(ankles * shoulders, axis=-1)
        angle = np.degrees(np.arctan2(cross, dot))
        return float(np.median(angle[max(0, contact - 1) : contact + 2]))

    learner_turn = shoulder_to_stance(learner)
    corrected_turn = shoulder_to_stance(corrected)
    return {
        "correction_shoulder_turn_shortfall_degrees": max(
            0.0, learner_turn - corrected_turn
        ),
        "correction_shoulder_turn_allowance_degrees": (
            _SERVE_CORRECTION_SHOULDER_TURN_ALLOWANCE_DEGREES
        ),
        "correction_learner_shoulder_stance_angle_degrees": learner_turn,
        "correction_corrected_shoulder_stance_angle_degrees": corrected_turn,

        "correction_learner_elbow_at_contact_degrees": learner_angle,
        "correction_corrected_elbow_at_contact_degrees": corrected_angle,
        "correction_elbow_at_contact_shortfall_degrees": max(
            0.0, corrected_angle - learner_angle
        ),
        "correction_elbow_allowance_degrees": _SERVE_CORRECTION_ELBOW_ALLOWANCE_DEGREES,
    }


def _serve_transfer_against_correction(
    semantic: dict[str, Any], residuals: dict[str, float], tolerance: float
) -> dict[str, Any]:
    """Decide the stance on the learner's own corrected skeleton."""
    scale = max(float(semantic["expert_scale_stance_retention"]), 1e-3)
    excess = max(
        0.0,
        residuals["correction_stance_retention_shortfall"]
        - _SERVE_CORRECTION_STANCE_ALLOWANCE,
    )
    # The allowance is already the experts' own spread against their corrections.
    stance_distance = tolerance + excess / scale if excess > 0.0 else 0.0
    early = max(
        0.0,
        residuals["correction_transfer_lead_excess_frames"]
        - _SERVE_CORRECTION_TRANSFER_LEAD_ALLOWANCE_FRAMES,
    )
    # Weight already on the front foot before the swing is the same fault as weight that never gets there.
    timing_distance = (
        tolerance + early / _SERVE_CORRECTION_TRANSFER_LEAD_SCALE_FRAMES
        if early > 0.0
        else 0.0
    )
    distance = float(
        max(
            semantic["transfer_chain_distance"],
            semantic["transfer_support_distance"],
            stance_distance,
            timing_distance,
        )
    )
    # The all-cues distance the rubric cap reads carries the stance too.
    weighted = (
        (2.5, semantic["standardized_shortfall_dominant_chain_excursion"]),
        (1.5, semantic["standardized_shortfall_dominant_chain_completion_change"]),
        (0.5, semantic["standardized_shortfall_pelvis_loading_shift"]),
        (1.0, stance_distance),
    )
    strict = float(
        np.sqrt(
            sum(weight * float(value) ** 2 for weight, value in weighted)
            / sum(weight for weight, _ in weighted)
        )
    )
    return {
        **semantic,
        **residuals,
        "strict_required_cue_distance": strict,
        "correction_stance_distance": stance_distance,
        "correction_transfer_timing_distance": timing_distance,
        "correction_transfer_lead_allowance_frames": (
            _SERVE_CORRECTION_TRANSFER_LEAD_ALLOWANCE_FRAMES
        ),
        "correction_stance_allowance": _SERVE_CORRECTION_STANCE_ALLOWANCE,
        "euclidean_distance": distance,
        "combined_distance": distance,
        "expert_pattern_distance": distance,
        "semantic_cue_aggregation": "dominant_chain_with_support_and_corrected_stance",
    }


def _serve_arms_at_corrected_shoulder_evidence(
    source_pose: NDArray[np.floating],
    corrected_pose: NDArray[np.floating],
    confidence: NDArray[np.floating],
) -> dict[str, float | bool]:
    """Check whether both hands reach the corrected shoulder level."""
    source = np.asarray(source_pose, dtype=np.float64)
    corrected = np.asarray(corrected_pose, dtype=np.float64)
    observed = np.asarray(confidence, dtype=np.float64)
    if source.shape != corrected.shape or source.ndim != 3:
        raise ValueError("serve arm evidence requires matching (T, J, 2) poses")
    start, end = motion_completion_bounds(len(source), 0.125, 0.34375)
    corrected_shoulders = 0.5 * (corrected[:, 5] + corrected[:, 6])
    corrected_hips = 0.5 * (corrected[:, 11] + corrected[:, 12])
    torso = np.maximum(
        np.linalg.norm(corrected_shoulders - corrected_hips, axis=-1), _EPS
    )
    margins: list[float] = []
    for shoulder, elbow, wrist in ((5, 7, 9), (6, 8, 10)):
        # Normalized pose y points from the pelvis toward the shoulders, so a larger y is visually higher.
        distal_y = np.where(
            observed[:, wrist] > 0.05,
            source[:, wrist, 1],
            source[:, elbow, 1],
        )
        distal_confidence = np.maximum(observed[:, wrist], observed[:, elbow])
        relative_height = (distal_y - corrected[:, shoulder, 1]) / torso
        margins.append(
            _robust_window_value(
                relative_height,
                distal_confidence,
                start=start,
                end=end,
            )
        )
    weaker_margin = float(min(margins))
    tolerance = 0.20
    return {
        "left_hand_corrected_shoulder_margin": float(margins[0]),
        "right_hand_corrected_shoulder_margin": float(margins[1]),
        "weaker_hand_corrected_shoulder_margin": weaker_margin,
        "corrected_shoulder_level_tolerance": tolerance,
        "passes_corrected_shoulder_height": weaker_margin >= -tolerance,
    }


def _serve_qualitative_factor(value: float, calibration: dict[str, float]) -> float:
    expert_scale = max(float(calibration["expert_scale"]), 1e-3)
    # Half an expert robust scale is a no-penalty detector/view margin.
    shortfall = max(
        float(calibration["expert_lower"]) - 0.5 * expert_scale - value,
        0.0,
    )
    return float(np.exp(-shortfall / min(expert_scale, 0.05)))


def _serve_required_motion_factor(value: float, calibration: dict[str, float]) -> float:
    """Score an absolute movement prerequisite against experts only."""
    expert_scale = max(float(calibration["expert_scale"]), 1e-3)
    shortfall = max(float(calibration["expert_lower"]) - value, 0.0)
    return float(np.exp(-shortfall / expert_scale))


def _serve_hip_rotation_components(
    source_pose: NDArray[np.floating],
    target_pose: NDArray[np.floating],
    confidence: NDArray[np.floating],
) -> dict[str, float]:
    source_values = np.asarray(source_pose, dtype=np.float64)
    target_values = np.asarray(target_pose, dtype=np.float64)
    source_angles = _serve_dominant_chain_angles(source_values)
    target_angles = _serve_dominant_chain_angles(target_values)
    preparation_start, preparation_end = motion_completion_bounds(
        len(source_values), 0.125, 0.34375
    )
    transfer_start, transfer_end = motion_completion_bounds(
        len(source_values), 0.25, 1.0
    )

    def coupled_trajectories(
        pose: NDArray[np.float64], angles: NDArray[np.float64]
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        hip_vector = pose[:, 12] - pose[:, 11]
        rotation = np.unwrap(np.arctan2(hip_vector[:, 1], hip_vector[:, 0]))
        rotation -= float(np.median(rotation[preparation_start:preparation_end]))
        angle_baseline = np.median(angles[preparation_start:preparation_end], axis=0)
        transfer = np.linalg.norm(angles - angle_baseline[None], axis=1)
        rotation = _smooth_trajectory(rotation[:, None])[:, 0]
        transfer = _smooth_trajectory(transfer[:, None])[:, 0]
        return rotation, transfer

    source_rotation, source_transfer = coupled_trajectories(
        source_values, source_angles
    )
    target_rotation, target_transfer = coupled_trajectories(
        target_values, target_angles
    )
    observed = np.min(np.asarray(confidence)[:, (6, 11, 12, 14, 16)], axis=1)
    valid = observed[transfer_start:transfer_end] >= 0.20
    if not np.any(valid):
        valid = np.ones(transfer_end - transfer_start, dtype=bool)

    def correlation(first: NDArray[np.floating], second: NDArray[np.floating]) -> float:
        left = np.asarray(first[transfer_start:transfer_end], dtype=np.float64)[valid]
        right = np.asarray(second[transfer_start:transfer_end], dtype=np.float64)[valid]
        left -= np.mean(left)
        right -= np.mean(right)
        denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
        return 0.0 if denominator <= _EPS else float(np.dot(left, right) / denominator)

    source_coupling = correlation(np.abs(source_rotation), source_transfer)
    target_coupling = correlation(np.abs(target_rotation), target_transfer)
    coupling_distance = abs(source_coupling - target_coupling) / 2.0
    rotation_distance = float(
        np.sqrt(
            np.mean(
                (
                    (
                        source_rotation[transfer_start:transfer_end][valid]
                        - target_rotation[transfer_start:transfer_end][valid]
                    )
                    / np.pi
                )
                ** 2
            )
        )
    )
    # Coupling is primary: hip rotation receives credit when it happens with the dominant-side weight-transfer chain, not as an isolated torso pose.
    distance = float(
        np.sqrt((0.5 * rotation_distance**2 + 2.0 * coupling_distance**2) / 2.5)
    )
    return {
        "euclidean_distance": distance,
        "target_angle_distance": rotation_distance,
        "combined_distance": distance,
        "hip_rotation_trajectory_distance": rotation_distance,
        "source_transfer_rotation_correlation": source_coupling,
        "target_transfer_rotation_correlation": target_coupling,
        "transfer_rotation_coupling_distance": coupling_distance,
    }


def _serve_wrist_motion_features(
    pose: NDArray[np.floating],
    confidence: NDArray[np.floating],
    *,
    start: int,
    end: int,
) -> NDArray[np.float64]:
    """Measure distal-arm speed; COCO-17 cannot observe wrist flexion directly."""
    values = np.asarray(pose, dtype=np.float64)
    observed = np.asarray(confidence, dtype=np.float64)
    wrist_to_shoulder = _smooth_trajectory(values[:, 10] - values[:, 6])
    wrist_to_elbow = _smooth_trajectory(values[:, 10] - values[:, 8])
    shoulder_speed = np.linalg.norm(np.diff(wrist_to_shoulder, axis=0), axis=-1)
    forearm_speed = np.linalg.norm(np.diff(wrist_to_elbow, axis=0), axis=-1)
    acceleration = np.linalg.norm(np.diff(wrist_to_shoulder, n=2, axis=0), axis=-1)
    joint_confidence = np.min(observed[:, (6, 8, 10)], axis=1)
    speed_confidence = np.minimum(joint_confidence[:-1], joint_confidence[1:])
    acceleration_confidence = np.minimum(
        np.minimum(joint_confidence[:-2], joint_confidence[1:-1]),
        joint_confidence[2:],
    )

    def summarize(
        signal: NDArray[np.floating],
        signal_confidence: NDArray[np.floating],
        left: int,
        right: int,
    ) -> tuple[float, float]:
        selected = np.asarray(signal, dtype=np.float64)[left:right]
        selected_confidence = np.asarray(signal_confidence, dtype=np.float64)[
            left:right
        ]
        valid = np.isfinite(selected) & (selected_confidence >= 0.20)
        if not np.any(valid):
            valid = np.isfinite(selected)
        available = selected[valid]
        if not len(available):
            return 0.0, 0.0
        return float(np.mean(available)), float(np.quantile(available, 0.90))

    shoulder_mean, shoulder_peak = summarize(
        shoulder_speed, speed_confidence, start, end - 1
    )
    forearm_mean, forearm_peak = summarize(
        forearm_speed, speed_confidence, start, end - 1
    )
    acceleration_mean, acceleration_peak = summarize(
        acceleration, acceleration_confidence, start, end - 2
    )
    return np.asarray(
        (
            shoulder_mean,
            shoulder_peak,
            forearm_mean,
            forearm_peak,
            acceleration_mean,
            acceleration_peak,
        ),
        dtype=np.float64,
    )


def _serve_wrist_action_components(
    source_pose: NDArray[np.floating],
    target_pose: NDArray[np.floating],
    confidence: NDArray[np.floating],
    *,
    start: int,
    end: int,
) -> dict[str, float]:
    source = _serve_wrist_motion_features(source_pose, confidence, start=start, end=end)
    target = _serve_wrist_motion_features(target_pose, confidence, start=start, end=end)
    delta = source - target
    # Sustained wrist-to-shoulder speed is weighted above isolated peaks.
    weights = np.asarray((3.0, 0.25, 0.25, 0.25, 0.25, 0.25))
    distance = float(np.sqrt(np.sum(weights * delta**2) / np.sum(weights)))
    return {
        "euclidean_distance": distance,
        "target_angle_distance": 0.0,
        "combined_distance": distance,
        "source_wrist_speed_mean": float(source[0]),
        "target_wrist_speed_mean": float(target[0]),
        "source_wrist_speed_p90": float(source[1]),
        "target_wrist_speed_p90": float(target[1]),
        "source_forearm_speed_p90": float(source[3]),
        "target_forearm_speed_p90": float(target[3]),
        "source_wrist_acceleration_p90": float(source[5]),
        "target_wrist_acceleration_p90": float(target[5]),
    }
