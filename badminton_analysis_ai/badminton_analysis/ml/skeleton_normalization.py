from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
from numpy.typing import NDArray

from badminton_analysis.models.types import (
    COCOKeypoints,
    CoordinateDict,
    Handedness,
    TrackingData,
)

COCO_JOINT_COUNT = 17
CANONICAL_PHASE_INDICES = np.asarray((0, 16, 32, 48, 63), dtype=np.int64)
_EPS = 1e-8
_LEFT_RIGHT_PAIRS = (
    (1, 2),
    (3, 4),
    (5, 6),
    (7, 8),
    (9, 10),
    (11, 12),
    (13, 14),
    (15, 16),
)
_NORMALIZATION_BONES = np.asarray(
    (
        (5, 6),
        (5, 7),
        (7, 9),
        (6, 8),
        (8, 10),
        (5, 11),
        (6, 12),
        (11, 12),
        (11, 13),
        (13, 15),
        (12, 14),
        (14, 16),
    ),
    dtype=np.int64,
)
_OUTLIER_PARENT_EDGES = (
    (5, 7),
    (7, 9),
    (6, 8),
    (8, 10),
    (11, 13),
    (13, 15),
    (12, 14),
    (14, 16),
)


@dataclass(frozen=True)
class NormalizedSkeletonMotion:
    """Canonical local pose plus the global motion removed by normalization."""

    skeleton: NDArray[np.float32]
    confidence: NDArray[np.float32]
    root_trajectory: NDArray[np.float32]
    body_basis: NDArray[np.float32]
    body_scale: float
    root_origin: NDArray[np.float32]


def landmark_dicts_to_array(
    frames: Iterable[CoordinateDict],
    dimensions: int,
) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
    """Convert sparse landmark dictionaries to dense arrays plus an observed mask."""
    frame_list = list(frames)
    coordinates = np.full(
        (len(frame_list), COCO_JOINT_COUNT, dimensions), np.nan, dtype=np.float64
    )
    confidence = np.zeros((len(frame_list), COCO_JOINT_COUNT), dtype=np.float64)
    for frame_index, landmarks in enumerate(frame_list):
        for keypoint, coordinate in landmarks.items():
            joint_index = int(keypoint)
            if not 0 <= joint_index < COCO_JOINT_COUNT:
                continue
            value = np.asarray(coordinate, dtype=np.float64)
            if value.ndim != 1 or len(value) < dimensions:
                continue
            value = value[:dimensions]
            if not np.all(np.isfinite(value)):
                continue
            coordinates[frame_index, joint_index] = value
            confidence[frame_index, joint_index] = 1.0
    return coordinates.astype(np.float32), confidence.astype(np.float32)


def tracking_body_arrays(
    tracking: TrackingData,
) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
    """Return dense body coordinates with continuous detector confidence.

    New RF-DETR tracking data carries dense coordinates and scores.  Older
    caches and test doubles only have sparse dictionaries, so they retain a
    compatibility fallback with a binary observed mask.  Keeping this choice
    in one function prevents inference, audit, and handedness paths from
    silently using different confidence semantics.
    """
    coordinates = tracking.get("body_keypoints_2d")
    confidence = tracking.get("body_confidence_2d")
    if coordinates is not None and confidence is not None:
        dense = np.asarray(coordinates, dtype=np.float32)
        scores = np.asarray(confidence, dtype=np.float32)
        if dense.ndim != 3 or dense.shape[1:] != (COCO_JOINT_COUNT, 2):
            raise ValueError("body_keypoints_2d must have shape (T, 17, 2)")
        if scores.shape != dense.shape[:2]:
            raise ValueError("body_confidence_2d must have shape (T, 17)")
        return dense, np.clip(scores, 0.0, 1.0).astype(np.float32)
    body_2d = tracking.get("body_landmarks_2d")
    if not body_2d:
        raise ValueError("aligned 2D body landmarks are unavailable")
    return landmark_dicts_to_array(body_2d, 2)


def _interpolate_missing(
    sequence: NDArray[np.floating], confidence: NDArray[np.floating]
) -> NDArray[np.float64]:
    coordinates = np.asarray(sequence, dtype=np.float64).copy()
    mask = np.asarray(confidence, dtype=np.float64)
    if coordinates.ndim != 3 or mask.shape != coordinates.shape[:2]:
        raise ValueError("sequence must be (T, J, D) and confidence must be (T, J)")
    timeline = np.arange(len(coordinates), dtype=np.float64)
    for joint in range(coordinates.shape[1]):
        for dimension in range(coordinates.shape[2]):
            values = coordinates[:, joint, dimension]
            valid = (mask[:, joint] > 0) & np.isfinite(values)
            count = int(np.count_nonzero(valid))
            if count == 0:
                coordinates[:, joint, dimension] = 0.0
            elif count == 1:
                coordinates[:, joint, dimension] = values[valid][0]
            else:
                coordinates[:, joint, dimension] = np.interp(
                    timeline, timeline[valid], values[valid]
                )
    return coordinates


# Proximal-to-distal bone pairs from skeleton_scoring.BONES, excluding the two
# lateral pairs ((5, 6) shoulder-shoulder, (11, 12) hip-hip) that have no
# parent/child relationship, and the torso connectors ((5, 11), (6, 12)) whose
# "child" is a normalization anchor rather than a joint this should touch.
_PARENT_CHILD_BONES = (
    (5, 7), (7, 9), (6, 8), (8, 10),
    (11, 13), (13, 15), (12, 14), (14, 16),
)
_STABILIZED_CHILD_CONFIDENCE = 0.5
EXPERT_RECONSTRUCTED_CONFIDENCE = 0.2


def complete_expert_joint_confidence(
    confidence: NDArray[np.floating],
    *,
    reconstructed_confidence: float = EXPERT_RECONSTRUCTED_CONFIDENCE,
) -> NDArray[np.float32]:
    """Keep all 17 interpolated expert joints weakly supervised.

    Expert coordinates have already been interpolated and projected onto one
    stable skeleton by :func:`normalize_skeleton_motion`.  Retaining a zero
    observation mask after that reconstruction makes the coordinate disappear
    from both the loss and review overlay, even though a finite kinematic
    estimate is available.  A deliberately low floor distinguishes those
    estimates from directly observed joints without discarding them.

    This must only be applied to expert references/targets.  Student confidence
    remains the real detector mask so the model still knows which input joints
    it needs to reconstruct.
    """
    values = np.asarray(confidence, dtype=np.float32)
    if values.ndim != 2 or values.shape[1] != COCO_JOINT_COUNT:
        raise ValueError("expert confidence must have shape (T, 17)")
    if not 0.0 < reconstructed_confidence <= 1.0:
        raise ValueError("reconstructed confidence must be in (0, 1]")
    return np.maximum(
        np.clip(values, 0.0, 1.0), np.float32(reconstructed_confidence)
    ).astype(np.float32)


def _raise_confidence_for_bone_stabilized_joints(
    confidence: NDArray[np.floating],
    *,
    parent_threshold: float = 0.5,
    child_threshold: float = 0.5,
) -> NDArray[np.float32]:
    """Let a confidently-tracked joint partially vouch for its stabilized child.

    `project_stable_bone_lengths` already anchors an undetected joint's
    position to a confidently-observed parent (e.g. the ankle to the knee)
    via the clip's own stable bone length, but leaves that joint's stored
    confidence untouched -- so a real, reasonably plausible reconstructed
    position still gets zero-weighted out of every downstream training loss
    and distance metric. If the parent is confidently observed in this exact
    frame, promote the child's confidence to a moderate synthetic value
    (never lowering it), reflecting "positionally anchored, not directly
    detected" rather than "unknown." Deliberately a single hop, always read
    from the original (pre-boost) array, so a boosted joint never vouches for
    a further joint down the same chain.
    """
    original = np.asarray(confidence, dtype=np.float64)
    boosted = np.asarray(confidence, dtype=np.float32).copy()
    for parent, child in _PARENT_CHILD_BONES:
        eligible = (original[:, parent] >= parent_threshold) & (
            original[:, child] < child_threshold
        )
        boosted[eligible, child] = np.maximum(
            boosted[eligible, child], _STABILIZED_CHILD_CONFIDENCE
        )
    return boosted


def reject_pose_outliers(
    sequence: NDArray[np.floating],
    confidence: NDArray[np.floating],
    *,
    minimum_length_ratio: float = 0.45,
    maximum_length_ratio: float = 1.8,
) -> NDArray[np.float32]:
    """Reject limb endpoints whose apparent bone length is temporally implausible."""
    coordinates = np.asarray(sequence, dtype=np.float64)
    observed = np.asarray(confidence, dtype=np.float64).copy()
    if coordinates.ndim != 3 or observed.shape != coordinates.shape[:2]:
        raise ValueError("sequence must be (T, J, D) and confidence must be (T, J)")
    if not 0.0 < minimum_length_ratio < 1.0 < maximum_length_ratio:
        raise ValueError("bone-length ratios must straddle one")

    for parent, child in _OUTLIER_PARENT_EDGES:
        lengths = np.linalg.norm(
            coordinates[:, child] - coordinates[:, parent], axis=-1
        )
        valid = (
            (observed[:, parent] > 0)
            & (observed[:, child] > 0)
            & np.isfinite(lengths)
            & (lengths > _EPS)
        )
        baseline = float(np.median(lengths[valid])) if np.any(valid) else 0.0
        if baseline <= _EPS:
            continue
        invalid = valid & (
            (lengths < minimum_length_ratio * baseline)
            | (lengths > maximum_length_ratio * baseline)
        )
        observed[invalid, child] = 0.0
    return observed.astype(np.float32)


def interpolate_pose_sequence(
    sequence: NDArray[np.floating], confidence: NDArray[np.floating]
) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
    """Reject isolated limb failures and interpolate them from adjacent frames."""
    filtered_confidence = reject_pose_outliers(sequence, confidence)
    interpolated = _interpolate_missing(sequence, filtered_confidence)
    return interpolated.astype(np.float32), filtered_confidence


def smooth_pose_sequence(
    sequence: NDArray[np.floating],
    confidence: NDArray[np.floating],
    *,
    window: int = 5,
    sigma: float = 1.0,
) -> NDArray[np.float32]:
    """Reduce per-frame detector jitter with a light, confidence-weighted
    temporal filter.

    A top-down, per-frame pose model (unlike a temporally-consistent 3D
    tracker) has no cross-frame constraint, so single-frame coordinate noise
    can make an otherwise-rigid bone's length swing by tens of percent
    frame-to-frame even though the joint itself barely moved. The Gaussian
    kernel's width (~1 frame std) is narrow relative to a badminton swing's
    multi-frame duration, so it damps single-frame spikes without smearing
    genuine fast motion. Frames get a confidence-weighted blend of their
    temporal neighborhood; a joint with no confident support anywhere in the
    window keeps its original (possibly already-interpolated) value.
    """
    coordinates = np.asarray(sequence, dtype=np.float64)
    weights = np.clip(np.asarray(confidence, dtype=np.float64), 0.0, 1.0)
    frames = len(coordinates)
    half = window // 2
    offsets = np.arange(-half, half + 1)
    kernel = np.exp(-0.5 * (offsets / sigma) ** 2)

    smoothed = np.empty_like(coordinates)
    for frame in range(frames):
        indices = np.clip(frame + offsets, 0, frames - 1)
        local_weights = kernel[:, None] * weights[indices]
        total = local_weights.sum(axis=0)
        has_support = total > _EPS
        safe_total = np.where(has_support, total, 1.0)
        weighted_sum = np.sum(local_weights[:, :, None] * coordinates[indices], axis=0)
        blended = weighted_sum / safe_total[:, None]
        smoothed[frame] = np.where(has_support[:, None], blended, coordinates[frame])
    return smoothed.astype(np.float32)


def _swap_left_right(
    sequence: NDArray[np.float64], confidence: NDArray[np.float64]
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    mirrored = sequence.copy()
    mirrored_confidence = confidence.copy()
    for left, right in _LEFT_RIGHT_PAIRS:
        mirrored[:, [left, right]] = mirrored[:, [right, left]]
        mirrored_confidence[:, [left, right]] = mirrored_confidence[:, [right, left]]
    return mirrored, mirrored_confidence


def stabilize_left_right_joint_labels(
    sequence: NDArray[np.floating],
    confidence: NDArray[np.floating],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Repair isolated whole-body left/right label flips over time.

    Top-down 2D detectors can reverse every paired anatomical label for one
    frame while keeping geometrically plausible coordinates. A temporal
    smoother cannot repair that semantic error and may choose the bad first
    frame as the clip's normalization basis. Dynamic programming chooses the
    original or fully swapped joint assignment per frame from torso/lower-body
    continuity. A small bias toward the detector's original labels resolves
    the otherwise ambiguous all-swapped solution and limits corrections to
    short discontinuous segments.
    """
    coordinates = np.asarray(sequence, dtype=np.float64)
    observed = np.asarray(confidence, dtype=np.float64)
    if coordinates.ndim != 3 or coordinates.shape[1:] != (COCO_JOINT_COUNT, 2):
        raise ValueError("sequence must have shape (T, 17, 2)")
    if observed.shape != coordinates.shape[:2]:
        raise ValueError("confidence must have shape (T, 17)")
    if len(coordinates) < 2:
        return coordinates.copy(), observed.copy()

    swapped, swapped_confidence = _swap_left_right(coordinates, observed)
    candidates = np.stack((coordinates, swapped), axis=1)
    candidate_confidence = np.stack((observed, swapped_confidence), axis=1)
    continuity_joints = np.asarray((5, 6, 11, 12, 13, 14, 15, 16), dtype=np.int64)
    shoulder_width = np.linalg.norm(coordinates[:, 6] - coordinates[:, 5], axis=-1)
    valid_width = shoulder_width[np.isfinite(shoulder_width) & (shoulder_width > _EPS)]
    scale = float(np.median(valid_width)) if len(valid_width) else 1.0
    scale = max(scale, 1.0)
    swap_bias = 0.02
    switch_penalty = 0.05
    frames = len(coordinates)
    cost = np.full((frames, 2), np.inf, dtype=np.float64)
    previous = np.zeros((frames, 2), dtype=np.int8)
    cost[0] = (0.0, swap_bias)
    for frame_index in range(1, frames):
        for state in (0, 1):
            for prior_state in (0, 1):
                weights = np.minimum(
                    candidate_confidence[frame_index, state, continuity_joints],
                    candidate_confidence[
                        frame_index - 1, prior_state, continuity_joints
                    ],
                )
                valid = weights > 0.05
                if np.any(valid):
                    delta = (
                        candidates[frame_index, state, continuity_joints[valid]]
                        - candidates[
                            frame_index - 1,
                            prior_state,
                            continuity_joints[valid],
                        ]
                    )
                    transition = float(
                        np.average(np.sum(delta * delta, axis=-1), weights=weights[valid])
                        / (scale * scale)
                    )
                else:
                    transition = 0.0
                candidate_cost = (
                    cost[frame_index - 1, prior_state]
                    + transition
                    + (swap_bias if state else 0.0)
                    + (switch_penalty if state != prior_state else 0.0)
                )
                if candidate_cost < cost[frame_index, state]:
                    cost[frame_index, state] = candidate_cost
                    previous[frame_index, state] = prior_state

    states = np.zeros(frames, dtype=np.int8)
    states[-1] = int(np.argmin(cost[-1]))
    for frame_index in range(frames - 1, 0, -1):
        states[frame_index - 1] = previous[frame_index, states[frame_index]]
    indices = np.arange(frames)
    return (
        candidates[indices, states].copy(),
        candidate_confidence[indices, states].copy(),
    )


def _body_frame_2d(frame: NDArray[np.float64]) -> tuple[NDArray[np.float64], float]:
    left_shoulder = frame[int(COCOKeypoints.LEFT_SHOULDER)]
    right_shoulder = frame[int(COCOKeypoints.RIGHT_SHOULDER)]
    left_hip = frame[int(COCOKeypoints.LEFT_HIP)]
    right_hip = frame[int(COCOKeypoints.RIGHT_HIP)]
    root = (left_hip + right_hip) / 2.0
    shoulder_midpoint = (left_shoulder + right_shoulder) / 2.0

    shoulder_vector = right_shoulder - left_shoulder
    scale = float(np.linalg.norm(shoulder_vector))
    if scale < _EPS:
        raise ValueError("shoulder width is zero")
    x_axis = shoulder_vector / scale
    spine = shoulder_midpoint - root
    spine -= np.dot(spine, x_axis) * x_axis
    spine_norm = float(np.linalg.norm(spine))
    if spine_norm < _EPS:
        y_axis = np.array((-x_axis[1], x_axis[0]), dtype=np.float64)
    else:
        y_axis = spine / spine_norm
    return np.vstack((x_axis, y_axis)), scale


def _body_scale(
    coordinates: NDArray[np.float64], confidence: NDArray[np.float64]
) -> float:
    starts = _NORMALIZATION_BONES[:, 0]
    ends = _NORMALIZATION_BONES[:, 1]
    lengths = np.linalg.norm(
        coordinates[:, starts] - coordinates[:, ends], axis=-1
    )
    observed = (confidence[:, starts] > 0) & (confidence[:, ends] > 0)
    valid = lengths[observed & np.isfinite(lengths) & (lengths > _EPS)]
    return float(np.median(valid)) if len(valid) else 1.0


def normalize_skeleton_sequence(
    sequence: NDArray[np.floating],
    confidence: NDArray[np.floating],
    handedness: Handedness | str,
) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
    """Interpolate, mirror anatomy, root-center, rotate, and body-scale a pose sequence.

    Right-dominant anatomy always occupies the right COCO joint slots. Confidence
    remains the original observation mask, including across interpolated gaps.
    """
    motion = normalize_skeleton_motion(sequence, confidence, handedness)
    return motion.skeleton, motion.confidence


def normalize_skeleton_motion(
    sequence: NDArray[np.floating],
    confidence: NDArray[np.floating],
    handedness: Handedness | str,
) -> NormalizedSkeletonMotion:
    """Normalize local pose while retaining root motion and camera-frame metadata."""
    stable_sequence, stable_confidence = stabilize_left_right_joint_labels(
        sequence, confidence
    )
    filtered_confidence = reject_pose_outliers(stable_sequence, stable_confidence)
    coordinates = _interpolate_missing(stable_sequence, filtered_confidence)
    observed = np.asarray(filtered_confidence, dtype=np.float64).copy()
    if coordinates.shape[1] != COCO_JOINT_COUNT or coordinates.shape[2] != 2:
        raise ValueError("sequence must have shape (T, 17, 2)")
    # A per-frame 2D detector has no cross-frame consistency constraint, so a
    # bone's detected length can drift tens of percent frame-to-frame from
    # coordinate noise alone; smooth the jitter, then project onto stable
    # per-clip bone lengths so the stored skeleton (and anything trained
    # against it) treats bone length as the anatomical constant it is. This
    # mirrors the same fix in `tracking_to_normalized_sequence`'s live path.
    from badminton_analysis.ml.skeleton_scoring import (
        TORSO_WIDTH_BONES,
        project_stable_bone_lengths,
    )

    coordinates = smooth_pose_sequence(coordinates, observed)
    coordinates = project_stable_bone_lengths(
        coordinates,
        coordinates,
        observed,
        expert_length_bones=TORSO_WIDTH_BONES,
    )
    observed = _raise_confidence_for_bone_stabilized_joints(observed)
    is_left = handedness == Handedness.LEFT or str(handedness).lower() == "left"
    if is_left:
        coordinates, observed = _swap_left_right(coordinates, observed)

    output = np.zeros_like(coordinates)
    roots = np.zeros((len(coordinates), coordinates.shape[2]), dtype=np.float64)
    basis: NDArray[np.float64] | None = None
    for frame in coordinates:
        try:
            basis, _ = _body_frame_2d(frame)
            break
        except ValueError:
            continue
    if basis is None:
        basis = np.eye(coordinates.shape[2], dtype=np.float64)

    scale = _body_scale(coordinates, observed)

    for frame_index, frame in enumerate(coordinates):
        left_hip = frame[int(COCOKeypoints.LEFT_HIP)]
        right_hip = frame[int(COCOKeypoints.RIGHT_HIP)]
        root = (left_hip + right_hip) / 2.0
        roots[frame_index] = root
        output[frame_index] = ((basis @ (frame - root).T).T) / scale
    root_origin = roots[0].copy()
    root_trajectory = ((basis @ (roots - root_origin).T).T) / scale
    return NormalizedSkeletonMotion(
        skeleton=output.astype(np.float32),
        confidence=observed.astype(np.float32),
        root_trajectory=root_trajectory.astype(np.float32),
        body_basis=np.asarray(basis, dtype=np.float32),
        body_scale=scale,
        root_origin=root_origin.astype(np.float32),
    )


def estimate_foot_contacts(
    skeleton: NDArray[np.floating],
    root_trajectory: NDArray[np.floating],
    confidence: NDArray[np.floating],
    *,
    foot_joints: tuple[int, int] = (15, 16),
) -> NDArray[np.float32]:
    """Estimate soft foot contacts from world-relative height and velocity."""
    local = np.asarray(skeleton, dtype=np.float64)
    root = np.asarray(root_trajectory, dtype=np.float64)
    observed = np.asarray(confidence, dtype=np.float64)
    if local.ndim != 3 or local.shape[-2:] != (COCO_JOINT_COUNT, 2):
        raise ValueError("skeleton must have shape (T, 17, 2)")
    if root.shape != (len(local), 2):
        raise ValueError("root_trajectory must have shape (T, 2)")
    if observed.shape != local.shape[:2]:
        raise ValueError("confidence must have shape (T, 17)")

    feet = local[:, foot_joints] + root[:, None, :]
    velocities = np.zeros(feet.shape[:2], dtype=np.float64)
    if len(feet) > 1:
        velocities[1:] = np.linalg.norm(np.diff(feet, axis=0), axis=-1)
        velocities[0] = velocities[1]
    valid = observed[:, foot_joints] > 0.0
    ground = np.zeros(2, dtype=np.float64)
    for index in range(2):
        heights = feet[:, index, 1]
        ground[index] = (
            float(np.quantile(heights[valid[:, index]], 0.05))
            if np.any(valid[:, index])
            else float(np.min(heights))
        )
    height = np.maximum(feet[..., 1] - ground[None, :], 0.0)
    valid_speeds = velocities[valid]
    speed_scale = (
        max(float(np.quantile(valid_speeds, 0.35)), 0.025)
        if len(valid_speeds)
        else 0.025
    )
    speed_score = np.exp(-np.square(velocities / (2.0 * speed_scale)))
    height_score = np.exp(-np.square(height / 0.12))
    return np.asarray(speed_score * height_score * valid, dtype=np.float32)


def resample_sequence(
    sequence: NDArray[np.floating], target_frames: int,
) -> NDArray[np.float32]:
    """Linearly resample the time axis to exactly ``target_frames``."""
    values = np.asarray(sequence, dtype=np.float64)
    if values.ndim < 1 or len(values) == 0:
        raise ValueError("sequence must contain at least one frame")
    if target_frames < 1:
        raise ValueError("target_frames must be positive")
    if len(values) == target_frames:
        return values.astype(np.float32, copy=True)
    source_time = np.linspace(0.0, 1.0, len(values))
    target_time = np.linspace(0.0, 1.0, target_frames)
    flattened = values.reshape(len(values), -1)
    result = np.empty((target_frames, flattened.shape[1]), dtype=np.float64)
    for column in range(flattened.shape[1]):
        result[:, column] = np.interp(target_time, source_time, flattened[:, column])
    return result.reshape((target_frames, *values.shape[1:])).astype(np.float32)


def _sample_sequence(
    sequence: NDArray[np.floating], positions: NDArray[np.floating]
) -> NDArray[np.float32]:
    values = np.asarray(sequence, dtype=np.float64)
    sample_positions = np.asarray(positions, dtype=np.float64)
    if values.ndim < 1 or len(values) < 2:
        raise ValueError("sequence must contain at least two frames")
    if sample_positions.ndim != 1:
        raise ValueError("sample positions must be one-dimensional")
    timeline = np.arange(len(values), dtype=np.float64)
    flattened = values.reshape(len(values), -1)
    sampled = np.empty((len(sample_positions), flattened.shape[1]), dtype=np.float64)
    for column in range(flattened.shape[1]):
        sampled[:, column] = np.interp(
            sample_positions, timeline, flattened[:, column]
        )
    return sampled.reshape((len(sample_positions), *values.shape[1:])).astype(
        np.float32
    )


def phase_align_sequence(
    sequence: NDArray[np.floating],
    phase_indices: NDArray[np.integer],
    *,
    canonical_indices: NDArray[np.integer] = CANONICAL_PHASE_INDICES,
) -> NDArray[np.float32]:
    """Warp five detected stroke phases onto a shared canonical timeline."""
    values = np.asarray(sequence)
    source_phases = np.asarray(phase_indices, dtype=np.float64)
    target_phases = np.asarray(canonical_indices, dtype=np.float64)
    if source_phases.shape != (5,) or target_phases.shape != (5,):
        raise ValueError("phase indices must contain five anchors")
    if len(values) != int(target_phases[-1]) + 1:
        raise ValueError("canonical phase indices must span the sequence")
    if np.any(np.diff(source_phases) <= 0) or np.any(np.diff(target_phases) <= 0):
        raise ValueError("phase indices must be strictly increasing")
    canonical_timeline = np.arange(len(values), dtype=np.float64)
    source_positions = np.interp(
        canonical_timeline, target_phases, source_phases
    )
    return _sample_sequence(values, source_positions)


def acceleration_event_phase_indices(
    sequence: NDArray[np.floating],
    *,
    wrist_index: int = 10,
    anchor_index: int = 6,
    smoothing_window: int = 7,
) -> NDArray[np.int64]:
    """Locate five overhead-stroke anchors from arm acceleration events.

    The returned anchors are the clip start, the midpoint before maximum
    directional acceleration, that acceleration event, the midpoint after it,
    and clip end.  The trajectory is the dominant wrist relative to the
    dominant shoulder, so
    camera translation and weight transfer do not masquerade as racket-arm
    acceleration.  This is a fixed event warp; it does not use DTW or compare
    the student to an expert while choosing time correspondences.
    """
    values = np.asarray(sequence, dtype=np.float64)
    if values.ndim != 3 or values.shape[1:] != (17, 2):
        raise ValueError("sequence must have shape (T, 17, 2)")
    frames = len(values)
    if frames < 9:
        raise ValueError("acceleration alignment requires at least nine frames")
    if not 0 <= wrist_index < 17 or not 0 <= anchor_index < 17:
        raise ValueError("wrist and anchor indices must be COCO-17 joints")
    if smoothing_window < 1 or smoothing_window % 2 == 0:
        raise ValueError("smoothing_window must be a positive odd integer")

    trajectory = values[:, wrist_index] - values[:, anchor_index]
    half = smoothing_window // 2
    kernel = np.full(smoothing_window, 1.0 / smoothing_window)
    smoothed = np.column_stack(
        [
            np.convolve(
                np.pad(trajectory[:, dimension], (half, half), mode="edge"),
                kernel,
                mode="valid",
            )
            for dimension in range(2)
        ]
    )
    velocity = np.diff(smoothed, axis=0)
    speed = np.linalg.norm(velocity, axis=-1)
    if not np.any(speed > 1e-8):
        return np.rint(np.linspace(0, frames - 1, 5)).astype(np.int64)

    # Identify a coherent directional episode before differentiating. A plain
    # global speed maximum frequently selects the faster recovery swing after
    # contact (clear CG1 was one concrete failure), which moves the expert's
    # impact pose into the student's ending phase.
    episode_window = min(7, len(velocity))
    if episode_window % 2 == 0:
        episode_window -= 1
    episode_window = max(episode_window, 1)
    episode_half = episode_window // 2
    episode_kernel = np.full(episode_window, 1.0 / episode_window)
    coherent_velocity = np.column_stack(
        [
            np.convolve(
                np.pad(
                    velocity[:, dimension],
                    (episode_half, episode_half),
                    mode="edge",
                ),
                episode_kernel,
                mode="valid",
            )
            for dimension in range(2)
        ]
    )
    mean_speed = np.convolve(
        np.pad(speed, (episode_half, episode_half), mode="edge"),
        episode_kernel,
        mode="valid",
    )
    coherent_speed = np.linalg.norm(coherent_velocity, axis=-1)
    consistency = coherent_speed / np.maximum(mean_speed, 1e-8)
    episode_score = (
        coherent_speed
        * np.square(consistency)
        * np.linspace(1.0, 0.65, len(coherent_speed))
    )
    if episode_half and len(episode_score) > 2 * episode_half:
        episode_score[:episode_half] = 0.0
        episode_score[-episode_half:] = 0.0
    episode_index = int(np.argmax(episode_score))
    axis = coherent_velocity[episode_index]
    axis_norm = float(np.linalg.norm(axis))
    if axis_norm <= 1e-8:
        episode_index = int(np.argmax(speed))
        axis = velocity[episode_index]
        axis_norm = float(np.linalg.norm(axis))
    forward_axis = axis / max(axis_norm, 1e-8)
    projected_velocity = velocity @ forward_axis
    directional_acceleration = np.diff(projected_velocity)
    direction_cosine = projected_velocity / np.maximum(speed, 1e-8)
    search_start = max(0, episode_index - episode_window)
    search_end = min(
        len(directional_acceleration), episode_index + episode_half + 1
    )
    score = (
        np.maximum(
            directional_acceleration[search_start:search_end], 0.0
        )
        * np.clip(direction_cosine[search_start + 1 : search_end + 1], 0.0, 1.0)
        * (projected_velocity[search_start + 1 : search_end + 1] > 0.0)
    )
    acceleration = (
        search_start + int(np.argmax(score)) + 2
        if score.size and np.any(score > 0.0)
        else int(np.argmax(projected_velocity)) + 1
    )

    # Only the acceleration peak is a detected temporal correspondence. The
    # two intermediate anchors are linear half-way points on either side, so
    # no one-frame detector event is expanded into an entire canonical phase.
    # This is deliberately simpler than DTW and robust to recovery speed.
    edge_margin = max(4, frames // 8)
    acceleration = int(
        np.clip(acceleration, edge_margin, frames - edge_margin - 1)
    )
    anchors = np.asarray(
        (
            0,
            acceleration // 2,
            acceleration,
            (acceleration + frames - 1) // 2,
            frames - 1,
        ),
        dtype=np.int64,
    )
    # Detector noise can put the strongest positive/negative derivative on an
    # adjacent sample. Preserve the event order without inventing a DTW path.
    for index in range(1, len(anchors)):
        anchors[index] = max(anchors[index], anchors[index - 1] + 1)
    for index in range(len(anchors) - 2, -1, -1):
        anchors[index] = min(anchors[index], anchors[index + 1] - 1)
    if anchors[0] != 0 or anchors[-1] != frames - 1 or np.any(np.diff(anchors) <= 0):
        raise RuntimeError("could not construct ordered acceleration anchors")
    return anchors


def overhead_acceleration_phase_indices(
    sequence: NDArray[np.floating],
    *,
    smoothing_window: int = 5,
    acceleration_index: int | None = None,
    dominant_shoulder_x: NDArray[np.floating] | None = None,
) -> NDArray[np.int64]:
    """End an overhead clip at maximum forward dominant-shoulder position.

    The forward-swing anchor is the directional wrist-acceleration event used
    by :func:`acceleration_event_phase_indices`. Completion is the post-event
    frame whose dominant shoulder (COCO 6) has the largest source-image x
    coordinate. Callers that retain the raw detector coordinates must pass
    that trajectory through ``dominant_shoulder_x``; the root-relative
    canonical coordinate is only a fallback for legacy callers.
    """
    values = np.asarray(sequence, dtype=np.float64)
    if values.ndim != 3 or values.shape[1:] != (17, 2):
        raise ValueError("sequence must have shape (T, 17, 2)")
    if smoothing_window < 1 or smoothing_window % 2 == 0:
        raise ValueError("smoothing_window must be a positive odd integer")
    frames = len(values)
    if acceleration_index is None:
        acceleration = int(acceleration_event_phase_indices(values)[2])
    else:
        acceleration = int(acceleration_index)
        if not 1 <= acceleration < frames - 2:
            raise ValueError("acceleration_index must be inside the sequence")
    # Two frames are required so the midpoint anchor remains strictly between
    # acceleration and completion.
    search_start = min(frames - 1, acceleration + 2)
    if dominant_shoulder_x is None:
        pelvis_x = 0.5 * (values[:, 11, 0] + values[:, 12, 0])
        shoulder_x = values[:, 6, 0] - pelvis_x
    else:
        shoulder_x = np.asarray(dominant_shoulder_x, dtype=np.float64)
        if shoulder_x.shape != (frames,):
            raise ValueError("dominant_shoulder_x must have shape (T,)")
    window = min(smoothing_window, frames)
    if window % 2 == 0:
        window -= 1
    if window > 1:
        padding = window // 2
        shoulder_x = np.convolve(
            np.pad(shoulder_x, (padding, padding), mode="edge"),
            np.ones(window, dtype=np.float64) / window,
            mode="valid",
        )
    candidates = shoulder_x[search_start:]
    completion = (
        frames - 1
        if not np.any(np.isfinite(candidates))
        else search_start + int(np.nanargmax(candidates))
    )
    output = np.asarray(
        (
            0,
            acceleration // 2,
            acceleration,
            (acceleration + completion) // 2,
            completion,
        ),
        dtype=np.int64,
    )
    if np.any(np.diff(output) <= 0):
        raise RuntimeError("could not construct ordered overhead anchors")
    return output


def restore_phase_timing(
    aligned_sequence: NDArray[np.floating],
    phase_indices: NDArray[np.integer],
    *,
    canonical_indices: NDArray[np.integer] = CANONICAL_PHASE_INDICES,
) -> NDArray[np.float32]:
    """Restore a canonical sequence to the source clip's phase timing."""
    values = np.asarray(aligned_sequence)
    target_phases = np.asarray(phase_indices, dtype=np.float64)
    source_phases = np.asarray(canonical_indices, dtype=np.float64)
    if target_phases.shape != (5,) or source_phases.shape != (5,):
        raise ValueError("phase indices must contain five anchors")
    if len(values) != int(source_phases[-1]) + 1:
        raise ValueError("canonical phase indices must span the sequence")
    if np.any(np.diff(target_phases) <= 0) or np.any(np.diff(source_phases) <= 0):
        raise ValueError("phase indices must be strictly increasing")
    target_timeline = np.arange(len(values), dtype=np.float64)
    aligned_positions = np.interp(
        target_timeline, target_phases, source_phases
    )
    return _sample_sequence(values, aligned_positions)


def _dtw_segment_positions(
    aligned_source: NDArray[np.floating],
    source: NDArray[np.floating],
    joint_weights: NDArray[np.floating],
) -> NDArray[np.float64]:
    aligned = np.asarray(aligned_source, dtype=np.float64)
    target = np.asarray(source, dtype=np.float64)
    weights = np.asarray(joint_weights, dtype=np.float64)
    if aligned.ndim != 3 or target.ndim != 3 or aligned.shape[1:] != target.shape[1:]:
        raise ValueError("DTW inputs must have compatible shapes (T, J, D)")
    if weights.shape != (aligned.shape[1],):
        raise ValueError("joint_weights must contain one value per joint")

    differences = aligned[:, None] - target[None, :]
    costs = np.sqrt(
        np.sum(np.square(differences) * weights[None, None, :, None], axis=(2, 3))
        / max(float(np.sum(weights)), _EPS)
    )
    rows, columns = costs.shape
    accumulated = np.full((rows, columns), np.inf, dtype=np.float64)
    accumulated[0, 0] = costs[0, 0]
    for row in range(rows):
        for column in range(columns):
            if row == 0 and column == 0:
                continue
            predecessors = []
            if row > 0 and column > 0:
                predecessors.append(accumulated[row - 1, column - 1])
            if row > 0:
                predecessors.append(accumulated[row - 1, column])
            if column > 0:
                predecessors.append(accumulated[row, column - 1])
            accumulated[row, column] = costs[row, column] + min(predecessors)

    path: list[tuple[int, int]] = []
    row, column = rows - 1, columns - 1
    while True:
        path.append((row, column))
        if row == 0 and column == 0:
            break
        candidates: list[tuple[float, int, int]] = []
        if row > 0 and column > 0:
            candidates.append((accumulated[row - 1, column - 1], row - 1, column - 1))
        if row > 0:
            candidates.append((accumulated[row - 1, column], row - 1, column))
        if column > 0:
            candidates.append((accumulated[row, column - 1], row, column - 1))
        _, row, column = min(candidates, key=lambda value: value[0])

    mapped: list[list[int]] = [[] for _ in range(columns)]
    for aligned_index, source_index in reversed(path):
        mapped[source_index].append(aligned_index)
    positions = np.asarray(
        [float(np.mean(indices)) for indices in mapped], dtype=np.float64
    )
    positions[0] = 0.0
    positions[-1] = float(rows - 1)
    return np.maximum.accumulate(positions)


def restore_phase_timing_dtw(
    aligned_sequence: NDArray[np.floating],
    aligned_source: NDArray[np.floating],
    source_sequence: NDArray[np.floating],
    phase_indices: NDArray[np.integer],
    *,
    joint_weights: NDArray[np.floating] | None = None,
    canonical_indices: NDArray[np.integer] = CANONICAL_PHASE_INDICES,
) -> NDArray[np.float32]:
    """Restore timing with DTW inside each detected phase interval."""
    values = np.asarray(aligned_sequence)
    aligned = np.asarray(aligned_source)
    source = np.asarray(source_sequence)
    target_phases = np.asarray(phase_indices, dtype=np.int64)
    canonical = np.asarray(canonical_indices, dtype=np.int64)
    if values.shape != aligned.shape or aligned.shape != source.shape:
        raise ValueError("aligned, source, and corrected sequences must have equal shapes")
    if target_phases.shape != (5,) or canonical.shape != (5,):
        raise ValueError("phase indices must contain five anchors")
    if np.any(np.diff(target_phases) <= 0) or np.any(np.diff(canonical) <= 0):
        raise ValueError("phase indices must be strictly increasing")
    weights = (
        np.ones(values.shape[1], dtype=np.float64)
        if joint_weights is None
        else np.asarray(joint_weights, dtype=np.float64)
    )
    sample_positions = np.zeros(len(values), dtype=np.float64)
    for segment in range(4):
        aligned_start, aligned_end = canonical[segment : segment + 2]
        source_start, source_end = target_phases[segment : segment + 2]
        local = _dtw_segment_positions(
            aligned[aligned_start : aligned_end + 1],
            source[source_start : source_end + 1],
            weights,
        )
        sample_positions[source_start : source_end + 1] = aligned_start + local
    sample_positions[target_phases] = canonical.astype(np.float64)
    sample_positions = np.maximum.accumulate(sample_positions)
    return _sample_sequence(values, sample_positions)


def resample_phase_indices(
    analysis_window: tuple[int, int, int], target_frames: int
) -> NDArray[np.int64]:
    start, peak, end = analysis_window
    if end <= start:
        raise ValueError("analysis window end must be after start")
    raw = np.asarray(
        (start, (start + peak) // 2, peak, (peak + end) // 2, end),
        dtype=np.float64,
    )
    mapped = np.rint((raw - start) * (target_frames - 1) / (end - start))
    return np.clip(mapped, 0, target_frames - 1).astype(np.int64)


def resample_detected_phase_indices(
    phase_frames: tuple[int, int, int, int, int], target_frames: int
) -> NDArray[np.int64]:
    """Map five detected source-frame anchors into a resampled sequence."""
    raw = np.asarray(phase_frames, dtype=np.float64)
    if target_frames < 5 or raw.shape != (5,) or np.any(np.diff(raw) <= 0):
        raise ValueError("five ordered phase frames and at least five targets are required")
    mapped = np.rint(
        (raw - raw[0]) * (target_frames - 1) / (raw[-1] - raw[0])
    )
    result: NDArray[np.int64] = np.asarray(
        np.clip(mapped, 0, target_frames - 1), dtype=np.int64
    )
    if np.any(np.diff(result) <= 0):
        raise ValueError("detected phases collapse after sequence resampling")
    return result


def refine_delayed_overhead_contact_phase_indices(
    sequence: NDArray[np.floating],
    phase_indices: NDArray[np.integer],
    *,
    minimum_delay_frames: int = 8,
    speed_ratio_threshold: float = 3.0,
) -> NDArray[np.int64]:
    """Move a false preparation anchor to a later overhead hitting event.

    A player can raise and then hold the racket overhead while waiting for a
    self-fed shuttle.  The broad overhead detector historically treated that
    first wrist-height maximum as contact even when the actual racket swing
    happened much later.  Phase-aligning an expert to that false event makes
    the corrected skeleton hit a shuttle that has not arrived yet.

    Detect only the unambiguous version of this failure: after the stored
    contact there must be a second dominant-wrist/shoulder-relative motion at
    least ``minimum_delay_frames`` later and more than
    ``speed_ratio_threshold`` times faster than motion at the old anchor.  A
    high threshold intentionally leaves ordinary 1--3 frame detector offsets
    unchanged.  The revised contact is one sample before the centred
    acceleration maximum, compensating for the derivative's one-frame lag.
    Preparation and follow-through anchors are then rebuilt as midpoints.
    """
    values = np.asarray(sequence, dtype=np.float64)
    phases = np.asarray(phase_indices, dtype=np.int64)
    if values.ndim != 3 or values.shape[1:] != (17, 2):
        raise ValueError("sequence must have shape (T, 17, 2)")
    if phases.shape != (5,) or np.any(np.diff(phases) <= 0):
        raise ValueError("phase_indices must contain five increasing frames")
    if phases[0] < 0 or phases[-1] >= len(values):
        raise ValueError("phase_indices are outside the sequence")
    if minimum_delay_frames < 2:
        raise ValueError("minimum_delay_frames must be at least two")
    if speed_ratio_threshold <= 1.0:
        raise ValueError("speed_ratio_threshold must be greater than one")

    contact = int(phases[2])
    completion = int(phases[4])
    relative_wrist = values[:, 10] - values[:, 6]
    smoothed = np.empty_like(relative_wrist)
    kernel = np.ones(3, dtype=np.float64) / 3.0
    for dimension in range(2):
        smoothed[:, dimension] = np.convolve(
            np.pad(relative_wrist[:, dimension], (1, 1), mode="edge"),
            kernel,
            mode="valid",
        )
    velocity = np.gradient(smoothed, axis=0)
    speed = np.linalg.norm(velocity, axis=1)
    acceleration = np.linalg.norm(np.gradient(velocity, axis=0), axis=1)

    search_start = contact + minimum_delay_frames
    search_stop = completion
    if search_stop - search_start < 3:
        return phases.copy()
    later_speed_index = search_start + int(
        np.argmax(speed[search_start:search_stop])
    )
    old_speed = float(
        np.median(speed[max(0, contact - 1) : min(len(speed), contact + 2)])
    )
    later_speed = float(speed[later_speed_index])
    if not np.isfinite(old_speed) or not np.isfinite(later_speed):
        return phases.copy()
    if later_speed <= speed_ratio_threshold * max(old_speed, _EPS):
        return phases.copy()

    acceleration_stop = min(search_stop, later_speed_index + 2)
    acceleration_index = search_start + int(
        np.argmax(acceleration[search_start:acceleration_stop])
    )
    revised_contact = max(search_start, acceleration_index - 1)
    if revised_contact >= completion - 2:
        return phases.copy()
    revised = np.asarray(
        (
            int(phases[0]),
            (int(phases[0]) + revised_contact) // 2,
            revised_contact,
            (revised_contact + completion) // 2,
            completion,
        ),
        dtype=np.int64,
    )
    return revised if np.all(np.diff(revised) > 0) else phases.copy()
