from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


# COCO has no explicit pelvis joint. ``-1`` means the midpoint of hips 11/12.
COCO_PARENTS = (-1, 0, 0, 1, 2, -1, -1, 5, 6, 7, 8, -1, -1, 11, 12, 13, 14)
COCO_CHAINS = (
    (0, 1, 3),
    (0, 2, 4),
    (5, 7, 9),
    (6, 8, 10),
    (11, 13, 15),
    (12, 14, 16),
)

_EPS = 1e-8




def implicit_pelvis(sequence: NDArray[np.floating]) -> NDArray[np.float64]:
    values = np.asarray(sequence, dtype=np.float64)
    if values.ndim != 3 or values.shape[1] != 17 or values.shape[2] not in (2, 3):
        raise ValueError("sequence must have shape (T, 17, 2|3)")
    return 0.5 * (values[:, 11] + values[:, 12])


def parent_offsets(sequence: NDArray[np.floating]) -> NDArray[np.float64]:
    """Return every COCO joint relative to its kinematic parent."""
    values = np.asarray(sequence, dtype=np.float64)
    pelvis = implicit_pelvis(values)
    offsets = np.empty_like(values)
    for joint, parent in enumerate(COCO_PARENTS):
        anchor = pelvis if parent < 0 else values[:, parent]
        offsets[:, joint] = values[:, joint] - anchor
    return offsets


def _weighted_median(values: NDArray[np.float64], weights: NDArray[np.float64]) -> float:
    valid = np.isfinite(values) & np.isfinite(weights) & (weights > 0.0)
    if not np.any(valid):
        return float("nan")
    selected_values = values[valid]
    selected_weights = weights[valid]
    order = np.argsort(selected_values)
    selected_values = selected_values[order]
    selected_weights = selected_weights[order]
    midpoint = 0.5 * float(np.sum(selected_weights))
    index = int(np.searchsorted(np.cumsum(selected_weights), midpoint, side="left"))
    return float(selected_values[min(index, len(selected_values) - 1)])


def stable_parent_lengths(
    sequence: NDArray[np.floating],
    confidence: NDArray[np.floating],
) -> NDArray[np.float64]:
    """Estimate one confidence-weighted length per kinematic edge."""
    values = np.asarray(sequence, dtype=np.float64)
    observed = np.asarray(confidence, dtype=np.float64)
    offsets = parent_offsets(values)
    if observed.shape != values.shape[:2]:
        raise ValueError("confidence must have shape (T, 17)")
    lengths = np.linalg.norm(offsets, axis=-1)
    pelvis_confidence = np.minimum(observed[:, 11], observed[:, 12])
    output = np.empty(17, dtype=np.float64)
    for joint, parent in enumerate(COCO_PARENTS):
        parent_confidence = pelvis_confidence if parent < 0 else observed[:, parent]
        weights = np.clip(observed[:, joint] * parent_confidence, 0.0, 1.0)
        output[joint] = _weighted_median(lengths[:, joint], weights)
        if not np.isfinite(output[joint]) or output[joint] <= _EPS:
            valid = lengths[:, joint][np.isfinite(lengths[:, joint])]
            output[joint] = float(np.median(valid)) if len(valid) else 1.0
    # The implicit pelvis is the exact midpoint of the two hips, so both hip
    # half-widths must be identical. Detector asymmetry must not turn that
    # identity into two incompatible FK constraints.
    hip_half_width = _weighted_median(
        np.concatenate((lengths[:, 11], lengths[:, 12])),
        np.concatenate((pelvis_confidence * observed[:, 11], pelvis_confidence * observed[:, 12])),
    )
    if np.isfinite(hip_half_width) and hip_half_width > _EPS:
        output[11] = output[12] = hip_half_width
    return output


def _fill_directions(
    directions: NDArray[np.float64],
    valid: NDArray[np.bool_],
    fallback: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Temporally fill detector gaps without inventing zero-length bones."""
    output = directions.copy()
    frames, joints = valid.shape
    for joint in range(joints):
        indices = np.flatnonzero(valid[:, joint])
        if not len(indices):
            output[:, joint] = fallback[:, joint]
            continue
        for axis in range(output.shape[-1]):
            output[:, joint, axis] = np.interp(
                np.arange(frames), indices, output[indices, joint, axis]
            )
    norms = np.linalg.norm(output, axis=-1, keepdims=True)
    fallback_norm = np.linalg.norm(fallback, axis=-1, keepdims=True)
    fallback_unit = np.divide(
        fallback,
        fallback_norm,
        out=np.zeros_like(fallback),
        where=fallback_norm > _EPS,
    )
    return np.divide(
        output,
        norms,
        out=fallback_unit,
        where=norms > _EPS,
    )


def _unit(values: NDArray[np.float64]) -> NDArray[np.float64]:
    norms = np.linalg.norm(values, axis=-1, keepdims=True)
    return np.divide(
        values,
        norms,
        out=np.zeros_like(values),
        where=norms > _EPS,
    )




def body_frame_axes(sequence: NDArray[np.floating]) -> NDArray[np.float64]:
    """Return rigid lateral/up axes from hips and shoulders for every frame."""
    values = np.asarray(sequence, dtype=np.float64)
    if values.ndim != 3 or values.shape[1] != 17 or values.shape[2] not in (2, 3):
        raise ValueError("sequence must have shape (T, 17, 2|3)")
    pelvis = implicit_pelvis(values)
    shoulder_center = 0.5 * (values[:, 5] + values[:, 6])
    lateral = _unit(values[:, 6] - values[:, 5]) + _unit(
        values[:, 12] - values[:, 11]
    )
    up = _unit(shoulder_center - pelvis)
    lateral -= np.sum(lateral * up, axis=-1, keepdims=True) * up
    lateral = _unit(lateral)
    if values.shape[-1] == 2:
        perpendicular = np.stack((up[:, 1], -up[:, 0]), axis=-1)
        # Choose chirality once, then unwrap it through time. Re-deciding the
        # sign independently in every frame lets one detector-side
        # shoulder/hip swap rotate the whole FK body by 180 degrees. In a 2D
        # correction that is never an observable weight-transfer cue; it is a
        # broken crossed skeleton. Temporal sign continuity preserves the
        # initial physical side assignment without smoothing any joint angle.
        initial_sign = (
            1.0
            if float(np.dot(perpendicular[0], lateral[0])) >= 0.0
            else -1.0
        )
        lateral = perpendicular.copy()
        lateral[0] *= initial_sign
        for frame in range(1, len(lateral)):
            if float(np.dot(lateral[frame], lateral[frame - 1])) < 0.0:
                lateral[frame] *= -1.0
        return np.stack((lateral, up), axis=-1)
    depth = _unit(np.cross(lateral, up))
    lateral = _unit(np.cross(up, depth))
    return np.stack((lateral, up, depth), axis=-1)


def _stable_torso_dimensions(
    sequence: NDArray[np.floating], confidence: NDArray[np.floating]
) -> tuple[float, float, float]:
    values = np.asarray(sequence, dtype=np.float64)
    observed = np.asarray(confidence, dtype=np.float64)
    shoulder_width = np.linalg.norm(values[:, 6] - values[:, 5], axis=-1)
    hip_width = np.linalg.norm(values[:, 12] - values[:, 11], axis=-1)
    torso_height = np.linalg.norm(
        0.5 * (values[:, 5] + values[:, 6]) - implicit_pelvis(values), axis=-1
    )
    shoulder_confidence = np.minimum(observed[:, 5], observed[:, 6])
    hip_confidence = np.minimum(observed[:, 11], observed[:, 12])
    torso_confidence = shoulder_confidence * hip_confidence
    dimensions = (
        _weighted_median(shoulder_width, shoulder_confidence),
        _weighted_median(hip_width, hip_confidence),
        _weighted_median(torso_height, torso_confidence),
    )
    if not all(np.isfinite(value) and value > _EPS for value in dimensions):
        raise ValueError("could not estimate stable student torso dimensions")
    return dimensions


def relative_projected_width_trajectory(
    sequence: NDArray[np.floating],
    confidence: NDArray[np.floating],
    *,
    left_joint: int = 5,
    right_joint: int = 6,
) -> NDArray[np.float64]:
    """Return a finite projected-width trajectory with a weighted median of 1.

    Unlike limb lengths, the distance between the two shoulder detections is
    not a rigid 2D bone. It narrows when the torso turns away from the camera
    and widens when the shoulders face the camera. Retargeting therefore must
    preserve the expert's *relative* projected-width change while scaling its
    median to the student's anatomy.
    """
    values = np.asarray(sequence, dtype=np.float64)
    observed = np.asarray(confidence, dtype=np.float64)
    if values.ndim != 3 or values.shape[1:] != (17, 2):
        raise ValueError("sequence must have shape (T, 17, 2)")
    if observed.shape != values.shape[:2]:
        raise ValueError("confidence must have shape (T, 17)")
    widths = np.linalg.norm(
        values[:, right_joint] - values[:, left_joint], axis=-1
    )
    weights = np.clip(
        observed[:, left_joint] * observed[:, right_joint], 0.0, 1.0
    )
    stable = _weighted_median(widths, weights)
    if not np.isfinite(stable) or stable <= _EPS:
        raise ValueError("could not estimate stable projected width")
    valid = np.isfinite(widths) & (widths > _EPS)
    if not np.any(valid):
        raise ValueError("projected width trajectory contains no finite values")
    filled = np.interp(np.arange(len(widths)), np.flatnonzero(valid), widths[valid])
    return filled / stable




def retarget_expert_body_local_rotations_fk(
    student: NDArray[np.floating],
    expert: NDArray[np.floating],
    student_confidence: NDArray[np.floating],
    expert_confidence: NDArray[np.floating],
    *,
    root_trajectory: NDArray[np.floating] | None = None,
) -> NDArray[np.float32]:
    """Transfer expert rotations in a torso-local frame, then run FK.

    Coordinates from two independently filmed clips do not share a camera
    frame. Copying raw bone vectors therefore transfers camera roll along with
    technique. This representation removes each expert frame's torso
    orientation, retains every expert joint direction relative to that torso,
    and composes the expert's *relative* torso turn onto the student's initial
    torso frame. It works directly in 2D as well as in lifted 3D; callers that
    request a 2D pipeline never need to create or consume 3D coordinates.
    """
    source = np.asarray(student, dtype=np.float64)
    reference = np.asarray(expert, dtype=np.float64)
    source_observed = np.asarray(student_confidence, dtype=np.float64)
    reference_observed = np.asarray(expert_confidence, dtype=np.float64)
    if source.shape != reference.shape or source.ndim != 3:
        raise ValueError("student and expert must have matching shape (T, 17, D)")
    if source.shape[1] != 17 or source.shape[2] not in (2, 3):
        raise ValueError(
            "body-local rotation transfer requires shape (T, 17, 2|3)"
        )
    if source_observed.shape != source.shape[:2]:
        raise ValueError("student_confidence must have shape (T, 17)")
    if reference_observed.shape != source.shape[:2]:
        raise ValueError("expert_confidence must have shape (T, 17)")

    source_offsets = parent_offsets(source)
    expert_offsets = parent_offsets(reference)
    lengths = stable_parent_lengths(source, source_observed)
    expert_norms = np.linalg.norm(expert_offsets, axis=-1, keepdims=True)
    source_norms = np.linalg.norm(source_offsets, axis=-1, keepdims=True)
    fallback = np.divide(
        source_offsets,
        source_norms,
        out=np.zeros_like(source_offsets),
        where=source_norms > _EPS,
    )
    valid = np.isfinite(expert_offsets).all(axis=-1) & (
        expert_norms[..., 0] > _EPS
    )
    raw_directions = np.divide(
        expert_offsets,
        expert_norms,
        out=np.zeros_like(expert_offsets),
        where=expert_norms > _EPS,
    )
    directions = _fill_directions(raw_directions, valid, fallback)

    expert_axes = body_frame_axes(reference)
    student_axes = body_frame_axes(source)
    # Columns are lateral/up(/depth) basis vectors in camera coordinates. Express each
    # expert edge in that basis, then map it into a target basis whose initial
    # orientation is the student's and whose temporal turn is the expert's.
    expert_relative_turn = np.einsum(
        "ij,tjk->tik", expert_axes[0].T, expert_axes
    )
    target_axes = np.einsum(
        "ij,tjk->tik", student_axes[0], expert_relative_turn
    )
    local_directions = np.einsum("tjd,tdk->tjk", directions, expert_axes)
    directions = np.einsum("tjk,tdk->tjd", local_directions, target_axes)
    directions = _unit(directions)

    if root_trajectory is None:
        root = np.zeros((len(source), source.shape[-1]), dtype=np.float64)
    else:
        root = np.asarray(root_trajectory, dtype=np.float64)
        if root.shape != (len(source), source.shape[-1]):
            raise ValueError("root_trajectory must have shape (T, D)")

    shoulder_width, hip_width, torso_height = _stable_torso_dimensions(
        source, source_observed
    )
    lateral = target_axes[..., 0]
    up = target_axes[..., 1]
    shoulder_center = root + torso_height * up
    output = np.empty_like(source)
    output[:, 11] = root - 0.5 * hip_width * lateral
    output[:, 12] = root + 0.5 * hip_width * lateral
    output[:, 5] = shoulder_center - 0.5 * shoulder_width * lateral
    output[:, 6] = shoulder_center + 0.5 * shoulder_width * lateral
    rigid_joints = {5, 6, 11, 12}
    for joint, parent in enumerate(COCO_PARENTS):
        if joint in rigid_joints:
            continue
        anchor = root if parent < 0 else output[:, parent]
        output[:, joint] = anchor + lengths[joint] * directions[:, joint]
    return output.astype(np.float32)


def retarget_expert_canonical_2d_fk(
    student: NDArray[np.floating],
    expert: NDArray[np.floating],
    student_confidence: NDArray[np.floating],
    expert_confidence: NDArray[np.floating],
    *,
    root_trajectory: NDArray[np.floating] | None = None,
) -> NDArray[np.float32]:
    """Transfer the complete expert motion in the shared canonical 2D frame.

    Skeleton archives are already root-centred, body-scale normalized, and
    rotated into the initial body frame by ``normalize_skeleton_motion``.
    Re-basing the expert onto the student's initial torso a second time (as
    ``retarget_expert_body_local_rotations_fk`` does) preserves the student's
    global shoulder orientation and defeats visible shoulder-turn correction.
    This variant copies every finite expert edge direction in the existing
    shared canonical frame and changes only bone lengths to the student's
    stable anatomy.  Screen translation remains an explicit, separate input.
    """
    source = np.asarray(student, dtype=np.float64)
    reference = np.asarray(expert, dtype=np.float64)
    source_observed = np.asarray(student_confidence, dtype=np.float64)
    reference_observed = np.asarray(expert_confidence, dtype=np.float64)
    if source.ndim != 3 or source.shape[1:] != (17, 2):
        raise ValueError("student must have shape (T, 17, 2)")
    if reference.shape != source.shape:
        raise ValueError("expert must match student shape")
    if source_observed.shape != source.shape[:2]:
        raise ValueError("student_confidence must have shape (T, 17)")
    if reference_observed.shape != source.shape[:2]:
        raise ValueError("expert_confidence must have shape (T, 17)")

    source_offsets = parent_offsets(source)
    expert_offsets = parent_offsets(reference)
    lengths = stable_parent_lengths(source, source_observed)
    source_norms = np.linalg.norm(source_offsets, axis=-1, keepdims=True)
    expert_norms = np.linalg.norm(expert_offsets, axis=-1, keepdims=True)
    fallback = np.divide(
        source_offsets,
        source_norms,
        out=np.zeros_like(source_offsets),
        where=source_norms > _EPS,
    )
    valid = np.isfinite(expert_offsets).all(axis=-1) & (
        expert_norms[..., 0] > _EPS
    )
    raw_directions = np.divide(
        expert_offsets,
        expert_norms,
        out=np.zeros_like(expert_offsets),
        where=expert_norms > _EPS,
    )
    directions = _fill_directions(raw_directions, valid, fallback)

    if root_trajectory is None:
        root = np.zeros((len(source), 2), dtype=np.float64)
    else:
        root = np.asarray(root_trajectory, dtype=np.float64)
        if root.shape != (len(source), 2):
            raise ValueError("root_trajectory must have shape (T, 2)")

    _, hip_width, torso_height = _stable_torso_dimensions(
        source, source_observed
    )
    shoulder_width_profile = relative_projected_width_trajectory(
        reference, reference_observed
    )
    # The correction should begin from the student's observed preparation,
    # not widen or narrow their shoulders merely because the selected expert
    # has a different first-frame projection. Preserve the expert's relative
    # torso-turn trajectory, but anchor its scale to the student's actual
    # first-frame shoulder span. Median anchoring previously enlarged smash
    # CG1 from 0.862 to 1.323 before the motion had even started.
    shoulder_width_profile /= max(float(shoulder_width_profile[0]), _EPS)
    initial_shoulder_width = float(
        np.linalg.norm(source[0, 6] - source[0, 5])
    )
    expert_pelvis = implicit_pelvis(reference)
    expert_shoulder_center = 0.5 * (reference[:, 5] + reference[:, 6])
    shoulder_lateral = _unit(reference[:, 6] - reference[:, 5])
    hip_lateral = _unit(reference[:, 12] - reference[:, 11])
    torso_direction = _unit(expert_shoulder_center - expert_pelvis)
    shoulder_center = root + torso_height * torso_direction

    output = np.empty_like(source)
    output[:, 11] = root - 0.5 * hip_width * hip_lateral
    output[:, 12] = root + 0.5 * hip_width * hip_lateral
    projected_shoulder_width = initial_shoulder_width * shoulder_width_profile
    output[:, 5] = (
        shoulder_center
        - 0.5 * projected_shoulder_width[:, None] * shoulder_lateral
    )
    output[:, 6] = (
        shoulder_center
        + 0.5 * projected_shoulder_width[:, None] * shoulder_lateral
    )
    rigid_joints = {5, 6, 11, 12}
    for joint, parent in enumerate(COCO_PARENTS):
        if joint in rigid_joints:
            continue
        anchor = root if parent < 0 else output[:, parent]
        output[:, joint] = anchor + lengths[joint] * directions[:, joint]
    return output.astype(np.float32)








