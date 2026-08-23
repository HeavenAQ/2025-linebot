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


def canonical_overhead_exemplar_quality_options(
    skill: str | None,
) -> dict[str, float]:
    """Return technique-specific balance limits without crossing rules.

    Clear has a longer, more inclined recovery than smash in the same-view
    expert set. The shoulder-x endpoint occurs earlier than neutral recovery;
    the corresponding coach-selected smash pose can reach roughly 14 degrees
    of torso lean, so a 10-degree legacy recovery limit would reject it.
    """
    return {
        "maximum_completion_torso_lean_degrees": (
            25.0 if str(skill).lower() == "clear" else 20.0
        ),
        # The verified smash exemplars reach contact with a slightly more
        # flexed hitting arm than clear. Keeping the clear floor for smash
        # rejected 戚佑仁7 despite the coach-approved completion trajectory.
        "minimum_contact_elbow_degrees": (
            135.0 if str(skill).lower() == "smash" else 140.0
        ),
    }


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


def contact_aware_root_trajectory(
    local_pose: NDArray[np.floating],
    *,
    transition_frames: int = 5,
) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
    """Recover root translation from a pelvis-centred pose and foot contacts.

    RTMW3D archives in this project are pelvis-centred and do not retain a
    world-space root trajectory.  Permanently pinning one ankle is incorrect
    for overhead strokes because the player changes support foot.  We infer a
    soft contact schedule from which ankle is lower, smooth the hand-off, and
    pin the weighted support point.  The result retains the expert's relative
    stepping and weight transfer without an instantaneous root jump.

    Returns ``(root, contact_weights)`` where contact weights have shape
    ``(T, 2)`` for COCO ankles 15 and 16.
    """
    pose = np.asarray(local_pose, dtype=np.float64)
    if pose.ndim != 3 or pose.shape[1] != 17 or pose.shape[2] not in (2, 3):
        raise ValueError("local_pose must have shape (T, 17, 2|3)")
    if transition_frames < 1:
        raise ValueError("transition_frames must be positive")
    ankles = pose[:, (15, 16)]
    # The project coordinate convention is y-up.  A lower ankle is therefore
    # the likelier support.  Normalize by leg length so the temperature is
    # anatomy independent.
    leg_lengths = np.linalg.norm(pose[:, (15, 16)] - pose[:, (13, 14)], axis=-1)
    scale = max(float(np.nanmedian(leg_lengths)), _EPS)
    relative_height = ankles[..., 1] - np.min(ankles[..., 1], axis=1, keepdims=True)
    logits = -relative_height / (0.055 * scale)
    logits -= np.max(logits, axis=1, keepdims=True)
    weights = np.exp(logits)
    weights /= np.sum(weights, axis=1, keepdims=True)

    if transition_frames > 1 and len(weights) > 1:
        radius = transition_frames - 1
        offsets = np.arange(-radius, radius + 1, dtype=np.float64)
        kernel = (radius + 1) - np.abs(offsets)
        kernel /= np.sum(kernel)
        padded = np.pad(weights, ((radius, radius), (0, 0)), mode="edge")
        weights = np.stack(
            [np.convolve(padded[:, foot], kernel, mode="valid") for foot in range(2)],
            axis=-1,
        )
        weights /= np.sum(weights, axis=1, keepdims=True)

    # Integrate the active contact foot's opposite local velocity.  Blending
    # absolute ankle locations would drag the pelvis between two distinct foot
    # placements during a support switch; blending velocities keeps the root
    # continuous and allows the stepping foot to land somewhere new.
    ankle_delta = np.diff(ankles, axis=0)
    midpoint_weights = 0.5 * (weights[:-1] + weights[1:])
    root_delta = -np.sum(midpoint_weights[..., None] * ankle_delta, axis=1)
    root = np.concatenate(
        (np.zeros((1, pose.shape[-1]), dtype=np.float64), np.cumsum(root_delta, axis=0)),
        axis=0,
    )
    return root.astype(np.float32), weights.astype(np.float32)


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


def retarget_expert_directions_fk(
    student: NDArray[np.floating],
    expert: NDArray[np.floating],
    student_confidence: NDArray[np.floating],
    expert_confidence: NDArray[np.floating],
    *,
    root_trajectory: NDArray[np.floating] | None = None,
    rigid_torso: bool = False,
) -> NDArray[np.float32]:
    """Transfer the complete expert pose through FK onto student proportions.

    The expert supplies every bone direction at every frame. The student
    supplies exactly one stable length for every parent edge. No student joint
    direction is blended back into the result. This makes the returned motion
    an explicit, testable full-expert target rather than a partial correction.

    The function supports both 2D and 3D keypoint skeletons. A global/root
    trajectory is optional and remains separate from local joint rotations.
    """
    source = np.asarray(student, dtype=np.float64)
    reference = np.asarray(expert, dtype=np.float64)
    source_observed = np.asarray(student_confidence, dtype=np.float64)
    reference_observed = np.asarray(expert_confidence, dtype=np.float64)
    if source.shape != reference.shape or source.ndim != 3:
        raise ValueError("student and expert must have matching shape (T, 17, D)")
    if source.shape[1] != 17 or source.shape[2] not in (2, 3):
        raise ValueError("student and expert must have shape (T, 17, 2|3)")
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
    valid = np.zeros(reference_observed.shape, dtype=bool)
    # Confidence controls loss/evaluation weighting, not whether finite expert
    # geometry is allowed to supervise correction.  RTMW3D still emits a
    # finite coordinate when confidence reaches zero.  Discarding that vector
    # used to copy the student's faulty direction into the target.
    valid[:] = np.isfinite(expert_offsets).all(axis=-1) & (
        expert_norms[..., 0] > _EPS
    )
    raw_directions = np.divide(
        expert_offsets,
        expert_norms,
        out=np.zeros_like(expert_offsets),
        where=expert_norms > _EPS,
    )
    directions = _fill_directions(raw_directions, valid, fallback)
    expert_hip_axis = reference[:, 12] - reference[:, 11]
    source_hip_axis = source[:, 12] - source[:, 11]
    expert_hip_norm = np.linalg.norm(expert_hip_axis, axis=-1, keepdims=True)
    source_hip_norm = np.linalg.norm(source_hip_axis, axis=-1, keepdims=True)
    hip_axis = np.divide(
        expert_hip_axis,
        expert_hip_norm,
        out=np.divide(
            source_hip_axis,
            source_hip_norm,
            out=np.zeros_like(source_hip_axis),
            where=source_hip_norm > _EPS,
        ),
        where=expert_hip_norm > _EPS,
    )
    directions[:, 11] = -hip_axis
    directions[:, 12] = hip_axis

    if root_trajectory is None:
        root = implicit_pelvis(reference)
    else:
        root = np.asarray(root_trajectory, dtype=np.float64)
        if root.shape != (len(source), source.shape[-1]):
            raise ValueError("root_trajectory must have shape (T, D)")

    output = np.empty_like(source)
    rigid_joints: set[int] = set()
    if rigid_torso:
        shoulder_width, hip_width, torso_height = _stable_torso_dimensions(
            source, source_observed
        )
        axes = body_frame_axes(reference)
        lateral = axes[..., 0]
        up = axes[..., 1]
        shoulder_center = root + torso_height * up
        output[:, 11] = root - 0.5 * hip_width * lateral
        output[:, 12] = root + 0.5 * hip_width * lateral
        output[:, 5] = shoulder_center - 0.5 * shoulder_width * lateral
        output[:, 6] = shoulder_center + 0.5 * shoulder_width * lateral
        rigid_joints = {5, 6, 11, 12}
    for joint, parent in enumerate(COCO_PARENTS):
        if joint in rigid_joints:
            continue
        if joint in (11, 12):
            output[:, joint] = root + lengths[joint] * directions[:, joint]
            continue
        anchor = root if parent < 0 else output[:, parent]
        output[:, joint] = anchor + lengths[joint] * directions[:, joint]
    return output.astype(np.float32)


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


def direction_error_degrees(
    first: NDArray[np.floating],
    second: NDArray[np.floating],
) -> NDArray[np.float64]:
    """Per-frame, per-joint angular error between parent-edge directions."""
    a = parent_offsets(first)
    b = parent_offsets(second)
    denominator = np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1)
    cosine = np.divide(
        np.sum(a * b, axis=-1),
        denominator,
        out=np.ones_like(denominator),
        where=denominator > _EPS,
    )
    return np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))


def canonical_overhead_exemplar_quality_mask(
    experts: NDArray[np.floating],
    *,
    preparation_frame: int = 0,
    contact_frame: int = 32,
    completion_frame: int = 63,
    minimum_dominant_step_x_ratio: float = 0.15,
    minimum_contact_shoulder_tilt_degrees: float = 15.0,
    minimum_contact_elbow_degrees: float = 140.0,
    minimum_contact_wrist_height_ratio: float = 0.75,
    maximum_completion_torso_lean_degrees: float = 10.0,
    minimum_completion_stance_width_ratio: float = 0.45,
) -> NDArray[np.bool_]:
    """Identify complete 2D overhead exemplars before reference retrieval.

    A technique label alone does not make every extracted expert clip a valid
    correction target.  In particular, a bad completion crop can collapse the
    arm while all 17 coordinates remain finite.
    This gate uses only qualitative geometry at the already-established event
    anchors.  It does not clamp or modify a selected expert pose.

    The bank is in the project's canonical right-handed 2D coordinates, so
    dominant shoulder/elbow/wrist are COCO joints 6/8/10. Completion shoulder
    order, shoulder tilt, and arm shape are deliberately unrestricted: a
    valid clear/smash follow-through can cross the body and finish with a bent
    arm, as the selected expert does.
    """
    bank = np.asarray(experts, dtype=np.float64)
    if bank.ndim != 4 or bank.shape[2:] != (17, 2):
        raise ValueError("experts must have shape (N, T, 17, 2)")
    if not (0 <= preparation_frame < bank.shape[1]):
        raise ValueError("preparation_frame is out of range")
    if not (0 <= contact_frame < bank.shape[1]):
        raise ValueError("contact_frame is out of range")
    if not (0 <= completion_frame < bank.shape[1]):
        raise ValueError("completion_frame is out of range")

    def _joint_angle(frame: NDArray[np.float64]) -> NDArray[np.float64]:
        first = frame[:, 6] - frame[:, 8]
        second = frame[:, 10] - frame[:, 8]
        denominator = np.linalg.norm(first, axis=-1) * np.linalg.norm(
            second, axis=-1
        )
        cosine = np.divide(
            np.sum(first * second, axis=-1),
            denominator,
            out=np.full(len(frame), np.nan, dtype=np.float64),
            where=denominator > _EPS,
        )
        return np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))

    preparation = bank[:, preparation_frame]
    contact = bank[:, contact_frame]
    completion = bank[:, completion_frame]
    contact_elbow = _joint_angle(contact)
    contact_pelvis = 0.5 * (contact[:, 11] + contact[:, 12])
    contact_shoulder_center = 0.5 * (contact[:, 5] + contact[:, 6])
    torso_height = np.linalg.norm(
        contact_shoulder_center - contact_pelvis, axis=-1
    )
    wrist_height_ratio = np.divide(
        contact[:, 10, 1] - contact[:, 6, 1],
        torso_height,
        out=np.full(len(bank), np.nan, dtype=np.float64),
        where=torso_height > _EPS,
    )
    contact_shoulder = contact[:, 6] - contact[:, 5]
    contact_shoulder_tilt = np.degrees(
        np.abs(np.arctan2(contact_shoulder[:, 1], contact_shoulder[:, 0]))
    )
    preparation_pelvis = 0.5 * (preparation[:, 11] + preparation[:, 12])
    contact_dominant_ankle = contact[:, 16] - contact_pelvis
    preparation_dominant_ankle = preparation[:, 16] - preparation_pelvis
    dominant_step_x_ratio = np.divide(
        contact_dominant_ankle[:, 0] - preparation_dominant_ankle[:, 0],
        torso_height,
        out=np.full(len(bank), np.nan, dtype=np.float64),
        where=torso_height > _EPS,
    )
    completion_pelvis = 0.5 * (completion[:, 11] + completion[:, 12])
    completion_shoulder_center = 0.5 * (completion[:, 5] + completion[:, 6])
    completion_torso = completion_shoulder_center - completion_pelvis
    completion_torso_lean = np.degrees(
        np.arctan2(
            np.abs(completion_torso[:, 0]),
            np.abs(completion_torso[:, 1]) + _EPS,
        )
    )
    completion_stance_width_ratio = np.divide(
        np.abs(completion[:, 16, 0] - completion[:, 15, 0]),
        torso_height,
        out=np.full(len(bank), np.nan, dtype=np.float64),
        where=torso_height > _EPS,
    )
    finite = np.isfinite(
        bank[:, (preparation_frame, contact_frame, completion_frame)]
    ).all(
        axis=(1, 2, 3)
    )
    return (
        finite
        & (dominant_step_x_ratio >= minimum_dominant_step_x_ratio)
        & (contact_shoulder_tilt >= minimum_contact_shoulder_tilt_degrees)
        & (contact_elbow >= minimum_contact_elbow_degrees)
        & (wrist_height_ratio >= minimum_contact_wrist_height_ratio)
        & (completion_torso_lean <= maximum_completion_torso_lean_degrees)
        & (
            completion_stance_width_ratio
            >= minimum_completion_stance_width_ratio
        )
    )


def select_torso_local_fk_2d_expert(
    student: NDArray[np.floating],
    experts: NDArray[np.floating],
    student_confidence: NDArray[np.floating],
    expert_confidence: NDArray[np.floating],
    *,
    preparation_frames: int = 17,
) -> tuple[int, NDArray[np.float32], float]:
    """Select an adjusted expert by preparation-phase 2D limb directions."""
    source = np.asarray(student, dtype=np.float32)
    bank = np.asarray(experts, dtype=np.float32)
    source_observed = np.asarray(student_confidence, dtype=np.float32)
    bank_observed = np.asarray(expert_confidence, dtype=np.float32)
    if source.ndim != 3 or source.shape[1:] != (17, 2):
        raise ValueError("student must have shape (T, 17, 2)")
    if bank.ndim != 4 or bank.shape[1:] != source.shape:
        raise ValueError("experts must have shape (N, T, 17, 2)")
    if source_observed.shape != source.shape[:2]:
        raise ValueError("student_confidence must have shape (T, 17)")
    if bank_observed.shape != bank.shape[:3]:
        raise ValueError("expert_confidence must have shape (N, T, 17)")
    if not len(bank):
        raise ValueError("at least one expert is required")
    stop = min(max(1, preparation_frames), len(source))
    body_joints = np.asarray((5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16))
    best: tuple[float, int, NDArray[np.float32]] | None = None
    for index, expert in enumerate(bank):
        target = retarget_expert_body_local_rotations_fk(
            source,
            expert,
            source_observed,
            bank_observed[index],
            root_trajectory=np.zeros((len(source), 2), dtype=np.float32),
        )
        error = direction_error_degrees(source, target)[:stop, body_joints] / 180.0
        weights = np.minimum(
            source_observed[:stop, body_joints],
            bank_observed[index, :stop][:, body_joints],
        )
        valid = np.isfinite(error) & np.isfinite(weights) & (weights > 0.0)
        score = (
            float(np.sum(error[valid] * weights[valid]) / np.sum(weights[valid]))
            if np.any(valid)
            else float("inf")
        )
        if best is None or score < best[0]:
            best = (score, index, target)
    assert best is not None
    return best[1], best[2], best[0]


def select_canonical_fk_2d_expert(
    student: NDArray[np.floating],
    experts: NDArray[np.floating],
    student_confidence: NDArray[np.floating],
    expert_confidence: NDArray[np.floating],
    *,
    preparation_frames: int = 17,
    require_overhead_quality: bool = False,
    skill: str | None = None,
) -> tuple[int, NDArray[np.float32], float]:
    """Select a canonical 2D expert without using the student's bad ending.

    Eligibility (including qualitative exemplar quality) is intentionally the
    caller's responsibility.  Among eligible references, retrieval considers
    preparation only; hit and completion are then copied fully from the expert
    instead of selecting an expert whose faulty ending resembles the student.
    """
    source = np.asarray(student, dtype=np.float32)
    bank = np.asarray(experts, dtype=np.float32)
    source_observed = np.asarray(student_confidence, dtype=np.float32)
    bank_observed = np.asarray(expert_confidence, dtype=np.float32)
    if source.ndim != 3 or source.shape[1:] != (17, 2):
        raise ValueError("student must have shape (T, 17, 2)")
    if bank.ndim != 4 or bank.shape[1:] != source.shape:
        raise ValueError("experts must have shape (N, T, 17, 2)")
    if source_observed.shape != source.shape[:2]:
        raise ValueError("student_confidence must have shape (T, 17)")
    if bank_observed.shape != bank.shape[:3]:
        raise ValueError("expert_confidence must have shape (N, T, 17)")
    if not len(bank):
        raise ValueError("at least one expert is required")
    stop = min(max(1, preparation_frames), len(source))
    body_joints = np.asarray((5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16))
    best: tuple[float, int, NDArray[np.float32]] | None = None
    eligible = (
        canonical_overhead_exemplar_quality_mask(
            bank, **canonical_overhead_exemplar_quality_options(skill)
        )
        if require_overhead_quality
        else np.ones(len(bank), dtype=bool)
    )
    if not np.any(eligible):
        raise ValueError("no canonical 2D expert passes the overhead quality gate")
    for index, expert in enumerate(bank):
        if not eligible[index]:
            continue
        target = retarget_expert_canonical_2d_fk(
            source,
            expert,
            source_observed,
            bank_observed[index],
            root_trajectory=np.zeros((len(source), 2), dtype=np.float32),
        )
        error = direction_error_degrees(source, target)[:stop, body_joints] / 180.0
        weights = np.minimum(
            source_observed[:stop, body_joints],
            bank_observed[index, :stop][:, body_joints],
        )
        valid = np.isfinite(error) & np.isfinite(weights) & (weights > 0.0)
        score = (
            float(np.sum(error[valid] * weights[valid]) / np.sum(weights[valid]))
            if np.any(valid)
            else float("inf")
        )
        if best is None or score < best[0]:
            best = (score, index, target)
    assert best is not None
    return best[1], best[2], best[0]
