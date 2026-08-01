from __future__ import annotations

from typing import Iterable

import numpy as np
from numpy.typing import NDArray

from badminton_analysis.models.types import COCOKeypoints, CoordinateDict, Handedness

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


def _swap_left_right(
    sequence: NDArray[np.float64], confidence: NDArray[np.float64]
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    mirrored = sequence.copy()
    mirrored_confidence = confidence.copy()
    for left, right in _LEFT_RIGHT_PAIRS:
        mirrored[:, [left, right]] = mirrored[:, [right, left]]
        mirrored_confidence[:, [left, right]] = mirrored_confidence[:, [right, left]]
    return mirrored, mirrored_confidence


def _body_frame_3d(frame: NDArray[np.float64]) -> tuple[NDArray[np.float64], float]:
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
        raise ValueError("spine is parallel to shoulder axis")
    y_axis = spine / spine_norm
    z_axis = np.cross(x_axis, y_axis)
    z_norm = float(np.linalg.norm(z_axis))
    if z_norm < _EPS:
        raise ValueError("body plane is degenerate")
    z_axis /= z_norm
    return np.vstack((x_axis, y_axis, z_axis)), scale


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


def normalize_skeleton_sequence(
    sequence: NDArray[np.floating],
    confidence: NDArray[np.floating],
    handedness: Handedness | str,
) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
    """Interpolate, mirror anatomy, root-center, rotate, and shoulder-scale a pose sequence.

    Right-dominant anatomy always occupies the right COCO joint slots. Confidence
    remains the original observation mask, including across interpolated gaps.
    """
    coordinates = _interpolate_missing(sequence, confidence)
    observed = np.asarray(confidence, dtype=np.float64).copy()
    if coordinates.shape[1] != COCO_JOINT_COUNT or coordinates.shape[2] not in (2, 3):
        raise ValueError("sequence must have shape (T, 17, 2) or (T, 17, 3)")
    is_left = handedness == Handedness.LEFT or str(handedness).lower() == "left"
    if is_left:
        coordinates, observed = _swap_left_right(coordinates, observed)

    output = np.zeros_like(coordinates)
    basis: NDArray[np.float64] | None = None
    for frame in coordinates:
        try:
            basis, _ = (
                _body_frame_3d(frame) if frame.shape[1] == 3 else _body_frame_2d(frame)
            )
            break
        except ValueError:
            continue
    if basis is None:
        basis = np.eye(coordinates.shape[2], dtype=np.float64)

    shoulder_widths = np.linalg.norm(
        coordinates[:, int(COCOKeypoints.RIGHT_SHOULDER)]
        - coordinates[:, int(COCOKeypoints.LEFT_SHOULDER)],
        axis=-1,
    )
    valid_widths = shoulder_widths[shoulder_widths > _EPS]
    scale = float(np.median(valid_widths)) if len(valid_widths) else 1.0

    for frame_index, frame in enumerate(coordinates):
        left_hip = frame[int(COCOKeypoints.LEFT_HIP)]
        right_hip = frame[int(COCOKeypoints.RIGHT_HIP)]
        root = (left_hip + right_hip) / 2.0
        output[frame_index] = ((basis @ (frame - root).T).T) / scale
    if is_left and output.shape[2] == 3:
        # A spatial mirror changes chirality; keep depth in the same canonical
        # dominant-side frame as right-handed sequences.
        output[..., 2] *= -1.0
    return output.astype(np.float32), observed.astype(np.float32)


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
