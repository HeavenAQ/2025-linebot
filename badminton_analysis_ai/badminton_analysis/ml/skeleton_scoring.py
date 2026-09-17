"""COCO-17 bone topology and stable-anatomy retargeting for generated poses."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

BONES = (
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
)

# Lateral torso spans are projection-dependent rather than rigid limb lengths.
# The expert-motion generator preserves their generated per-frame profile so
# shoulder/hip rotation remains visible after student-anatomy retargeting.
TORSO_WIDTH_BONES = ((5, 6), (11, 12))

ANGLE_TRIPLETS = (
    (5, 7, 9),
    (6, 8, 10),
    (7, 5, 11),
    (8, 6, 12),
    (5, 11, 13),
    (6, 12, 14),
    (11, 13, 15),
    (12, 14, 16),
)


def project_stable_bone_lengths(
    original: NDArray[np.floating],
    corrected: NDArray[np.floating],
    confidence: NDArray[np.floating],
    *,
    iterations: int = 20,
    expert_length_bones: tuple[tuple[int, int], ...] = (),
    preserve_target_pelvis: bool = False,
    preserve_direction_chains: tuple[tuple[int, ...], ...] = (),
) -> NDArray[np.float32]:
    """Retarget a motion using stable clip-level student anatomy.

    Rigid limbs use the student's median observed length. Projection-dependent
    torso widths may instead retain the generated expert profile. Optional
    distal chains are rebuilt after the global solve so their generated
    directions survive the length projection exactly.
    """
    source = np.asarray(original, dtype=np.float64)
    corrected_source = np.asarray(corrected, dtype=np.float64)
    projected = corrected_source.copy()
    observed = np.asarray(confidence, dtype=np.float64)
    if source.shape != projected.shape or source.ndim != 3 or source.shape[-1] != 2:
        raise ValueError("original and corrected must have matching shape (T, J, 2)")
    if observed.shape != source.shape[:2]:
        raise ValueError("confidence must have shape (T, J)")
    if iterations <= 0:
        raise ValueError("iterations must be positive")
    if not np.all(np.isfinite(projected)):
        raise ValueError("corrected skeleton must contain finite coordinates")

    free_bones = {frozenset(pair) for pair in expert_length_bones}
    bone_lookup = {
        frozenset((start, end)): index for index, (start, end) in enumerate(BONES)
    }
    direction_chains = tuple(tuple(chain) for chain in preserve_direction_chains)
    for chain in direction_chains:
        if len(chain) < 2:
            raise ValueError("preserved direction chains need at least two joints")
        for start, end in zip(chain[:-1], chain[1:], strict=True):
            if frozenset((start, end)) not in bone_lookup:
                raise ValueError(
                    f"preserved direction segment {start}-{end} is not a bone"
                )

    desired_lengths = np.empty((len(source), len(BONES)), dtype=np.float64)
    timeline = np.arange(len(source), dtype=np.float64)
    for bone_index, (start, end) in enumerate(BONES):
        is_free = frozenset((start, end)) in free_bones
        length_reference = corrected_source if is_free else source
        reference_vectors = length_reference[:, end] - length_reference[:, start]
        reference_lengths = np.linalg.norm(reference_vectors, axis=-1)
        visible = (
            (observed[:, start] > 0.05)
            & (observed[:, end] > 0.05)
            & np.isfinite(reference_lengths)
            & (reference_lengths > 1e-8)
        )
        fallback = np.isfinite(reference_lengths) & (reference_lengths > 1e-8)
        lengths = (
            reference_lengths[visible]
            if np.any(visible)
            else reference_lengths[fallback]
        )
        if not len(lengths):
            raise ValueError(f"reference bone {start}-{end} has no finite length")
        median_length = float(np.median(lengths))
        desired_lengths[:, bone_index] = (
            np.where(fallback, reference_lengths, median_length)
            if is_free
            else median_length
        )

        source_vectors = source[:, end] - source[:, start]
        source_lengths = np.linalg.norm(source_vectors, axis=-1)
        target_vectors = projected[:, end] - projected[:, start]
        target_lengths = np.linalg.norm(target_vectors, axis=-1)
        valid_direction = (
            np.isfinite(target_lengths)
            & (target_lengths > 1e-8)
            & (observed[:, start] > 0.05)
            & (observed[:, end] > 0.05)
        )
        if not np.any(valid_direction):
            valid_direction = fallback
            target_vectors = source_vectors.copy()
            target_lengths = source_lengths.copy()
        unit = np.zeros_like(target_vectors)
        unit[valid_direction] = (
            target_vectors[valid_direction] / target_lengths[valid_direction, None]
        )
        for dimension in range(unit.shape[-1]):
            unit[:, dimension] = np.interp(
                timeline, timeline[valid_direction], unit[valid_direction, dimension]
            )
        unit /= np.maximum(np.linalg.norm(unit, axis=-1, keepdims=True), 1e-8)
        unreliable = ~valid_direction
        projected[unreliable, end] = (
            projected[unreliable, start]
            + unit[unreliable] * desired_lengths[unreliable, bone_index][:, None]
        )

    bone_indices = np.asarray(BONES, dtype=np.int64)
    pelvis_reference = corrected_source if preserve_target_pelvis else source
    pelvis_anchor = (pelvis_reference[:, 11] + pelvis_reference[:, 12]) / 2.0
    for _ in range(iterations):
        for bone_index, (start, end) in enumerate(BONES):
            vector = projected[:, end] - projected[:, start]
            length = np.linalg.norm(vector, axis=-1)
            valid = length > 1e-8
            adjustment = np.zeros_like(vector)
            adjustment[valid] = (
                0.5
                * (
                    (length[valid] - desired_lengths[valid, bone_index]) / length[valid]
                )[:, None]
                * vector[valid]
            )
            projected[:, start] += adjustment
            projected[:, end] -= adjustment
        pelvis = (projected[:, 11] + projected[:, 12]) / 2.0
        projected += (pelvis_anchor - pelvis)[:, None, :]

    for chain in direction_chains:
        for start, end in zip(chain[:-1], chain[1:], strict=True):
            vectors = corrected_source[:, end] - corrected_source[:, start]
            lengths = np.linalg.norm(vectors, axis=-1)
            valid = np.isfinite(lengths) & (lengths > 1e-8)
            if not np.any(valid):
                vectors = projected[:, end] - projected[:, start]
                lengths = np.linalg.norm(vectors, axis=-1)
                valid = np.isfinite(lengths) & (lengths > 1e-8)
            units = np.zeros_like(vectors)
            units[valid] = vectors[valid] / lengths[valid, None]
            for dimension in range(2):
                units[:, dimension] = np.interp(
                    timeline, timeline[valid], units[valid, dimension]
                )
            units /= np.maximum(np.linalg.norm(units, axis=-1, keepdims=True), 1e-8)
            bone_index = bone_lookup[frozenset((start, end))]
            projected[:, end] = (
                projected[:, start] + units * desired_lengths[:, bone_index, None]
            )

    final_lengths = np.linalg.norm(
        projected[:, bone_indices[:, 1]] - projected[:, bone_indices[:, 0]], axis=-1
    )
    if np.any(final_lengths <= 1e-6):
        raise ValueError("stable bone projection produced a collapsed bone")
    return projected.astype(np.float32)
