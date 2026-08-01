"""Skeleton-correction training and inference utilities."""

from badminton_analysis.ml.skeleton_normalization import (
    COCO_JOINT_COUNT,
    normalize_skeleton_sequence,
    resample_sequence,
)

__all__ = [
    "COCO_JOINT_COUNT",
    "normalize_skeleton_sequence",
    "resample_sequence",
]
