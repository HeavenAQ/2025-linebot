"""Pick the expert clip a learner's corrected motion most resembles.

The diffusion prior generates an idealised movement rather than copying any one
expert, so there is no matched recording to show alongside it. This picks the
closest real demonstration instead: the learner sees a person performing what
their corrected skeleton is reaching towards.

Matching is on the canonical-space skeleton both sides already share, so no
further normalization is needed — the correction and the bank come out of the
same pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from badminton_analysis.ml.skeleton_normalization import (
    CANONICAL_PHASE_INDICES,
    _dtw_segment_positions,
    resample_detected_phase_indices,
)

Metric = Literal["cosine", "euclidean"]


@dataclass(frozen=True)
class ExpertReference:
    """One expert clip, with what playback needs to line it up."""

    skill: str
    handedness: str
    video_object_path: str
    subject_id: str
    fps: float
    analysis_window: tuple[int, int, int]
    source_phase_indices: tuple[int, ...]
    distance: float
    similarity: float
    # What the browser needs before it has loaded the clip. Banks built before
    # these were recorded report zero, which the player treats as "unknown".
    duration_seconds: float = 0.0
    width: int = 0
    height: int = 0
    # The clip's own canonical-space poses, kept so playback can be aligned
    # against them segment by segment rather than only at the checkpoints.
    skeleton: NDArray[np.float32] | None = None

    @property
    def motion_start_seconds(self) -> float:
        return float(self.source_phase_indices[0]) / self.fps

    @property
    def motion_end_seconds(self) -> float:
        return float(self.source_phase_indices[-1] + 1) / self.fps

    def phase_seconds(self) -> tuple[float, ...]:
        """Each checkpoint timed in this expert's own video."""
        return tuple(float(frame) / self.fps for frame in self.source_phase_indices)


class ExpertReferenceBank:
    """The expert clips the current checkpoints were trained on."""

    def __init__(self, path: str | Path) -> None:
        with np.load(Path(path), allow_pickle=False) as bank:
            self.skeletons: NDArray[np.float32] = bank["skeletons"].astype(np.float32)
            self.skill = bank["skill"].astype(str)
            self.handedness = bank["handedness"].astype(str)
            self.video_object_path = bank["video_object_path"].astype(str)
            self.subject_id = bank["subject_id"].astype(str)
            self.fps = bank["fps"].astype(np.float32)
            self.analysis_window = bank["analysis_window"].astype(np.int64)
            self.source_phase_indices = bank["source_phase_indices"].astype(np.int64)
            count = len(self.subject_id)
            self.duration_seconds = (
                bank["duration_seconds"].astype(np.float32)
                if "duration_seconds" in bank
                else np.zeros(count, dtype=np.float32)
            )
            self.width = (
                bank["width"].astype(np.int64) if "width" in bank else np.zeros(count, dtype=np.int64)
            )
            self.height = (
                bank["height"].astype(np.int64) if "height" in bank else np.zeros(count, dtype=np.int64)
            )
        if self.skeletons.ndim != 4 or self.skeletons.shape[1:] != (64, 17, 2):
            raise ValueError(f"unexpected expert bank shape: {self.skeletons.shape}")

    def __len__(self) -> int:
        return len(self.skeletons)

    def select(
        self,
        corrected_pose: NDArray[np.floating],
        *,
        skill: str,
        handedness: str,
        metric: Metric = "cosine",
    ) -> ExpertReference | None:
        """The closest expert of the same skill, preferring the same handedness.

        A left-handed learner is shown a left-handed demonstration where one
        exists; falling back to the other hand beats showing nothing, since the
        movement is mirrored but the technique is the same.
        """
        pose = np.asarray(corrected_pose, dtype=np.float32)
        if pose.shape != (64, 17, 2):
            raise ValueError(f"corrected pose must be (64, 17, 2), got {pose.shape}")

        candidates = np.flatnonzero(self.skill == skill)
        if not len(candidates):
            return None
        same_hand = candidates[self.handedness[candidates] == handedness]
        if len(same_hand):
            candidates = same_hand

        query = pose.reshape(-1)
        bank = self.skeletons[candidates].reshape(len(candidates), -1)
        if metric == "euclidean":
            distances = np.linalg.norm(bank - query, axis=1)
            best = int(np.argmin(distances))
            distance = float(distances[best])
            denominator = float(np.linalg.norm(query)) * float(np.linalg.norm(bank[best])) or 1.0
            similarity = float(bank[best] @ query / denominator)
        else:
            norms = np.linalg.norm(bank, axis=1) * float(np.linalg.norm(query))
            norms[norms == 0.0] = 1.0
            similarities = (bank @ query) / norms
            best = int(np.argmax(similarities))
            similarity = float(similarities[best])
            distance = float(np.linalg.norm(bank[best] - query))

        index = int(candidates[best])
        return ExpertReference(
            skill=str(self.skill[index]),
            handedness=str(self.handedness[index]),
            video_object_path=str(self.video_object_path[index]),
            subject_id=str(self.subject_id[index]),
            fps=float(self.fps[index]),
            analysis_window=tuple(int(v) for v in self.analysis_window[index]),
            source_phase_indices=tuple(int(v) for v in self.source_phase_indices[index]),
            distance=distance,
            similarity=similarity,
            duration_seconds=float(self.duration_seconds[index]),
            width=int(self.width[index]),
            height=int(self.height[index]),
            skeleton=self.skeletons[index],
        )


def segmental_alignment(
    reference: ExpertReference,
    corrected_pose: NDArray[np.floating],
    *,
    student_phase_indices: NDArray[np.integer] = CANONICAL_PHASE_INDICES,
) -> tuple[tuple[float, float], ...]:
    """Map the learner's motion onto the expert's clock, frame by frame.

    The checkpoints are fixed: the learner's k-th anchor lines up with the
    expert's k-th anchor, because they are the same moment of the stroke by
    definition. What happens between them is not fixed -- two people spend
    different fractions of a phase winding up -- so each segment is aligned by
    DTW over the poses themselves, the same warping the scorer uses, rather than
    assumed to run at a constant relative tempo.

    Returns (normalized_position, expert_seconds) pairs, one per learner frame,
    monotonic in both. An empty result means the alignment could not be built
    and playback should fall back to interpolating between the checkpoints.
    """
    if reference.skeleton is None:
        return ()
    student = np.asarray(corrected_pose, dtype=np.float64)
    expert = np.asarray(reference.skeleton, dtype=np.float64)
    if student.shape != expert.shape or student.ndim != 3:
        return ()
    frames = student.shape[0]
    student_anchors = np.asarray(student_phase_indices, dtype=np.int64)
    try:
        expert_anchors = resample_detected_phase_indices(
            tuple(reference.source_phase_indices), frames
        )
    except ValueError:
        return ()
    if student_anchors.shape != expert_anchors.shape or student_anchors.shape != (5,):
        return ()

    weights = np.ones(student.shape[1], dtype=np.float64)
    expert_positions = np.zeros(frames, dtype=np.float64)
    for segment in range(len(student_anchors) - 1):
        s0, s1 = int(student_anchors[segment]), int(student_anchors[segment + 1])
        e0, e1 = int(expert_anchors[segment]), int(expert_anchors[segment + 1])
        if s1 <= s0 or e1 <= e0:
            return ()
        local = _dtw_segment_positions(expert[e0 : e1 + 1], student[s0 : s1 + 1], weights)
        expert_positions[s0 : s1 + 1] = e0 + local
    # The anchors are the fixed points of the map; DTW fills in between them.
    expert_positions[student_anchors] = expert_anchors.astype(np.float64)
    expert_positions = np.maximum.accumulate(expert_positions)

    last = frames - 1
    start, end = int(reference.source_phase_indices[0]), int(reference.source_phase_indices[-1])
    span = end - start
    return tuple(
        (
            float(frame) / last,
            float(start + expert_positions[frame] * span / last) / reference.fps,
        )
        for frame in range(frames)
    )
