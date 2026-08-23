from __future__ import annotations

import hashlib
from dataclasses import dataclass

from google.cloud import firestore

from badminton_analysis.ml.expert_vector import (
    expert_phase_seconds,
    expert_source_frame_seconds,
)


def expert_document_id(skill: str, expert_id: str) -> str:
    digest = hashlib.sha256(f"{skill}\0{expert_id}".encode("utf-8")).hexdigest()
    return f"{skill}-{digest[:24]}"


@dataclass(frozen=True)
class ExpertRecord:
    expert_id: str
    display_name: str
    skill: str
    handedness: str
    video_object_path: str
    vector_object_path: str
    duration_seconds: float
    fps: float
    width: int
    height: int
    motion_start_seconds: float
    motion_end_seconds: float
    phase_indices: tuple[int, ...]
    analysis_window_frames: tuple[int, ...]
    source_phase_frames: tuple[int, ...]

    def phase_seconds(self, *, target_frames: int = 64) -> tuple[float, ...]:
        """Checkpoint timestamps in this expert's video, empty when unseeded.

        Source frames are authoritative because tracking drops frames; the
        window interpolation below is only for entries seeded before those
        frames were published.
        """
        if self.source_phase_frames:
            return expert_source_frame_seconds(self.source_phase_frames, self.fps)
        if not self.phase_indices or len(self.analysis_window_frames) != 3:
            return ()
        return expert_phase_seconds(
            self.phase_indices,
            self.analysis_window_frames,
            self.fps,
            target_frames=target_frames,
        )


class ExpertCatalog:
    def __init__(self, project_id: str, collection: str) -> None:
        self.collection = firestore.Client(project=project_id).collection(collection)

    def get(self, skill: str, expert_id: str) -> ExpertRecord:
        snapshot = self.collection.document(
            expert_document_id(skill, expert_id)
        ).get()
        if not snapshot.exists:
            raise KeyError(f"expert catalog entry not found: {skill}/{expert_id}")
        data = snapshot.to_dict() or {}
        return ExpertRecord(
            expert_id=str(data["expert_id"]),
            display_name=str(data.get("display_name", expert_id)),
            skill=str(data["skill"]),
            handedness=str(data.get("handedness", "unknown")),
            video_object_path=str(data["video_object_path"]),
            vector_object_path=str(data["vector_object_path"]),
            duration_seconds=float(data.get("duration_seconds", 0.0)),
            fps=float(data.get("fps", 0.0)),
            width=int(data.get("width", 0)),
            height=int(data.get("height", 0)),
            motion_start_seconds=float(data.get("motion_start_seconds", 0.0)),
            motion_end_seconds=float(
                data.get("motion_end_seconds", data.get("duration_seconds", 0.0))
            ),
            # Seeded alongside the vector. Catalog entries written before
            # checkpoint alignment lack them, and fall back to window-only sync.
            phase_indices=tuple(int(value) for value in data.get("phase_indices", ())),
            analysis_window_frames=tuple(
                int(value) for value in data.get("analysis_window_frames", ())
            ),
            source_phase_frames=tuple(
                int(value) for value in data.get("source_phase_frames", ())
            ),
        )
