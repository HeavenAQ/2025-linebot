from __future__ import annotations

import hashlib
from dataclasses import dataclass

from google.cloud import firestore


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
        )
