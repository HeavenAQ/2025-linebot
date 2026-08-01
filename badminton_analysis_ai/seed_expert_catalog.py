from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path
import subprocess
from typing import Any

import cv2
from google.auth.credentials import Credentials
from google.cloud import firestore, storage
from google.oauth2.credentials import Credentials as OAuthCredentials

from badminton_analysis.ml.expert_vector import expert_vector_payload
from service.expert_catalog import expert_document_id

SKILL_VIDEO_DIRS = {
    "clear": Path("scoring_videos/高遠球/專家高遠球"),
    "serve": Path("scoring_videos/發球/羽球隊同學"),
    "lift": Path("scoring_videos/挑球/專家挑球"),
    "smash": Path("scoring_videos/殺球/專家殺球"),
}
VIDEO_CATALOG_VERSION = "v1"
VECTOR_CATALOG_VERSION = "v2"


def _gcloud_user_credentials() -> Credentials:
    result = subprocess.run(
        ("gcloud", "auth", "print-access-token"),
        check=True,
        capture_output=True,
        text=True,
    )
    return OAuthCredentials(result.stdout.strip())


def _video_metadata(path: Path) -> dict[str, float | int]:
    capture = cv2.VideoCapture(str(path))
    try:
        fps = float(capture.get(cv2.CAP_PROP_FPS)) or 30.0
        count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        return {
            "duration_seconds": count / fps if fps > 0 else 0.0,
            "fps": fps,
            "width": int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height": int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        }
    finally:
        capture.release()


def seed(
    source_root: Path,
    *,
    project_id: str,
    bucket_name: str,
    collection_name: str,
    dry_run: bool,
    workers: int = 8,
    credentials: Credentials | None = None,
) -> dict[str, int]:
    storage_client = (
        None
        if dry_run
        else storage.Client(project=project_id, credentials=credentials)
    )
    firestore_client = (
        None
        if dry_run
        else firestore.Client(project=project_id, credentials=credentials)
    )
    bucket = None if storage_client is None else storage_client.bucket(bucket_name)
    collection = (
        None if firestore_client is None else firestore_client.collection(collection_name)
    )
    counts: dict[str, int] = {}
    documents: list[tuple[str, dict[str, Any]]] = []
    uploads: list[tuple[Path, str, str]] = []
    for skill, relative_video_dir in SKILL_VIDEO_DIRS.items():
        video_dir = source_root / relative_video_dir
        vector_dir = source_root / "datasets" / "skeleton_sequences" / skill / "experts"
        videos = {
            path.stem: path
            for path in video_dir.iterdir()
            if path.suffix.lower() in (".mp4", ".mov")
        }
        vectors = {path.stem: path for path in vector_dir.glob("*.npz")}
        if set(videos) != set(vectors):
            missing_vectors = sorted(set(videos) - set(vectors))
            missing_videos = sorted(set(vectors) - set(videos))
            raise ValueError(
                f"{skill} catalog mismatch: missing_vectors={missing_vectors} "
                f"missing_videos={missing_videos}"
            )
        if len(videos) != 50:
            raise ValueError(f"expected 50 {skill} experts, found {len(videos)}")
        for expert_id in sorted(videos):
            video_path = videos[expert_id]
            vector_path = vectors[expert_id]
            video_object = (
                f"experts/{VIDEO_CATALOG_VERSION}/{skill}/videos/{video_path.name}"
            )
            vector_object = (
                f"experts/{VECTOR_CATALOG_VERSION}/{skill}/vectors/{vector_path.name}"
            )
            video_metadata = _video_metadata(video_path)
            vector_payload, handedness = expert_vector_payload(
                vector_path, float(video_metadata["fps"])
            )
            document = {
                "expert_id": expert_id,
                "display_name": expert_id,
                "skill": skill,
                "handedness": handedness,
                "video_filename": video_path.name,
                "video_object_path": video_object,
                "video_gcs_uri": f"gs://{bucket_name}/{video_object}",
                "vector_object_path": vector_object,
                "vector_gcs_uri": f"gs://{bucket_name}/{vector_object}",
                **video_metadata,
                **vector_payload,
            }
            documents.append((expert_document_id(skill, expert_id), document))
            uploads.extend(
                (
                    (video_path, video_object, "video/mp4"),
                    (vector_path, vector_object, "application/octet-stream"),
                )
            )
        counts[skill] = len(videos)

    if not dry_run:
        assert bucket is not None and collection is not None
        existing = {blob.name for blob in bucket.list_blobs(prefix="experts/")}
        pending = [upload for upload in uploads if upload[1] not in existing]

        def upload(item: tuple[Path, str, str]) -> str:
            source, object_path, content_type = item
            bucket.blob(object_path).upload_from_filename(
                str(source), content_type=content_type, timeout=300
            )
            return object_path

        completed = 0
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(upload, item) for item in pending]
            for future in as_completed(futures):
                future.result()
                completed += 1
                if completed % 25 == 0 or completed == len(pending):
                    print(f"uploaded {completed}/{len(pending)} pending objects")

        for start in range(0, len(documents), 25):
            batch = firestore_client.batch()
            for document_id, document in documents[start : start + 25]:
                document["updated_at"] = firestore.SERVER_TIMESTAMP
                batch.set(collection.document(document_id), document)
            batch.commit()
            print(f"committed {min(start + 25, len(documents))}/{len(documents)} documents")
    return counts


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Upload expert videos/vectors and seed their Firestore catalog"
    )
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--project-id", default=os.getenv("GCP_PROJECT_ID", ""))
    parser.add_argument("--bucket", default=os.getenv("GCS_BUCKET_NAME", ""))
    parser.add_argument(
        "--collection",
        default=os.getenv("EXPERT_VIDEOS_COLLECTION", "badminton_experts_v2"),
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument(
        "--gcloud-user-credentials",
        action="store_true",
        help="Use the active gcloud user's in-memory access token",
    )
    args = parser.parse_args()
    if not args.project_id or not args.bucket:
        raise ValueError("--project-id and --bucket are required")
    if args.workers < 1:
        raise ValueError("--workers must be positive")
    credentials = _gcloud_user_credentials() if args.gcloud_user_credentials else None
    counts = seed(
        args.source_root,
        project_id=args.project_id,
        bucket_name=args.bucket,
        collection_name=args.collection,
        dry_run=args.dry_run,
        workers=args.workers,
        credentials=credentials,
    )
    print(json.dumps(counts, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
