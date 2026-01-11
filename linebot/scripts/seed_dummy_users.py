#!/usr/bin/env python3
"""
Seed a few dummy users into Firestore matching the backend UserData structure.

Prereqs:
  pip install google-cloud-firestore python-dotenv

Env used (loaded from ../.env if present):
  - GCP_PROJECT_ID (required)
  - FIREBASE_DATA_DB (defaults: users)
  - GOOGLE_APPLICATION_CREDENTIALS (defaults: ../sa-key.json)

Usage:
  python scripts/seed_dummy_users.py --count 3 --prefix UTEST
"""

from __future__ import annotations

import argparse
import os
import random
from datetime import datetime, timedelta, timezone

try:
    from dotenv import load_dotenv  # type: ignore
except Exception:
    load_dotenv = None  # optional

try:
    from google.cloud import firestore  # type: ignore
except Exception:
    print(
        "ERROR: Missing google-cloud-firestore.\nInstall with: pip install google-cloud-firestore python-dotenv"
    )
    raise


def load_env(project_root: str) -> None:
    # Load ../.env if present
    env_path = os.path.join(project_root, ".env")
    if load_dotenv and os.path.exists(env_path):
        load_dotenv(env_path)

    # Default GOOGLE_APPLICATION_CREDENTIALS to ../sa-key.json if not set
    if not os.getenv("GOOGLE_APPLICATION_CREDENTIALS"):
        sa_path = os.path.join(project_root, "sa-key.json")
        if os.path.exists(sa_path):
            os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = sa_path


def random_grade() -> int:
    return max(0, min(100, int(random.normalvariate(mu=80, sigma=10))))


def mk_grading_outcome(score: int) -> dict:
    # Minimal grading outcome compatible with backend types
    return {
        "grading_details": [
            {"description": "form", "grade": max(0, min(100, score - 3))},
            {"description": "timing", "grade": max(0, min(100, score - 2))},
            {"description": "power", "grade": max(0, min(100, score + 1))},
        ],
        "total_grade": score,
    }


def mk_work_item(date_iso: str, score: int, base_path: str) -> dict:
    return {
        "date": date_iso,
        "thumbnail": f"{base_path}/thumbnail.jpg",
        "skeleton_video": f"{base_path}/skeleton.mp4",
        "skeleton_comparison_video": f"{base_path}/compare.mp4",
        "reflection": "Test reflection",
        "preview_note": "Test preview note",
        "ai_note": "Test AI note",
        "grading_outcome": mk_grading_outcome(score),
    }


def mk_user_doc(user_id: str, user_name: str) -> dict:
    # Use today, yesterday, and two days ago for dates
    tz = timezone.utc
    today = datetime.now(tz).replace(hour=12, minute=0, second=0, microsecond=0)
    dates = [(today - timedelta(days=d)).strftime("%Y-%m-%d-%M-%S") for d in (0, 1, 2)]

    # Skills
    skills = ["serve", "smash", "clear"]
    portfolios = {s: {} for s in skills}

    # Add two works for 'serve' and one for 'smash'
    base = f"gs://dummy-bucket/{user_id}"
    serve_scores = [random_grade() for _ in range(2)]
    smash_scores = [random_grade() for _ in range(1)]

    portfolios["serve"][dates[2]] = mk_work_item(
        dates[2], serve_scores[0], f"{base}/serve/2025-01-0A"
    )
    portfolios["serve"][dates[0]] = mk_work_item(
        dates[0], serve_scores[1], f"{base}/serve/2025-01-0B"
    )
    portfolios["smash"][dates[1]] = mk_work_item(
        dates[1], smash_scores[0], f"{base}/smash/2025-01-0C"
    )
    # clear remains empty

    return {
        "portfolio": portfolios,
        "folder_paths": {
            "root": f"users/{user_id}/",
            "serve": f"users/{user_id}/serve/",
            "smash": f"users/{user_id}/smash/",
            "clear": f"users/{user_id}/clear/",
            "thumbnail": f"users/{user_id}/thumbnail",
        },
        "gpt_conversation_ids": {"serve": "", "smash": "", "clear": ""},
        "name": user_name,
        "id": user_id,
        # Right-handed = 1 (per backend enum)
        "handedness": 1,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--count", type=int, default=3, help="Number of users to create"
    )
    parser.add_argument("--prefix", type=str, default="UTEST", help="User ID prefix")
    parser.add_argument(
        "--project", type=str, default=None, help="Override GCP project id"
    )
    args = parser.parse_args()

    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
    load_env(project_root)

    project_id = args.project or os.getenv("GCP_PROJECT_ID")
    if not project_id:
        raise SystemExit(
            "GCP_PROJECT_ID is required (set in ../.env or pass --project)"
        )

    data_collection = os.getenv("FIREBASE_DATA_DB", "users")

    db = firestore.Client(project=project_id)
    coll = db.collection(data_collection)

    created = []
    for i in range(1, args.count + 1):
        user_id = f"{args.prefix}{i:03d}"
        doc = mk_user_doc(user_id, f"Test User {i}")
        coll.document(user_id).set(doc)
        created.append(user_id)

    print(f"Created {len(created)} users in collection '{data_collection}':")
    for uid in created:
        print(f" - {uid}")


if __name__ == "__main__":
    main()
