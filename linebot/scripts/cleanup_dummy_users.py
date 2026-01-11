#!/usr/bin/env python3
"""
Delete dummy users in Firestore whose document IDs start with a given prefix.

Prereqs:
  pip install google-cloud-firestore python-dotenv

Env used (loaded from ../.env if present):
  - GCP_PROJECT_ID (required unless --project is passed)
  - FIREBASE_DATA_DB (defaults: users)
  - GOOGLE_APPLICATION_CREDENTIALS (defaults: ../sa-key.json)

Usage:
  python scripts/cleanup_dummy_users.py --prefix UTEST --yes
  python scripts/cleanup_dummy_users.py --prefix UTEST --dry-run  # preview only
"""
from __future__ import annotations

import argparse
import os
from typing import List

try:
    from dotenv import load_dotenv  # type: ignore
except Exception:
    load_dotenv = None  # optional

try:
    from google.cloud import firestore  # type: ignore
except Exception:
    print("ERROR: Missing google-cloud-firestore.\nInstall with: pip install google-cloud-firestore python-dotenv")
    raise


def load_env(project_root: str) -> None:
    env_path = os.path.join(project_root, ".env")
    if load_dotenv and os.path.exists(env_path):
        load_dotenv(env_path)
    if not os.getenv("GOOGLE_APPLICATION_CREDENTIALS"):
        sa_path = os.path.join(project_root, "sa-key.json")
        if os.path.exists(sa_path):
            os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = sa_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prefix", required=True, help="Delete docs whose ID starts with this prefix")
    parser.add_argument("--project", default=None, help="Override GCP project id")
    parser.add_argument("--dry-run", action="store_true", help="Print matches without deleting")
    parser.add_argument("--yes", action="store_true", help="Confirm deletion without interactive prompt")
    args = parser.parse_args()

    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
    load_env(project_root)

    project_id = args.project or os.getenv("GCP_PROJECT_ID")
    if not project_id:
        raise SystemExit("GCP_PROJECT_ID is required (set in ../.env or pass --project)")

    data_collection = os.getenv("FIREBASE_DATA_DB", "users")

    db = firestore.Client(project=project_id)
    coll = db.collection(data_collection)

    # Firestore has no startswith query on IDs; list and filter client-side.
    # For large datasets you'd use a separate field to index prefixes.
    docs = list(coll.stream())
    matches: List[str] = [doc.id for doc in docs if doc.id.startswith(args.prefix)]

    if not matches:
        print(f"No documents start with prefix '{args.prefix}' in collection '{data_collection}'.")
        return

    print(f"Found {len(matches)} document(s) to delete in '{data_collection}':")
    for doc_id in matches:
        print(f" - {doc_id}")

    if args.dry_run:
        print("Dry run: no deletions performed.")
        return

    if not args.yes:
        print("Add --yes to confirm deletion, or use --dry-run to preview only.")
        return

    for doc_id in matches:
        coll.document(doc_id).delete()

    print(f"Deleted {len(matches)} document(s) from '{data_collection}'.")


if __name__ == "__main__":
    main()

