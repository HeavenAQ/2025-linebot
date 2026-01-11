#!/usr/bin/env python3
"""
Firestore migration: convert legacy camelCase field names to snake_case.

Targets:
1) Users collection (aka Data):
   - folderIDs -> folder_paths (object preserved)
   - gptConversationIDs -> gpt_conversation_ids (object preserved)
   - portfolio.*.* work items:
       comparisonVideo -> comparison_video
       previewNote -> preview_note
       aiNote -> ai_note
       gradingOutcome -> grading_outcome

2) Chat history collection (chat_history):
   - userId -> user_id
   - messages[*].conversationId -> messages[*].conversation_id

Safe to re-run (idempotent). Use --dry-run to preview changes.

Usage examples:
  python scripts/firestore_migrate_snake_case.py \
    --project YOUR_GCP_PROJECT \
    --data-collection users \
    --chat-collection chat_history \
    --credentials /path/to/service-account.json \
    --dry-run

Then remove --dry-run to apply.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from typing import Any, Dict, Tuple

try:
    from google.cloud import firestore  # type: ignore
    from google.oauth2 import service_account  # type: ignore
except Exception as e:
    print("ERROR: Missing google-cloud-firestore dependencies.\n"
          "Install with: pip install google-cloud-firestore google-auth",
          file=sys.stderr)
    raise


WorkKeyMap = {
    "comparisonVideo": "comparison_video",
    "previewNote": "preview_note",
    "aiNote": "ai_note",
    "gradingOutcome": "grading_outcome",
    # Ensure video field is standardized
    "skeletonVideo": "skeleton_video",
    "video": "skeleton_video",
}


def rename_keys(d: Dict[str, Any], mapping: Dict[str, str]) -> Tuple[Dict[str, Any], bool]:
    changed = False
    out = dict(d)
    for old, new in mapping.items():
        if old in out and new not in out:
            out[new] = out.pop(old)
            changed = True
        elif old in out and new in out:
            # Prefer new; drop old
            out.pop(old)
            changed = True
    return out, changed


def to_snake_case(name: str) -> str:
    """Robust camelCase/PascalCase -> snake_case with acronym handling.

    Examples:
        GPTConversationID -> gpt_conversation_id
        AINote -> ai_note
        previewNote -> preview_note
    """
    import re

    s = name
    # Insert underscore between a lower/digit and upper (e.g., previewNote -> preview_Note)
    s = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", s)
    # Insert underscore before last upper in acronym+Word (e.g., GPTConversation -> GPT_Conversation)
    s = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1_\2", s)
    return s.replace("-", "_").lower()


def correct_common_collapses(key: str) -> str:
    """Fix common acronym collapses after lowercasing.

    Handles cases like gptconversation_id -> gpt_conversation_id, ainote -> ai_note.
    """
    corrections = {
        "gptconversation_id": "gpt_conversation_id",
        "gptconversationids": "gpt_conversation_ids",
        "ainote": "ai_note",
    }
    if key in corrections:
        return corrections[key]
    for prefix in ("gpt", "ai"):
        if key.startswith(prefix) and not key.startswith(prefix + "_") and len(key) > len(prefix):
            return prefix + "_" + key[len(prefix):]
    return key


def deep_snake_keys(obj: Any) -> Any:
    if isinstance(obj, dict):
        new_map: Dict[str, Any] = {}
        for k, v in obj.items():
            new_k = to_snake_case(k) if any(c.isupper() for c in k) else k
            new_k = correct_common_collapses(new_k)
            # If collision, prefer normalized key
            if new_k in new_map and new_k != k:
                # drop original, keep existing
                pass
            else:
                new_map[new_k] = deep_snake_keys(v)
        return new_map
    if isinstance(obj, list):
        return [deep_snake_keys(x) for x in obj]
    return obj


def migrate_user_doc(doc: Dict[str, Any], deep: bool = False) -> Tuple[Dict[str, Any], bool]:
    changed = False
    new_doc = copy.deepcopy(doc)

    # Top-level renames
    top_map = {
        "folderIDs": "folder_paths",
        "gptConversationIDs": "gpt_conversation_ids",
    }
    for old, new in top_map.items():
        if old in new_doc:
            if new not in new_doc:
                new_doc[new] = new_doc[old]
            # Remove old regardless
            new_doc.pop(old, None)
            changed = True

    # Portfolio nested work-item renames
    portfolio = new_doc.get("portfolio")
    if isinstance(portfolio, dict):
        portfolio_changed = False
        # Iterate all skills present to be robust
        for skill, skill_map in list(portfolio.items()):
            if isinstance(skill_map, dict):
                for date_key, work in list(skill_map.items()):
                    if isinstance(work, dict):
                        new_work, wc = rename_keys(work, WorkKeyMap)
                        if wc:
                            skill_map[date_key] = new_work
                            portfolio_changed = True
        if portfolio_changed:
            new_doc["portfolio"] = portfolio
            changed = True

    if deep:
        deep_new = deep_snake_keys(new_doc)
        if deep_new != new_doc:
            new_doc = deep_new
            changed = True
    return new_doc, changed


def migrate_chat_doc(doc: Dict[str, Any], deep: bool = False) -> Tuple[Dict[str, Any], bool]:
    changed = False
    new_doc = copy.deepcopy(doc)

    # userId -> user_id
    if "userId" in new_doc:
        if "user_id" not in new_doc:
            new_doc["user_id"] = new_doc["userId"]
        new_doc.pop("userId", None)
        changed = True

    # messages[*].conversationId -> conversation_id
    msgs = new_doc.get("messages")
    if isinstance(msgs, list):
        msgs_changed = False
        for i, m in enumerate(msgs):
            if isinstance(m, dict) and "conversationId" in m:
                if "conversation_id" not in m:
                    m["conversation_id"] = m.get("conversationId")
                m.pop("conversationId", None)
                msgs_changed = True
        if msgs_changed:
            new_doc["messages"] = msgs
            changed = True

    if deep:
        deep_new = deep_snake_keys(new_doc)
        if deep_new != new_doc:
            new_doc = deep_new
            changed = True
    return new_doc, changed


def diff_dict(old: Dict[str, Any], new: Dict[str, Any]) -> Dict[str, Any]:
    # Shallow diff for display
    d = {}
    for k in sorted(set(list(old.keys()) + list(new.keys()))):
        if old.get(k) != new.get(k):
            d[k] = {"old": old.get(k), "new": new.get(k)}
    return d


def process_collection(col, dry_run: bool, deep: bool) -> int:
    """Process a collection and recursively process all subcollections."""
    updated = 0
    for snap in col.stream():
        doc = snap.to_dict() or {}
        new_doc = deep_snake_keys(doc) if deep else doc
        if new_doc != doc:
            updated += 1
            if dry_run:
                print(f"[DRY-RUN] {col.id}/{snap.id} (recursive) changes:")
                print(json.dumps(diff_dict(doc, new_doc), ensure_ascii=False, indent=2))
            else:
                col.document(snap.id).set(new_doc, merge=False)
                print(f"Updated {col.id}/{snap.id}")
        # Recurse into subcollections
        for sub in snap.reference.collections():
            updated += process_collection(sub, dry_run=dry_run, deep=deep)
    return updated


def run(project: str, data_collection: str, chat_collection: str, creds_path: str | None, dry_run: bool, deep: bool, all_collections: bool, only_collections: list[str] | None) -> None:
    if creds_path:
        creds = service_account.Credentials.from_service_account_file(creds_path)
        client = firestore.Client(project=project, credentials=creds)
    else:
        client = firestore.Client(project=project)

    total_updates = 0

    if all_collections:
        # Traverse all root collections and recurse into subcollections
        for col in client.collections():
            if only_collections and col.id not in only_collections:
                continue
            total_updates += process_collection(col, dry_run=dry_run, deep=deep)
    else:
        # Targeted collections (backward compatible path)
        data_col = client.collection(data_collection)
        for snap in data_col.stream():
            doc = snap.to_dict() or {}
            new_doc, changed = migrate_user_doc(doc, deep=deep)
            if not changed:
                continue
            total_updates += 1
            if dry_run:
                print(f"[DRY-RUN] {data_collection}/{snap.id} changes:")
                print(json.dumps(diff_dict(doc, new_doc), ensure_ascii=False, indent=2))
                continue
            data_col.document(snap.id).set(new_doc, merge=False)
            print(f"Updated {data_collection}/{snap.id}")

        chat_col = client.collection(chat_collection)
        for snap in chat_col.stream():
            doc = snap.to_dict() or {}
            new_doc, changed = migrate_chat_doc(doc, deep=deep)
            if not changed:
                continue
            total_updates += 1
            if dry_run:
                print(f"[DRY-RUN] {chat_collection}/{snap.id} changes:")
                print(json.dumps(diff_dict(doc, new_doc), ensure_ascii=False, indent=2))
                continue
            chat_col.document(snap.id).set(new_doc, merge=False)
            print(f"Updated {chat_collection}/{snap.id}")

    print(f"Done. Documents updated: {total_updates}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Migrate Firestore fields to snake_case.")
    parser.add_argument("--project", required=True, help="GCP project ID")
    parser.add_argument("--data-collection", required=True, help="Users/Data collection name (e.g., 'users')")
    parser.add_argument("--chat-collection", default="chat_history", help="Chat history collection name")
    parser.add_argument("--credentials", help="Path to service account JSON (optional if ADC configured)")
    parser.add_argument("--dry-run", action="store_true", help="Preview changes without writing")
    parser.add_argument("--deep", action="store_true", help="Recursively convert ALL camelCase/PascalCase keys to snake_case")
    parser.add_argument("--all-collections", action="store_true", help="Process all root collections and subcollections")
    parser.add_argument("--collections", nargs="*", help="Limit to these collection IDs when using --all-collections")
    args = parser.parse_args()

    run(
        project=args.project,
        data_collection=args.data_collection,
        chat_collection=args.chat_collection,
        creds_path=args.credentials,
        dry_run=args.dry_run,
        deep=args.deep,
        all_collections=args.all_collections,
        only_collections=args.collections,
    )


if __name__ == "__main__":
    main()
