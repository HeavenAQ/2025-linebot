"""Exercise real Cloud Tasks -> Go -> GPU -> Firestore, without sending LINE messages.

Uses gcloud impersonation, never a key file. Creates a uniquely named test user
and removes only that user's records and uploaded input after terminal success.
Generated output media remain available for inspection in the test user's path.
Run with uv --with google-cloud-firestore --with google-cloud-storage --with
python-dotenv, after configuring workers but before enabling queued uploads.
"""

import argparse
import base64
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import subprocess
import time
import urllib.request
import uuid

from dotenv import dotenv_values
from google.cloud import firestore, storage
from google.oauth2.credentials import Credentials


def gcloud(*args):
    return subprocess.check_output(["gcloud", *args], text=True).strip()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", type=Path)
    parser.add_argument("--variant", choices=["llm", "no-llm"], required=True)
    parser.add_argument("--skill", choices=["serve", "smash"], required=True)
    parser.add_argument("--handedness", choices=["left", "right"], default="right")
    parser.add_argument("--minimum", type=float, default=0)
    parser.add_argument("--expected", type=float, help="Assert score parity")
    parser.add_argument(
        "--tolerance",
        type=float,
        default=0.001,
        help="Absolute points on the 100-point scale; allows GPU round-off",
    )
    args = parser.parse_args()
    project = "nstc-linebot-2025"
    account = f"nstc-linebot-2025@{project}.iam.gserviceaccount.com"
    no_llm = args.variant == "no-llm"
    secret = "2025-linebot-noai-env" if no_llm else "2025-linebot-env"
    # Parse in memory; never print or persist secret contents.
    config = dotenv_values(
        stream=io.StringIO(
            gcloud(
                "secrets",
                "versions",
                "access",
                "latest",
                f"--secret={secret}",
                f"--project={project}",
                f"--impersonate-service-account={account}",
            )
        )
    )
    token = gcloud(
        "auth", "print-access-token", f"--impersonate-service-account={account}"
    )
    credentials = Credentials(token)
    db = firestore.Client(
        project=project,
        credentials=credentials,
        database=config.get("FIREBASE_DATABASE_ID") or "(default)",
    )
    bucket = storage.Client(project=project, credentials=credentials).bucket(
        config["GCS_BUCKET_NAME"]
    )
    service = "nstc-linebot-2025-noai" if no_llm else "nstc-linebot-2025"
    worker = gcloud(
        "run",
        "services",
        "describe",
        service,
        "--region=asia-east1",
        f"--project={project}",
        "--format=value(status.url)",
    )
    user_id = f"async-smoke-{uuid.uuid4().hex}"
    job_id = hashlib.sha256(user_id.encode()).hexdigest()
    date = "2099-01-01-00-00-00-" + job_id[:8]
    users = db.collection(config["FIREBASE_DATA_DB"])
    jobs = db.collection(config["FIREBASE_DATA_DB"] + "_analysis_jobs")
    source = bucket.blob(f"analyses/input/{user_id}/{job_id}.mp4")
    source.upload_from_filename(str(args.video), content_type="video/mp4")
    users.document(user_id).create(
        {
            "id": user_id,
            "portfolio": {
                args.skill: {
                    date: {
                        "analysis_status": "pending",
                        "reflection": "keep this test note",
                    }
                }
            },
        }
    )
    jobs.document(job_id).create(
        {
            "id": job_id,
            "user_id": user_id,
            "skill": args.skill,
            "handedness": args.handedness,
            "work_date": date,
            "input_object": source.name,
            "thumbnail": "",
            "status": "queued",
            "attempts": 0,
            "created_at": datetime.now(timezone.utc),
            "lease_until": datetime(1970, 1, 1, tzinfo=timezone.utc),
        }
    )
    queue = f"projects/{project}/locations/asia-east1/queues/analysis-{args.variant}"
    task = {
        "task": {
            "name": f"{queue}/tasks/{job_id}",
            "dispatchDeadline": "1200s",
            "httpRequest": {
                "httpMethod": "POST",
                "url": worker + "/internal/analysis/task",
                "headers": {"Content-Type": "application/json"},
                "body": base64.b64encode(
                    json.dumps({"job_id": job_id}).encode()
                ).decode(),
                "oidcToken": {"serviceAccountEmail": account, "audience": worker},
            },
        }
    }
    request = urllib.request.Request(
        f"https://cloudtasks.googleapis.com/v2/{queue}/tasks",
        data=json.dumps(task).encode(),
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        },
    )
    with urllib.request.urlopen(request, timeout=30):
        pass
    print(
        json.dumps({"variant": args.variant, "job": job_id, "test_user": user_id}),
        flush=True,
    )
    started = time.monotonic()
    previous = None
    # Poll only in this opt-in verification harness; the application never polls GPU jobs.
    while time.monotonic() - started < 1500:
        job = jobs.document(job_id).get().to_dict()
        if job["status"] != previous:
            previous = job["status"]
            print(
                json.dumps(
                    {
                        "status": previous,
                        "seconds": round(time.monotonic() - started, 1),
                    }
                ),
                flush=True,
            )
        if previous in {"completed", "failed"}:
            work = (
                users.document(user_id).get().to_dict()["portfolio"][args.skill][date]
            )
            print(
                json.dumps(
                    {
                        "variant": args.variant,
                        "status": previous,
                        "grade": work.get("grading_outcome"),
                        "error": work.get("analysis_error"),
                        "student_object": (work.get("student_video") or {}).get(
                            "object_path"
                        ),
                        "seconds": round(time.monotonic() - started, 1),
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
            assert (
                previous == "completed"
            ), "worker failed; records retained for inspection"
            assert work["reflection"] == "keep this test note"
            assert work["grading_outcome"]["total_grade"] >= args.minimum
            if args.expected is not None:
                difference = abs(work["grading_outcome"]["total_grade"] - args.expected)
                print(
                    json.dumps(
                        {"score_delta": difference, "tolerance": args.tolerance}
                    ),
                    flush=True,
                )
                assert difference <= args.tolerance
            assert work.get("student_video") and work.get("expert")
            if no_llm:
                assert not work.get("coaching_cues") and not work.get("ai_note")
            jobs.document(job_id).delete()
            users.document(user_id).delete()
            source.delete()
            print(
                "PASS; removed only test job, user and input. Output media retained.",
                flush=True,
            )
            return
        time.sleep(10)
    raise TimeoutError(f"job {job_id} still pending; records retained for inspection")


if __name__ == "__main__":
    main()
