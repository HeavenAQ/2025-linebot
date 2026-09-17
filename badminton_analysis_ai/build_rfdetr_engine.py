"""Build the RF-DETR TensorRT engine once, on the GPU that will serve it.

TensorRT engines are tied to the GPU they are built on, so this runs as a
Cloud Run job on the same L4 the analysis service uses. The result is published
to Artifact Registry as a generic artifact (see models/trt-engine.env) and baked
into the image by the deploy workflow, which is the whole point: without it the
service rebuilds the engine on every cold start, and it scales to zero.

Running this costs GPU-minutes on an L4, so it refuses to run when the version
it would publish already exists. Rebuilding is deliberate -- a new GPU type, a
new TensorRT or RF-DETR version -- and needs a new TRT_ENGINE_VERSION in
models/trt-engine.env. After publishing, record the printed SHA-256 in
models/trt-engines.sha256.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import google.auth
from google.auth.transport.requests import AuthorizedSession

CONFIG_PATH = Path(__file__).resolve().parent / "models" / "trt-engine.env"
API = "https://artifactregistry.googleapis.com"


def load_engine_config(path: Path = CONFIG_PATH) -> dict[str, str]:
    """KEY=VALUE lines, the same file the deploy workflow sources."""
    config: dict[str, str] = {}
    for line in path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            key, value = line.split("=", 1)
            config[key] = value
    return config


def repository_name(config: dict[str, str], project: str) -> str:
    return (
        f"projects/{project}/locations/{config['TRT_ENGINE_LOCATION']}"
        f"/repositories/{config['TRT_ENGINE_REPOSITORY']}"
    )


def version_exists(session: AuthorizedSession, config: dict[str, str], project: str) -> bool:
    url = (
        f"{API}/v1/{repository_name(config, project)}"
        f"/packages/{config['TRT_ENGINE_PACKAGE']}/versions/{config['TRT_ENGINE_VERSION']}"
    )
    response = session.get(url, timeout=30)
    if response.status_code == 404:
        return False
    response.raise_for_status()
    return True


def upload_engine(
    session: AuthorizedSession, config: dict[str, str], project: str, engine: Path
) -> None:
    """Publish one file as a generic artifact version (multipart media upload)."""
    boundary = "rfdetr-trt-engine-upload"
    metadata = json.dumps(
        {
            "packageId": config["TRT_ENGINE_PACKAGE"],
            "versionId": config["TRT_ENGINE_VERSION"],
            "filename": config["TRT_ENGINE_FILENAME"],
        }
    ).encode()
    body = b"".join(
        (
            f"--{boundary}\r\nContent-Type: application/json; charset=UTF-8\r\n\r\n".encode(),
            metadata,
            f"\r\n--{boundary}\r\nContent-Type: application/octet-stream\r\n\r\n".encode(),
            engine.read_bytes(),
            f"\r\n--{boundary}--\r\n".encode(),
        )
    )
    response = session.post(
        f"{API}/upload/v1/{repository_name(config, project)}/genericArtifacts:create",
        params={"alt": "json", "uploadType": "multipart"},
        headers={"Content-Type": f"multipart/related; boundary={boundary}"},
        data=body,
        timeout=600,
    )
    response.raise_for_status()
    operation = response.json().get("operation", {})
    if operation.get("error"):
        raise RuntimeError(f"artifact upload failed: {operation['error']}")


def main() -> int:
    import torch

    from badminton_analysis.services.pose_detector import BATCH_SIZE, PoseDetector

    if not torch.cuda.is_available():
        raise RuntimeError("no GPU visible; the engine must be built on the target GPU")
    gpu_name = torch.cuda.get_device_name(0)
    print(f"building on {gpu_name}", flush=True)

    config = load_engine_config()
    cache_root = Path(os.environ["BADMINTON_TRT_CACHE_DIR"])
    relative = Path(gpu_name.replace(" ", "_")) / f"batch{BATCH_SIZE}"
    # The artifact version names one GPU; refuse to publish an engine built on
    # different hardware under it.
    if relative != Path(config["TRT_ENGINE_SUBDIR"]):
        raise RuntimeError(
            f"this GPU produces {relative}, but {CONFIG_PATH.name} expects "
            f"{config['TRT_ENGINE_SUBDIR']}"
        )
    engine = cache_root / relative / config["TRT_ENGINE_FILENAME"]

    credentials, project = google.auth.default(
        scopes=["https://www.googleapis.com/auth/cloud-platform"]
    )
    project = os.environ.get("GCP_PROJECT_ID", project)
    session = AuthorizedSession(credentials)
    target = f"{repository_name(config, project)} {config['TRT_ENGINE_PACKAGE']}@{config['TRT_ENGINE_VERSION']}"

    # The guard: this job holds an L4 for the whole build, so an accidental
    # re-run -- a retried job, a workflow wired up twice -- is pure cost for an
    # engine that is already published.
    if version_exists(session, config, project):
        print(f"{target} already exists; refusing to rebuild.", flush=True)
        return 0

    detector = PoseDetector()
    # Compiles and caches the engine; ~2 minutes, and the reason this job exists.
    detector._load_or_build_batched_engine()

    if not engine.exists():
        raise RuntimeError(f"engine was not produced at {engine}")
    digest = hashlib.sha256(engine.read_bytes()).hexdigest()
    print(f"built {engine} ({engine.stat().st_size / 1e6:.1f} MB) sha256={digest}", flush=True)

    upload_engine(session, config, project, engine)
    print(f"published {target}", flush=True)
    print(f"record in models/trt-engines.sha256: {digest}  ./{relative}/{engine.name}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
