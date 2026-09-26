"""Build the two pose TensorRT engines once, on the GPU that will serve them.

The pipeline is two stages -- RF-DETR Medium for the player's box, ViTPose++-L
for the joints -- so there are two engines. TensorRT engines are tied to the
GPU they are built on, so this runs as a Cloud Run job on the same L4 the
analysis service uses. Each result is published to Artifact Registry as a
generic artifact (see models/trt-engine.env) and baked into the image by the
deploy workflow, which is the whole point: without them the service would
build engines on every cold start, and it scales to zero.

Running this costs GPU-minutes on an L4, so it skips an engine whose version is
already published. Rebuilding is deliberate -- a new GPU type, a new TensorRT,
RF-DETR or transformers version -- and needs a new version in
models/trt-engine.env. After publishing, record the printed SHA-256 lines in
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


def version_exists(
    session: AuthorizedSession,
    config: dict[str, str],
    project: str,
    package: str,
    version: str,
) -> bool:
    url = (
        f"{API}/v1/{repository_name(config, project)}"
        f"/packages/{package}/versions/{version}"
    )
    response = session.get(url, timeout=30)
    if response.status_code == 404:
        return False
    response.raise_for_status()
    return True


def upload_engine(
    session: AuthorizedSession,
    config: dict[str, str],
    project: str,
    engine: Path,
    package: str,
    version: str,
) -> None:
    """Publish one file as a generic artifact version (multipart media upload)."""
    boundary = "pose-trt-engine-upload"
    metadata = json.dumps(
        {"packageId": package, "versionId": version, "filename": engine.name}
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
        timeout=900,
    )
    response.raise_for_status()
    operation = response.json().get("operation", {})
    if operation.get("error"):
        raise RuntimeError(f"artifact upload failed: {operation['error']}")


def build_detector_engine(destination: Path, batch_size: int) -> Path:
    """RF-DETR Medium, exported through its own TensorRT export path."""
    from rfdetr import RFDETRMedium

    engine = destination / "rfdetr-medium.trt"
    if engine.exists():
        return engine
    destination.mkdir(parents=True, exist_ok=True)
    # The pose engine lands in the same directory and also ends in .trt, so
    # note what is there before the export rather than guessing afterwards.
    existing = set(destination.glob("*.trt"))
    model = RFDETRMedium(device="cuda")
    model.export(
        output_dir=str(destination),
        format="tensorrt",
        fp16=True,
        batch_size=batch_size,
        dynamic_batch=False,
        verbose=False,
    )
    produced = [path for path in destination.glob("*.trt") if path not in existing]
    if not produced:
        raise RuntimeError(f"RF-DETR export produced no engine in {destination}")
    if len(produced) > 1:
        raise RuntimeError(f"RF-DETR export produced several engines: {produced}")
    # The export names the file after the checkpoint; give it the name the
    # detector looks up.
    produced[0].rename(engine)
    return engine


def build_pose_engine(destination: Path, batch_size: int) -> Path:
    """ViTPose++-L, via ONNX.

    The checkpoint is a mixture of experts over its training datasets, so the
    router is pinned to the COCO expert before export: the service only ever
    asks for the 17 COCO joints, and a fixed route keeps the engine to a single
    static path.
    """
    import tensorrt as trt
    import torch
    from transformers import VitPoseForPoseEstimation

    from badminton_analysis.services.pose_detector import (
        POSE_CHECKPOINT,
        POSE_DATASET_INDEX,
        POSE_INPUT_HEIGHT,
        POSE_INPUT_WIDTH,
    )

    engine_path = destination / "vitpose-plus-large.trt"
    if engine_path.exists():
        return engine_path
    destination.mkdir(parents=True, exist_ok=True)
    onnx_path = destination / "vitpose-plus-large.onnx"

    class CocoViTPose(torch.nn.Module):
        def __init__(self, inner: torch.nn.Module) -> None:
            super().__init__()
            self.inner = inner

        def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
            index = torch.full(
                (pixel_values.shape[0],),
                POSE_DATASET_INDEX,
                dtype=torch.long,
                device=pixel_values.device,
            )
            return self.inner(pixel_values=pixel_values, dataset_index=index).heatmaps

    model = VitPoseForPoseEstimation.from_pretrained(
        POSE_CHECKPOINT, dtype=torch.float32
    ).eval()
    wrapped = CocoViTPose(model).eval()
    dummy = torch.randn(batch_size, 3, POSE_INPUT_HEIGHT, POSE_INPUT_WIDTH)
    torch.onnx.export(
        wrapped,
        (dummy,),
        str(onnx_path),
        input_names=["pixel_values"],
        output_names=["heatmaps"],
        opset_version=17,
        do_constant_folding=True,
    )
    print(f"exported {onnx_path} ({onnx_path.stat().st_size / 1e6:.0f} MB)", flush=True)

    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    )
    parser = trt.OnnxParser(network, logger)
    if not parser.parse(onnx_path.read_bytes()):
        errors = "; ".join(str(parser.get_error(i)) for i in range(parser.num_errors))
        raise RuntimeError(f"ONNX parse failed: {errors}")
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 4 << 30)
    # The detector engine is FP16 and this one reads its crops; matching the
    # precision keeps the two stages consistent and halves the pose cost.
    config.set_flag(trt.BuilderFlag.FP16)
    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError("TensorRT failed to build the ViTPose engine")
    engine_path.write_bytes(serialized)
    onnx_path.unlink()
    return engine_path


def main() -> int:
    import torch

    from badminton_analysis.services.pose_detector import BATCH_SIZE

    if not torch.cuda.is_available():
        raise RuntimeError("no GPU visible; the engines must be built on the target GPU")
    gpu_name = torch.cuda.get_device_name(0)
    print(f"building on {gpu_name}", flush=True)

    config = load_engine_config()
    cache_root = Path(os.environ["BADMINTON_TRT_CACHE_DIR"])
    relative = Path(gpu_name.replace(" ", "_")) / f"batch{BATCH_SIZE}"
    # The artifact versions name one GPU; refuse to publish engines built on
    # different hardware under them.
    if relative != Path(config["TRT_ENGINE_SUBDIR"]):
        raise RuntimeError(
            f"this GPU produces {relative}, but {CONFIG_PATH.name} expects "
            f"{config['TRT_ENGINE_SUBDIR']}"
        )
    destination = cache_root / relative

    credentials, project = google.auth.default(
        scopes=["https://www.googleapis.com/auth/cloud-platform"]
    )
    project = os.environ.get("GCP_PROJECT_ID", project)
    session = AuthorizedSession(credentials)

    stages = (
        ("detector", config["TRT_DETECTOR_PACKAGE"], config["TRT_DETECTOR_VERSION"],
         config["TRT_DETECTOR_FILENAME"], build_detector_engine),
        ("pose", config["TRT_POSE_PACKAGE"], config["TRT_POSE_VERSION"],
         config["TRT_POSE_FILENAME"], build_pose_engine),
    )
    digests: list[str] = []
    for stage, package, version, filename, build in stages:
        target = f"{package}@{version}"
        # The guard: this job holds an L4 for the whole build, so an accidental
        # re-run -- a retried job, a workflow wired up twice -- is pure cost for
        # an engine that is already published.
        if version_exists(session, config, project, package, version):
            print(f"{stage}: {target} already exists; skipping.", flush=True)
            continue
        engine = build(destination, BATCH_SIZE)
        if engine.name != filename:
            raise RuntimeError(
                f"{stage}: built {engine.name}, but the config expects {filename}"
            )
        digest = hashlib.sha256(engine.read_bytes()).hexdigest()
        print(
            f"{stage}: built {engine} ({engine.stat().st_size / 1e6:.1f} MB) "
            f"sha256={digest}",
            flush=True,
        )
        upload_engine(session, config, project, engine, package, version)
        print(f"{stage}: published {target}", flush=True)
        digests.append(f"{digest}  ./{relative}/{engine.name}")

    if digests:
        print("record in models/trt-engines.sha256:", flush=True)
        for line in digests:
            print(line, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
