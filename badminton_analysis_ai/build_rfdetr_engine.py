"""Build the RF-DETR TensorRT engine once, on the GPU that will serve it.

TensorRT engines are tied to the GPU they are built on, so this runs as a
Cloud Run job on the same L4 the analysis service uses. The result is uploaded
to GCS and baked into the image by the deploy workflow, which is the whole
point: without it the service rebuilds the engine on every cold start, and it
scales to zero.
"""

from __future__ import annotations

import os
from pathlib import Path

from google.cloud import storage


def main() -> int:
    import torch

    from badminton_analysis.services.pose_detector import BATCH_SIZE, PoseDetector

    if not torch.cuda.is_available():
        raise RuntimeError("no GPU visible; the engine must be built on the target GPU")
    gpu_name = torch.cuda.get_device_name(0)
    print(f"building on {gpu_name}", flush=True)

    cache_root = Path(os.environ["BADMINTON_TRT_CACHE_DIR"])
    detector = PoseDetector()
    # Compiles and caches the engine; ~2 minutes, and the reason this job exists.
    detector._load_or_build_batched_engine()

    engine = cache_root / gpu_name.replace(" ", "_") / f"batch{BATCH_SIZE}" / "rfdetr-keypoint-preview.trt"
    if not engine.exists():
        raise RuntimeError(f"engine was not produced at {engine}")
    print(f"built {engine} ({engine.stat().st_size / 1e6:.1f} MB)", flush=True)

    bucket_name = os.environ["GCS_BUCKET_NAME"]
    prefix = os.environ.get("ENGINE_UPLOAD_PREFIX", "models/rfdetr-trt-engines")
    relative = engine.relative_to(cache_root)
    blob_name = f"{prefix}/{relative}"
    storage.Client().bucket(bucket_name).blob(blob_name).upload_from_filename(str(engine))
    print(f"uploaded gs://{bucket_name}/{blob_name}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
