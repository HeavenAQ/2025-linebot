from __future__ import annotations

import argparse
import os
from pathlib import Path

import cv2
import numpy as np
import torch

from badminton_analysis.services.pose_detector import PoseDetector


def build_cache(
    cache_root: Path,
    pose_image: Path | None,
    detector_model: str | None,
    pose_model: str | None,
    device_id: int,
    reuse_pose_cache: bool,
) -> None:
    cache_root.mkdir(parents=True, exist_ok=True)
    torch.cuda.set_device(device_id)
    os.environ["POSE_EXECUTION_PROVIDER"] = "tensorrt"
    os.environ["POSE_TENSORRT_CACHE_DIR"] = str(cache_root.resolve())
    os.environ["ONNXRUNTIME_DEVICE_ID"] = str(device_id)

    if not reuse_pose_cache:
        detector = PoseDetector(
            detector_model=detector_model, model_path=pose_model
        )
        image = (
            cv2.imread(str(pose_image))
            if pose_image is not None
            else np.zeros((720, 1280, 3), dtype=np.uint8)
        )
        if image is None:
            raise ValueError(f"could not read pose image: {pose_image}")
        detector.get_pose(image)
        if not detector.active_execution_providers or not all(
            providers[0] == "TensorrtExecutionProvider"
            for providers in detector.active_execution_providers.values()
        ):
            raise RuntimeError("TensorRT did not activate for every pose component")

    expected = {
        "detector": cache_root / "detector",
        "pose": cache_root / "pose",
    }
    for name, directory in expected.items():
        engines = list(directory.glob("*.engine"))
        if not engines:
            raise RuntimeError(f"no TensorRT engine produced for {name}")
        print(f"{name}: {sum(path.stat().st_size for path in engines)} bytes")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build the production detector/pose TensorRT cache"
    )
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--pose-image", type=Path)
    parser.add_argument("--detector-model")
    parser.add_argument("--pose-model")
    parser.add_argument("--device-id", type=int, default=0)
    parser.add_argument(
        "--reuse-pose-cache",
        action="store_true",
        help="Verify existing detector/pose engines without rebuilding them",
    )
    args = parser.parse_args()
    build_cache(
        args.cache_root,
        args.pose_image,
        args.detector_model,
        args.pose_model,
        args.device_id,
        args.reuse_pose_cache,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
