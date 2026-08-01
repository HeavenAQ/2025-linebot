from __future__ import annotations

import argparse
import os
from pathlib import Path

import cv2
import numpy as np
import torch

from badminton_analysis.ml.skill_specs import get_skill_spec
from badminton_analysis.ml.skeleton_backend import SkeletonCorrectionBackend
from badminton_analysis.models.types import Skill
from badminton_analysis.services.pose_detector import PoseDetector


def build_cache(
    model_root: Path,
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
    os.environ["SKELETON_EXECUTION_PROVIDER"] = "tensorrt"
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

    for skill in (Skill.SERVE, Skill.LIFT, Skill.CLEAR, Skill.SMASH):
        spec = get_skill_spec(skill)
        backend = SkeletonCorrectionBackend(
            model_root / f"{spec.model_stem}.pt", device=f"cuda:{device_id}"
        )
        if (
            not backend.inference_providers
            or backend.inference_providers[0] != "TensorrtExecutionProvider"
            or backend.inference_session is None
        ):
            raise RuntimeError(f"TensorRT did not activate for {spec.slug}")
        input_metadata = backend.inference_session.get_inputs()[0]
        features = np.zeros((1, 64, 17, 7), dtype=np.float32)
        backend.inference_session.run(None, {input_metadata.name: features})
        print(f"built correction engine: {spec.slug}")

    expected = {
        "detector": cache_root / "detector",
        "pose": cache_root / "pose",
        **{
            f"corrector-{skill}": cache_root / "correctors" / skill
            for skill in ("serve", "lift", "clear", "smash")
        },
    }
    for name, directory in expected.items():
        engines = list(directory.glob("*.engine"))
        if not engines:
            raise RuntimeError(f"no TensorRT engine produced for {name}")
        print(f"{name}: {sum(path.stat().st_size for path in engines)} bytes")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build every production TensorRT engine and verify activation"
    )
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--pose-image", type=Path)
    parser.add_argument("--detector-model")
    parser.add_argument("--pose-model")
    parser.add_argument("--device-id", type=int, default=0)
    parser.add_argument(
        "--reuse-pose-cache",
        action="store_true",
        help="Verify existing detector/pose engines and build only correctors",
    )
    args = parser.parse_args()
    build_cache(
        args.model_root,
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
