from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from badminton_analysis.ml.infer_skeleton_corrector import load_corrector


def export_model(checkpoint_path: Path, output_path: Path) -> None:
    model, checkpoint = load_corrector(checkpoint_path, torch.device("cpu"))
    frames = int(checkpoint.get("sequence_frames", 64))
    features = int(checkpoint["model_config"]["input_features"])
    dummy = torch.zeros((1, frames, 17, features), dtype=torch.float32)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model,
        dummy,
        output_path,
        input_names=("features",),
        output_names=("corrected_skeleton",),
        opset_version=17,
        do_constant_folding=True,
    )

    import onnxruntime as ort

    session = ort.InferenceSession(
        str(output_path), providers=("CPUExecutionProvider",)
    )
    sample = np.random.default_rng(2026).normal(
        size=dummy.shape
    ).astype(np.float32)
    with torch.inference_mode():
        expected = model(torch.from_numpy(sample)).numpy()
    actual = session.run(None, {"features": sample})[0]
    np.testing.assert_allclose(actual, expected, rtol=2e-4, atol=2e-4)
    print(f"exported={output_path} bytes={output_path.stat().st_size}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Export and verify a skeleton corrector as fixed-shape ONNX"
    )
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    export_model(args.checkpoint, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
