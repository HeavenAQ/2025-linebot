"""Minimal torch-native TensorRT engine runner.

Binds engine inputs/outputs directly to `torch.cuda` tensor memory instead of
going through `pycuda` (which needs the full CUDA toolkit's dev headers to
build from source and isn't otherwise available here). This is the standard
zero-copy pattern for running a TensorRT engine from a process that already
has PyTorch/CUDA initialized.
"""

from __future__ import annotations

import time
from typing import Final

import tensorrt as trt
import torch

_TRT_TO_TORCH_DTYPE: Final = {
    trt.float32: torch.float32,
    trt.float16: torch.float16,
    trt.int32: torch.int32,
    trt.int8: torch.int8,
    trt.bool: torch.bool,
}


class TorchTRTEngine:
    def __init__(self, engine_path: str) -> None:
        logger = trt.Logger(trt.Logger.WARNING)
        trt.init_libnvinfer_plugins(logger, "")
        with open(engine_path, "rb") as engine_file, trt.Runtime(logger) as runtime:
            engine = runtime.deserialize_cuda_engine(engine_file.read())
        if engine is None:
            raise RuntimeError(f"failed to deserialize TensorRT engine: {engine_path}")
        self.engine = engine
        self.context = self.engine.create_execution_context()
        self.input_names = [
            name
            for name in self.engine
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT
        ]
        self.output_names = [
            name
            for name in self.engine
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.OUTPUT
        ]
        self.stream = torch.cuda.Stream()

    def __call__(self, inputs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        outputs: dict[str, torch.Tensor] = {}
        keepalive: list[torch.Tensor] = []
        for name in self.input_names:
            tensor = inputs[name].contiguous().cuda()
            self.context.set_input_shape(name, tuple(tensor.shape))
            self.context.set_tensor_address(name, tensor.data_ptr())
            keepalive.append(tensor)
        for name in self.output_names:
            shape = tuple(self.context.get_tensor_shape(name))
            dtype = _TRT_TO_TORCH_DTYPE[self.engine.get_tensor_dtype(name)]
            out = torch.empty(shape, dtype=dtype, device="cuda")
            self.context.set_tensor_address(name, out.data_ptr())
            outputs[name] = out
        with torch.cuda.stream(self.stream):
            self.context.execute_async_v3(self.stream.cuda_stream)
        self.stream.synchronize()
        return outputs

    def speed(self, inputs: dict[str, torch.Tensor], n: int = 30) -> float:
        for _ in range(5):
            self(inputs)
        torch.cuda.synchronize()
        start = time.time()
        for _ in range(n):
            self(inputs)
        torch.cuda.synchronize()
        return 1000.0 * (time.time() - start) / n
