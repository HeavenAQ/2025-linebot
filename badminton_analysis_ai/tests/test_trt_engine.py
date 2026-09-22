import importlib.machinery
import sys
import types
from unittest import mock

import pytest
import torch


@pytest.fixture
def trt_engine_module(monkeypatch):
    fake_trt = types.ModuleType("tensorrt")
    fake_trt.__spec__ = importlib.machinery.ModuleSpec("tensorrt", None)
    fake_trt.float32, fake_trt.float16, fake_trt.int32, fake_trt.int8, fake_trt.bool = (
        "f32", "f16", "i32", "i8", "b"
    )
    monkeypatch.setitem(sys.modules, "tensorrt", fake_trt)
    monkeypatch.delitem(sys.modules, "badminton_analysis.services.trt_engine", raising=False)
    import badminton_analysis.services.trt_engine as module

    yield module
    sys.modules.pop("badminton_analysis.services.trt_engine", None)


def test_engine_waits_for_the_callers_stream_before_it_executes(trt_engine_module, monkeypatch):
    events = []
    caller_stream = object()
    engine = object.__new__(trt_engine_module.TorchTRTEngine)
    engine.input_names = ["input"]
    engine.output_names = ["dets"]
    engine.engine = mock.Mock(get_tensor_dtype=lambda name: "f32")
    engine.context = mock.Mock(
        get_tensor_shape=lambda name: (1, 4),
        execute_async_v3=lambda handle: events.append("execute"),
    )
    engine.stream = mock.Mock(
        cuda_stream=7,
        wait_stream=lambda stream: events.append(("wait", stream)),
        synchronize=lambda: events.append("synchronize"),
    )
    monkeypatch.setattr(torch.cuda, "current_stream", lambda: caller_stream)
    monkeypatch.setattr(torch.cuda, "stream", lambda stream: mock.MagicMock())
    monkeypatch.setattr(torch.Tensor, "cuda", lambda self: self)
    real_empty = torch.empty
    monkeypatch.setattr(torch, "empty", lambda shape, dtype, device: real_empty(shape, dtype=dtype))

    engine({"input": torch.zeros(1, 3)})

    assert events == [("wait", caller_stream), "execute", "synchronize"]
