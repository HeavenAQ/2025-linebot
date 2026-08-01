from __future__ import annotations

import sys
from types import SimpleNamespace
from unittest.mock import MagicMock

from badminton_analysis.services.pose_detector import PoseDetector


def test_onnx_gpu_model_bootstraps_on_cpu_before_provider_assignment(
    monkeypatch,
) -> None:
    calls: dict[str, object] = {}

    def wholebody3d(**kwargs):
        calls.update(kwargs)
        return object()

    monkeypatch.setitem(
        sys.modules,
        "rtmlib",
        SimpleNamespace(Wholebody3d=wholebody3d),
    )
    detector = PoseDetector()
    detector.device = "cuda"
    detector.backend = "onnxruntime"
    detector._configure_execution_providers = MagicMock()

    model = detector._load_inferencer()

    assert calls["device"] == "cpu"
    detector._configure_execution_providers.assert_called_once_with(model)
