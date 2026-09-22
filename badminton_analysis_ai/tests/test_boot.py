"""The entry point has to open the port before the heavy imports: on a fresh
Cloud Run GPU instance those took 13 to 33 s, and the platform abandons an
instance whose port is not open by roughly 22 to 30 s."""

import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

import api.boot as boot
from api.config import Settings

ROOT = Path(__file__).resolve().parents[1]


def _settings() -> Settings:
    return Settings(
        port=0, max_video_bytes=1, grpc_api_key="key", gcp_project_id="p",
        gcs_bucket_name="b", gcp_service_account_email="", signed_url_minutes=60,
        expert_motion_model_root=Path("models"), device="cpu",
        openai_model="m", coaching_pause_seconds=2.0,
    )


def test_importing_the_entry_point_loads_nothing_heavy() -> None:
    probe = (
        "import sys, api.boot; "
        "print(','.join(m for m in ('torch', 'torchvision', 'cv2', 'numpy', "
        "'api.server', 'api.pipeline') if m in sys.modules))"
    )
    loaded = subprocess.run(
        [sys.executable, "-c", probe], cwd=ROOT, capture_output=True, text=True,
        env={"PYTHONPATH": f"{ROOT}:{ROOT / 'generated'}"}, check=True,
    ).stdout.strip()

    assert loaded == "", f"imported before the port opens: {loaded}"


class _Context:
    def abort(self, code, details):
        raise RuntimeError(f"{code}: {details}")


def test_calls_wait_for_the_service_and_then_reach_it() -> None:
    release = threading.Event()

    class Service:
        def Health(self, request, context):
            return ("healthy", request)

    def slow_load(_settings):
        release.wait(5)
        return Service()

    started = time.monotonic()
    deferred = boot.DeferredAnalysisService(_settings(), slow_load)
    assert time.monotonic() - started < 0.5, "construction must not wait for the service"

    answers = {}
    caller = threading.Thread(
        target=lambda: answers.setdefault("health", deferred.Health("ping", _Context()))
    )
    caller.start()
    caller.join(0.2)
    assert caller.is_alive(), "a call waits for the service instead of failing"

    release.set()
    caller.join(5)
    assert answers["health"] == ("healthy", "ping")


def test_a_service_that_fails_to_load_takes_the_instance_down(monkeypatch) -> None:
    exited = threading.Event()
    codes = []

    def fake_exit(code):
        codes.append(code)
        exited.set()
        raise SystemExit(code)

    def broken_load(_settings):
        raise RuntimeError("model file missing")

    monkeypatch.setattr(boot.os, "_exit", fake_exit)
    boot.DeferredAnalysisService(_settings(), broken_load)

    assert exited.wait(5)
    assert codes == [1]
