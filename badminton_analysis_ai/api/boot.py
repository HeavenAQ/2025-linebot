"""Entry point that opens the port before anything heavy is imported.

On a fresh Cloud Run GPU instance, importing torch and the analysis modules
took 13 to 33 s, and the platform abandons an instance whose port is not open
by roughly 22 to 30 s. Only grpc and the generated stubs load before binding;
the service itself loads behind the open port, and calls wait for it.
"""

from __future__ import annotations

import logging
import os
import signal
import threading
from concurrent import futures
from typing import Any, Callable

import grpc
from badminton.analysis.v1 import analysis_pb2_grpc

from api.config import Settings
from api.logging_config import configure_logging

LOGGER = logging.getLogger("badminton-analysis")

# Loading normally takes seconds; this only bounds a load that has hung.
_LOAD_TIMEOUT_SECONDS = 600


class DeferredAnalysisService(analysis_pb2_grpc.BadmintonAnalysisServicer):
    """Answers on the port at once and hands each call to the real service
    once it has loaded."""

    def __init__(self, settings: Settings, load: Callable[[Settings], Any]) -> None:
        self._settings = settings
        self._load = load
        self._service: Any = None
        self._ready = threading.Event()
        threading.Thread(target=self._build, name="service-load", daemon=True).start()

    def _build(self) -> None:
        try:
            self._service = self._load(self._settings)
        except BaseException:
            LOGGER.exception("analysis service failed to load")
            # An open port with nothing behind it would fail every call while
            # looking healthy; exiting lets Cloud Run replace the instance.
            os._exit(1)
        self._ready.set()
        LOGGER.info("analysis service loaded")

    def _loaded(self, context: grpc.ServicerContext) -> Any:
        if not self._ready.wait(_LOAD_TIMEOUT_SECONDS):
            context.abort(grpc.StatusCode.UNAVAILABLE, "analysis service is still loading")
        return self._service

    def AnalyzeVideo(self, request_iterator, context):
        return self._loaded(context).AnalyzeVideo(request_iterator, context)

    def RefreshPlaybackUrls(self, request, context):
        return self._loaded(context).RefreshPlaybackUrls(request, context)

    def Health(self, request, context):
        return self._loaded(context).Health(request, context)


def _load_analysis_service(settings: Settings) -> Any:
    from api.server import BadmintonAnalysisService

    return BadmintonAnalysisService(settings)


def serve() -> None:
    configure_logging(logging.INFO)
    try:
        settings = Settings.from_env()
    except Exception:
        LOGGER.exception("server failed to start")
        raise SystemExit(1) from None
    server = grpc.server(
        futures.ThreadPoolExecutor(max_workers=8),
        maximum_concurrent_rpcs=8,
        options=(
            ("grpc.max_receive_message_length", settings.max_video_bytes + 1024 * 1024),
            ("grpc.max_send_message_length", 8 * 1024 * 1024),
        ),
    )
    analysis_pb2_grpc.add_BadmintonAnalysisServicer_to_server(
        DeferredAnalysisService(settings, _load_analysis_service), server
    )
    server.add_insecure_port(f"[::]:{settings.port}")
    server.start()
    LOGGER.info("gRPC server listening", extra={"json_fields": {"port": settings.port}})

    def stop(*_: object) -> None:
        LOGGER.info("stopping gRPC server")
        server.stop(grace=30)

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    server.wait_for_termination()


if __name__ == "__main__":
    serve()
