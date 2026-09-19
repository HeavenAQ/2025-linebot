"""One-JSON-object-per-line logging for Cloud Logging.

Cloud Run ingests stdout line by line and promotes a JSON line's ``severity``,
``message``, ``time`` and ``logging.googleapis.com/*`` keys into the log entry,
so plain-text lines lose their level and cannot be joined to the caller's
trace. Only the stdlib is used so the service image gains no dependency.
"""

from __future__ import annotations

import json
import logging
import math
import os
import re
import sys
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import datetime, timezone
from typing import Any

TRACE_FIELD = "logging.googleapis.com/trace"
SPAN_FIELD = "logging.googleapis.com/spanId"

_request_id: ContextVar[str | None] = ContextVar("request_id", default=None)
_analysis_id: ContextVar[str | None] = ContextVar("analysis_id", default=None)
_trace_id: ContextVar[str | None] = ContextVar("trace_id", default=None)
_span_id: ContextVar[str | None] = ContextVar("span_id", default=None)

_TRACE_ID = re.compile(r"[0-9a-fA-F]{32}")
_SPAN_ID = re.compile(r"[0-9]{1,20}")
_MAX_REQUEST_ID_LENGTH = 128
# Fields the formatter owns; json_fields must not silently replace them.
_RESERVED = frozenset({"severity", "message", "time", "logger"})


def parse_cloud_trace_context(value: str | None) -> tuple[str | None, str | None]:
    """Split ``TRACE_ID/SPAN_ID;o=OPTIONS`` into (trace_id, span_id).

    The header comes from another process, so anything malformed yields None
    rather than an error: a bad header must cost the correlation, never the
    request. The span is decimal on the wire but Cloud Logging expects the
    16-hex-digit form, so it is converted here.
    """
    if not value:
        return None, None
    trace_part, _, rest = value.strip().partition("/")
    if not _TRACE_ID.fullmatch(trace_part) or int(trace_part, 16) == 0:
        return None, None
    trace_id = trace_part.lower()
    span_part = rest.partition(";")[0].strip()
    if not _SPAN_ID.fullmatch(span_part):
        return trace_id, None
    span = int(span_part)
    if not 0 < span < 2**64:
        return trace_id, None
    return trace_id, f"{span:016x}"


def sanitize_request_id(value: str | None) -> str | None:
    """Keep a caller-supplied id printable and bounded before it is logged."""
    if not value:
        return None
    cleaned = "".join(char for char in value.strip() if char.isprintable())
    return cleaned[:_MAX_REQUEST_ID_LENGTH] or None


@contextmanager
def request_context(
    *,
    request_id: str | None = None,
    trace_id: str | None = None,
    span_id: str | None = None,
) -> Iterator[None]:
    """Bind correlation fields to every log line emitted during one RPC.

    gRPC reuses worker threads, so the values are reset on exit instead of
    being left for the next request on the same thread to inherit.
    """
    tokens = (
        (_request_id, _request_id.set(request_id)),
        (_trace_id, _trace_id.set(trace_id)),
        (_span_id, _span_id.set(span_id)),
        # Reset too, so an id set mid-request by set_analysis_id is undone.
        (_analysis_id, _analysis_id.set(None)),
    )
    try:
        yield
    finally:
        for variable, token in reversed(tokens):
            variable.reset(token)


def set_analysis_id(value: str) -> None:
    _analysis_id.set(value)


def set_request_id_if_absent(value: str | None) -> None:
    if _request_id.get() is None:
        _request_id.set(sanitize_request_id(value))


def current_request_id() -> str | None:
    return _request_id.get()


def _severity(levelno: int) -> str:
    if levelno >= logging.CRITICAL:
        return "CRITICAL"
    if levelno >= logging.ERROR:
        return "ERROR"
    if levelno >= logging.WARNING:
        return "WARNING"
    if levelno >= logging.INFO:
        return "INFO"
    if levelno >= logging.DEBUG:
        return "DEBUG"
    return "DEFAULT"


def _json_safe(value: Any) -> Any:
    # NaN/Infinity are not JSON; one such latency would turn the whole line
    # back into unstructured text in Cloud Logging.
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


class CloudLoggingJsonFormatter(logging.Formatter):
    def __init__(self, project_id: str | None = None) -> None:
        super().__init__()
        self.project_id = (
            project_id if project_id is not None else os.getenv("GCP_PROJECT_ID", "")
        )

    def format(self, record: logging.LogRecord) -> str:
        message = record.getMessage()
        # Error Reporting only recognises a stack trace inside the message.
        if record.exc_info:
            if not record.exc_text:
                record.exc_text = self.formatException(record.exc_info)
        if record.exc_text:
            message = f"{message}\n{record.exc_text}"
        if record.stack_info:
            message = f"{message}\n{self.formatStack(record.stack_info)}"

        entry: dict[str, Any] = {}
        extra_fields = getattr(record, "json_fields", None)
        if isinstance(extra_fields, Mapping):
            entry.update(
                (str(key), value)
                for key, value in extra_fields.items()
                if key not in _RESERVED
            )
        entry.update(
            {
                "severity": _severity(record.levelno),
                "message": message,
                "time": datetime.fromtimestamp(record.created, tz=timezone.utc)
                .isoformat(timespec="microseconds")
                .replace("+00:00", "Z"),
                "logger": record.name,
            }
        )
        request_id = _request_id.get()
        if request_id:
            entry["request_id"] = request_id
        analysis_id = _analysis_id.get()
        if analysis_id:
            entry["analysis_id"] = analysis_id
        trace_id = _trace_id.get()
        # The trace field must be a full resource name to link in the console;
        # without a project it would only be a dangling string.
        if trace_id and self.project_id:
            entry[TRACE_FIELD] = f"projects/{self.project_id}/traces/{trace_id}"
            span_id = _span_id.get()
            if span_id:
                entry[SPAN_FIELD] = span_id
        try:
            return json.dumps(_json_safe(entry), ensure_ascii=False, default=str)
        except (TypeError, ValueError):
            # A log call must never raise; keep the essentials.
            return json.dumps(
                {key: entry[key] for key in ("severity", "message", "time", "logger")},
                ensure_ascii=False,
                default=str,
            )


def json_stream_handler(level: int = logging.NOTSET) -> logging.Handler:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(CloudLoggingJsonFormatter())
    handler.setLevel(level)
    return handler


def configure_logging(level: int = logging.INFO) -> None:
    """Route the root logger to stdout as JSON, replacing prior handlers."""
    logging.basicConfig(level=level, handlers=[json_stream_handler()], force=True)
