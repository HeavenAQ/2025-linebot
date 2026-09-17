from __future__ import annotations

import json
import logging
import sys

import pytest

from service.logging_config import (
    SPAN_FIELD,
    TRACE_FIELD,
    CloudLoggingJsonFormatter,
    parse_cloud_trace_context,
    request_context,
    sanitize_request_id,
    set_analysis_id,
)

_TRACE = "105445aa7843bc8bf206b12000100000"


def _record(
    message: str = "hello %s",
    args: tuple[object, ...] = ("world",),
    *,
    level: int = logging.INFO,
    exc_info=None,
    **extra: object,
) -> logging.LogRecord:
    record = logging.LogRecord(
        "badminton-analysis", level, __file__, 1, message, args, exc_info
    )
    record.__dict__.update(extra)
    return record


def _format(record: logging.LogRecord, project_id: str = "proj") -> dict:
    line = CloudLoggingJsonFormatter(project_id=project_id).format(record)
    assert "\n" not in line
    return json.loads(line)


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (f"{_TRACE}/1;o=1", (_TRACE, "0000000000000001")),
        (f"{_TRACE}/18446744073709551615", (_TRACE, "ffffffffffffffff")),
        (f"{_TRACE.upper()}/255;o=0", (_TRACE, "00000000000000ff")),
        (_TRACE, (_TRACE, None)),
        (f"{_TRACE}/", (_TRACE, None)),
        (f"{_TRACE}/abc;o=1", (_TRACE, None)),
        (f"{_TRACE}/0;o=1", (_TRACE, None)),
        (f"{_TRACE}/18446744073709551616", (_TRACE, None)),
        ("", (None, None)),
        (None, (None, None)),
        ("not-a-trace/1;o=1", (None, None)),
        ("0" * 32 + "/1", (None, None)),
        (f"{_TRACE}0/1", (None, None)),
    ],
)
def test_cloud_trace_context_parsing_tolerates_malformed_headers(
    value, expected
) -> None:
    assert parse_cloud_trace_context(value) == expected


def test_formatter_emits_cloud_logging_core_fields() -> None:
    entry = _format(_record(level=logging.WARNING))

    assert entry["severity"] == "WARNING"
    assert entry["message"] == "hello world"
    assert entry["logger"] == "badminton-analysis"
    assert entry["time"].endswith("Z") and "T" in entry["time"]
    assert "request_id" not in entry
    assert TRACE_FIELD not in entry


def test_formatter_adds_request_context_and_trace_resource_name() -> None:
    with request_context(request_id="job-1", trace_id=_TRACE, span_id="00000000000000ff"):
        set_analysis_id("a1")
        entry = _format(_record())

    assert entry["request_id"] == "job-1"
    assert entry["analysis_id"] == "a1"
    assert entry[TRACE_FIELD] == f"projects/proj/traces/{_TRACE}"
    assert entry[SPAN_FIELD] == "00000000000000ff"
    # Reset on exit so a reused worker thread does not inherit the ids.
    after = _format(_record())
    assert "request_id" not in after and "analysis_id" not in after


def test_trace_is_omitted_without_a_project() -> None:
    with request_context(trace_id=_TRACE, span_id="1"):
        entry = _format(_record(), project_id="")

    assert TRACE_FIELD not in entry and SPAN_FIELD not in entry


def test_exception_traceback_is_part_of_the_message() -> None:
    try:
        raise RuntimeError("boom")
    except RuntimeError:
        entry = _format(_record("failed", (), level=logging.ERROR, exc_info=sys.exc_info()))

    assert entry["severity"] == "ERROR"
    assert entry["message"].startswith("failed\nTraceback")
    assert "RuntimeError: boom" in entry["message"]


def test_json_fields_merge_without_replacing_core_fields() -> None:
    entry = _format(
        _record(
            "done",
            (),
            json_fields={
                "latency_service_seconds": 1.5,
                "latency_bad_seconds": float("nan"),
                "severity": "DEBUG",
            },
        )
    )

    assert entry["latency_service_seconds"] == 1.5
    assert entry["latency_bad_seconds"] is None
    assert entry["severity"] == "INFO"


def test_request_id_is_bounded_and_printable() -> None:
    assert sanitize_request_id("  job\n-1 ") == "job-1"
    assert sanitize_request_id("") is None
    assert len(sanitize_request_id("x" * 500) or "") == 128


def test_core_logger_wrapper_writes_json(capsys) -> None:
    from badminton_analysis.core.logger import Logger

    logger = Logger("test-core-logger-json")
    logger.debug("hidden at the default level")
    logger.info("visible")

    lines = capsys.readouterr().out.strip().splitlines()
    assert len(lines) == 1
    entry = json.loads(lines[0])
    assert entry["severity"] == "INFO"
    assert entry["message"].startswith("visible - ")
