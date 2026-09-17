from __future__ import annotations

import json
import logging
import tempfile
from concurrent import futures
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import grpc
import pytest

from badminton.analysis.v1 import analysis_pb2, analysis_pb2_grpc
from badminton_analysis.ml.expert_reference_bank import ExpertReference
from badminton_analysis.ml.skill_specs import get_skill_spec
from badminton_analysis.models.types import Handedness, Skill
from service.pipeline import AnalysisResult, PhaseResult, SkillMismatchError
from service.logging_config import CloudLoggingJsonFormatter
from service.server import BadmintonAnalysisService, _PROTO_TO_SKILL
from service.storage import SignedObject


class _Backend:
    target_frames = 64

    def __init__(self) -> None:
        self.spec = get_skill_spec(Skill.SERVE)


class _Pipeline:
    def __init__(self) -> None:
        self.backends = {Skill.SERVE: _Backend()}
        self.skip_coaching_seen: bool | None = None

    def analyze(
        self,
        *,
        video_path: Path,
        output_path: Path,
        skeleton_overlay_path: Path,
        filename: str,
        skill: Skill,
        requested_handedness: str,
        skip_coaching: bool = False,
    ) -> AnalysisResult:
        self.skip_coaching_seen = skip_coaching
        assert video_path.read_bytes() == b"video-bytes"
        assert filename == "serve.mp4"
        assert skill == Skill.SERVE
        assert requested_handedness == "right"
        output_path.write_bytes(b"feedback-video")
        skeleton_overlay_path.write_bytes(b"overlay-video")
        spec = self.backends[skill].spec
        return AnalysisResult(
            skill=skill,
            handedness=Handedness.RIGHT,
            grade={
                "total_grade": 82.0,
                "grading_details": [
                    {"description": rule.name_zh_tw, "grade": rule.maximum * 0.8}
                    for rule in spec.rules
                ],
            },
            diagnostics={
                "correction_distance": 0.2,
                "expert_reference_distance": 0.1,
            },
            expert_id="nearest-expert-a",
            expert_distance=0.1,
            phases=tuple(
                PhaseResult(
                    id=rule.id,
                    label=rule.name_zh_tw,
                    normalized_frame=index * 10,
                    normalized_position=index * 10 / 63.0,
                    timestamp_seconds=index / 3.0,
                )
                for index, rule in enumerate(spec.rules)
            ),
            overall_feedback="保持動作連續。",
            coaching_problems=(),
            pause_seconds=2.0,
        )


class _Storage:
    def upload_file(
        self, source: Path, object_path: str, *, content_type: str
    ) -> SignedObject:
        assert source.exists()
        assert content_type == "video/mp4"
        return SignedObject(
            object_path=object_path,
            gcs_uri=f"gs://test/{object_path}",
            signed_url=f"https://media.test/{source.name}",
            expires_at_unix=2_000_000_000,
        )

    def sign(self, object_path: str) -> SignedObject:
        raise AssertionError(f"serve must not fetch a catalog video: {object_path}")


class _Catalog:
    def get(self, skill: str, expert_id: str):
        raise AssertionError(
            f"serve must use the generated prior, not catalog {skill}/{expert_id}"
        )


def test_api_exposes_only_serve_and_smash_analysis() -> None:
    assert set(_PROTO_TO_SKILL) == {
        analysis_pb2.SKILL_SERVE,
        analysis_pb2.SKILL_SMASH,
    }


def test_skill_mismatch_is_reported_as_invalid_argument() -> None:
    class MismatchPipeline(_Pipeline):
        def analyze(self, **kwargs):
            del kwargs
            raise SkillMismatchError("requested serve conflicts with smash support")

    class NoUploadStorage(_Storage):
        def upload_file(self, *args, **kwargs):
            del args, kwargs
            raise AssertionError("mismatched motion reached object upload")

    service = BadmintonAnalysisService.__new__(BadmintonAnalysisService)
    service.settings = SimpleNamespace(grpc_api_key="test-key", max_video_bytes=1024)
    service.pipeline = MismatchPipeline()
    service.storage = NoUploadStorage()
    service.catalog = _Catalog()

    grpc_server = grpc.server(futures.ThreadPoolExecutor(max_workers=1))
    analysis_pb2_grpc.add_BadmintonAnalysisServicer_to_server(service, grpc_server)
    port = grpc_server.add_insecure_port("127.0.0.1:0")
    grpc_server.start()
    try:
        with grpc.insecure_channel(f"127.0.0.1:{port}") as channel:
            stub = analysis_pb2_grpc.BadmintonAnalysisStub(channel)
            with pytest.raises(grpc.RpcError) as failure:
                stub.AnalyzeVideo(
                    iter(
                        (
                            analysis_pb2.AnalyzeVideoChunk(
                                header=analysis_pb2.AnalyzeVideoHeader(
                                    request_id="mismatch",
                                    user_id="user",
                                    filename="unknown.mp4",
                                    skill=analysis_pb2.SKILL_SERVE,
                                    handedness=analysis_pb2.HANDEDNESS_RIGHT,
                                )
                            ),
                            analysis_pb2.AnalyzeVideoChunk(data=b"video-bytes"),
                        )
                    ),
                    metadata=(("x-api-key", "test-key"),),
                )
    finally:
        grpc_server.stop(None)

    assert failure.value.code() == grpc.StatusCode.INVALID_ARGUMENT


def test_streamed_grpc_api_returns_feedback_and_clean_overlay(monkeypatch) -> None:
    metadata = {
        "duration_seconds": 2.0,
        "fps": 30.0,
        "width": 720,
        "height": 1280,
    }
    monkeypatch.setattr("service.server.probe_video", lambda _: metadata)
    service = BadmintonAnalysisService.__new__(BadmintonAnalysisService)
    service.settings = SimpleNamespace(
        grpc_api_key="test-key",
        max_video_bytes=1024,
    )
    service.pipeline = _Pipeline()
    service.storage = _Storage()
    service.catalog = _Catalog()

    server = grpc.server(futures.ThreadPoolExecutor(max_workers=1))
    analysis_pb2_grpc.add_BadmintonAnalysisServicer_to_server(service, server)
    port = server.add_insecure_port("127.0.0.1:0")
    server.start()
    try:
        with grpc.insecure_channel(f"127.0.0.1:{port}") as channel:
            stub = analysis_pb2_grpc.BadmintonAnalysisStub(channel)
            response = stub.AnalyzeVideo(
                iter(
                    (
                        analysis_pb2.AnalyzeVideoChunk(
                            header=analysis_pb2.AnalyzeVideoHeader(
                                request_id="request-1",
                                user_id="user-1",
                                filename="serve.mp4",
                                skill=analysis_pb2.SKILL_SERVE,
                                handedness=analysis_pb2.HANDEDNESS_RIGHT,
                            )
                        ),
                        analysis_pb2.AnalyzeVideoChunk(data=b"video-bytes"),
                    )
                ),
                metadata=(("x-api-key", "test-key"),),
            )
    finally:
        server.stop(grace=None).wait()

    assert response.student_video == response.feedback_video
    assert response.feedback_video.object_path.endswith("student_corrected.mp4")
    assert response.skeleton_overlay_video.object_path.endswith(
        "student_skeleton_overlay.mp4"
    )
    assert response.feedback_video.signed_url.endswith("student_corrected.mp4")
    assert response.skeleton_overlay_video.signed_url.endswith(
        "student_skeleton_overlay.mp4"
    )
    assert response.grade.score_status == "expert_only_generated_distribution"
    assert response.expert.display_name == "Generated expert prior"
    assert not response.expert.HasField("video")


class _ExpertStorage(_Storage):
    def sign(self, object_path: str) -> SignedObject:
        return SignedObject(
            object_path=object_path,
            gcs_uri=f"gs://test/{object_path}",
            signed_url=f"https://media.test/{object_path}",
            expires_at_unix=2_000_000_000,
        )


def _matched_service() -> BadmintonAnalysisService:
    service = BadmintonAnalysisService.__new__(BadmintonAnalysisService)
    service.pipeline = _Pipeline()
    service.storage = _ExpertStorage()
    service.catalog = _Catalog()
    return service


def _expert_reference() -> ExpertReference:
    return ExpertReference(
        skill="serve",
        handedness="right",
        video_object_path="experts/v3/serve/videos/test.mp4",
        subject_id="test-expert",
        fps=30.0,
        analysis_window=(0, 70, 70),
        source_phase_indices=(12, 27, 42, 51, 60),
        distance=0.1,
        similarity=0.9,
    )


def _matched_response(alignment: tuple[tuple[float, float], ...]):
    service = _matched_service()
    metadata = {"duration_seconds": 2.0, "fps": 30.0, "width": 720, "height": 1280}
    signed = SignedObject(
        object_path="student_corrected.mp4",
        gcs_uri="gs://test/student_corrected.mp4",
        signed_url="https://media.test/student_corrected.mp4",
        expires_at_unix=2_000_000_000,
    )
    pipeline = service.pipeline
    result = pipeline.analyze(
        video_path=_video(),
        output_path=Path(tempfile.mkstemp(suffix=".mp4")[1]),
        skeleton_overlay_path=Path(tempfile.mkstemp(suffix=".mp4")[1]),
        filename="serve.mp4",
        skill=Skill.SERVE,
        requested_handedness="right",
    )
    result = replace(
        result, expert_reference=_expert_reference(), expert_alignment=alignment
    )
    return service._response("analysis-1", result, signed, signed, metadata, metadata)


def _video() -> Path:
    path = Path(tempfile.mkstemp(suffix=".mp4")[1])
    path.write_bytes(b"video-bytes")
    return path


def test_a_matched_expert_carries_its_warped_alignment() -> None:
    samples = ((0.0, 0.4), (0.5, 1.1), (1.0, 2.0))

    response = _matched_response(samples)

    assert [
        (sample.normalized_position, sample.expert_seconds)
        for sample in response.expert.alignment
    ] == list(samples)


# The alignment is a refinement of playback, so a clip the warp could not
# handle costs the analysis nothing beyond it.
def test_an_expert_without_an_alignment_still_returns() -> None:
    response = _matched_response(())

    assert list(response.expert.alignment) == []
    assert response.expert.expert_id == "test-expert"
    assert len(response.expert.timeline) == len(response.timeline)


def test_coaching_cue_uses_analysis_clip_clock_not_source_upload_clock() -> None:
    service = _matched_service()
    metadata = {
        "duration_seconds": 3.4,
        "fps": 30.0,
        "width": 720,
        "height": 1280,
    }
    signed = SignedObject(
        object_path="student_corrected.mp4",
        gcs_uri="gs://test/student_corrected.mp4",
        signed_url="https://media.test/student_corrected.mp4",
        expires_at_unix=2_000_000_000,
    )
    result = service.pipeline.analyze(
        video_path=_video(),
        output_path=Path(tempfile.mkstemp(suffix=".mp4")[1]),
        skeleton_overlay_path=Path(tempfile.mkstemp(suffix=".mp4")[1]),
        filename="serve.mp4",
        skill=Skill.SERVE,
        requested_handedness="right",
    )
    result = replace(
        result,
        diagnostics={
            **result.diagnostics,
            "analysis_window_start_frame": 30,
            "analysis_window_end_frame": 71,
            "normalized_sequence_length": 64,
            "source_frame_count": 100,
        },
        coaching_problems=(
            {
                "frame_index": 63,
                "title": "重心轉移至非持拍腳",
                "feedback": "完成擊球時將重心轉移到非持拍腳。",
                "joint_ids": [5, 6, 11, 12, 13, 14, 15, 16],
            },
        ),
    )

    response = service._response(
        "analysis-1", result, signed, signed, metadata, metadata
    )

    cue = response.coaching_cues[0]
    assert cue.normalized_position == pytest.approx(1.0)
    assert cue.student_timestamp_seconds == pytest.approx(41 / 30)


def test_skip_coaching_reaches_the_pipeline(monkeypatch) -> None:
    """The header's skip_coaching must survive the wire.

    It is the switch that decides whether frames of the learner are uploaded to
    a third-party model, so a deployment that sets it and is silently ignored
    would be sending imagery it promised not to.
    """
    monkeypatch.setattr(
        "service.server.probe_video",
        lambda _: {"duration_seconds": 2.0, "fps": 30.0, "width": 720, "height": 1280},
    )
    service = BadmintonAnalysisService.__new__(BadmintonAnalysisService)
    service.settings = SimpleNamespace(grpc_api_key="test-key", max_video_bytes=1024)
    pipeline = _Pipeline()
    service.pipeline = pipeline
    service.storage = _Storage()
    service.catalog = _Catalog()

    grpc_server = grpc.server(futures.ThreadPoolExecutor(max_workers=1))
    analysis_pb2_grpc.add_BadmintonAnalysisServicer_to_server(service, grpc_server)
    port = grpc_server.add_insecure_port("127.0.0.1:0")
    grpc_server.start()
    try:
        with grpc.insecure_channel(f"127.0.0.1:{port}") as channel:
            stub = analysis_pb2_grpc.BadmintonAnalysisStub(channel)
            stub.AnalyzeVideo(
                iter(
                    (
                        analysis_pb2.AnalyzeVideoChunk(
                            header=analysis_pb2.AnalyzeVideoHeader(
                                request_id="r",
                                user_id="u",
                                filename="serve.mp4",
                                skill=analysis_pb2.SKILL_SERVE,
                                handedness=analysis_pb2.HANDEDNESS_RIGHT,
                                skip_coaching=True,
                            )
                        ),
                        analysis_pb2.AnalyzeVideoChunk(data=b"video-bytes"),
                    )
                ),
                metadata=(("x-api-key", "test-key"),),
            )
    finally:
        grpc_server.stop(None)

    assert pipeline.skip_coaching_seen is True


def test_analysis_root_separates_deployments_sharing_the_bucket() -> None:
    """Two deployments share this service, its bucket, and their learner ids.

    The login channel is shared, so the same learner is the same string in
    both, and only the caller's own prefix can say which product a recording
    belongs to. An empty prefix must keep the original layout, because the
    first deployment's objects are already stored that way.
    """
    from service.server import _analysis_root

    assert _analysis_root("", "U1", "req9") == "analyses/v1/U1/req9"
    assert _analysis_root("noai", "U1", "req9") == "noai/analyses/v1/U1/req9"
    # A prefix arrives over the wire, so it may not escape its own directory.
    assert _analysis_root("../../etc", "U1", "req9").startswith("etc/")
    assert ".." not in _analysis_root("../../etc", "U1", "req9")


class _JsonCapture(logging.Handler):
    """Formats at emit time, on the RPC's own thread, where its context lives."""

    def __init__(self) -> None:
        super().__init__()
        self.setFormatter(CloudLoggingJsonFormatter(project_id="proj"))
        self.entries: list[dict] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.entries.append(json.loads(self.format(record)))


@pytest.fixture
def json_logs():
    handler = _JsonCapture()
    logger = logging.getLogger("badminton-analysis")
    previous = logger.level
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    try:
        yield handler.entries
    finally:
        logger.removeHandler(handler)
        logger.setLevel(previous)


def _call_analyze(service, metadata, header_request_id="request-1"):
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=1))
    analysis_pb2_grpc.add_BadmintonAnalysisServicer_to_server(service, server)
    port = server.add_insecure_port("127.0.0.1:0")
    server.start()
    try:
        with grpc.insecure_channel(f"127.0.0.1:{port}") as channel:
            stub = analysis_pb2_grpc.BadmintonAnalysisStub(channel)
            return stub.AnalyzeVideo(
                iter(
                    (
                        analysis_pb2.AnalyzeVideoChunk(
                            header=analysis_pb2.AnalyzeVideoHeader(
                                request_id=header_request_id,
                                user_id="user-1",
                                filename="serve.mp4",
                                skill=analysis_pb2.SKILL_SERVE,
                                handedness=analysis_pb2.HANDEDNESS_RIGHT,
                            )
                        ),
                        analysis_pb2.AnalyzeVideoChunk(data=b"video-bytes"),
                    )
                ),
                metadata=metadata,
            )
    finally:
        server.stop(grace=None).wait()


def _logging_service(pipeline) -> BadmintonAnalysisService:
    service = BadmintonAnalysisService.__new__(BadmintonAnalysisService)
    service.settings = SimpleNamespace(grpc_api_key="test-key", max_video_bytes=1024)
    service.pipeline = pipeline
    service.storage = _Storage()
    service.catalog = _Catalog()
    return service


_TRACE = "105445aa7843bc8bf206b12000100000"


def test_request_and_trace_metadata_correlate_the_completion_log(
    monkeypatch, json_logs
) -> None:
    """The Go client's ids must land on every line the analysis writes.

    Log-based metrics and the Cloud Trace link are keyed on these exact field
    names, so they are asserted literally.
    """
    monkeypatch.setattr(
        "service.server.probe_video",
        lambda _: {"duration_seconds": 2.0, "fps": 30.0, "width": 720, "height": 1280},
    )

    class LoggingPipeline(_Pipeline):
        def analyze(self, **kwargs):
            logging.getLogger("badminton-analysis").info("inside pipeline")
            result = super().analyze(**kwargs)
            result.diagnostics.update(
                {
                    "latency_pose_seconds": 0.5,
                    "latency_llm_inference_seconds": 2.0,
                    "pose_tensorrt_active": 1.0,
                    "coaching_source": "openai",
                }
            )
            return result

    response = _call_analyze(
        _logging_service(LoggingPipeline()),
        (
            ("x-api-key", "test-key"),
            ("x-request-id", "job-42"),
            ("x-cloud-trace-context", f"{_TRACE}/255;o=1"),
        ),
    )

    inside = next(e for e in json_logs if e["message"] == "inside pipeline")
    completed = next(e for e in json_logs if e["message"] == "analysis completed")
    for entry in (inside, completed):
        assert entry["request_id"] == "job-42"
        assert entry["analysis_id"] == response.analysis_id
        assert entry["logging.googleapis.com/trace"] == f"projects/proj/traces/{_TRACE}"
        assert entry["logging.googleapis.com/spanId"] == "00000000000000ff"
    assert completed["severity"] == "INFO"
    assert completed["skill"] == "serve"
    assert completed["handedness"] == "right"
    assert completed["total_grade"] == 82.0
    assert completed["coaching_source"] == "openai"
    assert completed["pose_tensorrt_active"] is True
    assert completed["input_video_bytes"] == len(b"video-bytes")
    assert completed["latency_pose_seconds"] == 0.5
    assert completed["latency_llm_inference_seconds"] == 2.0
    assert completed["latency_service_seconds"] >= 0.0
    assert completed["latency_storage_seconds"] >= 0.0


def test_malformed_trace_and_missing_request_id_fall_back_to_header(
    monkeypatch, json_logs
) -> None:
    monkeypatch.setattr(
        "service.server.probe_video",
        lambda _: {"duration_seconds": 2.0, "fps": 30.0, "width": 720, "height": 1280},
    )

    _call_analyze(
        _logging_service(_Pipeline()),
        (("x-api-key", "test-key"), ("x-cloud-trace-context", "garbage")),
        header_request_id="header-job",
    )

    completed = next(e for e in json_logs if e["message"] == "analysis completed")
    assert completed["request_id"] == "header-job"
    assert "logging.googleapis.com/trace" not in completed
    assert "coaching_source" not in completed


def test_failed_analysis_logs_grpc_code_and_error_type(json_logs) -> None:
    class FailingPipeline(_Pipeline):
        def analyze(self, **kwargs):
            del kwargs
            raise RuntimeError("gpu fell over")

    with pytest.raises(grpc.RpcError) as failure:
        _call_analyze(
            _logging_service(FailingPipeline()),
            (("x-api-key", "test-key"), ("x-request-id", "job-7")),
        )

    assert failure.value.code() == grpc.StatusCode.INTERNAL
    failed = next(e for e in json_logs if e["message"].startswith("analysis failed"))
    assert failed["severity"] == "ERROR"
    assert failed["grpc_code"] == "INTERNAL"
    assert failed["error_type"] == "RuntimeError"
    assert failed["request_id"] == "job-7"
    assert "RuntimeError: gpu fell over" in failed["message"]


def test_rejected_analysis_logs_a_warning(json_logs) -> None:
    with pytest.raises(grpc.RpcError) as failure:
        _call_analyze(_logging_service(_Pipeline()), (("x-api-key", "wrong"),))

    assert failure.value.code() == grpc.StatusCode.UNAUTHENTICATED
    failed = next(e for e in json_logs if e["message"] == "analysis failed")
    assert failed["severity"] == "WARNING"
    assert failed["grpc_code"] == "UNAUTHENTICATED"
    assert failed["error_type"] == "invalid_api_key"


def test_health_binds_request_context(json_logs) -> None:
    seen: dict[str, str | None] = {}

    class WarmPipeline:
        loaded_skills = (Skill.SERVE,)

        def warmup(self) -> None:
            from service.logging_config import current_request_id

            seen["request_id"] = current_request_id()

    service = _logging_service(WarmPipeline())
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=1))
    analysis_pb2_grpc.add_BadmintonAnalysisServicer_to_server(service, server)
    port = server.add_insecure_port("127.0.0.1:0")
    server.start()
    try:
        with grpc.insecure_channel(f"127.0.0.1:{port}") as channel:
            stub = analysis_pb2_grpc.BadmintonAnalysisStub(channel)
            stub.Health(
                analysis_pb2.HealthRequest(),
                metadata=(("x-api-key", "test-key"), ("x-request-id", "health-1")),
            )
    finally:
        server.stop(grace=None).wait()

    assert seen["request_id"] == "health-1"
