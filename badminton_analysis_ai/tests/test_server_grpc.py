from __future__ import annotations

from concurrent import futures
from pathlib import Path
from types import SimpleNamespace

import grpc

from badminton.analysis.v1 import analysis_pb2, analysis_pb2_grpc
from badminton_analysis.ml.skill_specs import get_skill_spec
from badminton_analysis.models.types import Handedness, Skill
from service.pipeline import AnalysisResult, PhaseResult
from service.server import BadmintonAnalysisService, _PROTO_TO_SKILL
from service.storage import SignedObject


class _Backend:
    target_frames = 64

    def __init__(self) -> None:
        self.spec = get_skill_spec(Skill.SERVE)


class _Pipeline:
    def __init__(self) -> None:
        self.backends = {Skill.SERVE: _Backend()}

    def analyze(
        self,
        *,
        video_path: Path,
        output_path: Path,
        skeleton_overlay_path: Path,
        filename: str,
        skill: Skill,
        requested_handedness: str,
    ) -> AnalysisResult:
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
            output_path=output_path,
            skeleton_overlay_path=skeleton_overlay_path,
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
