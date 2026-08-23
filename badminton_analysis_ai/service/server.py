from __future__ import annotations

import logging
import re
import secrets
import signal
import tempfile
import time
import uuid
from concurrent import futures
from pathlib import Path
from typing import Iterable

import grpc

from badminton.analysis.v1 import analysis_pb2, analysis_pb2_grpc
from badminton_analysis.models.types import Handedness, Skill

from service.config import Settings
from service.pipeline import (
    expert_phase_results,
    AnalysisResult,
    SkeletonAnalysisPipeline,
)
from service.renderer import probe_video
from service.storage import ObjectStorage, SignedObject

LOGGER = logging.getLogger("badminton-analysis")
_SAFE_SEGMENT = re.compile(r"[^A-Za-z0-9._-]+")

_PROTO_TO_SKILL = {
    analysis_pb2.SKILL_SERVE: Skill.SERVE,
    analysis_pb2.SKILL_SMASH: Skill.SMASH,
}
_SKILL_TO_PROTO = {value: key for key, value in _PROTO_TO_SKILL.items()}
_PROTO_TO_HANDEDNESS = {
    analysis_pb2.HANDEDNESS_AUTO: "auto",
    analysis_pb2.HANDEDNESS_RIGHT: "right",
    analysis_pb2.HANDEDNESS_LEFT: "left",
}
_HANDEDNESS_TO_PROTO = {
    Handedness.RIGHT: analysis_pb2.HANDEDNESS_RIGHT,
    Handedness.LEFT: analysis_pb2.HANDEDNESS_LEFT,
}


def _safe_segment(value: str, fallback: str) -> str:
    cleaned = _SAFE_SEGMENT.sub("_", value).strip("._")
    return cleaned[:96] or fallback


class BadmintonAnalysisService(analysis_pb2_grpc.BadmintonAnalysisServicer):
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.storage = ObjectStorage(
            settings.gcp_project_id,
            settings.gcs_bucket_name,
            service_account_email=settings.gcp_service_account_email,
            signed_url_minutes=settings.signed_url_minutes,
        )
        self.pipeline = SkeletonAnalysisPipeline(
            settings.expert_motion_model_root,
            device=settings.device,
            openai_model=settings.openai_model,
            pause_seconds=settings.coaching_pause_seconds,
        )

    def _authorize(self, context: grpc.ServicerContext) -> None:
        supplied = dict(context.invocation_metadata()).get("x-api-key", "")
        if not secrets.compare_digest(supplied, self.settings.grpc_api_key):
            context.abort(grpc.StatusCode.UNAUTHENTICATED, "invalid API key")

    @staticmethod
    def _stored_video(
        signed: SignedObject, metadata: dict[str, float | int]
    ) -> analysis_pb2.StoredVideo:
        return analysis_pb2.StoredVideo(
            object_path=signed.object_path,
            gcs_uri=signed.gcs_uri,
            signed_url=signed.signed_url,
            signed_url_expires_at_unix=signed.expires_at_unix,
            duration_seconds=float(metadata.get("duration_seconds", 0.0)),
            fps=float(metadata.get("fps", 0.0)),
            width=int(metadata.get("width", 0)),
            height=int(metadata.get("height", 0)),
        )

    def AnalyzeVideo(
        self,
        request_iterator: Iterable[analysis_pb2.AnalyzeVideoChunk],
        context: grpc.ServicerContext,
    ) -> analysis_pb2.AnalyzeVideoResponse:
        self._authorize(context)
        service_started = time.perf_counter()
        header = None
        total_bytes = 0
        analysis_id = uuid.uuid4().hex
        with tempfile.TemporaryDirectory(prefix="badminton-analysis-") as temp_value:
            temp_dir = Path(temp_value)
            input_path = temp_dir / "input.mp4"
            with input_path.open("wb") as handle:
                for chunk in request_iterator:
                    payload = chunk.WhichOneof("payload")
                    if payload == "header":
                        if header is not None or total_bytes:
                            context.abort(
                                grpc.StatusCode.INVALID_ARGUMENT,
                                "header must be the first and only header chunk",
                            )
                        header = chunk.header
                    elif payload == "data":
                        if header is None:
                            context.abort(
                                grpc.StatusCode.INVALID_ARGUMENT,
                                "video header must be sent first",
                            )
                        total_bytes += len(chunk.data)
                        if total_bytes > self.settings.max_video_bytes:
                            context.abort(
                                grpc.StatusCode.RESOURCE_EXHAUSTED,
                                "video exceeds configured size limit",
                            )
                        handle.write(chunk.data)
                    else:
                        context.abort(grpc.StatusCode.INVALID_ARGUMENT, "empty chunk")
            if header is None or total_bytes == 0:
                context.abort(grpc.StatusCode.INVALID_ARGUMENT, "video is empty")
            skill = _PROTO_TO_SKILL.get(header.skill)
            handedness = _PROTO_TO_HANDEDNESS.get(header.handedness)
            if skill is None:
                context.abort(
                    grpc.StatusCode.INVALID_ARGUMENT,
                    "only serve and smash are currently supported",
                )
            if handedness is None:
                context.abort(grpc.StatusCode.INVALID_ARGUMENT, "unsupported handedness")

            output_path = temp_dir / "student_corrected.mp4"
            skeleton_overlay_path = temp_dir / "student_skeleton_overlay.mp4"
            try:
                result = self.pipeline.analyze(
                    video_path=input_path,
                    output_path=output_path,
                    skeleton_overlay_path=skeleton_overlay_path,
                    filename=header.filename or "video.mp4",
                    skill=skill,
                    requested_handedness=handedness,
                )
                storage_started = time.perf_counter()
                user_segment = _safe_segment(header.user_id, "anonymous")
                request_segment = _safe_segment(header.request_id, analysis_id)
                object_path = (
                    f"analyses/v1/{user_segment}/{request_segment}/"
                    "student_corrected.mp4"
                )
                student_signed = self.storage.upload_file(
                    output_path, object_path, content_type="video/mp4"
                )
                overlay_object_path = (
                    f"analyses/v1/{user_segment}/{request_segment}/"
                    "student_skeleton_overlay.mp4"
                )
                overlay_signed = self.storage.upload_file(
                    skeleton_overlay_path,
                    overlay_object_path,
                    content_type="video/mp4",
                )
                storage_finished = time.perf_counter()
                result.diagnostics.update(
                    {
                        "latency_storage_seconds": storage_finished - storage_started,
                        "latency_service_seconds": storage_finished - service_started,
                        "input_video_bytes": float(total_bytes),
                    }
                )
                return self._response(
                    analysis_id,
                    result,
                    student_signed,
                    overlay_signed,
                    probe_video(output_path),
                    probe_video(skeleton_overlay_path),
                )
            except (ValueError, KeyError) as exc:
                LOGGER.warning("analysis rejected id=%s error=%s", analysis_id, exc)
                context.abort(grpc.StatusCode.FAILED_PRECONDITION, str(exc))
            except Exception:
                LOGGER.exception("analysis failed id=%s", analysis_id)
                context.abort(grpc.StatusCode.INTERNAL, "analysis failed")
        raise AssertionError("unreachable")

    def _expert_timeline(self, spec, reference) -> list[analysis_pb2.PhaseMarker]:
        """The chosen expert's checkpoints, timed in that expert's own video.

        Built from the same rule anchors as the learner's timeline so marker i
        means the same moment on both sides, which is what lets playback line
        the two clips up segment by segment rather than stretching one evenly
        across the other. A clip whose phases cannot be read costs the
        alignment, not the analysis.
        """
        try:
            markers = expert_phase_results(
                spec,
                phase_indices=tuple(range(len(reference.source_phase_indices))),
                phase_seconds=reference.phase_seconds(),
                sequence_length=len(reference.source_phase_indices),
            )
        except ValueError as exc:
            LOGGER.warning(
                "expert checkpoints unusable expert=%s error=%s", reference.subject_id, exc
            )
            return []
        return [
            analysis_pb2.PhaseMarker(
                id=marker.id,
                label=marker.label,
                normalized_frame=marker.normalized_frame,
                normalized_position=marker.normalized_position,
                timestamp_seconds=marker.timestamp_seconds,
            )
            for marker in markers
        ]

    def _response(
        self,
        analysis_id: str,
        result: AnalysisResult,
        student_signed: SignedObject,
        overlay_signed: SignedObject,
        student_metadata: dict[str, float | int],
        overlay_metadata: dict[str, float | int],
    ) -> analysis_pb2.AnalyzeVideoResponse:
        backend = self.pipeline.backends[result.skill]
        spec = backend.spec
        details = [
            analysis_pb2.GradingDetail(
                criterion_id=rule.id,
                description=detail["description"],
                grade=float(detail["grade"]),
                maximum=rule.maximum,
            )
            for rule, detail in zip(
                spec.rules, result.grade["grading_details"], strict=True
            )
        ]
        problems_by_frame: dict[int, list[dict[str, object]]] = {}
        for problem in result.coaching_problems:
            problems_by_frame.setdefault(int(problem["frame_index"]), []).append(problem)
        fps = float(student_metadata["fps"])
        pause_frames = round(fps * result.pause_seconds)
        cues: list[analysis_pb2.CoachingCue] = []
        pauses_before = 0
        for frame in sorted(problems_by_frame):
            start_time = (frame + pauses_before) / fps
            for problem in problems_by_frame[frame]:
                cues.append(
                    analysis_pb2.CoachingCue(
                        title=str(problem["title"]),
                        feedback=str(problem["feedback"]),
                        normalized_frame=frame,
                        normalized_position=frame / 63.0,
                        student_timestamp_seconds=start_time,
                        pause_duration_seconds=result.pause_seconds,
                        joint_ids=[int(value) for value in problem["joint_ids"]],
                    )
                )
            pauses_before += pause_frames
        diagnostics = [
            analysis_pb2.DiagnosticValue(key=key, value=float(value))
            for key, value in result.diagnostics.items()
            if isinstance(value, (int, float)) and not isinstance(value, bool)
        ]
        # The correction is generated rather than copied, so the clip beside it
        # is the closest real demonstration to what the learner is reaching
        # towards. Without one the panel simply has no video, as before.
        reference = result.expert_reference
        if reference is None:
            expert_message = analysis_pb2.ExpertMatch(
                expert_id=result.expert_id,
                display_name="Generated expert prior",
                correction_distance=result.expert_distance,
            )
        else:
            expert_signed = self.storage.sign(reference.video_object_path)
            expert_message = analysis_pb2.ExpertMatch(
                expert_id=reference.subject_id,
                display_name=reference.subject_id,
                correction_distance=reference.distance,
                video=self._stored_video(expert_signed, {"fps": reference.fps}),
                motion_start_seconds=reference.motion_start_seconds,
                motion_end_seconds=reference.motion_end_seconds,
                timeline=self._expert_timeline(spec, reference),
            )
        feedback_video = self._stored_video(student_signed, student_metadata)
        overlay_video = self._stored_video(overlay_signed, overlay_metadata)
        return analysis_pb2.AnalyzeVideoResponse(
            analysis_id=analysis_id,
            skill=_SKILL_TO_PROTO[result.skill],
            handedness=_HANDEDNESS_TO_PROTO[result.handedness],
            grade=analysis_pb2.GradingOutcome(
                total_grade=float(result.grade["total_grade"]),
                grading_details=details,
                score_status="expert_only_generated_distribution",
            ),
            # Backward-compatible alias: old clients continue to receive the
            # GPT feedback version through student_video.
            student_video=feedback_video,
            feedback_video=feedback_video,
            skeleton_overlay_video=overlay_video,
            expert=expert_message,
            timeline=[
                analysis_pb2.PhaseMarker(
                    id=phase.id,
                    label=phase.label,
                    normalized_frame=phase.normalized_frame,
                    normalized_position=phase.normalized_position,
                    timestamp_seconds=phase.timestamp_seconds,
                )
                for phase in result.phases
            ],
            diagnostics=diagnostics,
            coaching_cues=cues,
            overall_feedback=result.overall_feedback,
        )

    def RefreshPlaybackUrls(
        self,
        request: analysis_pb2.RefreshPlaybackUrlsRequest,
        context: grpc.ServicerContext,
    ) -> analysis_pb2.RefreshPlaybackUrlsResponse:
        self._authorize(context)
        if not request.object_paths or len(request.object_paths) > 8:
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, "request one to eight objects")
        videos = []
        for object_path in request.object_paths:
            if ".." in object_path or not object_path.startswith(("analyses/", "experts/")):
                context.abort(grpc.StatusCode.PERMISSION_DENIED, "object path is not playable")
            videos.append(self._stored_video(self.storage.sign(object_path), {}))
        return analysis_pb2.RefreshPlaybackUrlsResponse(videos=videos)

    def Health(
        self, request: analysis_pb2.HealthRequest, context: grpc.ServicerContext
    ) -> analysis_pb2.HealthResponse:
        return analysis_pb2.HealthResponse(
            status="serving",
            loaded_skills=[_SKILL_TO_PROTO[value] for value in self.pipeline.loaded_skills],
        )


def serve() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    settings = Settings.from_env()
    service = BadmintonAnalysisService(settings)
    server = grpc.server(
        futures.ThreadPoolExecutor(max_workers=2),
        options=(
            ("grpc.max_receive_message_length", settings.max_video_bytes + 1024 * 1024),
            ("grpc.max_send_message_length", 8 * 1024 * 1024),
        ),
    )
    analysis_pb2_grpc.add_BadmintonAnalysisServicer_to_server(service, server)
    server.add_insecure_port(f"[::]:{settings.port}")
    server.start()
    LOGGER.info("gRPC server listening port=%d", settings.port)

    def stop(*_: object) -> None:
        LOGGER.info("stopping gRPC server")
        server.stop(grace=30)

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    server.wait_for_termination()


if __name__ == "__main__":
    serve()
