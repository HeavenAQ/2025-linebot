"""Process beginner videos once for the GPT feedback validation study.

Runs as a Cloud Run job on an L4 with the analysis image, entirely outside the
production analysis service: each task builds its own pipeline, analyzes its
shard of uploaded videos with GPT coaching on, renders the RF-DETR pose-only
reference video experts watch before GPT is revealed, uploads every render
under the study's own prefix, and freezes the result as one Firestore item.

The study design and the Firestore schema are in
research/gpt-validation/DESIGN.md at the repository root.

Configuration (environment):
  GCP_PROJECT_ID, GCS_BUCKET_NAME    where inputs, renders and items live
  VALIDATION_BATCH_ID                e.g. beginners-2026-09
  VALIDATION_PREFIX                  default gpt-validation/<batch id>
  CLOUD_RUN_TASK_INDEX / _COUNT      set by Cloud Run; each task takes one shard
  VALIDATION_MAX_ATTEMPTS            default 2; failed items are retried up to it
  VALIDATION_FORCE=1                 reprocess items that are already ready
  OPENAI_API_KEY, OPENAI_COACHING_MODEL, COACHING_* as for the service

Reruns are safe: ready items are skipped, so a job can simply be executed again
after a partial failure.
"""

from __future__ import annotations

import hashlib
import logging
import os
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Iterable

LOGGER = logging.getLogger("gpt-validation-batch")

SKILL_CODES = {"serve": "SV", "smash": "SM"}
ITEMS_COLLECTION = "gpt_validation_items"


@dataclass(frozen=True)
class InputVideo:
    object_path: str
    skill: str
    source_file: str

    @property
    def item_id(self) -> str:
        return item_id_for(self.skill, self.source_file)


def item_id_for(skill: str, source_file: str) -> str:
    """Stable across reruns and independent of listing order."""
    digest = hashlib.sha1(source_file.encode("utf-8")).hexdigest()[:10]
    return f"{skill}-{digest}"


def parse_input(object_path: str, prefix: str) -> InputVideo | None:
    """`<prefix>/inputs/<skill>/<file>.mp4` → InputVideo; anything else is ignored.

    Mirrored `_left` copies repeat an attempt already in the set, so they are
    excluded even if they were uploaded.
    """
    root = PurePosixPath(prefix.strip("/")) / "inputs"
    path = PurePosixPath(object_path)
    try:
        relative = path.relative_to(root)
    except ValueError:
        return None
    if len(relative.parts) != 2:
        return None
    skill, source_file = relative.parts
    if skill not in SKILL_CODES or path.suffix.lower() != ".mp4":
        return None
    if path.stem.endswith("_left"):
        return None
    return InputVideo(object_path=object_path, skill=skill, source_file=source_file)


def display_codes(videos: Iterable[InputVideo]) -> dict[str, str]:
    """Neutral labels (SV-001, SM-001, …) that reveal nothing about the file.

    Numbered in item-id order, which is a hash of the filename, so the label
    carries no hint of student codes or the 較佳 (better attempt) suffix.
    """
    codes: dict[str, str] = {}
    for skill, code in SKILL_CODES.items():
        ids = sorted(video.item_id for video in videos if video.skill == skill)
        for number, item_id in enumerate(ids, start=1):
            codes[item_id] = f"{code}-{number:03d}"
    return codes


def shard(videos: list[InputVideo], task_index: int, task_count: int) -> list[InputVideo]:
    if task_count < 1 or not 0 <= task_index < task_count:
        raise ValueError(f"invalid shard {task_index}/{task_count}")
    ordered = sorted(videos, key=lambda video: video.item_id)
    return [video for position, video in enumerate(ordered) if position % task_count == task_index]


def should_process(existing: dict[str, Any] | None, *, force: bool, max_attempts: int) -> bool:
    if existing is None or force:
        return True
    if existing.get("status") == "ready":
        return False
    if existing.get("status") == "failed":
        return int(existing.get("attempts", 0)) < max_attempts
    return True


def build_item(
    *,
    video: InputVideo,
    batch_id: str,
    display_code: str,
    rules: list[tuple[str, str, float]],
    grade: dict[str, Any],
    handedness: str,
    coaching_source: str,
    coaching_model: str,
    overall_feedback: str,
    problems: list[dict[str, Any]],
    media: dict[str, str],
    attempts: int,
) -> dict[str, Any]:
    """The frozen Firestore record of one processed video (DESIGN.md schema)."""
    details = list(grade.get("grading_details", []))
    if len(details) != len(rules):
        raise ValueError("grading details do not match the skill's rubric")
    criteria = [
        {"id": rule_id, "name_zh": name, "grade": float(detail["grade"]), "maximum": float(maximum)}
        for (rule_id, name, maximum), detail in zip(rules, details)
    ]
    rule_ids = {rule_id for rule_id, _, _ in rules}
    cues = []
    for index, problem in enumerate(problems, start=1):
        criterion_id = str(problem.get("rule_reference", ""))
        if criterion_id not in rule_ids:
            raise ValueError(f"GPT cue {index} names an unknown criterion {criterion_id!r}")
        cues.append(
            {
                "index": index,
                "criterion_id": criterion_id,
                "title": str(problem.get("title", "")),
                "feedback": str(problem.get("feedback", "")),
            }
        )
    flagged = []
    for cue in cues:
        if cue["criterion_id"] not in flagged:
            flagged.append(cue["criterion_id"])
    return {
        "item_id": video.item_id,
        "batch_id": batch_id,
        "skill": video.skill,
        "source_file": video.source_file,
        "display_code": display_code,
        "status": "ready",
        "error": "",
        "attempts": attempts,
        # Only real GPT output is under test: the rule-based fallback and the
        # score gate (no model call) are recorded but never shown to experts.
        "eligible": coaching_source == "openai",
        "coaching_source": coaching_source,
        "coaching_model": coaching_model,
        "handedness": handedness,
        "total_grade": float(grade["total_grade"]),
        "criteria": criteria,
        "gpt_overall_feedback": overall_feedback,
        "gpt_cues": cues,
        "gpt_flagged_criteria": flagged,
        "media": media,
    }


def render_detected_overlay(video_path: Path, output_path: Path, detector: Any) -> None:
    """The step-1 reference video: RF-DETR's detected skeleton on the full clip.

    Deliberately shows nothing else -- no generated expert skeleton, score or
    GPT markers -- so it cannot hint at what the experts are asked to judge.
    """
    import cv2

    from badminton_analysis.services.video_processor import VideoProcessor
    from service.renderer import (
        _draw_skeleton,
        _prepare_detected_pose_for_render,
        _transcode_preserving_frame_rate,
        source_fps,
        source_frame_rate,
    )

    tracking = VideoProcessor(str(video_path), detector).process_frames_batched(None)
    frames = tracking["frames"]
    if not frames:
        raise ValueError("video has no frames")
    coordinates, confidence = _prepare_detected_pose_for_render(tracking)
    height, width = frames[0].shape[:2]
    raw_path = output_path.with_suffix(".raw.mp4")
    writer = cv2.VideoWriter(
        str(raw_path), cv2.VideoWriter_fourcc(*"mp4v"), source_fps(video_path), (width, height)
    )
    if not writer.isOpened():
        raise RuntimeError("could not open the overlay video writer")
    try:
        thickness = max(3, round(min(width, height) / 270))
        for frame, points, scores in zip(frames, coordinates, confidence):
            canvas = frame.copy()
            _draw_skeleton(canvas, points, scores, (30, 210, 255), thickness)
            writer.write(canvas)
    finally:
        writer.release()
    try:
        _transcode_preserving_frame_rate(raw_path, output_path, source_frame_rate(video_path))
    finally:
        raw_path.unlink(missing_ok=True)


def _process(video: InputVideo, *, context: dict[str, Any], display_code: str, attempts: int) -> dict[str, Any]:
    from badminton_analysis.ml.skill_specs import get_skill_spec
    from badminton_analysis.models.types import Skill

    pipeline = context["pipeline"]
    storage = context["storage"]
    prefix = context["prefix"]
    skill = Skill.convert_to_enum(video.skill)
    with tempfile.TemporaryDirectory(prefix="gpt-validation-") as raw_directory:
        directory = Path(raw_directory)
        input_path = directory / "input.mp4"
        context["bucket"].blob(video.object_path).download_to_filename(str(input_path))
        feedback_path = directory / "feedback.mp4"
        overlay_path = directory / "skeleton_overlay.mp4"
        detected_path = directory / "detected_overlay.mp4"

        try:
            result = _analyze(pipeline, input_path, feedback_path, overlay_path, display_code, skill, "auto")
        except ValueError as exc:
            if "ambiguous" not in str(exc):
                raise
            # The beginner originals are right-handed recordings; mirrored
            # left-handed copies are excluded from the study.
            result = _analyze(pipeline, input_path, feedback_path, overlay_path, display_code, skill, "right")
        render_detected_overlay(input_path, detected_path, pipeline.pose_detector)

        renders = f"{prefix}/renders/{video.item_id}"
        media = {
            "detected_overlay": f"{renders}/detected_overlay.mp4",
            "feedback_video": f"{renders}/feedback.mp4",
            "skeleton_overlay": f"{renders}/skeleton_overlay.mp4",
        }
        storage.upload_file(detected_path, media["detected_overlay"], content_type="video/mp4")
        storage.upload_file(feedback_path, media["feedback_video"], content_type="video/mp4")
        storage.upload_file(overlay_path, media["skeleton_overlay"], content_type="video/mp4")

    spec = get_skill_spec(skill)
    return build_item(
        video=video,
        batch_id=context["batch_id"],
        display_code=display_code,
        rules=[(rule.id, rule.name_zh_tw, rule.maximum) for rule in spec.rules],
        grade=dict(result.grade),
        handedness=str(result.handedness),
        coaching_source=str(result.diagnostics.get("coaching_source", "")),
        coaching_model=pipeline.coaching.model,
        overall_feedback=result.overall_feedback,
        problems=[dict(problem) for problem in result.coaching_problems],
        media=media,
        attempts=attempts,
    )


def _analyze(pipeline: Any, video: Path, feedback: Path, overlay: Path, filename: str, skill: Any, handedness: str) -> Any:
    return pipeline.analyze(
        video_path=video,
        output_path=feedback,
        skeleton_overlay_path=overlay,
        # Drawn into the videos' header, so use the neutral code, not the file.
        filename=filename,
        skill=skill,
        requested_handedness=handedness,
        skip_coaching=False,
    )


def main() -> int:
    from google.cloud import firestore, storage as gcs

    from service.config import Settings
    from service.logging_config import configure_logging
    from service.pipeline import SkeletonAnalysisPipeline
    from service.storage import ObjectStorage

    configure_logging(logging.INFO)
    project = os.environ["GCP_PROJECT_ID"]
    bucket_name = os.environ["GCS_BUCKET_NAME"]
    batch_id = os.environ["VALIDATION_BATCH_ID"]
    prefix = os.environ.get("VALIDATION_PREFIX", f"gpt-validation/{batch_id}").strip("/")
    task_index = int(os.environ.get("CLOUD_RUN_TASK_INDEX", "0"))
    task_count = int(os.environ.get("CLOUD_RUN_TASK_COUNT", "1"))
    max_attempts = int(os.environ.get("VALIDATION_MAX_ATTEMPTS", "2"))
    force = os.environ.get("VALIDATION_FORCE") == "1"

    client = gcs.Client(project=project)
    bucket = client.bucket(bucket_name)
    videos = [
        video
        for blob in client.list_blobs(bucket_name, prefix=f"{prefix}/inputs/")
        if (video := parse_input(blob.name, prefix)) is not None
    ]
    codes = display_codes(videos)
    mine = shard(videos, task_index, task_count)
    LOGGER.info(
        "validation shard starting",
        extra={"json_fields": {"batch_id": batch_id, "task_index": task_index, "task_count": task_count, "videos_total": len(videos), "videos_in_shard": len(mine)}},
    )

    items = firestore.Client(project=project).collection(ITEMS_COLLECTION)
    settings = Settings.from_env()
    context = {
        "pipeline": SkeletonAnalysisPipeline(
            settings.expert_motion_model_root,
            device=settings.device,
            openai_model=settings.openai_model,
            pause_seconds=settings.coaching_pause_seconds,
        ),
        "storage": ObjectStorage(project, bucket_name),
        "bucket": bucket,
        "prefix": prefix,
        "batch_id": batch_id,
    }

    failures = 0
    for video in mine:
        reference = items.document(video.item_id)
        snapshot = reference.get()
        existing = snapshot.to_dict() if snapshot.exists else None
        if not should_process(existing, force=force, max_attempts=max_attempts):
            continue
        attempts = int((existing or {}).get("attempts", 0)) + 1
        now = datetime.now(timezone.utc)
        reference.set(
            {
                "item_id": video.item_id,
                "batch_id": batch_id,
                "skill": video.skill,
                "source_file": video.source_file,
                "display_code": codes[video.item_id],
                "status": "processing",
                "eligible": False,
                "attempts": attempts,
                "updated_at": now,
                **({} if existing else {"created_at": now}),
            },
            merge=True,
        )
        try:
            item = _process(video, context=context, display_code=codes[video.item_id], attempts=attempts)
        except Exception as exc:  # noqa: BLE001 - one bad video must not stop the shard
            failures += 1
            LOGGER.exception(
                "validation item failed",
                extra={"json_fields": {"item_id": video.item_id, "error_type": type(exc).__name__}},
            )
            reference.set(
                {"status": "failed", "error": f"{type(exc).__name__}: {exc}"[:500], "eligible": False, "updated_at": datetime.now(timezone.utc)},
                merge=True,
            )
            continue
        item["updated_at"] = datetime.now(timezone.utc)
        reference.set(item, merge=True)
        LOGGER.info(
            "validation item ready",
            extra={"json_fields": {"item_id": video.item_id, "skill": video.skill, "eligible": item["eligible"], "coaching_source": item["coaching_source"], "n_cues": len(item["gpt_cues"])}},
        )
    LOGGER.info("validation shard finished", extra={"json_fields": {"task_index": task_index, "failures": failures}})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
