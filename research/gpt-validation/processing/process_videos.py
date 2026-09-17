"""Process the study's beginner videos once, on a local Mac, for GPT validation.

Runs the production analysis pipeline in-process (RF-DETR pose on Apple MPS,
EIMD diffusion, grading and GPT coaching) for every video in the manifest,
renders an RF-DETR pose-only overlay of the full clip for experts to judge
before GPT is revealed, uploads the renders under the study's Cloud Storage
prefix, and freezes each result as one Firestore item. No cloud GPU is used.

The study design and Firestore schema are in ../DESIGN.md; `build_manifest.py`
writes the manifest of the 100 workbook-listed beginner videos.

Usage (from the repository root, with Application Default Credentials and
OPENAI_API_KEY set):

  PYTHONPATH=badminton_analysis_ai:badminton_analysis_ai/generated \
    python research/gpt-validation/processing/process_videos.py \
      --manifest research/gpt-validation/processing/manifest.csv \
      --videos-root ~/dev/badminton-analysis/scoring_videos

Reruns are safe: ready items are skipped and failed ones are retried within
--max-attempts, so an interrupted run can simply be started again.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import logging
import os
import tempfile
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

LOGGER = logging.getLogger("gpt-validation-batch")

SKILL_CODES = {"serve": "SV", "smash": "SM"}
ITEMS_COLLECTION = "gpt_validation_items"


@dataclass(frozen=True)
class InputVideo:
    skill: str
    workbook_id: str
    source_file: str
    source_path: str

    @property
    def item_id(self) -> str:
        return item_id_for(self.skill, self.source_file)


def item_id_for(skill: str, source_file: str) -> str:
    """Stable across reruns and independent of manifest order."""
    digest = hashlib.sha1(source_file.encode("utf-8")).hexdigest()[:10]
    return f"{skill}-{digest}"


def read_manifest(path: Path) -> list[InputVideo]:
    """Rows of `skill, workbook_id, source_file, source_path` (see build_manifest.py)."""
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    videos = []
    for row in rows:
        if row["skill"] not in SKILL_CODES:
            raise ValueError(f"unsupported skill in manifest: {row['skill']!r}")
        videos.append(
            InputVideo(
                skill=row["skill"],
                workbook_id=row["workbook_id"],
                source_file=row["source_file"],
                source_path=row["source_path"],
            )
        )
    ids = [video.item_id for video in videos]
    if len(ids) != len(set(ids)):
        raise ValueError("manifest lists the same video twice")
    return videos


def recorded_handedness(video: InputVideo) -> str:
    """The cohort's handedness as recorded in its files.

    A `_left` file with no unmirrored original is the student's own
    left-handed recording; everything else in the cohort is right-handed.
    """
    return "left" if Path(video.source_file).stem.endswith("_left") else "right"


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


# Coaching sources that mean GPT could not be reached. score_gate is a real
# outcome (the swing needs no feedback) and stays ready but ineligible.
RETRYABLE_COACHING_SOURCES = frozenset({"deterministic_fallback", "skipped"})


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
        "workbook_id": video.workbook_id,
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


@contextmanager
def capture_pose_pass() -> Iterator[list[tuple[Path, Any]]]:
    """Record the analysis's own pose extraction so the overlay can reuse it.

    Running RF-DETR a second time just for the step-1 video would double the
    slowest stage on a Mac. The pipeline builds exactly one VideoProcessor per
    analysis (for smash, over its 30 fps copy of the source), so wrapping it
    yields that video path and its tracking data unchanged.
    """
    import service.pipeline as pipeline_module

    captured: list[tuple[Path, Any]] = []
    original = pipeline_module.VideoProcessor

    class RecordingVideoProcessor(original):  # type: ignore[misc, valid-type]
        def process_frames_batched(self, handedness):  # noqa: ANN001, ANN202
            tracking = super().process_frames_batched(handedness)
            captured.append((Path(self.video_path), tracking))
            return tracking

    pipeline_module.VideoProcessor = RecordingVideoProcessor
    try:
        yield captured
    finally:
        pipeline_module.VideoProcessor = original


def render_detected_overlay(video_path: Path, output_path: Path, tracking: Any) -> None:
    """The step-1 reference video: RF-DETR's detected skeleton on the full clip.

    Deliberately shows nothing else -- no generated expert skeleton, score or
    GPT markers -- so it cannot hint at what the experts are asked to judge.
    `tracking` is the pose pass over `video_path`.
    """
    import cv2

    from service.renderer import (
        _draw_skeleton,
        _prepare_detected_pose_for_render,
        _transcode_preserving_frame_rate,
        source_fps,
        source_frame_rate,
    )

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
    input_path = Path(context["videos_root"]) / video.source_path
    if not input_path.is_file():
        raise FileNotFoundError(f"video not found: {input_path}")
    with tempfile.TemporaryDirectory(prefix="gpt-validation-") as raw_directory:
        directory = Path(raw_directory)
        feedback_path = directory / "feedback.mp4"
        overlay_path = directory / "skeleton_overlay.mp4"
        detected_path = directory / "detected_overlay.mp4"

        # Handedness is known for this cohort, exactly as production passes the
        # learner's chosen hand; estimating it wastes a pose pass and is often
        # ambiguous on beginner clips.
        with capture_pose_pass() as pose_passes:
            result = _analyze(
                pipeline, input_path, feedback_path, overlay_path, display_code, skill, recorded_handedness(video)
            )
        if len(pose_passes) != 1:
            raise RuntimeError(f"expected one pose pass, captured {len(pose_passes)}")
        posed_video, tracking = pose_passes[0]
        render_detected_overlay(posed_video, detected_path, tracking)

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


def release_memory() -> None:
    """Return one video's frames and GPU buffers before the next.

    A 1080p clip decodes to over a gigabyte of frames, and MPS keeps its
    allocations cached, so a long run on a 16 GB Mac otherwise grows until the
    OS kills it.
    """
    import gc

    gc.collect()
    try:
        import torch

        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
    except Exception:  # noqa: BLE001 - freeing memory is best effort
        pass


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--videos-root", type=Path, required=True, help="the scoring_videos directory")
    parser.add_argument("--batch-id", default="beginners-2026-09")
    parser.add_argument("--project", default=os.environ.get("GCP_PROJECT_ID", "nstc-linebot-2025"))
    parser.add_argument("--bucket", default=os.environ.get("GCS_BUCKET_NAME", "nstc-2025-storage"))
    parser.add_argument("--max-attempts", type=int, default=2)
    parser.add_argument("--force", action="store_true", help="reprocess items that are already ready")
    parser.add_argument("--limit", type=int, default=0, help="process at most N videos (0 = all)")
    parser.add_argument("--only", action="append", default=[], help="process only this item id (repeatable)")
    arguments = parser.parse_args()

    from google.cloud import firestore

    from service.config import Settings
    from service.logging_config import configure_logging
    from service.pipeline import SkeletonAnalysisPipeline
    from service.storage import ObjectStorage

    configure_logging(logging.INFO)
    if not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY is required: GPT coaching is what this study evaluates")
    videos = read_manifest(arguments.manifest)
    videos_root = arguments.videos_root.expanduser().resolve()
    codes = display_codes(videos)
    selected = [video for video in videos if not arguments.only or video.item_id in arguments.only]
    prefix = f"gpt-validation/{arguments.batch_id}"

    items = firestore.Client(project=arguments.project).collection(ITEMS_COLLECTION)
    # The pipeline reads its model paths relative to badminton_analysis_ai/.
    service_root = Path(__file__).resolve().parents[3] / "badminton_analysis_ai"
    os.chdir(service_root)
    os.environ.setdefault("ANALYSIS_GRPC_API_KEY", "local-validation-run")
    os.environ.setdefault("GCP_PROJECT_ID", arguments.project)
    os.environ.setdefault("GCS_BUCKET_NAME", arguments.bucket)
    settings = Settings.from_env()
    context = {
        "pipeline": SkeletonAnalysisPipeline(
            settings.expert_motion_model_root,
            device=settings.device,
            openai_model=settings.openai_model,
            pause_seconds=settings.coaching_pause_seconds,
        ),
        "storage": ObjectStorage(arguments.project, arguments.bucket),
        "videos_root": videos_root,
        "prefix": prefix,
        "batch_id": arguments.batch_id,
    }
    LOGGER.info(
        "validation run starting",
        extra={"json_fields": {"batch_id": arguments.batch_id, "videos_total": len(videos), "videos_selected": len(selected)}},
    )

    processed = failures = 0
    for video in selected:
        if arguments.limit and processed >= arguments.limit:
            break
        reference = items.document(video.item_id)
        snapshot = reference.get()
        existing = snapshot.to_dict() if snapshot.exists else None
        if not should_process(existing, force=arguments.force, max_attempts=arguments.max_attempts):
            continue
        processed += 1
        attempts = int((existing or {}).get("attempts", 0)) + 1
        now = datetime.now(timezone.utc)
        reference.set(
            {
                "item_id": video.item_id,
                "batch_id": arguments.batch_id,
                "skill": video.skill,
                "source_file": video.source_file,
                "workbook_id": video.workbook_id,
                "display_code": codes[video.item_id],
                "status": "processing",
                "eligible": False,
                "attempts": attempts,
                "updated_at": now,
                **({} if existing else {"created_at": now}),
            },
            merge=True,
        )
        started = time.perf_counter()
        try:
            item = _process(video, context=context, display_code=codes[video.item_id], attempts=attempts)
        except Exception as exc:  # noqa: BLE001 - one bad video must not stop the run
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
        finally:
            release_memory()
        if item["coaching_source"] in RETRYABLE_COACHING_SOURCES:
            # GPT was unreachable (out of credits, rate limited, down), not
            # judged unnecessary. Keep the item out of the study and stop, so
            # the remaining videos are not spent on feedback nobody can rate.
            LOGGER.error(
                "coaching unavailable; stopping run",
                extra={"json_fields": {"item_id": video.item_id, "coaching_source": item["coaching_source"]}},
            )
            reference.set(
                {"status": "failed", "error": "coaching unavailable: GPT did not return feedback", "eligible": False, "attempts": 0, "updated_at": datetime.now(timezone.utc)},
                merge=True,
            )
            failures += 1
            break
        item["updated_at"] = datetime.now(timezone.utc)
        reference.set(item, merge=True)
        LOGGER.info(
            "validation item ready",
            extra={"json_fields": {"item_id": video.item_id, "display_code": item["display_code"], "eligible": item["eligible"], "coaching_source": item["coaching_source"], "n_cues": len(item["gpt_cues"]), "seconds": round(time.perf_counter() - started, 1)}},
        )
    LOGGER.info("validation run finished", extra={"json_fields": {"processed": processed, "failures": failures}})
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
