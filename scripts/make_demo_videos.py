"""Build the demonstration videos students watch before recording themselves.

One video per expert per skill, in both handednesses. Each is the same expert
stroke three times:

  1. 正常速度 -- the whole stroke at recording speed
  2. 慢動作 -- quarter speed, freezing 6 s on every rubric checkpoint with its
     name on screen, in the order the stroke reaches them rather than the order
     the rubric scores them. Two criteria judged at the same moment freeze
     twice, once per criterion, because the learner is marked on each
     separately.
  3. 再看一次 -- normal speed again, now that the checkpoints are known

Checkpoint frames come from the production expert reference bank, so the
moments shown are exactly the ones the grader scores against. Left-handed
videos are the right-handed expert mirrored: there are too few left-handed
experts to demonstrate from.

Usage:
    python scripts/make_demo_videos.py --videos-root ~/dev/badminton-analysis/scoring_videos
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

REPO = Path(__file__).resolve().parents[1]
ANALYSIS = REPO / "badminton_analysis_ai"
sys.path.insert(0, str(ANALYSIS))
sys.path.insert(0, str(ANALYSIS / "generated"))

from badminton_analysis.ml.skill_specs import SKILL_SPECS  # noqa: E402
from badminton_analysis.models.types import Skill  # noqa: E402

# Where each skill's expert clips live under the scoring-videos root.
EXPERT_FOLDERS = {"serve": "發球/專家發球", "smash": "殺球/專家殺球"}

SKILL_ZH = {"smash": "殺球", "serve": "發球"}
MIRROR = {"right": "left", "left": "right"}
HAND_ZH = {"right": "右手", "left": "左手"}

FONT_CANDIDATES = [
    "/System/Library/Fonts/Hiragino Sans GB.ttc",
    "/System/Library/Fonts/PingFang.ttc",
    "/System/Library/Fonts/STHeiti Medium.ttc",
]

SLOW_FACTOR = 4  # quarter speed
PAUSE_SECONDS = 6

# How the slow motion is laid out, per skill. Not derived from the rubric: a
# rule's anchor is the frame the scorer measures against, not where the movement
# happens, so ordering by anchors played the stroke backwards.
#
# Each entry is (criterion ids, keyframes): one keyframe freezes, two play the
# stroke between them at a crawl with each criterion captioned across its share.
DEMO_SEQUENCES = {
    "serve": (
        (("arms_raised",), (0,)),
        (("racket_foot_weight",), (1,)),
        (("weight_transfer",), (1, 2)),
        (("wrist_flick",), (2,)),
        (("hip_rotation", "shoulder_rotation"), (3, 4)),
    ),
    "smash": (
        (("preparation",), (0,)),
        (("body_rotation", "arm_balance"), (1, 2)),
        (("elbow_forward", "wrist_flick"), (2,)),
        (("follow_through",), (4,)),
    ),
}
TITLE_SECONDS = 2.5
FPS = 30


def font(size: int) -> ImageFont.FreeTypeFont:
    for path in FONT_CANDIDATES:
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    raise SystemExit("no Traditional Chinese font found")


def run(*command: str | Path) -> None:
    result = subprocess.run([str(part) for part in command], capture_output=True, text=True)
    if result.returncode != 0:
        raise SystemExit(f"{command[0]} failed:\n{result.stderr[-2000:]}")


def probe(video: Path) -> tuple[int, int]:
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v", "-show_entries",
         "stream=width,height", "-of", "json", str(video)],
        capture_output=True, text=True, check=True,
    )
    stream = json.loads(result.stdout)["streams"][0]
    return int(stream["width"]), int(stream["height"])


@dataclass
class Checkpoint:
    """One criterion as the learner sees it named on screen."""

    name: str
    points: float
    index: int
    total: int


@dataclass
class Step:
    """A stretch of the slow motion: a freeze when `end` is None, else a span."""

    marks: list[Checkpoint]
    start: int
    end: int | None


def checkpoints(skill: str, phase_frames: list[int]) -> list[Step]:
    """The slow motion, step by step, in the order the learner watches it."""
    rules = {rule.id: rule for rule in SKILL_SPECS[Skill[skill.upper()]].rules}
    sequence = DEMO_SEQUENCES[skill]

    listed = [rule_id for ids, _ in sequence for rule_id in ids]
    if sorted(listed) != sorted(rules):
        # The rubric changed under the layout; a silently dropped criterion is
        # a demonstration that teaches the wrong thing.
        missing = sorted(set(rules) - set(listed)) or "none"
        unknown = sorted(set(listed) - set(rules)) or "none"
        raise SystemExit(f"{skill}: DEMO_SEQUENCES missing {missing}, unknown {unknown}")

    steps, number = [], 0
    for ids, phases in sequence:
        marks = []
        for rule_id in ids:
            number += 1
            rule = rules[rule_id]
            marks.append(Checkpoint(name=rule.name_zh_tw, points=rule.maximum,
                                    index=number, total=len(listed)))
        steps.append(Step(
            marks=marks,
            start=int(phase_frames[phases[0]]),
            end=int(phase_frames[phases[-1]]) if len(phases) > 1 else None,
        ))
    return steps


def title_card(path: Path, size: tuple[int, int], heading: str, detail: str) -> None:
    width, height = size
    image = Image.new("RGB", size, (16, 20, 26))
    draw = ImageDraw.Draw(image)
    heading_font, detail_font = font(int(width * 0.085)), font(int(width * 0.045))
    for text, chosen, offset, colour in (
        (heading, heading_font, -int(width * 0.05), (245, 245, 245)),
        (detail, detail_font, int(width * 0.04), (150, 200, 175)),
    ):
        box = draw.textbbox((0, 0), text, font=chosen)
        draw.text(((width - box[2] + box[0]) / 2, (height - box[3] + box[1]) / 2 + offset),
                  text, font=chosen, fill=colour)
    image.save(path)


def caption_band(size: tuple[int, int], mark: Checkpoint) -> Image.Image:
    """The checkpoint's name and worth, as a band to sit at the foot of frame."""
    width, height = size
    band = int(height * 0.16)
    overlay = Image.new("RGBA", (width, band), (10, 14, 20, 225))
    draw = ImageDraw.Draw(overlay)
    name_font, note_font = font(int(width * 0.062)), font(int(width * 0.034))
    draw.text((int(width * 0.05), int(band * 0.16)), mark.name, font=name_font, fill=(255, 255, 255))
    draw.text((int(width * 0.05), int(band * 0.62)),
              f"檢核點 {mark.index} / {mark.total}　滿分 {mark.points:.0f} 分",
              font=note_font, fill=(150, 200, 175))
    return overlay


def caption(frame_image: Path, output: Path, mark: Checkpoint) -> None:
    """Write the checkpoint's name over the frozen frame."""
    image = Image.open(frame_image).convert("RGB")
    band = caption_band(image.size, mark)
    image.paste(band, (0, image.size[1] - band.size[1]), band)
    image.save(output)


def caption_overlay(path: Path, size: tuple[int, int], mark: Checkpoint) -> None:
    """The same band on transparency, for overlaying on moving video."""
    caption_band(size, mark).save(path)


def encode_still(image: Path, output: Path, seconds: float, size: tuple[int, int]) -> None:
    run("ffmpeg", "-y", "-loop", "1", "-framerate", FPS, "-i", image, "-t", seconds,
        "-vf", f"scale={size[0]}:{size[1]},format=yuv420p", "-c:v", "libx264",
        "-preset", "veryfast", "-crf", "20", "-r", FPS, output)


def build(skill: str, handedness: str, mirrored: bool, source: Path, phase_frames: list[int],
          window: list[int], work: Path, output: Path) -> None:
    work.mkdir(parents=True, exist_ok=True)
    steps = checkpoints(skill, phase_frames)

    # One normalised copy drives every segment: same size, same frame rate, no
    # audio, mirrored once here for the left-handed version.
    base = work / "base.mp4"
    mirror = "hflip," if mirrored else ""
    run("ffmpeg", "-y", "-i", source, "-vf", f"{mirror}fps={FPS},format=yuv420p",
        "-an", "-c:v", "libx264", "-preset", "veryfast", "-crf", "20", base)
    size = probe(base)

    pieces: list[Path] = []

    def add_title(heading: str, detail: str) -> None:
        index = len(pieces)
        card = work / f"title-{index}.png"
        clip = work / f"title-{index}.mp4"
        title_card(card, size, heading, detail)
        encode_still(card, clip, TITLE_SECONDS, size)
        pieces.append(clip)

    def add_normal(tag: str) -> None:
        clip = work / f"normal-{tag}.mp4"
        shutil.copy(base, clip)
        pieces.append(clip)

    add_title(f"{SKILL_ZH[skill]}　{HAND_ZH[handedness]}示範", "動作示範影片")
    add_title("① 正常速度", "先看一次完整動作")
    add_normal("first")

    add_title("② 慢動作　停格檢核", "逐一看過每個檢核點")
    start, end = int(window[0]), int(window[-1])
    cursor = start

    def catch_up(number: int, target: int) -> None:
        """Quarter speed from wherever the clip is up to the next checkpoint."""
        nonlocal cursor
        if target <= cursor:
            return
        slow = work / f"slow-{number}.mp4"
        run("ffmpeg", "-y", "-i", base, "-vf",
            f"trim=start_frame={cursor}:end_frame={target},setpts=(PTS-STARTPTS)*{SLOW_FACTOR}",
            "-an", "-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-r", FPS, slow)
        pieces.append(slow)
        cursor = target

    for number, step in enumerate(steps):
        begin = max(start, min(step.start, end))
        catch_up(number, begin)

        # A criterion judged over a movement is watched over that movement: the
        # span plays far slower than the surrounding slow motion, each caption
        # taking its share of it.
        if step.end is not None:
            finish = max(begin, min(step.end, end))
            share = max(1, (finish - begin) // len(step.marks))
            for position, mark in enumerate(step.marks):
                first = begin + position * share
                last = finish if position == len(step.marks) - 1 else first + share
                # Stretch each share to the dwell a freeze would have had, so
                # the section keeps the pace of the rest of the segment.
                factor = max(SLOW_FACTOR, round(PAUSE_SECONDS * int(FPS) / max(1, last - first)))
                band = work / f"span-{number}-{position}.png"
                clip = work / f"span-{number}-{position}.mp4"
                caption_overlay(band, size, mark)
                run("ffmpeg", "-y", "-i", base, "-loop", "1", "-i", band,
                    "-filter_complex",
                    f"[0:v]trim=start_frame={first}:end_frame={last},"
                    f"setpts=(PTS-STARTPTS)*{factor}[slow];"
                    f"[slow][1:v]overlay=0:H-h:shortest=1,format=yuv420p",
                    "-an", "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
                    "-r", FPS, clip)
                pieces.append(clip)
            cursor = max(cursor, finish)
            continue

        for position, mark in enumerate(step.marks):
            still = work / f"freeze-{number}-{position}.png"
            captioned = work / f"freeze-{number}-{position}-caption.png"
            frozen = work / f"freeze-{number}-{position}.mp4"
            run("ffmpeg", "-y", "-i", base, "-vf", f"select=eq(n\\,{begin})", "-vsync", "0",
                "-frames:v", "1", still)
            caption(still, captioned, mark)
            encode_still(captioned, frozen, PAUSE_SECONDS, size)
            pieces.append(frozen)

    if end > cursor:
        tail = work / "slow-tail.mp4"
        run("ffmpeg", "-y", "-i", base, "-vf",
            f"trim=start_frame={cursor}:end_frame={end},setpts=(PTS-STARTPTS)*{SLOW_FACTOR}",
            "-an", "-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-r", FPS, tail)
        pieces.append(tail)

    add_title("③ 再看一次　正常速度", "把剛才的檢核點串起來")
    add_normal("second")

    listing = work / "pieces.txt"
    listing.write_text("".join(f"file '{piece.resolve()}'\n" for piece in pieces))
    output.parent.mkdir(parents=True, exist_ok=True)
    run("ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", listing,
        "-c:v", "libx264", "-preset", "medium", "-crf", "21", "-pix_fmt", "yuv420p",
        "-movflags", "+faststart", output)


@dataclass
class Demo:
    """One expert's stroke, ready to render."""

    skill: str
    person: str
    source: Path
    handedness: str
    phase_frames: list[int]
    window: list[int]


def expert_demos(bank, root: Path) -> list[Demo]:
    """One demonstration per expert per skill, using their first recorded take.

    The bank holds several takes per person; a learner only needs to watch one,
    and the first is the one whose checkpoints are listed first.
    """
    seen: set[tuple[str, str]] = set()
    demos: list[Demo] = []
    for index in range(len(bank["subject_id"])):
        subject, skill = str(bank["subject_id"][index]), str(bank["skill"][index])
        person = subject.rsplit("-", 1)[0] if subject.startswith("expert-") else subject.rstrip("0123456789")
        if (skill, person) in seen:
            continue
        seen.add((skill, person))
        source = root / EXPERT_FOLDERS[skill] / Path(str(bank["video_object_path"][index])).name
        if not source.exists():
            raise SystemExit(f"missing expert video: {source}")
        demos.append(Demo(
            skill=skill,
            person=person,
            source=source,
            handedness=str(bank["handedness"][index]),
            phase_frames=[int(value) for value in bank["source_phase_indices"][index]],
            window=[int(value) for value in bank["analysis_window"][index]],
        ))
    return demos


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--videos-root", default="~/dev/badminton-analysis/scoring_videos")
    parser.add_argument("--output", default=str(REPO / "demo-videos"))
    parser.add_argument("--bank", default=str(ANALYSIS / "models" / "expert_reference_bank.npz"))
    parser.add_argument("--only", action="append", default=[], help="build only this expert (repeatable)")
    parser.add_argument("--skill", action="append", default=[], help="build only this skill (repeatable)")
    parser.add_argument("--keep-work", action="store_true", help="keep the intermediate clips")
    arguments = parser.parse_args()

    root = Path(arguments.videos_root).expanduser()
    bank = np.load(arguments.bank, allow_pickle=True)
    work_root = Path(arguments.output) / ".work"

    demos = expert_demos(bank, root)
    if arguments.only:
        demos = [demo for demo in demos if demo.person in arguments.only]
    if arguments.skill:
        demos = [demo for demo in demos if demo.skill in arguments.skill]
    print(f"{len(demos)} experts -> {len(demos) * 2} videos")

    for demo in demos:
        # The expert's own handedness first, then the mirror for the other
        # hand: there are too few left-handed experts to record both.
        for handedness in (demo.handedness, MIRROR[demo.handedness]):
            mirrored = handedness != demo.handedness
            output = Path(arguments.output) / demo.skill / f"{demo.person}-{handedness}.mp4"
            work = work_root / f"{demo.skill}-{demo.person}-{handedness}"
            if work.exists():
                shutil.rmtree(work)
            print(f"building {output.relative_to(Path(arguments.output))} from {demo.source.name}"
                  f"{' (mirrored)' if mirrored else ''}")
            build(demo.skill, handedness, mirrored, demo.source, demo.phase_frames, demo.window, work, output)

    if not arguments.keep_work and work_root.exists():
        shutil.rmtree(work_root)


if __name__ == "__main__":
    main()
