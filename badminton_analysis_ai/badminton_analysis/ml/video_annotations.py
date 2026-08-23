from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from badminton_analysis.models.types import Handedness


BETTER_PERFORMANCE_MARKER = "較佳"
MIRROR_MARKER = "mirror"
SYNTHETIC_MIRROR_PREFIX = "synthetic_left_from_"

# Delimiter-aware markers. The trailing delimiter is a lookahead so that
# removing one marker cannot swallow the delimiter that introduces the next.
_LEFT_MARKER = re.compile(r"(?:^|[-_\s])left(?=$|[-_\s])", re.IGNORECASE)
_MIRROR_MARKER = re.compile(r"(?:^|[-_\s])mirror(?=$|[-_\s])", re.IGNORECASE)

# Participant conventions. ``CG13`` and ``EG2`` number the participants
# themselves, whereas lift filenames such as ``A1``/``A2``/``A3`` number the
# takes of participant ``A``.
_PARTICIPANT_NUMBER = re.compile(r"^([A-Za-z]+\d+)")
_PARTICIPANT_LETTER = re.compile(r"^([A-Za-z]+)\d+")

PARTICIPANT_NUMBER_GROUPING = "participant_number"
PARTICIPANT_LETTER_GROUPING = "participant_letter"
SUBJECT_GROUPINGS = (PARTICIPANT_NUMBER_GROUPING, PARTICIPANT_LETTER_GROUPING)


@dataclass(frozen=True)
class VideoAnnotations:
    subject_id: str
    handedness: Handedness | None
    performance_tier: str
    is_mirror: bool


def strip_handedness_and_mirror_markers(stem: str) -> str:
    """Return a clip stem with mirror and handedness markers removed.

    ``EG12``, ``EG12_left``, ``EG12_mirror`` and ``EG12_mirror_left`` all
    collapse onto ``EG12`` so that a horizontally flipped copy of a clip is
    never treated as a different participant or a different source clip.
    """

    canonical = stem.removeprefix(SYNTHETIC_MIRROR_PREFIX)
    canonical = _MIRROR_MARKER.sub("", canonical)
    canonical = _LEFT_MARKER.sub("", canonical)
    return canonical.strip("-_ ")


def canonical_clip_stem(name: str | Path) -> str:
    """Return the source-clip identity shared by a clip and its mirror."""

    return strip_handedness_and_mirror_markers(Path(name).stem)


def has_left_handed_marker(name: str | Path) -> bool:
    """Report whether a filename annotates its performer as left-handed."""

    return bool(_LEFT_MARKER.search(Path(name).stem))


def is_mirrored_name(name: str | Path) -> bool:
    stem = Path(name).stem
    return bool(_MIRROR_MARKER.search(stem)) or stem.startswith(
        SYNTHETIC_MIRROR_PREFIX
    )


def expert_subject_identity(filename: str | Path) -> str:
    """Return a recoverable expert subject, otherwise the individual clip.

    Named expert clips use names such as ``林國欽1`` through ``林國欽10``; their
    trailing take number is removed. Serve clips use an explicit
    ``<label>-<subject>-<take>`` form such as ``expert-6-22``, whose subject is
    also recoverable. Anonymous identifiers with no subject field (for example
    ``team_right_41``) deliberately stay clip-level, because inventing a shared
    identity there would merge unrelated people into one split unit.
    """

    stem = canonical_clip_stem(filename)
    labelled_take = re.match(r"^([A-Za-z]+[-_]\d+)[-_]\d+$", stem, flags=re.UNICODE)
    if labelled_take:
        return labelled_take.group(1)
    named_take = re.match(r"^(.*[^\W\d_])\d+$", stem, flags=re.UNICODE)
    return named_take.group(1) if named_take else stem


def subject_id_from_stem(
    stem: str, *, subject_grouping: str = PARTICIPANT_NUMBER_GROUPING
) -> str:
    if subject_grouping not in SUBJECT_GROUPINGS:
        raise ValueError(
            f"unknown subject grouping {subject_grouping!r}; "
            f"expected one of {SUBJECT_GROUPINGS}"
        )
    canonical = strip_handedness_and_mirror_markers(stem)
    pattern = (
        _PARTICIPANT_LETTER
        if subject_grouping == PARTICIPANT_LETTER_GROUPING
        else _PARTICIPANT_NUMBER
    )
    match = pattern.match(canonical)
    return match.group(1).upper() if match else canonical


def parse_video_annotations(
    path: str | Path,
    *,
    label: str,
    subject_grouping: str = PARTICIPANT_NUMBER_GROUPING,
) -> VideoAnnotations:
    """Parse authoritative human annotations encoded in a source filename.

    ``_left`` marks a left-handed performer. ``較佳`` marks a beginner who
    performs better than the ordinary beginner cohort. Performance annotations
    never change an expert/student dataset role. ``_mirror`` marks a
    horizontally flipped copy; it shares its source clip's participant and
    performance tier, and its handedness is the opposite of the source, which
    the ``_left`` marker already encodes on the mirrored filename itself.
    """

    stem = Path(path).stem
    handedness = Handedness.LEFT if _LEFT_MARKER.search(stem) else None
    performance_tier = (
        "better"
        if label == "beginners" and BETTER_PERFORMANCE_MARKER in stem
        else "unannotated"
    )
    return VideoAnnotations(
        subject_id=subject_id_from_stem(stem, subject_grouping=subject_grouping),
        handedness=handedness,
        performance_tier=performance_tier,
        is_mirror=is_mirrored_name(stem),
    )
