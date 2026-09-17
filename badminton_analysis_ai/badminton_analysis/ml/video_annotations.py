from __future__ import annotations

import re
from pathlib import Path

SYNTHETIC_MIRROR_PREFIX = "synthetic_left_from_"

# Delimiter-aware markers. The trailing delimiter is a lookahead so that
# removing one marker cannot swallow the delimiter that introduces the next.
_LEFT_MARKER = re.compile(r"(?:^|[-_\s])left(?=$|[-_\s])", re.IGNORECASE)
_MIRROR_MARKER = re.compile(r"(?:^|[-_\s])mirror(?=$|[-_\s])", re.IGNORECASE)


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
