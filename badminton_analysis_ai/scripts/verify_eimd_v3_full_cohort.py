"""Recompute all 203 localhost review records under one runtime contract.

The review JSON remains the media manifest and visible-score oracle.  This
verifier independently reruns every cached EIMD-v3 score, records all
per-checkpoint values and correction hashes, and verifies both input and
overlay media at the declared 30 fps/frame count/duration.  It never reruns
RF-DETR and never fits a threshold or scorer.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.verify_eimd_v3_review_parity import (  # noqa: E402
    _load_backend,
    _media_facts,
    _score_case,
)


EXPECTED_TABS = {
    "serve_beginner": 50,
    "serve_expert": 53,
    "smash_beginner": 50,
    "smash_expert": 50,
}


def _case(
    row: dict[str, Any], *, artifact_root: Path, validation_root: Path
) -> dict[str, Any]:
    tab = str(row["tab"])
    skill = str(row["skill"])
    stem = Path(str(row["file"])).stem
    generation = {
        "serve_beginner": (
            artifact_root
            / "error-isolated-motion-v3-scoring-data/serve/beginners"
            / f"{stem}.npz"
        ),
        "serve_expert": (
            artifact_root
            / "expert_score_stability/full_cohort_body_frame_v4"
            / "samples/serve/expert"
            / f"{stem}.npz"
        ),
        "smash_beginner": (
            artifact_root
            / "error-isolated-motion-v3-scoring-data/smash/beginners"
            / f"{stem}.npz"
        ),
        "smash_expert": (
            artifact_root
            / "expert_score_stability/full_cohort_body_frame_v4"
            / "samples/smash/expert"
            / f"{stem}.npz"
        ),
    }[tab]
    # A source video can be trimmed after its first cache was made.  The
    # review replaces such stale smash windows with the current v17 cache;
    # detect that from the visible analysis window instead of naming a clip.
    if skill == "smash":
        with np.load(generation, allow_pickle=False) as archive:
            cached_window = [int(value) for value in archive["analysis_window"]]
        fallback = (
            artifact_root
            / "smash-v6-rebuild-inputs/smash_uniform_v17_audit"
            / "samples/smash"
            / ("student" if tab.endswith("beginner") else "expert")
            / generation.name
        )
        if cached_window != [int(value) for value in row["window"]]:
            if not fallback.exists():
                raise FileNotFoundError(
                    f"review window changed but current cache is absent: {fallback}"
                )
            generation = fallback
    result: dict[str, Any] = {
        "skill": skill,
        "generation_sample": (
            "artifact:" + str(generation.relative_to(artifact_root))
        ),
    }
    if tab == "serve_beginner":
        result["scoring_sample"] = (
            "validation:audit_30fps_labelled/samples/serve/student/"
            f"{stem}.npz"
        )
    elif tab == "serve_expert":
        result["scoring_sample"] = (
            f"validation:rfdetr_expert_bank_v14/serve/{stem}.npz"
        )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--validation-root", type=Path, required=True)
    parser.add_argument("--review-root", type=Path, required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--candidates", type=int, default=8)
    parser.add_argument("--seed", type=int, default=19)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--skip-media", action="store_true")
    args = parser.parse_args()

    rendered = json.loads((args.review_root / "rendered.json").read_text())
    counts = Counter(str(row["tab"]) for row in rendered)
    if len(rendered) != 203 or dict(counts) != EXPECTED_TABS:
        raise RuntimeError(
            f"expected full 203-record review {EXPECTED_TABS}, got {dict(counts)}"
        )
    model_root = PROJECT_ROOT / "models/error_isolated_motion"
    backends = {
        skill: _load_backend(model_root, skill=skill, device=args.device)
        for skill in ("serve", "smash")
    }
    failures: list[str] = []
    records = []
    for index, row in enumerate(rendered, 1):
        label = f"{row['tab']}:{row['file']}"
        case = _case(
            row,
            artifact_root=args.artifact_root,
            validation_root=args.validation_root,
        )
        actual = _score_case(
            case,
            model_root=model_root,
            artifact_root=args.artifact_root,
            validation_root=args.validation_root,
            device=args.device,
            candidates=args.candidates,
            seed=args.seed,
            backend=backends[str(row["skill"])],
        )
        expected_score = float(row["score"])
        if not np.isclose(actual["score"], expected_score, atol=1e-8):
            failures.append(
                f"{label} score {actual['score']} != review {expected_score}"
            )
        media = {}
        if not args.skip_media:
            expected_frames = int(row["frames"])
            for key in ("input", "overlay"):
                rate, frames, duration = _media_facts(
                    args.review_root / str(row[key])
                )
                media[key] = {
                    "fps": rate,
                    "frames": frames,
                    "duration": duration,
                }
                if rate != "30/1":
                    failures.append(f"{label} {key} fps={rate}")
                if frames != expected_frames:
                    failures.append(
                        f"{label} {key} frames={frames} expected={expected_frames}"
                    )
                if abs(duration - expected_frames / 30.0) > 1 / 30 + 1e-6:
                    failures.append(
                        f"{label} {key} duration={duration:.6f}"
                    )
        records.append(
            {
                "tab": row["tab"],
                "file": row["file"],
                "window": row["window"],
                "frames": row["frames"],
                "review_score": expected_score,
                **actual,
                "media": media,
            }
        )
        if index % 10 == 0 or index == len(rendered):
            print(f"completed {index}/{len(rendered)}", flush=True)

    means = {
        tab: float(
            np.mean([record["score"] for record in records if record["tab"] == tab])
        )
        for tab in EXPECTED_TABS
    }
    payload = {
        "contract": {
            "phase_contract": "eimd_v3",
            "device": args.device,
            "candidates": args.candidates,
            "seed": args.seed,
            "fps": "30/1",
            "placement": "fixed_clip_level_ankle_spine_ankle_knee_hip",
            "score_and_overlay_share_generation": True,
        },
        "counts": dict(counts),
        "means": means,
        "records": records,
        "failures": failures,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    if failures:
        raise SystemExit(
            f"full cohort parity failed with {len(failures)} differences; "
            f"see {args.output}"
        )
    print(f"verified 203 score/media records; means={means}")


if __name__ == "__main__":
    main()
