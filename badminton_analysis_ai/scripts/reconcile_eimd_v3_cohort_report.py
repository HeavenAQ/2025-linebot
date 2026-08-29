"""Rebuild cohort ICC from the same 203 scores shown by the review site.

Human ratings are read-only validation labels.  They do not select a scorer,
threshold, diffusion candidate, or skill-gate margin.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


def _icc(values: list[list[float]]) -> tuple[float, float]:
    matrix = np.asarray(values, dtype=np.float64)
    n, k = matrix.shape
    grand = float(matrix.mean())
    target_means = matrix.mean(axis=1)
    rater_means = matrix.mean(axis=0)
    ms_targets = k * float(np.square(target_means - grand).sum()) / (n - 1)
    ms_raters = n * float(np.square(rater_means - grand).sum()) / (k - 1)
    residual = matrix - target_means[:, None] - rater_means[None] + grand
    ms_error = float(np.square(residual).sum()) / ((n - 1) * (k - 1))
    absolute = (ms_targets - ms_error) / (
        ms_targets
        + (k - 1) * ms_error
        + k * (ms_raters - ms_error) / n
    )
    consistency = (ms_targets - ms_error) / (
        ms_targets + (k - 1) * ms_error
    )
    return float(absolute), float(consistency)


def _serve_key(stem: str) -> str:
    return stem.removesuffix("_left").removesuffix("-較佳")


def _summary(
    learner: list[list[float]],
    expert_scores: list[float],
    *,
    maximum: float,
) -> dict[str, Any]:
    expert = [
        [score * maximum / 100.0, maximum, maximum]
        for score in expert_scores
    ]
    learner_absolute, learner_consistency = _icc(learner)
    pooled_absolute, pooled_consistency = _icc(learner + expert)
    return {
        "learner_n": len(learner),
        "learner_icc_2_1": learner_absolute,
        "learner_icc_3_1": learner_consistency,
        "expert_n": len(expert),
        "expert_icc_2_1": None,
        "expert_icc_3_1": None,
        "expert_icc_note": (
            "undefined because both human raters are constant at maximum "
            "for every expert clip"
        ),
        "expert_system_mean_100": float(np.mean(expert_scores)),
        "expert_system_min_100": float(np.min(expert_scores)),
        "expert_system_max_100": float(np.max(expert_scores)),
        "learner_system_mean_100": float(
            np.mean(np.asarray(learner)[:, 0]) * 100.0 / maximum
        ),
        "pooled_n": len(learner) + len(expert),
        "pooled_icc_2_1": pooled_absolute,
        "pooled_icc_3_1": pooled_consistency,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--full-parity", type=Path, required=True)
    parser.add_argument("--serve-ratings", type=Path, required=True)
    parser.add_argument("--smash-ratings", type=Path, required=True)
    parser.add_argument("--previous-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    parity = json.loads(args.full_parity.read_text())
    if parity.get("failures"):
        raise ValueError("cannot reconcile a failed full-cohort parity run")
    if sum(parity["counts"].values()) != 203:
        raise ValueError("authoritative review must contain 203 clips")
    records = parity["records"]
    scores = {
        (str(row["tab"]), Path(str(row["file"])).stem): float(row["score"])
        for row in records
    }

    serve_source = json.loads(args.serve_ratings.read_text())
    serve_rows = []
    for row in serve_source["rows"]:
        matches = [
            score
            for (tab, stem), score in scores.items()
            if tab == "serve_beginner" and _serve_key(stem) == row["key"]
        ]
        if len(matches) != 1:
            raise ValueError(f"serve rating mapping failed for {row['key']}")
        serve_rows.append(
            [matches[0] * 6.0 / 100.0, row["human_1"], row["human_2"]]
        )

    smash_source = json.loads(args.smash_ratings.read_text())
    smash_rows = []
    for row in smash_source["rows"]:
        stem = Path(str(row["archive"])).stem
        smash_rows.append(
            [
                scores[("smash_beginner", stem)] * 7.0 / 100.0,
                row["expert_1"],
                row["expert_2"],
            ]
        )

    result = {
        "contract": parity["contract"],
        "human_ratings_usage": "validation_only_no_parameter_selection",
        "serve": _summary(
            serve_rows,
            [
                float(row["score"])
                for row in records
                if row["tab"] == "serve_expert"
            ],
            maximum=6.0,
        ),
        "smash": _summary(
            smash_rows,
            [
                float(row["score"])
                for row in records
                if row["tab"] == "smash_expert"
            ],
            maximum=7.0,
        ),
    }
    previous = json.loads(args.previous_report.read_text())
    result["reconciliation"] = {
        skill: {
            "previous_learner_mean": previous[skill][
                "learner_system_mean_100"
            ]
            if "learner_system_mean_100" in previous[skill]
            else (
                float(serve_source["summary"]["old_w0"]["mean"])
                if skill == "serve"
                else float(smash_source["system_mean_100"])
            ),
            "authoritative_learner_mean": result[skill][
                "learner_system_mean_100"
            ],
        }
        for skill in ("serve", "smash")
    }
    result["reconciliation"]["serve"]["cause"] = (
        "previous benchmark hard-coded rigid-fit slice 0:16; the frozen rubric "
        "preparation.bounds(64) is 0:24 and is shared by runtime/review"
    )
    result["reconciliation"]["smash"]["cause"] = (
        "previous report used the stale pre-trim EG40 cache; the review and "
        "runtime select the current v17 cache when the old window exceeds "
        "the source video"
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
