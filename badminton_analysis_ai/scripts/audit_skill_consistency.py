"""Audit the frozen dual-window skill guard from cached RF-DETR poses."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
from pathlib import Path
import sys
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from badminton_analysis.ml.expert_motion_preprocessing import (  # noqa: E402
    prepare_expert_motion_sample,
)
from badminton_analysis.ml.expert_reference_bank import (  # noqa: E402
    ExpertReferenceBank,
)
from badminton_analysis.models.types import Skill  # noqa: E402
from scripts.build_expert_skill_support import (  # noqa: E402
    _tracking_from_cache,
)


def _case(task: tuple[str, str, str, str]) -> dict[str, Any]:
    actual, requested, path_value, bank_value = task
    path = Path(path_value)
    tracking, handedness = _tracking_from_cache(path)
    poses = {}
    errors = {}
    for skill_name in ("serve", "smash"):
        try:
            sample, _, _ = prepare_expert_motion_sample(
                tracking,
                handedness,
                Skill.convert_to_enum(skill_name),
                "motion.npz",
                target_frames=64,
                phase_contract="eimd_v3",
            )
            poses[skill_name] = sample.pose
        except ValueError as exc:
            errors[skill_name] = str(exc)
    if requested not in poses:
        return {
            "skill": actual,
            "requested_skill": requested,
            "file": path.name,
            "status": "reject_requested_unavailable",
            "errors": errors,
        }
    alternative = "smash" if requested == "serve" else "serve"
    if alternative not in poses:
        return {
            "skill": actual,
            "requested_skill": requested,
            "file": path.name,
            "status": "accept_alternative_unavailable",
            "errors": errors,
        }
    bank = ExpertReferenceBank(bank_value)
    support = bank.temporal_skill_support(
        poses[requested], poses[alternative], requested_skill=requested
    )
    return {
        "skill": actual,
        "requested_skill": requested,
        "file": path.name,
        "status": "reject" if support.mismatch else "accept",
        "requested_distance": support.requested_distance,
        "alternative_distance": support.alternative_distance,
        "alternative_advantage": support.alternative_advantage,
        "rejection_margin": support.rejection_margin,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bank", type=Path, required=True)
    parser.add_argument("--serve-root", type=Path, required=True)
    parser.add_argument("--smash-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--include-wrong-labels",
        action="store_true",
        help="also submit every clip under the opposite requested skill",
    )
    args = parser.parse_args()
    tasks = [
        (skill, skill, str(path), str(args.bank.resolve()))
        for skill, root in (
            ("serve", args.serve_root),
            ("smash", args.smash_root),
        )
        for path in sorted(root.glob("*.npz"))
    ]
    if args.include_wrong_labels:
        tasks.extend(
            (
                skill,
                "smash" if skill == "serve" else "serve",
                str(path),
                str(args.bank.resolve()),
            )
            for skill, root in (
                ("serve", args.serve_root),
                ("smash", args.smash_root),
            )
            for path in sorted(root.glob("*.npz"))
        )
    rows = []
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(_case, task) for task in tasks]
        for index, future in enumerate(as_completed(futures), 1):
            rows.append(future.result())
            if index % 10 == 0 or index == len(futures):
                print(f"completed {index}/{len(futures)}", flush=True)
    rows.sort(
        key=lambda row: (row["skill"], row["requested_skill"], row["file"])
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n")
    correct = [row for row in rows if row["skill"] == row["requested_skill"]]
    wrong = [row for row in rows if row["skill"] != row["requested_skill"]]
    invalid_correct = [
        row
        for row in correct
        if row["status"] in {"reject", "reject_requested_unavailable"}
    ]
    invalid_wrong = [
        row
        for row in wrong
        if row["status"]
        not in {"reject", "reject_requested_unavailable"}
    ]
    for skill in ("serve", "smash"):
        selected = [row for row in correct if row["skill"] == skill]
        counts = {
            status: sum(row["status"] == status for row in selected)
            for status in sorted({row["status"] for row in selected})
        }
        print(f"{skill}: {counts}")
    if wrong:
        print(
            "wrong labels: "
            + json.dumps(
                {
                    status: sum(row["status"] == status for row in wrong)
                    for status in sorted({row["status"] for row in wrong})
                },
                sort_keys=True,
            )
        )
    if invalid_correct or invalid_wrong:
        raise SystemExit(
            "skill consistency audit failed:\n"
            + json.dumps(
                {
                    "correct_label_failures": invalid_correct,
                    "wrong_label_failures": invalid_wrong,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
    print(
        f"verified {len(correct)} correctly labelled motions"
        + (f" and {len(wrong)} wrong labels" if wrong else "")
    )


if __name__ == "__main__":
    main()
