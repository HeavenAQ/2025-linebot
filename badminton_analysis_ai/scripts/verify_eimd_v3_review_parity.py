"""Verify runtime EIMD scoring/trajectory parity with the localhost review.

This is a runtime-only verifier: it consumes frozen expert/learner pose caches
and never imports the pruned training toolchain.  The checked-in oracle is MPS
specific because diffusion candidate ranking is device dependent.  Produce and
review a separate CUDA oracle on the deployment L4 before promotion.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from badminton_analysis.ml.error_isolated_motion import (  # noqa: E402
    correct_student_motion_error_isolated,
    load_error_isolated_bundle,
)
from badminton_analysis.ml.expert_motion_backend import (  # noqa: E402
    _dual_window_scoring_correction,
    _score_smash_correction,
    _serve_single_head_score,
)
from badminton_analysis.ml.expert_motion_preprocessing import (  # noqa: E402
    prepare_expert_motion_sample,
)
from badminton_analysis.ml.expert_reference_bank import (  # noqa: E402
    ExpertReferenceBank,
)
from badminton_analysis.ml.expert_phase_baseline import (  # noqa: E402
    align_expert_correction_to_ankle_spine_view,
    load_expert_phase_model,
    load_motion_sample,
    score_expert_correction,
)
from badminton_analysis.ml.smash_expert_scoring import (  # noqa: E402
    load_smash_distribution,
)
from badminton_analysis.ml.trajectory_distance import (  # noqa: E402
    load_smash_trajectory_scorer,
)
from badminton_analysis.models.types import Skill  # noqa: E402
from scripts.build_expert_skill_support import _tracking_from_cache  # noqa: E402


def _digest(value: np.ndarray) -> str:
    return hashlib.sha256(
        np.asarray(value, dtype=np.float32).tobytes()
    ).hexdigest()


def _resolve_sample(
    value: str, *, artifact_root: Path, validation_root: Path
) -> Path:
    prefix, relative = value.split(":", 1)
    roots = {"artifact": artifact_root, "validation": validation_root}
    if prefix not in roots:
        raise ValueError(f"unsupported sample root {prefix!r}")
    result = roots[prefix] / relative
    if not result.exists():
        raise FileNotFoundError(result)
    return result


def _score_case(
    case: dict[str, Any],
    *,
    model_root: Path,
    artifact_root: Path,
    validation_root: Path,
    device: str,
    candidates: int,
    seed: int,
    backend: dict[str, Any] | None = None,
) -> dict[str, Any]:
    skill = str(case["skill"])
    loaded = backend or _load_backend(model_root, skill=skill, device=device)
    bundle = loaded["bundle"]
    score_model = loaded["score_model"]
    sample = load_motion_sample(
        _resolve_sample(
            str(case["generation_sample"]),
            artifact_root=artifact_root,
            validation_root=validation_root,
        )
    )
    correction = correct_student_motion_error_isolated(
        bundle, sample, candidates=candidates, seed=seed
    )
    preparation = next(
        phase
        for phase in score_model.spec.phase_windows
        if phase.name == "preparation"
    )
    start, end = preparation.bounds(len(correction.aligned_student_pose))
    correction, _ = align_expert_correction_to_ankle_spine_view(
        correction, start=start, end=end, placement_mode="fixed"
    )
    base_score = score_expert_correction(score_model, correction)
    if skill == "serve":
        scoring = correction
        scoring_sample = case.get("scoring_sample")
        if scoring_sample:
            current = load_motion_sample(
                _resolve_sample(
                    str(scoring_sample),
                    artifact_root=artifact_root,
                    validation_root=validation_root,
                )
            )
            scoring = correct_student_motion_error_isolated(
                bundle, current, candidates=candidates, seed=seed
            )
            scoring, _ = align_expert_correction_to_ankle_spine_view(
                scoring, start=start, end=end, placement_mode="fixed"
            )
            scoring = _dual_window_scoring_correction(
                correction, scoring, start=start, end=end
            )
        score = _serve_single_head_score(
            score_expert_correction(score_model, scoring)
        )
    else:
        score = _score_smash_correction(
            base_score,
            sample,
            correction,
            distribution=loaded["distribution"],
            variant=loaded["variant"],
            trajectory_scorer=loaded["trajectory_scorer"],
            spec=score_model.spec,
        )
    return {
        "score": float(score["total_score"]),
        "criteria": {
            str(item["rule_reference"]): float(item["score"])
            for item in score["criteria"]
        },
        "aligned_corrected_sha256": _digest(
            correction.aligned_corrected_pose
        ),
        "corrected_sha256": _digest(correction.corrected_pose),
    }


def _load_backend(
    model_root: Path, *, skill: str, device: str
) -> dict[str, Any]:
    root = model_root / skill
    loaded: dict[str, Any] = {
        "bundle": load_error_isolated_bundle(
            root / "error_isolated_motion.pt", device=device
        ),
        "score_model": load_expert_phase_model(
            root / "expert_score_model.npz"
        ),
    }
    if skill == "smash":
        distribution, variant = load_smash_distribution(
            root / "expert_semantic_score_model.npz"
        )
        loaded.update(
            distribution=distribution,
            variant=variant,
            trajectory_scorer=load_smash_trajectory_scorer(
                root / "expert_trajectory_score_model.npz"
            ),
        )
    return loaded


def _media_facts(path: Path) -> tuple[str, int, float]:
    payload = json.loads(
        subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=avg_frame_rate,nb_frames,duration",
                "-of",
                "json",
                str(path),
            ],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    )["streams"][0]
    return (
        str(payload["avg_frame_rate"]),
        int(payload["nb_frames"]),
        float(payload["duration"]),
    )


def _verify_skill_consistency_cases(
    *,
    oracle_path: Path,
    bank_path: Path,
    artifact_root: Path,
    validation_root: Path,
) -> list[str]:
    """Verify label consistency without running or steering a generator."""
    oracle = json.loads(oracle_path.read_text())
    forbidden = set(oracle.get("forbidden_inputs", ()))
    if forbidden != {"filename", "cohort", "learner_rating"}:
        return ["skill-consistency oracle no longer forbids cohort-specific inputs"]
    bank = ExpertReferenceBank(bank_path)
    failures: list[str] = []
    for case in oracle["cases"]:
        cache_path = _resolve_sample(
            str(case["sample"]),
            artifact_root=artifact_root,
            validation_root=validation_root,
        )
        tracking, handedness = _tracking_from_cache(cache_path)
        hypotheses = {}
        for skill_name in ("serve", "smash"):
            try:
                sample, _, _ = prepare_expert_motion_sample(
                    tracking,
                    handedness,
                    Skill.convert_to_enum(skill_name),
                    cache_path.name,
                    target_frames=64,
                    phase_contract="eimd_v3",
                )
            except ValueError as exc:
                failures.append(
                    f"skill-consistency:{case['name']} could not form "
                    f"{skill_name} hypothesis: {exc}"
                )
                break
            hypotheses[skill_name] = sample.pose
        if len(hypotheses) != 2:
            continue
        requested = str(case["requested_skill"])
        alternative = "smash" if requested == "serve" else "serve"
        support = bank.temporal_skill_support(
            hypotheses[requested],
            hypotheses[alternative],
            requested_skill=requested,
        )
        expected_rejection = case["expected"] == "reject_before_generation"
        if support.mismatch != expected_rejection:
            failures.append(
                f"skill-consistency:{case['name']} mismatch={support.mismatch}, "
                f"expected={expected_rejection}, "
                f"advantage={support.alternative_advantage:.6f}, "
                f"margin={support.rejection_margin:.6f}"
            )
            continue
        status = "REJECT" if support.mismatch else "ACCEPT"
        print(
            f"PASS skill-consistency:{case['name']} {status} "
            f"advantage={support.alternative_advantage:.6f}",
            flush=True,
        )
    return failures


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--validation-root", type=Path, required=True)
    parser.add_argument("--review-root", type=Path, required=True)
    parser.add_argument(
        "--oracle",
        type=Path,
        default=PROJECT_ROOT / "tests/fixtures/eimd_v3_review_mps.json",
    )
    parser.add_argument(
        "--model-root",
        type=Path,
        default=PROJECT_ROOT / "models/error_isolated_motion",
    )
    parser.add_argument(
        "--expert-bank",
        type=Path,
        default=PROJECT_ROOT / "models/expert_reference_bank.npz",
    )
    parser.add_argument(
        "--skill-consistency-oracle",
        type=Path,
        default=(
            PROJECT_ROOT
            / "tests/fixtures/eimd_v3_skill_consistency_cases.json"
        ),
    )
    parser.add_argument("--device", required=True)
    parser.add_argument("--skip-media", action="store_true")
    args = parser.parse_args()

    oracle = json.loads(args.oracle.read_text())
    contract = oracle["contract"]
    if args.device != contract["device"]:
        raise RuntimeError(
            f"oracle device is {contract['device']}, got {args.device}; "
            "generate a reviewed device-specific oracle instead"
        )
    rendered = json.loads((args.review_root / "rendered.json").read_text())
    by_key = {
        (str(row["skill"]), Path(str(row["file"])).stem): row
        for row in rendered
    }
    failures = _verify_skill_consistency_cases(
        oracle_path=args.skill_consistency_oracle,
        bank_path=args.expert_bank,
        artifact_root=args.artifact_root,
        validation_root=args.validation_root,
    )
    for case in oracle["cases"]:
        label = f"{case['skill']}:{case['name']}"
        actual = _score_case(
            case,
            model_root=args.model_root,
            artifact_root=args.artifact_root,
            validation_root=args.validation_root,
            device=args.device,
            candidates=int(contract["candidates"]),
            seed=int(contract["seed"]),
        )
        if not np.isclose(actual["score"], case["score"], atol=1e-8):
            failures.append(
                f"{label} score {actual['score']} != {case['score']}"
            )
        if actual["criteria"].keys() != case["criteria"].keys():
            failures.append(f"{label} criterion IDs changed")
        for criterion, expected in case["criteria"].items():
            value = actual["criteria"].get(criterion, float("nan"))
            if not np.isclose(value, expected, atol=1e-8):
                failures.append(
                    f"{label}:{criterion} {value} != {expected}"
                )
        for key in ("aligned_corrected_sha256", "corrected_sha256"):
            if actual[key] != case[key]:
                failures.append(f"{label} {key} changed")
        row = by_key.get((str(case["skill"]), str(case["name"])))
        if row is None:
            failures.append(f"{label} absent from rendered.json")
            continue
        if not np.isclose(float(row["score"]), actual["score"], atol=1e-8):
            failures.append(f"{label} service/review score diverged")
        if args.skip_media:
            continue
        expected_frames = int(row["frames"])
        expected_duration = expected_frames / 30.0
        for media_key in ("input", "overlay"):
            rate, frames, duration = _media_facts(
                args.review_root / str(row[media_key])
            )
            if rate != contract["fps"]:
                failures.append(f"{label} {media_key} fps={rate}")
            if frames != expected_frames:
                failures.append(
                    f"{label} {media_key} frames={frames}, expected {expected_frames}"
                )
            if abs(duration - expected_duration) > 1.0 / 30.0 + 1e-6:
                failures.append(
                    f"{label} {media_key} duration={duration:.6f}, "
                    f"expected {expected_duration:.6f}"
                )
        print(f"PASS {label} {actual['score']:.6f}", flush=True)
    if failures:
        raise SystemExit("parity failed:\n- " + "\n- ".join(failures))
    print(f"verified {len(oracle['cases'])} EIMD-v3 parity cases")


if __name__ == "__main__":
    main()
