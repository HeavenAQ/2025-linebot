"""Append a frozen expert-only serve/smash consistency model to the bank.

This is an offline fitting utility.  It needs expert caches that retain the
full RF-DETR skeleton/confidence sequence so both EIMD-v3 phase hypotheses can
be evaluated from one pose pass. Learner videos, clip filenames, cohort labels,
and human scores never enter the descriptor or boundary. Expert ``subject_id``
metadata is used only for the required leave-one-identity-out split.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from badminton_analysis.ml.expert_motion_preprocessing import (  # noqa: E402
    prepare_expert_motion_sample,
)
from badminton_analysis.ml.expert_reference_bank import (  # noqa: E402
    _SKILL_SUPPORT_CONTRACT,
    skill_temporal_descriptor,
    skill_temporal_distance,
)
from badminton_analysis.ml.handedness import interpolated_keypoint  # noqa: E402
from badminton_analysis.ml.video_annotations import (  # noqa: E402
    expert_subject_identity,
)
from badminton_analysis.models.types import (  # noqa: E402
    COCOKeypoints,
    Handedness,
    Skill,
    TrackingData,
)


def _tracking_from_cache(path: Path) -> tuple[TrackingData, Handedness]:
    with np.load(path, allow_pickle=False) as archive:
        required = {"source_skeleton_2d", "source_confidence", "handedness"}
        missing = required.difference(archive.files)
        if missing:
            raise ValueError(f"{path} lacks full expert pose: {sorted(missing)}")
        skeleton = archive["source_skeleton_2d"].astype(np.float32)
        confidence = archive["source_confidence"].astype(np.float32)
        handedness = Handedness.convert_to_enum(str(archive["handedness"].item()))
    if skeleton.ndim != 3 or skeleton.shape[1:] != (17, 2):
        raise ValueError(f"{path} has invalid source skeleton {skeleton.shape}")
    if confidence.shape != skeleton.shape[:2]:
        raise ValueError(f"{path} has invalid source confidence {confidence.shape}")
    sparse = [
        {
            COCOKeypoints(index): coordinates[index]
            for index in range(17)
            if confidence_value[index] > 0
        }
        for coordinates, confidence_value in zip(skeleton, confidence, strict=True)
    ]
    wrist = (
        COCOKeypoints.RIGHT_WRIST
        if handedness == Handedness.RIGHT
        else COCOKeypoints.LEFT_WRIST
    )
    elbow = (
        COCOKeypoints.RIGHT_ELBOW
        if handedness == Handedness.RIGHT
        else COCOKeypoints.LEFT_ELBOW
    )
    tracking: TrackingData = {
        "frames": [None] * len(skeleton),  # type: ignore[list-item]
        "body_landmarks_2d": sparse,
        "body_keypoints_2d": list(skeleton),
        "body_confidence_2d": list(confidence),
        "hand_positions": list(
            interpolated_keypoint(skeleton, confidence, wrist)
        ),
        "elbow_positions": list(
            interpolated_keypoint(skeleton, confidence, elbow)
        ),
    }
    return tracking, handedness


def _identity_from_cache(path: Path) -> str:
    with np.load(path, allow_pickle=False) as archive:
        if "subject_id" not in archive.files:
            raise ValueError(f"{path} lacks expert subject identity metadata")
        subject_id = str(archive["subject_id"].item())
    identity = expert_subject_identity(subject_id)
    if not identity:
        raise ValueError(f"{path} has empty expert subject identity")
    return identity


def _expert_rows(
    roots: dict[str, Path],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for skill_name, root in roots.items():
        skill = Skill.convert_to_enum(skill_name)
        paths = sorted(root.glob("*.npz"))
        if not paths:
            raise FileNotFoundError(f"no {skill_name} experts under {root}")
        for path in paths:
            tracking, handedness = _tracking_from_cache(path)
            hypotheses = {}
            for hypothesis_name in ("serve", "smash"):
                hypothesis, _, _ = prepare_expert_motion_sample(
                    tracking,
                    handedness,
                    Skill.convert_to_enum(hypothesis_name),
                    "expert-motion.npz",
                    target_frames=64,
                    phase_contract="eimd_v3",
                )
                hypotheses[hypothesis_name] = skill_temporal_descriptor(
                    hypothesis.pose
                )
            rows.append(
                {
                    "skill": skill_name,
                    "subject_id": _identity_from_cache(path),
                    "hypotheses": hypotheses,
                }
            )
    return rows


def _nearest(
    descriptor: np.ndarray,
    rows: list[dict[str, object]],
    *,
    skill: str,
    exclude_subject: str | None = None,
) -> float:
    distances = []
    for row in rows:
        if row["skill"] != skill or row["subject_id"] == exclude_subject:
            continue
        hypotheses = row["hypotheses"]
        assert isinstance(hypotheses, dict)
        distances.append(
            skill_temporal_distance(descriptor, hypotheses[skill])
        )
    if not distances:
        raise ValueError(f"no independent expert support for {skill}")
    return min(distances)


def build(
    *,
    reference_bank: Path,
    serve_experts: Path,
    smash_experts: Path,
    output: Path,
) -> tuple[int, float]:
    rows = _expert_rows({"serve": serve_experts, "smash": smash_experts})
    separations = []
    for row in rows:
        skill = str(row["skill"])
        alternative = "smash" if skill == "serve" else "serve"
        hypotheses = row["hypotheses"]
        assert isinstance(hypotheses, dict)
        own_distance = _nearest(
            hypotheses[skill],
            rows,
            skill=skill,
            exclude_subject=str(row["subject_id"]),
        )
        alternative_distance = _nearest(
            hypotheses[alternative], rows, skill=alternative
        )
        separations.append(alternative_distance - own_distance)
    minimum = float(np.min(separations))
    if not np.isfinite(minimum) or minimum <= 0.0:
        raise ValueError(
            "expert-only dual-window support does not separate serve and smash; "
            f"minimum={minimum}"
        )
    # Maximal expert-only safe boundary: every identity-held-out expert is
    # still rejected under the wrong label, while ambiguous inputs below the
    # smallest observed expert separation remain accepted.  The fixed 1e-6
    # subtraction is only numerical headroom; no learner result chooses it.
    numerical_headroom = 1e-6
    margin = minimum - numerical_headroom
    if margin <= 0.0:
        raise ValueError("expert separation is smaller than numerical headroom")

    with np.load(reference_bank, allow_pickle=False) as bank:
        payload = {key: bank[key] for key in bank.files}
    payload.update(
        {
            "skill_support_features": np.stack(
                [
                    row["hypotheses"][str(row["skill"])]  # type: ignore[index]
                    for row in rows
                ]
            ).astype(np.float32),
            "skill_support_skill": np.asarray(
                [row["skill"] for row in rows]
            ),
            "skill_support_subject_id": np.asarray(
                [row["subject_id"] for row in rows]
            ),
            "skill_rejection_margin": np.asarray(margin, dtype=np.float64),
            "skill_support_feature_contract": np.asarray(
                _SKILL_SUPPORT_CONTRACT
            ),
            "skill_support_fit_policy": np.asarray(
                "expert_only_leave_one_identity_out"
            ),
            "skill_support_expert_count": np.asarray(len(rows), dtype=np.int64),
            "skill_support_minimum_loo_separation": np.asarray(
                minimum, dtype=np.float64
            ),
            "skill_support_numerical_headroom": np.asarray(
                numerical_headroom, dtype=np.float64
            ),
            "skill_support_student_data_used": np.asarray(False),
        }
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **payload)
    return len(rows), margin


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-bank", type=Path, required=True)
    parser.add_argument("--serve-experts", type=Path, required=True)
    parser.add_argument("--smash-experts", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    count, margin = build(
        reference_bank=args.reference_bank,
        serve_experts=args.serve_experts,
        smash_experts=args.smash_experts,
        output=args.output,
    )
    print(f"wrote {args.output}: experts={count}, rejection_margin={margin:.9f}")


if __name__ == "__main__":
    main()
