"""Replay ported geometry on all saved local source poses, including learners.

This isolates the numeric port. It does NOT claim fresh-generation/CUDA parity:
historical heads and generated overlays are explicit frozen upstream inputs.
"""

import argparse
import json
from pathlib import Path
import hashlib

import numpy as np

from badminton_analysis.ml.smash_current_scoring import (
    CurrentSmashCalibration,
    score_source_evidence,
)
from badminton_analysis.ml.smash_current_placement import (
    first_frame_ankle_spine_map,
    first_frame_standing_offsets,
)


def verify(research, calibration_path):
    root = research / ".artifacts"
    read = lambda path: json.loads(path.read_text())
    calibration = CurrentSmashCalibration.load(calibration_path)
    latest = root / "smash-arm-balance-height-20260913"
    expected = read(latest / "frozen_predictions.json")["balance_height"]
    media = {
        r["file"]: r
        for r in read(root / "smash-source-media-audit-20260905.json")["records"]
    }
    intervals = {
        r["file"]: r["intervals"]
        for r in read(
            root
            / "smash-marked-interval-historical-pairs-20260912/intervals_and_coverage.json"
        )
    }
    baseline = {
        r["file"]: r
        for r in read(root / "smash-full-cohort-best-endpoints-20260912/results.json")
    }
    lineage = {
        r["file"]: r
        for r in read(root / "smash-historical-overlay-intervals-20260912/lineage.json")
    }
    records = []
    for row in expected:
        name = row["file"]
        sample, sidecar = Path(media[name]["sample"]), Path(lineage[name]["sidecar"])
        if (
            hashlib.sha256(sample.read_bytes()).hexdigest()
            != lineage[name]["sample_sha256"]
        ):
            raise ValueError("Source cache hash mismatch")
        if hashlib.sha256(sidecar.read_bytes()).hexdigest() != lineage[name]["sha256"]:
            raise ValueError("Correction cache hash mismatch")
        with (
            np.load(sample, allow_pickle=False) as s,
            np.load(sidecar, allow_pickle=False) as z,
        ):
            p, c, frames = (
                s["source_skeleton_2d"],
                s["source_confidence"],
                z["source_indices"],
            )
            q = z["corrected_pixels"].astype(float)
            matrix, translation, _ = first_frame_ankle_spine_map(
                q[0], p[frames[0]], c[frames[0]]
            )
            q = q @ matrix + translation
            q += first_frame_standing_offsets(q[0], p[frames[0]], c[frames[0]])
            actual = score_source_evidence(
                poses=p,
                confidence=c,
                corrected_pixels=q,
                generated_frames=frames,
                intervals=intervals[name],
                contact=int(s["source_phase_indices"][2]),
                endpoint_acceleration=int(s["analysis_window"][1]),
                historical_points=baseline[name]["before_points"],
                calibration=calibration,
                handedness=str(s["handedness"]),
                fps=float(s["fps"]),
                rescore_interval=False,
            )
        error = float(np.max(np.abs(np.array(actual["points"]) - row["points"])))
        records.append(
            dict(
                file=name,
                expert=row["expert"],
                maximum_error=error,
                actual=actual["points"],
                expected=row["points"],
            )
        )
    for row in read(latest / "desktop/results.json"):
        with np.load(row["npz"], allow_pickle=False) as z:
            # Desktop cached prior already includes the reviewed interval score.
            prior = row["previous_candidate_points"]
            actual = score_source_evidence(
                poses=z["source_skeleton_2d"],
                confidence=z["source_confidence"],
                corrected_pixels=z["corrected_pixels"],
                generated_frames=z["source_indices"],
                intervals=row["original_intervals"],
                contact=row["source_phases"][2],
                endpoint_acceleration=row["analysis_window"][1],
                historical_points=prior,
                calibration=calibration,
                rescore_interval=False,
            )
        error = float(
            np.max(np.abs(np.array(actual["points"]) - row["candidate_points"]))
        )
        records.append(
            dict(
                file=row["file"],
                desktop=True,
                maximum_error=error,
                actual=actual["points"],
                expected=row["candidate_points"],
            )
        )
    report = dict(
        clips=len(records),
        maximum_error=max(r["maximum_error"] for r in records),
        cached_inputs_only=True,
        failures=[r for r in records if r["maximum_error"] > 1e-6],
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if report["failures"] or len(records) != 100:
        raise SystemExit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--research-root", type=Path, required=True)
    parser.add_argument(
        "--calibration",
        type=Path,
        default=Path(
            "models/error_isolated_motion/smash/checkpoint_scorer_v1/calibration.json"
        ),
    )
    args = parser.parse_args()
    verify(args.research_root, args.calibration)
