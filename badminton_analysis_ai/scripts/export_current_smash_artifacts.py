"""Package frozen local smash artifacts; never fit from learners or ratings.

Run with the research repository as --research-root and a fresh --output.
The deployed scorer needs only the resulting directory, not the research tree.
"""

import argparse
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def export(research, output):
    if output.exists():
        raise ValueError("Fresh artifact output required")
    root = research / ".artifacts"
    read = lambda path: json.loads(path.read_text())
    graph = root / "smash-two-state-heads-nonwang-global-20260906"
    matched = root / "smash-published-scorer-matched-refit-20260905"
    rotation = root / "smash-ankle-shoulder-rotation-20260913/calibration.json"
    prep = root / "smash-preparation-height-20260913/calibration.json"
    balance = root / "smash-arm-balance-height-20260913/calibration.json"
    span = root / "smash-endpoint-shoulder-span-20260913/calibration.json"
    endpoint = (
        root
        / "smash-historical-overlay-intervals-20260912/calibration_before_learners.json"
    )
    legacy_rotation = (
        root / "smash-directional-torso-progress-20260913/calibration.json"
    )
    annotations = (
        root
        / "smash-marked-interval-historical-pairs-20260912/annotations.snapshot.json"
    )
    reference = (
        root
        / "smash-torso-anchored-training-inputs-20260908/samples/smash/expert/張宸愷1.npz"
    )
    graph_report, matched_report = (
        read(graph / "report.json"),
        read(matched / "report.json"),
    )
    semantic = matched / "expert_semantic_score_model.npz"
    if digest(graph / "metric_graph.pt") != graph_report["model_sha256"]:
        raise ValueError("Graph checkpoint differs from reviewed run")
    if digest(semantic) != matched_report["model_sha256"][semantic.name]:
        raise ValueError("Semantic model differs from reviewed run")
    fit = read(rotation)["full"]
    if len(fit["files"]) != 42 or any("王鶴勳" in name for name in fit["files"]):
        raise ValueError("Wrong rotation calibration cohort")
    output.mkdir(parents=True)
    for path in (graph / "metric_graph.pt", semantic):
        shutil.copy2(path, output / path.name)
    with np.load(graph / "evidence.npz", allow_pickle=False) as z:
        # No learner poses, names, probabilities, or ratings are shipped.
        support = z["support"].copy()
    with np.load(reference, allow_pickle=False) as z:
        np.savez_compressed(
            output / "checkpoint_reference.npz",
            poses=z["source_skeleton_2d"],
            confidence=z["source_confidence"],
            fps=z["fps"],
        )
    dependencies = [
        rotation,
        prep,
        balance,
        span,
        endpoint,
        legacy_rotation,
        annotations,
        reference,
        graph / "metric_graph.pt",
        graph / "evidence.npz",
        graph / "contract.json",
        semantic,
    ]
    # Unsupported graph heads carry NaN support; store null, not invalid JSON.
    data = dict(
        version="smash_local_20260913",
        maxima=[5, 20, 5, 20, 30, 20],
        calibration=dict(
            rotation_lower=fit["lower"],
            rotation_full=fit["full"],
            preparation_height=read(prep)["full_credit_height"],
            balance_gap=read(balance)["expert_tolerance"],
            shoulder_span_reference=read(span)["q10_squared/full"]["reference"],
            legacy_rotation_reference=read(legacy_rotation)["directional_min/full"][
                "reference"
            ],
            endpoint_map=read(endpoint)["maps"]["shape_dtw"][5],
        ),
        graph_support=[float(v) if np.isfinite(v) else None for v in support],
        graph_rules=read(graph / "contract.json")["rules"],
        intervals=read(annotations)["videos"]["張宸愷1.mp4"],
        source_sha256={
            str(path.relative_to(research)): digest(path) for path in dependencies
        },
        trajectory_sha256=matched_report["model_sha256"][
            "expert_trajectory_score_model.npz"
        ],
        generation=dict(
            device="mps",
            candidates=8,
            seed=19,
            checkpoint_sha256="956b567e407d88eff23ebc936f264e03d16543a550c08bf879268fbb85353977",
        ),
        limitations=[
            "Cached local evidence is not live CUDA parity.",
            "Image-right-forward reference convention; do not select a view by maximizing grades.",
            "Calibration support is full-fit, not unseen-person validation.",
        ],
    )
    (output / "calibration.json").write_text(
        json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    )
    manifest = {p.name: digest(p) for p in sorted(output.iterdir()) if p.is_file()}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--research-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    export(args.research_root.resolve(), args.output)
