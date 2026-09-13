"""Replay all eight reviewed new-video caches through the deployed runtime.

Recomputes historical heads, interval transfer, projection, EMA, and all new
caps from saved native generations; no research imports. Diffusion/pose parity
must be checked separately on the target device before promotion.
"""

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from badminton_analysis.ml.expert_motion_backend import _score_smash_correction
from badminton_analysis.ml.skill_specs import get_skill_spec
from badminton_analysis.ml.smash_current_runtime import CurrentSmashScorer
from badminton_analysis.ml.trajectory_distance import load_smash_trajectory_scorer


def verify(research, models, device, regenerate=False, cache_directory=None):
    root = models / "smash"
    scorer = CurrentSmashScorer(
        root / "checkpoint_scorer_v1",
        generator_path=root / "error_isolated_motion.pt",
        trajectory_path=root / "expert_trajectory_score_model.npz",
        candidates=8,
        seed=19,
        device=device,
    )
    trajectory = load_smash_trajectory_scorer(
        root / "expert_trajectory_score_model.npz"
    )
    if regenerate:
        from badminton_analysis.ml.expert_motion_backend import (
            ExpertMotionGeneratorBackend,
        )
        from badminton_analysis.ml.handedness import interpolated_keypoint
        from badminton_analysis.models.types import Skill, Handedness

        backend = ExpertMotionGeneratorBackend(
            models,
            Skill.SMASH,
            device=device,
            current_smash=True,
            candidates=8,
            seed=19,
            align_ankle_spine_view=True,
        )
    rows = json.loads(
        (
            (cache_directory / "results.json")
            if cache_directory
            else research
            / ".artifacts/smash-arm-balance-height-20260913/desktop/results.json"
        ).read_text()
    )
    reports = []
    for row in rows:
        cache = (
            cache_directory / Path(row["npz"]).name
            if cache_directory
            else Path(row["npz"])
        )
        with np.load(cache, allow_pickle=False) as z:
            sample = SimpleNamespace(
                pose=z["skeleton"],
                confidence=z["confidence"],
                phase_indices=z["phase_indices"],
                handedness="right",
            )
            correction = SimpleNamespace(aligned_corrected_pose=z["native_corrected"])
            generation_error = None
            if regenerate:
                full, conf = z["source_skeleton_2d"], z["source_confidence"]
                tracking = dict(
                    body_landmarks_2d=[{} for _ in full],
                    body_keypoints_2d=full,
                    body_confidence_2d=conf,
                    hand_positions=list(interpolated_keypoint(full, conf, 10)),
                    elbow_positions=list(interpolated_keypoint(full, conf, 8)),
                )
                prepared = backend.prepare(tracking, Handedness.RIGHT, row["file"])
                sample, selected_window, indices = prepared
                np.testing.assert_array_equal(
                    indices[sample.phase_indices], z["source_phase_indices"]
                )
                assert tuple(selected_window) == tuple(row["analysis_window"])
                generated = backend.infer(
                    tracking, Handedness.RIGHT, row["file"], prepared=prepared, fps=30.0
                )
                correction = generated.correction
                generation_error = float(
                    np.max(
                        np.abs(
                            correction.aligned_corrected_pose - z["native_corrected"]
                        )
                    )
                )
            score, pixels, window = scorer.score(
                sample=sample,
                correction=correction,
                source_phases=z["source_phase_indices"],
                native_phases=z["native_phases"],
                window=tuple(row["analysis_window"]),
                poses=z["source_skeleton_2d"],
                confidence=z["source_confidence"],
                handedness="right",
                fps=30.0,
                spec=get_skill_spec("smash"),
                trajectory_scorer=trajectory,
                score_baseline=_score_smash_correction,
            )
            score_error = float(
                np.max(
                    np.abs(
                        np.array([c["score"] for c in score["criteria"]])
                        - row["candidate_points"]
                    )
                )
            )
            if regenerate:
                np.testing.assert_allclose(
                    generated.grade["total_grade"],
                    score["total_score"],
                    atol=1e-8,
                    rtol=0,
                )
                np.testing.assert_array_equal(generated.corrected_pixels, pixels)
                assert generated.window == window
            overlay_error = float(
                np.max(np.abs(pixels[z["source_indices"]] - z["corrected_pixels"]))
            )
            assert window == (
                0,
                row["analysis_window"][1],
                len(z["source_skeleton_2d"]) - 1,
            )
            assert score["checkpoint_source_frames"]["wrist_flick"] == int(
                z["source_phase_indices"][2]
            )
            reports.append(
                dict(
                    file=row["file"],
                    score_error=score_error,
                    overlay_error_px=overlay_error,
                    native_generation_error=generation_error,
                )
            )
            print(row["file"], score_error, overlay_error, flush=True)
    print(
        json.dumps(
            dict(
                cached_generation=not regenerate,
                cached_poses=True,
                device=device,
                records=reports,
            ),
            ensure_ascii=False,
            indent=2,
        )
    )
    if len(reports) != 8 or any(
        r["score_error"] > 1e-4 or r["overlay_error_px"] > 1e-4 for r in reports
    ):
        raise SystemExit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--research-root", type=Path, required=True)
    parser.add_argument(
        "--model-root", type=Path, default=Path("models/error_isolated_motion")
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--regenerate", action="store_true")
    parser.add_argument("--cache-directory", type=Path)
    args = parser.parse_args()
    verify(
        args.research_root,
        args.model_root,
        args.device,
        args.regenerate,
        args.cache_directory,
    )
