"""Pack the diffusion models' own expert clips into a reference bank.

The comparison video shown next to a learner is chosen by matching their
corrected skeleton against these experts, so the bank has to be exactly the
clips the models were trained on — not the older catalogue, which came from a
different pipeline and a different pose estimator.

The skeletons are small enough to ship inside the image, which keeps expert
selection a local array operation instead of a Firestore round trip on a
service that scales to zero. Only the videos live in Cloud Storage, because
only they are streamed to the browser.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

# The datasets the current serve and smash checkpoints were trained from.
SOURCES = {
    "serve": (
        ".artifacts/shoulder-angle-aligned-v1/serve/experts",
        "scoring_videos/發球/專家發球",
    ),
    "smash": (
        ".artifacts/smash-2d-ending-range-v4/experts",
        "scoring_videos/殺球/專家殺球",
    ),
}

VIDEO_PREFIX = "experts/v3"


def _clip_metadata(video: Path) -> tuple[float, int, int]:
    """The clip's duration and frame size, read from the file itself.

    Only the videos live in Cloud Storage, and the service hands the browser a
    signed URL without ever opening them, so anything the player needs before it
    has loaded the clip has to be recorded here. Without a duration the player
    cannot bound the expert's motion window, and without the frame size it
    guesses an aspect ratio and reflows once the clip arrives.
    """
    capture = cv2.VideoCapture(str(video))
    try:
        if not capture.isOpened():
            raise RuntimeError(f"could not open {video}")
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        frames = float(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    finally:
        capture.release()
    if fps <= 0 or frames <= 0 or width <= 0 or height <= 0:
        raise RuntimeError(f"{video} reports no usable metadata")
    return frames / fps, width, height


def build(source_root: Path, output: Path) -> dict[str, int]:
    skeletons: list[np.ndarray] = []
    meta: dict[str, list] = {
        "skill": [], "handedness": [], "video_object_path": [], "subject_id": [],
        "fps": [], "analysis_window": [], "source_phase_indices": [],
        "duration_seconds": [], "width": [], "height": [],
    }
    counts: dict[str, int] = {}

    for skill, (npz_dir, video_dir) in SOURCES.items():
        paths = sorted((source_root / npz_dir).glob("*.npz"))
        if not paths:
            raise FileNotFoundError(f"no expert vectors under {source_root / npz_dir}")
        for path in paths:
            with np.load(path, allow_pickle=False) as sample:
                if str(sample["skill"].item()) != skill:
                    raise ValueError(f"{path} is not a {skill} clip")
                if bool(np.asarray(sample.get("is_mirror", False)).item()):
                    raise ValueError(f"synthetic mirror is not a real expert: {path}")
                skeleton = sample["skeleton"].astype(np.float32)
                if skeleton.shape != (64, 17, 2):
                    raise ValueError(f"unexpected skeleton shape in {path}: {skeleton.shape}")
                video_name = str(sample["video_name"].item())
                video = source_root / video_dir / video_name
                if not video.exists():
                    matches = list((source_root / video_dir).glob(Path(video_name).stem + ".*"))
                    if not matches:
                        raise FileNotFoundError(f"no video for {path.name}: {video_name}")
                    video = matches[0]
                skeletons.append(skeleton)
                meta["skill"].append(skill)
                meta["handedness"].append(str(sample["handedness"].item()))
                meta["video_object_path"].append(f"{VIDEO_PREFIX}/{skill}/videos/{video.name}")
                meta["subject_id"].append(str(sample["subject_id"].item()))
                meta["fps"].append(float(sample["fps"].item()))
                meta["analysis_window"].append(sample["analysis_window"].astype(np.int64))
                meta["source_phase_indices"].append(sample["source_phase_indices"].astype(np.int64))
                duration, width, height = _clip_metadata(video)
                meta["duration_seconds"].append(duration)
                meta["width"].append(width)
                meta["height"].append(height)
        counts[skill] = len(paths)

    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        skeletons=np.stack(skeletons),
        skill=np.asarray(meta["skill"]),
        handedness=np.asarray(meta["handedness"]),
        video_object_path=np.asarray(meta["video_object_path"]),
        subject_id=np.asarray(meta["subject_id"]),
        fps=np.asarray(meta["fps"], dtype=np.float32),
        analysis_window=np.stack(meta["analysis_window"]),
        source_phase_indices=np.stack(meta["source_phase_indices"]),
        duration_seconds=np.asarray(meta["duration_seconds"], dtype=np.float32),
        width=np.asarray(meta["width"], dtype=np.int64),
        height=np.asarray(meta["height"], dtype=np.int64),
    )
    return counts


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--output", type=Path, default=Path("models/expert_reference_bank.npz"))
    args = parser.parse_args()
    counts = build(args.source_root, args.output)
    size = args.output.stat().st_size / 1024
    print(f"wrote {args.output} ({size:.0f} KB): " + ", ".join(f"{k}={v}" for k, v in counts.items()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
