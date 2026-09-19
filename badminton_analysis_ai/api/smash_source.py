"""Establish the frozen smash scorer's 30fps clock before pose extraction."""

from pathlib import Path
import subprocess

from api.renderer import source_fps


def normalize_smash_source(source: Path, output: Path) -> Path:
    fps = source_fps(source)
    if abs(fps - 30.0) < 1e-6:
        return source
    output.parent.mkdir(parents=True, exist_ok=True)
    # fps filter works on both container ffmpeg 4.x and current local builds.
    subprocess.run(
        [
            "ffmpeg",
            "-nostdin",
            "-v",
            "error",
            "-y",
            "-i",
            str(source),
            "-map",
            "0:v:0",
            "-vf",
            "fps=30",
            "-an",
            "-c:v",
            "libx264",
            "-crf",
            "18",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(output),
        ],
        check=True,
        capture_output=True,
        timeout=300,
    )
    if abs(source_fps(output) - 30.0) >= 1e-6:
        raise ValueError("Smash frame-rate normalization did not produce 30fps")
    return output
