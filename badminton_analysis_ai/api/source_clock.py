"""Put a source video on the frame rate a skill's scorer was built on, before pose extraction."""

from pathlib import Path
import subprocess

from api.renderer import source_fps


def normalize_source_fps(source: Path, output: Path, fps: float) -> Path:
    if abs(source_fps(source) - fps) < 1e-6:
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
            f"fps={fps:g}",
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
    if abs(source_fps(output) - fps) >= 1e-6:
        raise ValueError(f"frame-rate normalization did not produce {fps:g}fps")
    return output
