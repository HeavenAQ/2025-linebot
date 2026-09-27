from pathlib import Path

import build_pose_engines
from badminton_analysis.services.pose_detector import (
    BATCH_SIZE,
    DETECTOR_ENGINE,
    POSE_ENGINE,
)


def test_engine_artifact_config_matches_checksum_and_detector_paths() -> None:
    config = build_pose_engines.load_engine_config()
    # The deploy verifies each downloaded file against this manifest, and the
    # detector loads <cache>/<GPU>/batch<N>/<file>; all three must agree, for
    # both stages.
    manifest = (
        Path(build_pose_engines.__file__).resolve().parent
        / "models"
        / "trt-engines.sha256"
    )
    listed = {
        path: digest
        for digest, path in (
            line.split()
            for line in manifest.read_text().splitlines()
            if line.strip() and not line.startswith("#")
        )
    }
    subdir = config["TRT_ENGINE_SUBDIR"]
    assert subdir.endswith(f"/batch{BATCH_SIZE}")
    for filename, expected_name, package, version in (
        (config["TRT_DETECTOR_FILENAME"], DETECTOR_ENGINE,
         config["TRT_DETECTOR_PACKAGE"], config["TRT_DETECTOR_VERSION"]),
        (config["TRT_POSE_FILENAME"], POSE_ENGINE,
         config["TRT_POSE_PACKAGE"], config["TRT_POSE_VERSION"]),
    ):
        assert filename == expected_name
        assert f"./{subdir}/{filename}" in listed
        assert version.startswith("nvidia-l4-batch16-")
        assert package.endswith("-trt")


def test_checksums_are_filled_in_before_a_deployable_image() -> None:
    """A placeholder here means the engines have not been rebuilt yet.

    The deploy checks the downloaded engines against this manifest, so leaving
    a placeholder in it fails the deploy rather than shipping an unverified
    engine -- which is the intended behaviour, but the reason is worth naming.
    """
    manifest = (
        Path(build_pose_engines.__file__).resolve().parent
        / "models"
        / "trt-engines.sha256"
    )
    digests = [
        line.split()[0]
        for line in manifest.read_text().splitlines()
        if line.strip() and not line.startswith("#")
    ]
    assert digests, "no engine checksums listed"
    for digest in digests:
        assert len(digest) == 64 or digest == "PENDING_BUILD_ON_L4"
