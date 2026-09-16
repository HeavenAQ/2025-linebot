from pathlib import Path

import build_rfdetr_engine
from badminton_analysis.services.pose_detector import BATCH_SIZE


def test_engine_artifact_config_matches_checksum_and_detector_path() -> None:
    config = build_rfdetr_engine.load_engine_config()
    # The deploy verifies the downloaded file against this manifest, and the
    # detector loads <cache>/<GPU>/batch<N>/<file>; all three must agree.
    manifest = (
        Path(build_rfdetr_engine.__file__).resolve().parent
        / "models"
        / "trt-engines.sha256"
    )
    digest, path = manifest.read_text().split()
    assert len(digest) == 64
    assert path == f"./{config['TRT_ENGINE_SUBDIR']}/{config['TRT_ENGINE_FILENAME']}"
    assert config["TRT_ENGINE_SUBDIR"].endswith(f"/batch{BATCH_SIZE}")
    assert config["TRT_ENGINE_FILENAME"] == "rfdetr-keypoint-preview.trt"
    assert config["TRT_ENGINE_VERSION"].startswith("nvidia-l4-batch16-")
