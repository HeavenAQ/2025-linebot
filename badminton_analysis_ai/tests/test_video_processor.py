import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest

from badminton_analysis.services.video_analyzer import VideoAnalyzer
from badminton_analysis.services.video_processor import VideoProcessor
from badminton_analysis.services.pose_detector import PoseDetector
from badminton_analysis.models.types import Skill


def test_video_processor_accepts_shared_pose_detector() -> None:
    detector = MagicMock()
    processor = VideoProcessor("test.mp4", "output.mp4", "/tmp", detector)

    assert processor.video_path == "test.mp4"
    assert processor.out_filename == "output.mp4"
    assert processor.output_folder == "/tmp"
    assert processor.pose_detector is detector


def test_rtmw3d_camera_conversion_preserves_body_shape() -> None:
    keypoints_3d = np.zeros((17, 3), dtype=np.float64)
    keypoints_3d[:, 2] = np.linspace(-0.2, 0.2, 17)
    keypoints_2d = np.column_stack(
        (np.linspace(400, 680, 17), np.linspace(700, 1200, 17))
    )

    converted = PoseDetector._camera_coordinates(
        keypoints_3d, keypoints_2d, (1920, 1080, 3)
    )

    assert converted.shape == (17, 3)
    assert np.isfinite(converted).all()
    assert converted[:, 2].min() == pytest.approx(0.0)
    assert np.ptp(converted[:, 0]) > 0
    assert np.ptp(converted[:, 1]) > 0


def test_rtmw3d_detector_reuses_bbox_between_detection_frames() -> None:
    detector = PoseDetector(detection_frequency=7)
    runtime = MagicMock()
    runtime.det_model.return_value = np.array([[100, 200, 900, 1700, 0.99]])
    raw_3d = np.zeros((1, 133, 3), dtype=np.float32)
    raw_2d = np.zeros((1, 133, 2), dtype=np.float32)
    raw_2d[0, :, 0] = np.linspace(200, 800, 133)
    raw_2d[0, :, 1] = np.linspace(300, 1600, 133)
    runtime.pose_model.return_value = (
        raw_3d,
        np.ones((1, 133), dtype=np.float32),
        raw_3d.copy(),
        raw_2d,
    )
    detector.model = runtime
    frame = np.zeros((1920, 1080, 3), dtype=np.uint8)

    first = detector.get_pose(frame)
    second = detector.get_pose(frame)

    assert len(first) == len(second) == 1
    assert len(first[0]["keypoints"]) == 17
    assert len(detector._last_wholebody_predictions[0]["keypoints"]) == 133
    runtime.det_model.assert_called_once()
    assert runtime.pose_model.call_count == 2


def test_rtmw3d_rejects_unknown_gpu_execution_provider(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    detector = PoseDetector(backend="onnxruntime")
    detector.device = "cuda"
    detector.execution_provider = "metal"
    monkeypatch.setitem(
        sys.modules,
        "onnxruntime",
        SimpleNamespace(get_available_providers=lambda: ["CUDAExecutionProvider"]),
    )

    with pytest.raises(ValueError, match="must be tensorrt, cuda, or cpu"):
        detector._configure_execution_providers(SimpleNamespace())


def test_rtmw3d_configures_tensorrt_with_cuda_fallback(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    detector = PoseDetector(backend="onnxruntime")
    detector.device = "cuda"
    detector.execution_provider = "tensorrt"
    monkeypatch.setenv("POSE_TENSORRT_CACHE_DIR", str(tmp_path))
    monkeypatch.setenv("TENSORRT_ENGINE_HW_COMPATIBLE", "false")
    monkeypatch.setitem(
        sys.modules,
        "onnxruntime",
        SimpleNamespace(
            get_available_providers=lambda: [
                "TensorrtExecutionProvider",
                "CUDAExecutionProvider",
                "CPUExecutionProvider",
            ]
        ),
    )
    components = {
        name: SimpleNamespace(session=MagicMock()) for name in ("detector", "pose")
    }
    for component in components.values():
        component.session.get_providers.return_value = [
            "TensorrtExecutionProvider",
            "CUDAExecutionProvider",
            "CPUExecutionProvider",
        ]
    runtime = SimpleNamespace(
        det_model=components["detector"], pose_model=components["pose"]
    )

    detector._configure_execution_providers(runtime)

    for name, component in components.items():
        providers = component.session.set_providers.call_args.args[0]
        assert providers[0][0] == "TensorrtExecutionProvider"
        assert providers[1][0] == "CUDAExecutionProvider"
        assert providers[2] == "CPUExecutionProvider"
        options = providers[0][1]
        assert options["trt_fp16_enable"] is True
        assert options["trt_engine_cache_enable"] is True
        assert options["trt_engine_cache_path"] == str(tmp_path / name)
        assert options["trt_engine_hw_compatible"] is False
        assert "TopK" in options["trt_op_types_to_exclude"]
        assert detector.active_execution_providers[name][0] == (
            "TensorrtExecutionProvider"
        )


def test_moving_average_preserves_shape() -> None:
    positions = np.asarray(
        [(0, 0), (1, 1), (2, 2), (3, 3), (4, 4)], dtype=float
    )
    smoothed = VideoAnalyzer.moving_average(positions, window_size=3)

    assert smoothed.shape == positions.shape


def test_moving_average_uses_edge_padding() -> None:
    positions = np.asarray([(0, 0), (10, 10)], dtype=float)
    smoothed = VideoAnalyzer.moving_average(
        positions, window_size=3, pad_mode="edge"
    )

    np.testing.assert_allclose(smoothed[0], np.array([3.33, 3.33]), atol=0.1)


def test_velocity_uses_coordinate_distance() -> None:
    positions = np.asarray([(0, 0), (3, 4), (6, 8)], dtype=float)
    velocities = VideoAnalyzer.calc_velocity(positions, 1, 1)

    assert velocities[0] == pytest.approx(50.0, rel=1e-2)


def test_acceleration_uses_velocity_delta() -> None:
    accelerations = VideoAnalyzer.calc_acceleration(
        np.asarray([10, 20, 30], dtype=float), 1, 1
    )

    assert accelerations[0] == pytest.approx(100.0, rel=1e-2)


def test_compute_angles_handles_missing_landmarks() -> None:
    result = VideoAnalyzer().compute_angles({})

    assert all(angle == 0.0 for angle in result.values())


@pytest.mark.parametrize("skill", (Skill.SERVE, Skill.LIFT))
def test_analysis_window_keeps_follow_through_after_peak(
    monkeypatch: pytest.MonkeyPatch, skill: Skill
) -> None:
    monkeypatch.setattr(
        VideoAnalyzer,
        "find_acc_analysis_window",
        classmethod(lambda cls, positions: (10, 40, 70)),
    )
    hand_positions = np.zeros((100, 2), dtype=np.float64)
    hand_positions[40, 1] = 10.0
    elbow_positions = np.zeros((100, 2), dtype=np.float64)

    start, peak, end = VideoAnalyzer.find_analysis_window(
        skill=skill,
        hand_positions=list(hand_positions),
        elbow_positions=list(elbow_positions),
    )

    assert (start, peak, end) == (10, 40, 70)


@pytest.mark.parametrize("skill", (Skill.CLEAR, Skill.SMASH))
def test_overhead_window_never_ends_at_impact(skill: Skill) -> None:
    hand_positions = np.zeros((100, 2), dtype=np.float64)
    hand_positions[50, 1] = -10.0
    elbow_positions = np.zeros((100, 2), dtype=np.float64)

    _, peak, end = VideoAnalyzer.find_analysis_window(
        skill=skill,
        hand_positions=list(hand_positions),
        elbow_positions=list(elbow_positions),
    )

    assert end - peak >= 2
