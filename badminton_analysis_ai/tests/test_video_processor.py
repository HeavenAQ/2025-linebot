from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from badminton_analysis.models.types import COCOKeypoints, Handedness, Skill
from badminton_analysis.services.pose_detector import BATCH_SIZE
from badminton_analysis.services.video_analyzer import VideoAnalyzer
from badminton_analysis.services.video_processor import VideoProcessor


def test_video_processor_accepts_shared_pose_detector() -> None:
    detector = MagicMock()
    processor = VideoProcessor("test.mp4", "output.mp4", "/tmp", detector)

    assert processor.video_path == "test.mp4"
    assert processor.out_filename == "output.mp4"
    assert processor.output_folder == "/tmp"
    assert processor.pose_detector is detector


def _fake_pose_prediction(x: float) -> list[dict]:
    keypoints = np.zeros((17, 2), dtype=np.float64)
    keypoints[int(COCOKeypoints.RIGHT_WRIST)] = (x, 1.0)
    keypoints[int(COCOKeypoints.RIGHT_ELBOW)] = (x, 2.0)
    scores = np.full(17, 0.9, dtype=np.float64)
    wholebody_keypoints = np.zeros((133, 2), dtype=np.float64)
    wholebody_keypoints[:17] = keypoints
    wholebody_scores = np.zeros(133, dtype=np.float64)
    wholebody_scores[:17] = scores
    return [
        {
            "bbox": [0.0, 0.0, 10.0, 10.0],
            "keypoints": keypoints,
            "keypoint_scores": scores,
            "wholebody_keypoints": wholebody_keypoints,
            "wholebody_scores": wholebody_scores,
        }
    ]


class _FakeCapture:
    """Minimal cv2.VideoCapture stand-in yielding a fixed number of frames."""

    def __init__(self, frame_count: int) -> None:
        self._remaining = frame_count

    def isOpened(self) -> bool:
        return self._remaining > 0

    def read(self):
        if self._remaining <= 0:
            return False, None
        self._remaining -= 1
        return True, np.zeros((4, 4, 3), dtype=np.uint8)

    def release(self) -> None:
        pass


def test_process_frames_batched_chunks_and_records_results() -> None:
    frame_count = BATCH_SIZE + 3  # forces one full batch plus one partial one
    call_sizes: list[int] = []
    detector = MagicMock()
    detector.min_detection_confidence = 0.5

    def _fake_batch(frames):
        call_sizes.append(len(frames))  # record now: the caller clears frames after
        return [_fake_pose_prediction(1.0) for _ in frames]

    detector.get_poses_batch = MagicMock(side_effect=_fake_batch)
    detector.get_2d_landmarks = MagicMock(
        side_effect=lambda results: {
            COCOKeypoints(i): results[0]["keypoints"][i] for i in range(17)
        }
    )
    detector.get_wholebody_2d_landmarks = MagicMock(return_value={})
    detector.get_wholebody_2d_keypoints = MagicMock(
        return_value=(np.zeros((133, 2)), np.zeros(133))
    )
    processor = VideoProcessor("test.mp4", "out.mp4", "/tmp", detector)

    with patch(
        "badminton_analysis.services.video_processor.cv2.VideoCapture",
        return_value=_FakeCapture(frame_count),
    ):
        tracking = processor.process_frames_batched(Handedness.RIGHT)

    assert len(tracking["frames"]) == frame_count
    assert len(tracking["original_landmarks"]) == frame_count
    assert tracking["source_frame_indices"] == list(range(frame_count))
    # Batched in chunks of BATCH_SIZE: one full call, one partial call.
    assert call_sizes == [BATCH_SIZE, 3]


def test_process_frames_preserves_continuous_rfdetr_body_confidence() -> None:
    detector = MagicMock()
    detector.min_detection_confidence = 0.15
    detector.get_poses_batch = MagicMock(return_value=[_fake_pose_prediction(1.0)])
    detector.get_2d_landmarks = MagicMock(
        return_value={
            COCOKeypoints.RIGHT_WRIST: np.asarray((1.0, 1.0)),
            COCOKeypoints.RIGHT_ELBOW: np.asarray((1.0, 2.0)),
        }
    )
    detector.get_wholebody_2d_landmarks = MagicMock(return_value={})
    scores = np.zeros(133, dtype=np.float64)
    scores[int(COCOKeypoints.RIGHT_WRIST)] = 0.42
    scores[int(COCOKeypoints.RIGHT_ELBOW)] = 0.73
    scores[int(COCOKeypoints.LEFT_ELBOW)] = 0.10
    scores[int(COCOKeypoints.LEFT_WRIST)] = 0.10
    detector.elbow_detection_confidence = 0.05
    detector.get_wholebody_2d_keypoints = MagicMock(
        return_value=(np.zeros((133, 2), dtype=np.float64), scores)
    )
    processor = VideoProcessor("test.mp4", "out.mp4", "/tmp", detector)

    with patch(
        "badminton_analysis.services.video_processor.cv2.VideoCapture",
        return_value=_FakeCapture(1),
    ):
        tracking = processor.process_frames_batched(Handedness.RIGHT)

    confidence = tracking["body_confidence_2d"][0]
    assert confidence[int(COCOKeypoints.RIGHT_WRIST)] == pytest.approx(0.42)
    assert confidence[int(COCOKeypoints.RIGHT_ELBOW)] == pytest.approx(0.73)
    # Elbows use a lower acceptance threshold than other body joints.
    assert confidence[int(COCOKeypoints.LEFT_ELBOW)] == pytest.approx(0.10)
    assert confidence[int(COCOKeypoints.LEFT_WRIST)] == 0.0


def test_process_frames_batched_skips_frames_missing_expected_hand() -> None:
    detector = MagicMock()
    detector.min_detection_confidence = 0.5
    detector.get_poses_batch = MagicMock(return_value=[[]])
    detector.get_2d_landmarks = MagicMock(return_value=None)
    detector.get_wholebody_2d_landmarks = MagicMock(return_value=None)
    detector.get_wholebody_2d_keypoints = MagicMock(return_value=None)
    processor = VideoProcessor("test.mp4", "out.mp4", "/tmp", detector)

    with patch(
        "badminton_analysis.services.video_processor.cv2.VideoCapture",
        return_value=_FakeCapture(1),
    ):
        tracking = processor.process_frames_batched(Handedness.RIGHT)

    assert tracking["frames"] == []


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


def _directional_swing_with_recovery_spike() -> np.ndarray:
    horizontal = (
        [0.0] * 12
        + list(np.linspace(0.0, -12.0, 13)[1:])
        + list(np.linspace(-12.0, 35.0, 17)[1:])
        + [35.0] * 15
        + [35.0, -40.0, 35.0]
        + [35.0] * 20
    )
    return np.column_stack((horizontal, np.zeros(len(horizontal))))


def test_directional_acceleration_rejects_opposite_recovery_spike() -> None:
    positions = _directional_swing_with_recovery_spike()

    _, peak, _ = VideoAnalyzer.find_acc_analysis_window(list(positions))

    assert peak == 24


@pytest.mark.parametrize(
    "transform",
    (
        np.asarray(((-1.0, 0.0), (0.0, 1.0))),
        np.asarray(((-1.0, 0.0), (0.0, -1.0))),
    ),
)
def test_directional_acceleration_is_invariant_to_reversed_and_mirrored_swing(
    transform: np.ndarray,
) -> None:
    positions = _directional_swing_with_recovery_spike() @ transform

    _, peak, _ = VideoAnalyzer.find_acc_analysis_window(list(positions))

    assert peak == 24


def test_directional_acceleration_uses_body_relative_wrist_motion() -> None:
    relative = _directional_swing_with_recovery_spike()
    frames = len(relative)
    anchor = np.column_stack(
        (np.linspace(0.0, 120.0, frames), np.linspace(0.0, 30.0, frames))
    )
    wrist = relative + anchor

    _, peak, _ = VideoAnalyzer.find_acc_analysis_window(
        list(wrist), list(anchor)
    )

    assert peak == 24


def test_explicit_forward_axis_rejects_faster_backward_wrist_swing() -> None:
    positions = _directional_swing_with_recovery_spike()

    peak = VideoAnalyzer._directional_acceleration_peak(
        positions,
        forward_axis=np.asarray((1.0, 0.0)),
    )
    mirrored_peak = VideoAnalyzer._directional_acceleration_peak(
        positions @ np.asarray(((-1.0, 0.0), (0.0, 1.0))),
        forward_axis=np.asarray((-1.0, 0.0)),
    )

    velocity = np.diff(positions, axis=0)
    assert peak > 20
    assert velocity[peak - 1, 0] > 0.0
    assert mirrored_peak == peak


@pytest.mark.parametrize("skill", (Skill.SERVE,))
def test_analysis_window_keeps_follow_through_after_peak(
    monkeypatch: pytest.MonkeyPatch, skill: Skill
) -> None:
    monkeypatch.setattr(
        VideoAnalyzer,
        "find_acc_analysis_window",
        classmethod(lambda cls, positions, anchors=None: (10, 40, 70)),
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


def test_serve_elbow_heuristic_cannot_shorten_acceleration_tail(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        VideoAnalyzer,
        "find_acc_analysis_window",
        classmethod(lambda cls, positions, anchors=None: (10, 40, 70)),
    )
    hand_positions = np.zeros((100, 2), dtype=np.float64)
    hand_positions[40, 1] = 10.0
    elbow_positions = np.zeros((100, 2), dtype=np.float64)
    # This makes the legacy x-y heuristic select frame 50: long enough to
    # bypass its two-frame fallback, but still before the reserved tail.
    elbow_positions[50, 0] = 10.0

    start, peak, end = VideoAnalyzer.find_analysis_window(
        skill=Skill.SERVE,
        hand_positions=list(hand_positions),
        elbow_positions=list(elbow_positions),
    )

    assert (start, peak, end) == (10, 40, 70)


def test_serve_reserves_follow_through_after_late_low_hand_peak(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        VideoAnalyzer,
        "find_acc_analysis_window",
        classmethod(lambda cls, positions, anchors=None: (10, 40, 70)),
    )
    hand_positions = np.zeros((100, 2), dtype=np.float64)
    hand_positions[60, 1] = 10.0
    elbow_positions = np.zeros((100, 2), dtype=np.float64)

    start, peak, end = VideoAnalyzer.find_analysis_window(
        skill=Skill.SERVE,
        hand_positions=list(hand_positions),
        elbow_positions=list(elbow_positions),
    )

    assert (start, peak, end) == (30, 60, 90)


def test_lift_phases_include_return_to_ready_position(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    hand_positions = np.full((100, 2), 100.0, dtype=np.float64)
    hand_positions[20:36, 1] = np.linspace(100.0, 130.0, 16)
    hand_positions[36:41, 1] = 130.0
    hand_positions[40:51, 1] = np.linspace(130.0, 70.0, 11)
    hand_positions[50:71, 1] = np.linspace(70.0, 100.0, 21)
    hand_positions[71:, 1] = 100.0
    elbow_positions = np.zeros((100, 2), dtype=np.float64)

    phases = VideoAnalyzer.find_analysis_phases(
        skill=Skill.LIFT,
        hand_positions=list(hand_positions),
        elbow_positions=list(elbow_positions),
    )
    window = VideoAnalyzer.find_analysis_window(
        skill=Skill.LIFT,
        hand_positions=list(hand_positions),
        elbow_positions=list(elbow_positions),
    )

    assert phases[0] < phases[1] < phases[2] < phases[3] < phases[4]
    assert 34 <= phases[2] <= 41
    assert 43 <= phases[3] <= 51
    assert 68 <= phases[4] <= 80
    assert window == (phases[0], phases[2], phases[4])


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

    assert end - peak >= 15


@pytest.mark.parametrize("skill", (Skill.CLEAR, Skill.SMASH))
def test_overhead_window_keeps_two_second_preparation_context(
    monkeypatch: pytest.MonkeyPatch, skill: Skill
) -> None:
    monkeypatch.setattr(
        VideoAnalyzer,
        "find_acc_analysis_window",
        classmethod(lambda cls, positions, anchors=None: (70, 100, 130)),
    )
    hand_positions = np.zeros((180, 2), dtype=np.float64)
    hand_positions[100, 1] = -10.0
    elbow_positions = np.zeros((180, 2), dtype=np.float64)

    start, peak, end = VideoAnalyzer.find_analysis_window(
        skill=skill,
        hand_positions=list(hand_positions),
        elbow_positions=list(elbow_positions),
    )

    assert start == 40
    assert peak == 100
    assert end >= 130


@pytest.mark.parametrize("skill", (Skill.CLEAR, Skill.SMASH))
def test_overhead_window_includes_slow_wrist_follow_through_after_acceleration(
    monkeypatch: pytest.MonkeyPatch, skill: Skill
) -> None:
    monkeypatch.setattr(
        VideoAnalyzer,
        "find_acc_analysis_window",
        classmethod(lambda cls, positions, anchors=None: (10, 40, 70)),
    )
    hand_positions = np.zeros((100, 2), dtype=np.float64)
    hand_positions[40, 1] = -10.0
    hand_positions[41:87, 1] = np.linspace(-8.0, 12.0, 46)
    hand_positions[87:, 1] = 12.0
    elbow_positions = np.zeros((100, 2), dtype=np.float64)

    _, peak, end = VideoAnalyzer.find_analysis_window(
        skill=skill,
        hand_positions=list(hand_positions),
        elbow_positions=list(elbow_positions),
    )

    assert peak == 40
    assert 85 <= end <= 89


def test_smash_ends_at_first_low_wrist_velocity_stop_not_later_recovery(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        VideoAnalyzer,
        "find_acc_analysis_window",
        classmethod(lambda cls, positions, anchors=None: (10, 40, 70)),
    )
    hand_positions = np.zeros((120, 2), dtype=np.float64)
    hand_positions[:, 1] = 100.0
    hand_positions[40, 1] = 40.0
    # First real follow-through: the wrist descends and stops around frame 59.
    hand_positions[41:59, 1] = np.linspace(45.0, 150.0, 18)
    hand_positions[59:70, 1] = 150.0
    hand_positions[70:90, 1] = np.linspace(150.0, 110.0, 20)
    # A later rest/recovery position is even lower, but is not part of the
    # completed smash and must not become the analysis endpoint.
    hand_positions[90:101, 1] = np.linspace(110.0, 180.0, 11)
    hand_positions[101:, 1] = 180.0
    elbow_positions = np.zeros((120, 2), dtype=np.float64)
    elbow_positions[110, 1] = 250.0

    start, peak, end = VideoAnalyzer.find_analysis_window(
        skill=Skill.SMASH,
        hand_positions=list(hand_positions),
        elbow_positions=list(elbow_positions),
    )

    assert start == 0
    assert peak == 40
    assert 58 <= end <= 61


def test_smash_wrist_stop_requires_a_velocity_trend_not_single_frame_jitter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        VideoAnalyzer,
        "find_acc_analysis_window",
        classmethod(lambda cls, positions, anchors=None: (10, 40, 70)),
    )
    hand_positions = np.zeros((100, 2), dtype=np.float64)
    hand_positions[40, 1] = -10.0
    hand_positions[48, 1] = 1.0  # isolated detector spike
    elbow_positions = np.zeros((100, 2), dtype=np.float64)

    _, peak, end = VideoAnalyzer.find_analysis_window(
        skill=Skill.SMASH,
        hand_positions=list(hand_positions),
        elbow_positions=list(elbow_positions),
    )

    assert peak == 40
    assert end == 70
