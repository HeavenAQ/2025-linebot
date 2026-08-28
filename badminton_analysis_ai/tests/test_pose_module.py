from types import SimpleNamespace
from unittest.mock import patch, MagicMock

import numpy as np
import pytest
from badminton_analysis.services.pose_detector import PoseDetector
from badminton_analysis.models.types import COCOKeypoints


def _fake_keypoints_result(
    boxes_xyxy, class_ids, keypoints=None, keypoint_confidence=None
):
    """Minimal stand-in for supervision.KeyPoints, duck-typed to what
    PoseDetector actually reads (.class_id, .xy, .keypoint_confidence,
    .data["xyxy"])."""
    class_id = np.asarray(class_ids, dtype=np.int64)
    count = len(class_id)
    if keypoints is None:
        keypoints = np.zeros((count, 17, 2), dtype=np.float64)
    if keypoint_confidence is None:
        keypoint_confidence = np.full((count, 17), 0.9, dtype=np.float64)
    return SimpleNamespace(
        class_id=class_id,
        xy=np.asarray(keypoints, dtype=np.float64),
        keypoint_confidence=np.asarray(keypoint_confidence, dtype=np.float64),
        data={"xyxy": np.asarray(boxes_xyxy, dtype=np.float64)},
    )


class TestPoseDetector:
    def setup_method(self, method):
        with patch("badminton_analysis.core.logger.Logger") as mock_logger:
            mock_logger.return_value.info = MagicMock()
            self.detector = PoseDetector()

    def test_pose_detector_initialization(self):
        assert self.detector.min_detection_confidence == 0.15
        assert self.detector.elbow_detection_confidence == 0.05
        assert self.detector.person_detection_threshold == 0.5
        assert self.detector._model is None
        assert hasattr(self.detector, "logger")

    def test_fps_calculation_zero_diff(self):
        with patch("time.time", return_value=1.0):
            self.detector.fps
            fps2 = self.detector.fps
            assert fps2 > 0

    def test_compute_angle_valid_points(self):
        point_a = (0, 1)
        point_b = (0, 0)
        point_c = (1, 0)
        angle = self.detector.compute_angle(point_a, point_b, point_c)
        assert angle == pytest.approx(90.0, rel=1e-2)

    def test_compute_angle_straight_line(self):
        point_a = (0, 0)
        point_b = (1, 0)
        point_c = (2, 0)
        angle = self.detector.compute_angle(point_a, point_b, point_c)
        assert angle == pytest.approx(180.0, rel=1e-2)

    def test_compute_angle_zero_vector(self):
        point_a = (0, 0)
        point_b = (0, 0)
        point_c = (1, 0)
        angle = self.detector.compute_angle(point_a, point_b, point_c)
        assert angle is None

    def test_get_2d_landmarks_no_results(self):
        assert self.detector.get_2d_landmarks(None) is None
        assert self.detector.get_2d_landmarks([]) is None

    def test_get_2d_landmarks_filters_by_confidence(self):
        keypoints = np.zeros((17, 2), dtype=np.float64)
        keypoints[0] = (10.0, 20.0)
        scores = np.full(17, 0.9, dtype=np.float64)
        scores[1] = 0.1  # below the default 0.15 threshold
        results = [{"keypoints": keypoints, "keypoint_scores": scores}]

        landmarks = self.detector.get_2d_landmarks(results)

        assert landmarks is not None
        assert COCOKeypoints.NOSE in landmarks
        np.testing.assert_allclose(landmarks[COCOKeypoints.NOSE], (10.0, 20.0))
        assert COCOKeypoints.LEFT_EYE not in landmarks

    def test_get_2d_landmarks_uses_lower_elbow_threshold(self):
        keypoints = np.arange(34, dtype=np.float64).reshape(17, 2)
        scores = np.full(17, 0.9, dtype=np.float64)
        scores[int(COCOKeypoints.LEFT_ELBOW)] = 0.08
        scores[int(COCOKeypoints.RIGHT_ELBOW)] = 0.04
        scores[int(COCOKeypoints.LEFT_WRIST)] = 0.08
        results = [{"keypoints": keypoints, "keypoint_scores": scores}]

        landmarks = self.detector.get_2d_landmarks(results)

        assert landmarks is not None
        assert COCOKeypoints.LEFT_ELBOW in landmarks
        assert COCOKeypoints.RIGHT_ELBOW not in landmarks
        # The same score is still too low for a non-elbow joint.
        assert COCOKeypoints.LEFT_WRIST not in landmarks

    def test_elbow_threshold_cannot_exceed_general_threshold(self):
        with pytest.raises(ValueError, match="elbow_detection_confidence"):
            PoseDetector(
                min_detection_confidence=0.15,
                elbow_detection_confidence=0.2,
            )

    def test_get_wholebody_keypoints_preserves_coordinates_and_confidence(self):
        keypoints = np.zeros((133, 2), dtype=np.float64)
        keypoints[10] = (120.0, 45.0)
        scores = np.linspace(0.0, 1.0, 133)
        self.detector._last_predictions = [
            {
                "bbox": [0.0, 0.0, 100.0, 200.0],
                "keypoints": np.zeros((17, 2)),
                "keypoint_scores": np.zeros(17),
                "wholebody_keypoints": keypoints,
                "wholebody_scores": scores,
            }
        ]

        result = self.detector.get_wholebody_2d_keypoints()

        assert result is not None
        coordinates, confidence = result
        assert coordinates.shape == (133, 2)
        assert confidence.shape == (133,)
        np.testing.assert_allclose(coordinates[10], (120.0, 45.0))
        assert confidence[10] == pytest.approx(scores[10])

    def test_get_wholebody_2d_keypoints_no_predictions(self):
        assert self.detector.get_wholebody_2d_keypoints() is None
        assert self.detector.get_wholebody_2d_landmarks() is None

    def test_get_pose_prefers_largest_person_and_ignores_other_classes(self):
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        keypoints = np.zeros((3, 17, 2), dtype=np.float64)
        keypoints[2, 0] = (55.0, 60.0)  # nose of the larger person
        self.detector._model = MagicMock(
            predict=MagicMock(
                return_value=_fake_keypoints_result(
                    boxes_xyxy=[
                        [0.0, 0.0, 20.0, 20.0],  # small person
                        [100.0, 100.0, 500.0, 500.0],  # large, but a bench
                        [50.0, 50.0, 250.0, 350.0],  # larger person
                    ],
                    class_ids=[1, 15, 1],
                    keypoints=keypoints,
                )
            )
        )

        result = self.detector.get_pose(img)

        assert len(result) == 1
        assert result[0]["bbox"] == pytest.approx([50.0, 50.0, 250.0, 350.0])
        np.testing.assert_allclose(result[0]["keypoints"][0], (55.0, 60.0))

    def test_get_pose_returns_empty_list_when_no_person_found(self):
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        self.detector._model = MagicMock(
            predict=MagicMock(
                return_value=_fake_keypoints_result(
                    boxes_xyxy=[[0.0, 0.0, 20.0, 20.0]], class_ids=[15]
                )
            )
        )

        result = self.detector.get_pose(img)

        assert result == []
        assert self.detector.get_2d_landmarks(result) is None

    def test_get_pose_runs_end_to_end_with_no_hand_keypoints(self):
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        keypoints = np.zeros((1, 17, 2), dtype=np.float64)
        keypoints[0] = np.arange(34, dtype=np.float64).reshape(17, 2)
        confidence = np.full((1, 17), 0.9, dtype=np.float64)
        self.detector._model = MagicMock(
            predict=MagicMock(
                return_value=_fake_keypoints_result(
                    boxes_xyxy=[[10.0, 20.0, 210.0, 320.0]],
                    class_ids=[1],
                    keypoints=keypoints,
                    keypoint_confidence=confidence,
                )
            )
        )

        result = self.detector.get_pose(img)

        assert len(result) == 1
        assert result[0]["keypoints"].shape == (17, 2)
        assert result[0]["wholebody_keypoints"].shape == (133, 2)
        # RFDETRKeypointPreview's native order already matches COCOKeypoints,
        # so no schema adapter is needed.
        np.testing.assert_allclose(
            result[0]["keypoints"][int(COCOKeypoints.RIGHT_WRIST)], (20.0, 21.0)
        )
        # No hand/face/feet keypoints exist, so only the first 17 wholebody
        # slots carry real (nonzero-confidence) data.
        np.testing.assert_allclose(
            result[0]["wholebody_keypoints"][:17], result[0]["keypoints"]
        )
        assert np.all(result[0]["wholebody_scores"][17:] == 0.0)
        assert len(self.detector._last_predictions) == 1
        self.detector._model.predict.assert_called_once()

    def test_reset_tracking_clears_cached_state(self):
        self.detector._last_predictions = [{"keypoints": np.zeros((17, 2))}]
        self.detector._target_bbox_center = np.array((1.0, 2.0))

        self.detector.reset_tracking()

        assert self.detector._last_predictions == []
        assert self.detector._target_bbox_center is None
