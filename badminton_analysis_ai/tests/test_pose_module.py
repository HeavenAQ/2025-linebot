from types import SimpleNamespace
from unittest.mock import patch, MagicMock

import numpy as np
import pytest
from badminton_analysis.services.pose_detector import PoseDetector
from badminton_analysis.models.types import COCOKeypoints


def _fake_detection_result(boxes_xyxy, class_ids):
    """Minimal stand-in for an RF-DETR detection result, duck-typed to what
    the detection stage reads (.class_id, .xyxy). The detector now returns a
    box only; the joints come from the pose stage."""
    return SimpleNamespace(
        class_id=np.asarray(class_ids, dtype=np.int64),
        xyxy=np.asarray(boxes_xyxy, dtype=np.float64),
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
        assert self.detector._detector is None
        assert self.detector._pose_model is None
        assert hasattr(self.detector, "logger")

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

    def test_get_dense_keypoints_preserves_coordinates_and_confidence(self):
        keypoints = np.zeros((17, 2), dtype=np.float64)
        keypoints[10] = (120.0, 45.0)
        scores = np.linspace(0.0, 1.0, 17)
        self.detector._last_predictions = [
            {
                "bbox": [0.0, 0.0, 100.0, 200.0],
                "keypoints": keypoints,
                "keypoint_scores": scores,
            }
        ]

        result = self.detector.get_dense_2d_keypoints()

        assert result is not None
        coordinates, confidence = result
        assert coordinates.shape == (17, 2)
        assert confidence.shape == (17,)
        np.testing.assert_allclose(coordinates[10], (120.0, 45.0))
        assert confidence[10] == pytest.approx(scores[10])

    def test_get_dense_2d_keypoints_no_predictions(self):
        assert self.detector.get_dense_2d_keypoints() is None

    def test_detection_prefers_largest_person_and_ignores_other_classes(self):
        self.detector.device = "mps"
        self.detector._detector = SimpleNamespace(
            predict=lambda images, **kwargs: [
                _fake_detection_result(
                    boxes_xyxy=[
                        [0.0, 0.0, 20.0, 20.0],  # small person
                        [100.0, 100.0, 500.0, 500.0],  # large, but a bench
                        [50.0, 50.0, 250.0, 350.0],  # larger person
                    ],
                    class_ids=[1, 15, 1],
                )
            ]
        )

        boxes = self.detector._person_boxes([np.zeros((10, 10, 3), dtype=np.uint8)])

        assert len(boxes) == 1
        assert boxes[0] == pytest.approx((50.0, 50.0, 250.0, 350.0))

    def test_detection_is_empty_when_no_person_found(self):
        self.detector.device = "mps"
        self.detector._detector = SimpleNamespace(
            predict=lambda images, **kwargs: [
                _fake_detection_result(boxes_xyxy=[[0.0, 0.0, 20.0, 20.0]], class_ids=[15])
            ]
        )

        boxes = self.detector._person_boxes([np.zeros((10, 10, 3), dtype=np.uint8)])

        assert boxes == [None]

    def test_prediction_keeps_the_seventeen_body_keypoints(self):
        keypoints = np.arange(34, dtype=np.float64).reshape(17, 2)
        scores = np.full(17, 0.9, dtype=np.float64)

        prediction = self.detector._build_prediction(
            (10.0, 20.0, 210.0, 320.0), keypoints, scores
        )

        assert prediction["keypoints"].shape == (17, 2)
        # ViTPose emits COCO-17 in this repository's own index order, so no
        # schema adapter is needed.
        np.testing.assert_allclose(
            prediction["keypoints"][int(COCOKeypoints.RIGHT_WRIST)], (20.0, 21.0)
        )
        # The pose stage predicts these 17 joints and nothing else: no padded
        # WholeBody slots are carried alongside them.
        assert set(prediction) == {"bbox", "keypoints", "keypoint_scores"}

    def test_reset_tracking_clears_cached_state(self):
        self.detector._last_predictions = [{"keypoints": np.zeros((17, 2))}]

        self.detector.reset_tracking()

        assert self.detector._last_predictions == []
