from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from badminton_analysis.services.pose_detector import PoseDetector


def _result(boxes: list[list[float]]) -> SimpleNamespace:
    """One RF-DETR result holding several detected people."""
    count = len(boxes)
    return SimpleNamespace(
        class_id=np.ones(count, dtype=np.int64),
        data={"xyxy": np.asarray(boxes, dtype=np.float64)},
        xy=np.zeros((count, 17, 2), dtype=np.float64),
        keypoint_confidence=np.ones((count, 17), dtype=np.float64),
    )


# A court holds spectators and opponents; the athlete being graded is the one
# nearest the camera, so the largest box wins regardless of detection order.
def test_multiple_people_select_the_largest_bounding_box() -> None:
    detector = object.__new__(PoseDetector)
    smaller = [0.0, 0.0, 40.0, 50.0]
    largest = [100.0, 80.0, 260.0, 300.0]

    for boxes, expected in (
        ([smaller, largest], largest),
        ([largest, smaller], largest),
    ):
        predictions = detector._largest_person_prediction(_result(boxes))
        assert len(predictions) == 1
        assert list(predictions[0]["bbox"]) == expected


def test_no_person_detected_yields_no_prediction() -> None:
    detector = object.__new__(PoseDetector)
    empty = _result([[0.0, 0.0, 10.0, 10.0]])
    empty.class_id = np.zeros(1, dtype=np.int64)  # not the person class

    assert detector._largest_person_prediction(empty) == []
