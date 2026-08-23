from __future__ import annotations

from badminton_analysis.services.pose_detector import PoseDetector


def test_multiple_people_select_the_largest_bounding_box() -> None:
    detector = object.__new__(PoseDetector)
    smaller = {
        "bbox": [0.0, 0.0, 40.0, 50.0],
        "keypoints": [[10.0, 10.0], [30.0, 40.0]],
    }
    largest = {
        "bbox": [100.0, 80.0, 260.0, 300.0],
        "keypoints": [[120.0, 100.0], [240.0, 280.0]],
    }

    assert detector._select_target([smaller, largest]) is largest
    assert detector._select_target([largest, smaller]) is largest
