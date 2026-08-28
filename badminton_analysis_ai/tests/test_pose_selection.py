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


# The batched engine path reads raw postprocess tensors rather than a
# supervision result, so it selects through the same helper. It once did not,
# and picked the largest detection of any class -- which produces a plausible
# pose for the wrong object and a silently wrong grade, never an error.
def test_largest_person_index_ignores_bigger_non_person_detections() -> None:
    from badminton_analysis.services.pose_detector import _largest_person_index

    class_ids = np.asarray([0, 1, 1], dtype=np.int64)  # 0 is not a person
    boxes = np.asarray(
        [
            [0.0, 0.0, 900.0, 900.0],  # biggest overall, but not a person
            [10.0, 10.0, 60.0, 70.0],
            [100.0, 80.0, 260.0, 300.0],  # biggest person
        ],
        dtype=np.float64,
    )
    scores = np.asarray([0.99, 0.80, 0.90], dtype=np.float64)

    assert _largest_person_index(class_ids, boxes, scores, 0.5) == 2
    assert _largest_person_index(class_ids, boxes) == 2


def test_largest_person_index_applies_the_score_threshold() -> None:
    from badminton_analysis.services.pose_detector import _largest_person_index

    class_ids = np.asarray([1, 1], dtype=np.int64)
    boxes = np.asarray(
        [[0.0, 0.0, 500.0, 500.0], [10.0, 10.0, 60.0, 70.0]], dtype=np.float64
    )
    scores = np.asarray([0.10, 0.90], dtype=np.float64)

    assert _largest_person_index(class_ids, boxes, scores, 0.5) == 1
    assert _largest_person_index(class_ids, boxes, scores, 0.95) is None
