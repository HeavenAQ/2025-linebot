from concurrent.futures import ThreadPoolExecutor
import threading
import time

import pytest

from api.pose_batcher import PoseBatcher


class Detector:
    def __init__(self):
        self.batches = []
        self._last_predictions = ["shared"]

    def reset_tracking(self):
        self._last_predictions = []

    def get_poses_batch(self, images):
        self.batches.append(list(images))
        return [image * 10 for image in images]


class BatchSensitiveDetector(Detector):
    """A frame's result depends on what else is in its batch, as a GPU engine's
    may. Any mixing of requests shows up as a different answer."""

    def get_poses_batch(self, images):
        self.batches.append(list(images))
        time.sleep(0.01)
        return [image * 10 + sum(images) for image in images]


def test_concurrent_requests_never_share_an_engine_batch():
    detector = Detector()
    batcher = PoseBatcher(detector, threading.Lock())
    try:
        with ThreadPoolExecutor(4) as pool:
            calls = [pool.submit(batcher.predict, [base, base + 1]) for base in (1, 10, 100, 1000)]
            results = [call.result(timeout=2) for call in calls]
        assert results == [[10, 20], [100, 110], [1000, 1010], [10000, 10010]]
        requests = [{1, 2}, {10, 11}, {100, 101}, {1000, 1001}]
        for batch in detector.batches:
            assert any(set(batch) <= request for request in requests), batch
    finally:
        batcher.close()


def test_a_grade_does_not_depend_on_who_else_is_uploading():
    alone = PoseBatcher(BatchSensitiveDetector(), threading.Lock())
    try:
        expected = alone.predict([1, 2, 3])
    finally:
        alone.close()

    shared = PoseBatcher(BatchSensitiveDetector(), threading.Lock())
    try:
        with ThreadPoolExecutor(3) as pool:
            mine = pool.submit(shared.predict, [1, 2, 3])
            others = [pool.submit(shared.predict, [50, 60]) for _ in range(2)]
            assert mine.result(timeout=2) == expected
            for other in others:
                assert other.result(timeout=2) == [50 * 10 + 110, 60 * 10 + 110]
    finally:
        shared.close()


def test_a_call_larger_than_the_engine_batch_is_split_in_order():
    detector = Detector()
    batcher = PoseBatcher(detector, threading.Lock(), size=16)
    try:
        assert batcher.predict(list(range(20))) == [i * 10 for i in range(20)]
        assert detector.batches == [list(range(16)), list(range(16, 20))]
    finally:
        batcher.close()


def test_a_lone_request_runs_without_waiting_for_company():
    batcher = PoseBatcher(Detector(), threading.Lock())
    try:
        start = time.monotonic()
        assert batcher.predict([7]) == [70]
        assert time.monotonic() - start < 0.03
    finally:
        batcher.close()


def test_request_detectors_do_not_share_tracking_state():
    detector = Detector()
    batcher = PoseBatcher(detector, threading.Lock())
    try:
        left, right = batcher.request_detector(), batcher.request_detector()
        left._last_predictions.append("left")
        assert not right._last_predictions
        assert detector._last_predictions == ["shared"]
    finally:
        batcher.close()


def test_batch_exception_reaches_its_caller_and_worker_survives():
    class Broken(Detector):
        def get_poses_batch(self, images):
            if 0 in images:
                raise ValueError("bad frame")
            return super().get_poses_batch(images)

    batcher = PoseBatcher(Broken(), threading.Lock())
    try:
        with pytest.raises(ValueError, match="bad frame"):
            batcher.predict([0, 1])
        assert batcher.predict([2]) == [20]
    finally:
        batcher.close()
    with pytest.raises(RuntimeError, match="stopped"):
        batcher.predict([1])


def test_invalid_batch_size_rejected():
    with pytest.raises(ValueError):
        PoseBatcher(Detector(), threading.Lock(), size=17)
