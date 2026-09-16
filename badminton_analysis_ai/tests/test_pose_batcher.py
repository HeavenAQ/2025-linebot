from concurrent.futures import ThreadPoolExecutor
import threading
import time

import pytest

from service.pose_batcher import PoseBatcher


class Detector:
    def __init__(self):
        self.batches = []
        self._last_predictions = ["shared"]

    def reset_tracking(self):
        self._last_predictions = []

    def get_poses_batch(self, images):
        self.batches.append(images)
        return [image * 10 for image in images]


def test_multiple_requests_share_a_batch_without_mixing_their_results():
    detector = Detector()
    batcher = PoseBatcher(detector, threading.Lock(), size=4)
    try:
        with ThreadPoolExecutor(2) as pool:
            one = pool.submit(batcher.predict, [1, 2])
            two = pool.submit(batcher.predict, [3, 4])
            assert one.result(timeout=1) == [10, 20]
            assert two.result(timeout=1) == [30, 40]
        assert len(detector.batches) == 1
        left, right = batcher.request_detector(), batcher.request_detector()
        left._last_predictions.append("left")
        assert not right._last_predictions
        assert detector._last_predictions == ["shared"]
    finally:
        batcher.close()


def test_idle_partial_batch_fires_without_another_request():
    detector = Detector()
    batcher = PoseBatcher(detector, threading.Lock())
    try:
        start = time.monotonic()
        assert batcher.predict([7]) == [70]
        elapsed = time.monotonic() - start
        # Leave scheduler jitter headroom; configured formation deadline is 50ms.
        assert 0.04 <= elapsed < 0.3
    finally:
        batcher.close()


def test_full_batch_does_not_wait_for_timer():
    detector = Detector()
    batcher = PoseBatcher(detector, threading.Lock(), max_wait=0.05)
    try:
        assert batcher.predict(list(range(16))) == [i * 10 for i in range(16)]
        assert detector.batches == [list(range(16))]
    finally:
        batcher.close()


def test_batch_exception_reaches_all_callers_and_worker_survives():
    class Broken(Detector):
        def get_poses_batch(self, images):
            if 0 in images:
                raise ValueError("bad frame")
            return super().get_poses_batch(images)

    batcher = PoseBatcher(Broken(), threading.Lock(), max_wait=0)
    try:
        with pytest.raises(ValueError, match="bad frame"):
            batcher.predict([0, 1])
        assert batcher.predict([2]) == [20]
    finally:
        batcher.close()
    with pytest.raises(RuntimeError, match="stopped"):
        batcher.predict([1])


def test_invalid_batch_settings_rejected():
    with pytest.raises(ValueError):
        PoseBatcher(Detector(), threading.Lock(), max_wait=0.051)
