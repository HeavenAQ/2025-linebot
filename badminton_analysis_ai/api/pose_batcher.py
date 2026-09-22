"""Serialised pose inference on the shared GPU detector, one request per batch.

Frames from different requests are never placed in the same engine batch. A
GPU engine does not promise a frame the same output whatever shares its batch,
and the pipeline turns small differences into different grades through its
thresholds, so a learner's score came to depend on who else was uploading at
that moment. Each call is run as its caller chunked it, padded with its own
last frame, which is what an upload analysed alone gets.
"""

from collections import deque
from concurrent.futures import Future
import copy
import logging
import threading

LOGGER = logging.getLogger("badminton-analysis")


class PoseBatcher:
    def __init__(self, detector, execution_lock, *, size=16):
        if not 1 <= size <= 16:
            raise ValueError("pose batching requires a size of 1..16")
        self.detector = detector
        self.execution_lock = execution_lock
        self.size = size
        self.condition = threading.Condition()
        self.pending = deque()
        self.closed = False
        self.thread = threading.Thread(
            target=self._run, name="pose-inference", daemon=True
        )
        self.thread.start()

    def request_detector(self):
        # Getters/reset_tracking mutate these fields. Give each video its own
        # view; only stateless get_poses_batch runs on the shared GPU detector.
        view = copy.copy(self.detector)
        view.reset_tracking()
        view.get_poses_batch = self.predict
        return view

    def predict(self, images):
        if not images:
            return []
        future = Future()
        with self.condition:
            if self.closed:
                raise RuntimeError("pose batcher stopped")
            self.pending.append((list(images), future))
            self.condition.notify()
        return future.result()

    def _run(self):
        while True:
            with self.condition:
                while not self.pending and not self.closed:
                    self.condition.wait()
                if self.closed and not self.pending:
                    return
                images, future = self.pending.popleft()
            try:
                results = []
                for start in range(0, len(images), self.size):
                    chunk = images[start : start + self.size]
                    with self.execution_lock:
                        output = self.detector.get_poses_batch(chunk)
                    if len(output) != len(chunk):
                        raise RuntimeError("pose engine returned wrong batch size")
                    results.extend(output)
                future.set_result(results)
            except Exception as exc:
                future.set_exception(exc)

    def close(self):
        with self.condition:
            self.closed = True
            self.condition.notify_all()
        self.thread.join(timeout=30)
