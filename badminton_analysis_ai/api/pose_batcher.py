"""Bounded cross-request pose microbatches, with a size OR oldest-item timer.

The 50 ms budget is batch-formation delay while the GPU is available, not a
promise about Cloud Tasks delivery, cold starts, or an already busy GPU.
Diffusion shares the execution lock but not RNG state with concurrent requests.
"""

from collections import deque
from concurrent.futures import Future
import copy
import logging
import threading
import time

LOGGER = logging.getLogger("badminton-analysis")


class PoseBatcher:
    def __init__(self, detector, execution_lock, *, size=16, max_wait=0.05):
        if not 1 <= size <= 16 or not 0 <= max_wait <= 0.05:
            raise ValueError("pose batching requires size 1..16 and wait <= 50ms")
        self.detector = detector
        self.execution_lock = execution_lock
        self.size, self.max_wait = size, max_wait
        self.condition = threading.Condition()
        self.pending = deque()
        self.closed = False
        self.thread = threading.Thread(
            target=self._run, name="pose-microbatch", daemon=True
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
        futures = [Future() for _ in images]
        with self.condition:
            if self.closed:
                raise RuntimeError("pose batcher stopped")
            queued_at = time.monotonic()
            self.pending.extend(zip(images, futures, [queued_at] * len(images)))
            self.condition.notify()
        return [future.result() for future in futures]

    def _run(self):
        while True:
            with self.condition:
                while not self.pending and not self.closed:
                    self.condition.wait()
                if self.closed and not self.pending:
                    return
                deadline = self.pending[0][2] + self.max_wait
                while len(self.pending) < self.size and not self.closed:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        break
                    self.condition.wait(remaining)
                batch = [
                    self.pending.popleft()
                    for _ in range(min(self.size, len(self.pending)))
                ]
            try:
                with self.execution_lock:
                    results = self.detector.get_poses_batch([item[0] for item in batch])
                if len(results) != len(batch):
                    raise RuntimeError("pose engine returned wrong batch size")
                for (_, future, _), result in zip(batch, results):
                    future.set_result(result)
            except Exception as exc:
                for _, future, _ in batch:
                    future.set_exception(exc)

    def close(self):
        with self.condition:
            self.closed = True
            self.condition.notify_all()
        self.thread.join(timeout=30)
