import threading
import time
from typing import Any
import cv2
import numpy as np
from queue import Queue
from typing import final

from numpy.typing import NDArray

from badminton_analysis.core.logger import Logger
from badminton_analysis.models.types import (
    COCOKeypoints,
    Coordinate2D,
    Coordinate2DDict,
    CoordinateDict,
    Handedness,
    TrackingData,
    WholeBodyCoordinateDict,
)
from badminton_analysis.services.pose_detector import BATCH_SIZE, PoseDetector


@final
class VideoProcessor:
    """Extracts pose/position data from a video. Does not grade."""

    def __init__(
        self,
        video_path: str,
        out_filename: str,
        output_folder: str,
        pose_detector: PoseDetector | None = None,
    ) -> None:
        self.video_path = video_path
        self.out_filename = out_filename
        self.output_folder = output_folder
        self.logger = Logger(self.__class__.__name__)
        self.pose_detector = pose_detector or PoseDetector()
        self.time_intervals: list[float] = []
        self.source_frame_indices: list[int] = []

        # Buffers
        self.frames: list[NDArray[np.uint8]] = []
        self.landmarks: list[CoordinateDict] = []
        self.body_landmarks_2d: list[Coordinate2DDict] = []
        self.wholebody_landmarks: list[WholeBodyCoordinateDict] = []
        self.wholebody_keypoints_2d: list[NDArray[np.float64]] = []
        self.wholebody_confidence: list[NDArray[np.float64]] = []
        self.hand_positions: list[Coordinate2D] = []
        self.elbow_positions: list[Coordinate2D] = []

    def __frame_capture(
        self,
        cap: cv2.VideoCapture,
        frame_queue: Queue[NDArray[Any]],
        timestamp_queue: Queue[float],
        frame_index_queue: Queue[int],
    ) -> None:
        prev_time = time.perf_counter()
        frame_index = 0
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break

            current_time = time.perf_counter()
            time_interval = current_time - prev_time
            prev_time = current_time
            if not frame_queue.full():
                frame_queue.put(frame.copy())
                timestamp_queue.put(time_interval)
                frame_index_queue.put(frame_index)
            frame_index += 1
        cap.release()

    def _record_frame_result(
        self,
        frame: NDArray[np.uint8],
        source_frame_index: int,
        landmark_2d: Coordinate2DDict | None,
        wholebody_2d: WholeBodyCoordinateDict | None,
        wholebody_keypoints: tuple[NDArray[np.float64], NDArray[np.float64]] | None,
        handedness: int | None,
    ) -> None:
        """Shared bookkeeping for one frame's pose result, used by both the
        single-frame and batched extraction paths so they cannot silently
        diverge in what they accept/record."""
        if not landmark_2d:
            return
        wrist = (
            COCOKeypoints.RIGHT_WRIST
            if handedness == Handedness.RIGHT
            else COCOKeypoints.LEFT_WRIST
        )
        elbow = (
            COCOKeypoints.RIGHT_ELBOW
            if handedness == Handedness.RIGHT
            else COCOKeypoints.LEFT_ELBOW
        )

        if handedness is not None:
            if landmark_2d.get(wrist) is None or landmark_2d.get(elbow) is None:
                return

        self.landmarks.append(landmark_2d)
        self.body_landmarks_2d.append(landmark_2d)
        self.wholebody_landmarks.append(wholebody_2d or {})
        if wholebody_keypoints is None:
            self.wholebody_keypoints_2d.append(np.empty((0, 2), dtype=np.float64))
            self.wholebody_confidence.append(np.empty((0,), dtype=np.float64))
        else:
            coordinates, scores = wholebody_keypoints
            self.wholebody_keypoints_2d.append(coordinates)
            self.wholebody_confidence.append(scores)

        if handedness is not None:
            self.hand_positions.append(
                np.asarray(landmark_2d[wrist], dtype=np.float64)
            )
            self.elbow_positions.append(
                np.asarray(landmark_2d[elbow], dtype=np.float64)
            )
        self.frames.append(frame.copy())
        self.source_frame_indices.append(source_frame_index)

    def _tracking_data(self) -> TrackingData:
        return {
            "frames": self.frames,
            "original_landmarks": self.landmarks,
            "body_landmarks_2d": self.body_landmarks_2d,
            "hand_positions": self.hand_positions,
            "elbow_positions": self.elbow_positions,
            "time_intervals": self.time_intervals,
            "source_frame_indices": self.source_frame_indices,
            "wholebody_landmarks": self.wholebody_landmarks,
            "wholebody_keypoints_2d": self.wholebody_keypoints_2d,
            "wholebody_confidence": self.wholebody_confidence,
        }

    def process_frames(self, handedness: int | None) -> TrackingData:
        """Process video frames, detect pose, and return extracted data only."""
        self.logger.info("Starting video frame processing (extraction only)")
        self.pose_detector.reset_tracking()
        cap = cv2.VideoCapture(self.video_path)

        frame_queue: Queue[NDArray[Any]] = Queue()
        timestamp_queue: Queue[float] = Queue()
        frame_index_queue: Queue[int] = Queue()
        capture_thread = threading.Thread(
            target=self.__frame_capture,
            daemon=True,
            args=(
                cap,
                frame_queue,
                timestamp_queue,
                frame_index_queue,
            ),
        )
        capture_thread.start()

        while True:
            if not frame_queue.empty():
                frame = frame_queue.get()
                time_interval = timestamp_queue.get()
                source_frame_index = frame_index_queue.get()
                self.time_intervals.append(time_interval)
                results = self.pose_detector.get_pose(frame)
                landmark_2d = self.pose_detector.get_2d_landmarks(results)
                wholebody_2d = self.pose_detector.get_wholebody_2d_landmarks()
                wholebody_keypoints = self.pose_detector.get_wholebody_2d_keypoints()
                self._record_frame_result(
                    frame,
                    source_frame_index,
                    landmark_2d,
                    wholebody_2d,
                    wholebody_keypoints,
                    handedness,
                )
            else:
                if not capture_thread.is_alive():
                    break

        cap.release()
        self.logger.info(f"Extraction complete: frames={len(self.frames)}")
        return self._tracking_data()

    def process_frames_batched(self, handedness: int | None) -> TrackingData:
        """Batched variant of `process_frames` that hands `BATCH_SIZE` frames
        per pose-detector call so the TensorRT engine serves them, instead of
        the unbatched, non-TensorRT path `process_frames` uses. It reads the
        whole video up front, so it suits any caller that already has the
        entire clip -- offline extraction, and the analysis service -- but not
        a frame-at-a-time interactive one.

        Deliberately synchronous, not threaded: measured on real clips, a
        background decode thread (mirroring `process_frames`'s producer) made
        this slower, not faster — OpenCV/PyTorch already use their own
        internal multi-threading for decode/tensor ops, so an added
        Python-level thread mostly contended with that (high aggregate CPU
        time, worse wall time) rather than overlapping anything.
        """
        self.logger.info(
            "Starting video frame processing (extraction only, batched)"
        )
        self.pose_detector.reset_tracking()
        cap = cv2.VideoCapture(self.video_path)

        chunk_frames: list[NDArray[np.uint8]] = []
        chunk_indices: list[int] = []
        source_frame_index = 0
        inference_batch_size = (
            4 if getattr(self.pose_detector, "device", "cuda") == "mps" else BATCH_SIZE
        )

        def flush_chunk() -> None:
            if not chunk_frames:
                return
            batch_results = self.pose_detector.get_poses_batch(chunk_frames)
            for frame, index, results in zip(
                chunk_frames, chunk_indices, batch_results
            ):
                # Route each frame's batch result through the same getters
                # `get_pose()` feeds via `_last_predictions`, so wholebody
                # extraction logic (confidence filtering, shapes) has exactly
                # one implementation regardless of which path produced it.
                self.pose_detector._last_predictions = results
                landmark_2d = self.pose_detector.get_2d_landmarks(results)
                wholebody_2d = self.pose_detector.get_wholebody_2d_landmarks()
                wholebody_keypoints = self.pose_detector.get_wholebody_2d_keypoints()
                self._record_frame_result(
                    frame,
                    index,
                    landmark_2d,
                    wholebody_2d,
                    wholebody_keypoints,
                    handedness,
                )
            chunk_frames.clear()
            chunk_indices.clear()

        while True:
            success, frame = cap.read()
            if not success:
                break
            self.time_intervals.append(0.0)
            chunk_frames.append(frame.copy())
            chunk_indices.append(source_frame_index)
            source_frame_index += 1
            if len(chunk_frames) == inference_batch_size:
                flush_chunk()
        flush_chunk()

        cap.release()
        self.logger.info(f"Extraction complete: frames={len(self.frames)}")
        return self._tracking_data()
