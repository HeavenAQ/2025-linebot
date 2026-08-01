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
from badminton_analysis.services.pose_detector import PoseDetector


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
                results_3d = self.pose_detector.get_pose(frame)
                landmark_3d = self.pose_detector.get_3d_landmarks(results_3d)
                landmark_2d = self.pose_detector.get_2d_landmarks()
                wholebody_2d = self.pose_detector.get_wholebody_2d_landmarks()
                if not landmark_3d or not landmark_2d:
                    continue
                else:
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
                        if (
                            landmark_3d.get(wrist) is None
                            or landmark_3d.get(elbow) is None
                        ):
                            continue
                        if (
                            landmark_2d.get(wrist) is None
                            or landmark_2d.get(elbow) is None
                        ):
                            continue

                    self.landmarks.append(landmark_3d)
                    self.body_landmarks_2d.append(landmark_2d)
                    self.wholebody_landmarks.append(wholebody_2d or {})

                    if handedness is not None:
                        self.hand_positions.append(
                            np.asarray(landmark_2d[wrist], dtype=np.float64)
                        )
                        self.elbow_positions.append(
                            np.asarray(landmark_2d[elbow], dtype=np.float64)
                        )
                    self.frames.append(frame.copy())
                    self.source_frame_indices.append(source_frame_index)
            else:
                if not capture_thread.is_alive():
                    break

        cap.release()
        self.logger.info(f"Extraction complete: frames={len(self.frames)}")
        return {
            "frames": self.frames,
            "original_landmarks": self.landmarks,
            "body_landmarks_2d": self.body_landmarks_2d,
            "hand_positions": self.hand_positions,
            "elbow_positions": self.elbow_positions,
            "time_intervals": self.time_intervals,
            "source_frame_indices": self.source_frame_indices,
            "wholebody_landmarks": self.wholebody_landmarks,
        }
