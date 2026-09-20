from typing import final

import cv2
import numpy as np
from numpy.typing import NDArray

from badminton_analysis.core.logger import Logger
from badminton_analysis.models.types import (
    COCOKeypoints,
    Coordinate2D,
    Coordinate2DDict,
    Handedness,
    TrackingData,
)
from badminton_analysis.services.pose_detector import BATCH_SIZE, PoseDetector


@final
class VideoProcessor:
    """Extracts pose/position data from a video. Does not grade."""

    def __init__(
        self,
        video_path: str,
        pose_detector: PoseDetector | None = None,
    ) -> None:
        self.video_path = video_path
        self.logger = Logger(self.__class__.__name__)
        self.pose_detector = pose_detector or PoseDetector()
        self.source_frame_indices: list[int] = []

        # Buffers
        self.frames: list[NDArray[np.uint8]] = []
        self.body_landmarks_2d: list[Coordinate2DDict] = []
        self.body_keypoints_2d: list[NDArray[np.float64]] = []
        self.body_confidence_2d: list[NDArray[np.float64]] = []
        self.hand_positions: list[Coordinate2D] = []
        self.elbow_positions: list[Coordinate2D] = []

    def _record_frame_result(
        self,
        frame: NDArray[np.uint8],
        source_frame_index: int,
        landmark_2d: Coordinate2DDict | None,
        dense_keypoints: tuple[NDArray[np.float64], NDArray[np.float64]] | None,
        handedness: int | None,
    ) -> None:
        """Record one frame's pose result, skipping frames without a person
        (or, when handedness is known, without its wrist and elbow)."""
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

        self.body_landmarks_2d.append(landmark_2d)
        if dense_keypoints is None:
            # This branch is defensive: a valid RF-DETR body result normally
            # always carries its dense scores.  Keep the buffers aligned even
            # for test doubles or alternate backends.
            dense = np.full((17, 2), np.nan, dtype=np.float64)
            observed = np.zeros(17, dtype=np.float64)
            for keypoint, coordinate in landmark_2d.items():
                dense[int(keypoint)] = np.asarray(coordinate, dtype=np.float64)[:2]
                observed[int(keypoint)] = 1.0
            self.body_keypoints_2d.append(dense)
            self.body_confidence_2d.append(observed)
        else:
            coordinates, scores = dense_keypoints
            body_coordinates = np.asarray(coordinates, dtype=np.float64).copy()
            body_scores = np.clip(np.asarray(scores, dtype=np.float64), 0.0, 1.0)
            # Match get_2d_landmarks' validity decision while retaining the
            # detector's continuous score for every accepted keypoint.
            general_threshold = float(self.pose_detector.min_detection_confidence)
            elbow_threshold_value = getattr(
                self.pose_detector, "elbow_detection_confidence", general_threshold
            )
            elbow_threshold = (
                float(elbow_threshold_value)
                if isinstance(elbow_threshold_value, (int, float))
                else general_threshold
            )
            thresholds = np.full(17, general_threshold, dtype=np.float64)
            thresholds[
                [
                    int(COCOKeypoints.LEFT_ELBOW),
                    int(COCOKeypoints.RIGHT_ELBOW),
                ]
            ] = elbow_threshold
            body_scores = np.where(
                body_scores > thresholds,
                body_scores,
                0.0,
            )
            self.body_keypoints_2d.append(body_coordinates)
            self.body_confidence_2d.append(body_scores)

        if handedness is not None:
            self.hand_positions.append(np.asarray(landmark_2d[wrist], dtype=np.float64))
            self.elbow_positions.append(
                np.asarray(landmark_2d[elbow], dtype=np.float64)
            )
        self.frames.append(frame.copy())
        self.source_frame_indices.append(source_frame_index)

    def _tracking_data(self) -> TrackingData:
        return {
            "frames": self.frames,
            "body_landmarks_2d": self.body_landmarks_2d,
            "body_keypoints_2d": self.body_keypoints_2d,
            "body_confidence_2d": self.body_confidence_2d,
            "hand_positions": self.hand_positions,
            "elbow_positions": self.elbow_positions,
            "source_frame_indices": self.source_frame_indices,
        }

    def process_frames_batched(self, handedness: int | None) -> TrackingData:
        """Extract poses from the whole video in pose-detector batches.

        Each call hands up to `BATCH_SIZE` frames to the detector, which runs
        the TensorRT engine on CUDA and RF-DETR ``predict()`` on MPS/CPU.

        Deliberately synchronous, not threaded: measured on real clips, a
        background decode thread made this slower, not faster — OpenCV/PyTorch
        already use their own
        internal multi-threading for decode/tensor ops, so an added
        Python-level thread mostly contended with that (high aggregate CPU
        time, worse wall time) rather than overlapping anything.
        """
        self.logger.info("Starting video frame processing (extraction only, batched)")
        self.pose_detector.reset_tracking()
        cap = cv2.VideoCapture(self.video_path)

        chunk_frames: list[NDArray[np.uint8]] = []
        chunk_indices: list[int] = []
        source_frame_index = 0
        inference_batch_size = (
            # RF-DETR's native MPS path accepts variable batches. Eight keeps
            # peak unified-memory use bounded while avoiding the severe
            # per-call overhead of the former four-frame extraction batch.
            8
            if getattr(self.pose_detector, "device", "cuda") == "mps"
            else BATCH_SIZE
        )

        def flush_chunk() -> None:
            if not chunk_frames:
                return
            batch_results = self.pose_detector.get_poses_batch(chunk_frames)
            for frame, index, results in zip(
                chunk_frames, chunk_indices, batch_results
            ):
                # The dense-keypoint getter reads `_last_predictions`, so
                # point it at this frame's result before reading landmarks.
                self.pose_detector._last_predictions = results
                landmark_2d = self.pose_detector.get_2d_landmarks(results)
                dense_keypoints = self.pose_detector.get_dense_2d_keypoints()
                self._record_frame_result(
                    frame,
                    index,
                    landmark_2d,
                    dense_keypoints,
                    handedness,
                )
            chunk_frames.clear()
            chunk_indices.clear()

        while True:
            success, frame = cap.read()
            if not success:
                break
            chunk_frames.append(frame.copy())
            chunk_indices.append(source_frame_index)
            source_frame_index += 1
            if len(chunk_frames) == inference_batch_size:
                flush_chunk()
        flush_chunk()

        cap.release()
        self.logger.info(f"Extraction complete: frames={len(self.frames)}")
        return self._tracking_data()
