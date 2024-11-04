import time
from typing import Any, Optional

import cv2
import numpy as np
from cv2.typing import MatLike
from ultralytics import YOLO

from Logger import Logger
from Types import Body2DCoordinates, COCOKeypoints


class PoseDetector:
    """
    A class to detect and handle human poses using YOLOv8.
    It supports pose detection from images and video streams, and provides additional
    functionalities such as drawing pose landmarks and displaying the current FPS.

    Attributes:
        logger (Logger): A custom logger instance for logging information.
        model (YOLO): The YOLOv8 model for pose estimation.
        __cur_time (float): Stores the current time for FPS calculation.
        __prev_time (float): Stores the previous time for FPS calculation.
    """

    def __init__(
        self,
        model_path: str = "yolov8m-pose.pt",
        min_detection_confidence: float = 0.5,
    ):
        """
        Initializes the PoseDetector with specified parameters for pose detection.

        Args:
            model_path (str): Path to the YOLOv8 pose model.
            min_detection_confidence (float): Minimum confidence for detections.
        """
        self.logger = Logger(self.__class__.__name__)
        self.min_detection_confidence = min_detection_confidence

        # Initialize the YOLOv8 pose model
        self.model = YOLO(model_path)

        # FPS settings
        self.__cur_time = 0
        self.__prev_time = 0

        self.logger.info(
            f"PoseDetector initialized with model={model_path}, "
            f"min_detection_confidence={min_detection_confidence}"
        )

    @property
    def fps(self) -> float:
        """
        Calculates and returns the current frames per second (FPS).

        Returns:
            float: The current FPS based on the time difference between the current
                   and previous frames.
        """
        self.__cur_time = time.time()
        time_diff = self.__cur_time - self.__prev_time
        if time_diff == 0:
            time_diff = 1e-6  # Avoid division by zero
        cur_fps = 1 / time_diff
        self.__prev_time = self.__cur_time
        self.logger.debug(f"Current FPS calculated: {cur_fps}")
        return cur_fps

    def get_pose(self, img: MatLike) -> Any:
        """
        Detects the pose in the given image and returns the results.

        Args:
            img (MatLike): The input image in which the pose needs to be detected.

        Returns:
            Any: The results from YOLOv8's pose detection, containing keypoints
                 and other information if a pose is detected.
        """
        self.logger.debug(
            f"Processing image for pose detection, image shape: {img.shape}"
        )
        # Run pose estimation
        results = self.model.predict(img, conf=self.min_detection_confidence)
        self.logger.debug("Pose estimation completed")
        return results

    def get_2d_landmarks(self, results: Any) -> Optional[Body2DCoordinates]:
        """
        Retrieves the pose landmarks as a dictionary with body part numbers as keys
        with their corresponding x, y coordinates being the value.

        Args:
            img (MatLike): The input image from which landmarks are to be extracted.

        Returns:
            Optional[Body2DCoordinates]: A dictionary containing the landmark index
                                        and its corresponding x, y coordinates
                                        or None if no landmarks are detected.
        """
        if results and results[0].keypoints is not None:
            self.logger.debug("Getting pose landmarks")
            # Get the keypoints for the first detection
            keypoints_xy = (
                results[0].keypoints.xy[0].cpu().numpy()
            )  # Shape: (num_keypoints, 2)
            keypoints_conf = (
                results[0].keypoints.conf[0].cpu().numpy()
            )  # Shape: (num_keypoints,)
            body_coordinates = {}
            for idx, (x, y) in enumerate(keypoints_xy):
                conf = keypoints_conf[idx]
                if conf > self.min_detection_confidence:
                    body_coordinates[COCOKeypoints(idx)] = (float(x), float(y))
            self.logger.debug("Retrieved pose landmarks")
            return body_coordinates
        else:
            self.logger.error("No pose landmarks detected")
            return None

    def compute_angle(
        self,
        point_a: tuple[float, float],
        point_b: tuple[float, float],
        point_c: tuple[float, float],
    ) -> Optional[float]:
        """
        Computes the angle between three 2D points.

        Args:
            point_a (tuple[float, float]): The first point (x, y).
            point_b (tuple[float, float]): The second point (x, y).
            point_c (tuple[float, float]): The third point (x, y).

        Returns:
            float: The angle in degrees between the three points.
        """
        # Get the coordinates of the points
        a = np.array(point_a, dtype=np.float64)
        b = np.array(point_b, dtype=np.float64)
        c = np.array(point_c, dtype=np.float64)

        # Get vectors
        vector_ba = a - b
        vector_bc = c - b

        # Compute the norms
        norm_ba = np.linalg.norm(vector_ba)
        norm_bc = np.linalg.norm(vector_bc)

        # Avoid division by zero
        if norm_ba == 0 or norm_bc == 0:
            return None

        # Compute the cosine and clip it to the range [-1, 1]
        # ba @ bc = magnitude of vector ba * magnitude of vector bc * cos(theta)
        cos_theta = (vector_ba @ vector_bc) / (norm_ba * norm_bc)
        cos_theta = np.clip(cos_theta, -1, 1)

        # Compute the angle in radians and convert it to degree
        angle_radian = np.arccos(cos_theta)
        return np.degrees(angle_radian)

    def show_pose(self, img: MatLike, results: Any) -> None:
        """
        Draws only the pose skeleton (landmarks and connections) on the given image.

        Args:
            img (MatLike): The input image on which the landmarks are to be drawn.
            results (Any): The results from YOLOv8's pose detection.

        Returns:
            None
        """
        if results and len(results) > 0:
            self.logger.debug("Drawing pose skeleton on the image")

            # Define the connections between keypoints for the skeleton
            skeleton_connections = [
                (COCOKeypoints.LEFT_SHOULDER, COCOKeypoints.RIGHT_SHOULDER),
                (COCOKeypoints.LEFT_SHOULDER, COCOKeypoints.LEFT_ELBOW),
                (COCOKeypoints.LEFT_ELBOW, COCOKeypoints.LEFT_WRIST),
                (COCOKeypoints.RIGHT_SHOULDER, COCOKeypoints.RIGHT_ELBOW),
                (COCOKeypoints.RIGHT_ELBOW, COCOKeypoints.RIGHT_WRIST),
                (COCOKeypoints.LEFT_HIP, COCOKeypoints.RIGHT_HIP),
                (COCOKeypoints.LEFT_HIP, COCOKeypoints.LEFT_KNEE),
                (COCOKeypoints.LEFT_KNEE, COCOKeypoints.LEFT_ANKLE),
                (COCOKeypoints.RIGHT_HIP, COCOKeypoints.RIGHT_KNEE),
                (COCOKeypoints.RIGHT_KNEE, COCOKeypoints.RIGHT_ANKLE),
                (COCOKeypoints.LEFT_SHOULDER, COCOKeypoints.LEFT_HIP),
                (COCOKeypoints.RIGHT_SHOULDER, COCOKeypoints.RIGHT_HIP),
            ]

            # Iterate over each detected person in the results
            for result in results:
                keypoints = (
                    result.keypoints.xy[0].gpu().numpy()
                )  # Get keypoints (x, y) for each detected person

                # Draw each connection in the skeleton
                for start, end in skeleton_connections:
                    if start.value < len(keypoints) and end.value < len(keypoints):
                        x1, y1 = keypoints[start.value]
                        x2, y2 = keypoints[end.value]
                        if (
                            x1 > 0 and y1 > 0 and x2 > 0 and y2 > 0
                        ):  # Ensure points are valid
                            cv2.line(
                                img,
                                (int(x1), int(y1)),
                                (int(x2), int(y2)),
                                (255, 255, 255),
                                2,
                            )

                # Draw each keypoint
                for point in keypoints:
                    x, y = point
                    if x > 0 and y > 0:  # Ensure point is valid
                        cv2.circle(img, (int(x), int(y)), 3, (249, 210, 60), 1)
        else:
            self.logger.error("No landmarks provided to show_pose method")

    def show_fps(self, img: MatLike) -> None:
        """
        Displays the current FPS on the image.

        Args:
            img (MatLike): The input image on which the FPS will be displayed.

        Returns:
            None
        """
        fps = int(self.fps)
        self.logger.debug(f"Displaying FPS: {fps}")
        cv2.putText(
            img,
            f"FPS: {fps}",
            (10, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 0, 255),
            2,
        )

    def show_angle_arc(
        self,
        img: np.ndarray,
        point_a: tuple[float, float],
        point_b: tuple[float, float],
        point_c: tuple[float, float],
        angle: float,
        color: tuple = (255, 200, 200),  # Light blue in BGR
        thickness: int = 2,
    ) -> None:
        # Convert points to NumPy arrays
        a = np.array(point_a, dtype=np.float64)
        b = np.array(point_b, dtype=np.float64)
        c = np.array(point_c, dtype=np.float64)

        # Vectors from point_b to point_a and point_b to point_c
        ba = a - b
        bc = c - b

        # Calculate the angles of the vectors
        angle_ba = np.degrees(np.arctan2(ba[1], ba[0]))
        angle_bc = np.degrees(np.arctan2(bc[1], bc[0]))

        # Normalize angles to [0, 360)
        start_angle = (angle_ba + 360) % 360
        end_angle = (angle_bc + 360) % 360

        # Determine the direction to draw the arc
        if end_angle < start_angle:
            end_angle += 360

        # Compute the arc span and adjust if necessary
        arc_span = end_angle - start_angle
        if arc_span > 180:
            start_angle, end_angle = end_angle, start_angle
            start_angle -= 360  # Adjust for OpenCV's ellipse function

        # Set the radius of the arc (smaller radius)
        radius = int(max(img.shape[0], img.shape[1]) * 0.04)

        # Draw the arc
        center = (int(b[0]), int(b[1]))
        axes = (radius, radius)
        cv2.ellipse(
            img,
            center,
            axes,
            0,  # No rotation of the ellipse
            start_angle,
            end_angle,
            color,
            thickness,
            cv2.LINE_AA,
        )

        # Display the angle value near the arc
        text_offset = radius + 5
        text_position = (center[0] + text_offset, center[1] - text_offset)
        cv2.putText(
            img,
            f"{int(angle)} deg",
            text_position,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,  # Smaller font size
            color,
            1,  # Thinner text
            cv2.LINE_AA,
        )
