import os
import time
from typing import Any, Optional, final

import cv2
import numpy as np
from cv2.typing import MatLike
from ultralytics import YOLO
import torch

from Logger import Logger
from Types import Body2DCoordinates, COCOKeypoints
from PIL import Image, ImageDraw, ImageFont
from Joints import SKELETON_CONNECTIONS


@final
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
        model_path: str = "yolo11m-pose.pt",
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

        # Select compute device: prefer CUDA, then MPS (Apple Silicon), else CPU
        if torch.cuda.is_available():
            self.device = "cuda"
        elif torch.mps.is_available():
            self.device = "mps"
        else:
            self.device = "cpu"

        # Move model to the selected device
        try:
            self.model.to(self.device)
        except Exception:
            # In case Ultralytics handles device internally, keep a safe fallback
            self.device = "cpu"

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
        # Run pose estimation on the selected device
        results = self.model.predict(
            img,
            conf=self.min_detection_confidence,
            device=self.device,
        )
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
        try:
            # Validate results and keypoints container
            if not results or len(results) == 0:
                self.logger.debug("No results returned from model")
                return None

            keypoints = getattr(results[0], "keypoints", None)
            if keypoints is None:
                self.logger.debug("No keypoints in the first result")
                return None

            # Extract XY coordinates for the first detection
            kpt_xy = getattr(keypoints, "xy", None)
            if kpt_xy is None or len(kpt_xy) == 0:
                self.logger.debug("Keypoints.xy missing or empty")
                return None

            xy = kpt_xy[0]
            keypoints_xy = xy.cpu().numpy() if hasattr(xy, "cpu") else xy

            # Confidence scores are optional; treat as 1.0 if absent
            kpt_conf = getattr(keypoints, "conf", None)
            keypoints_conf = None
            if kpt_conf is not None and len(kpt_conf) > 0 and kpt_conf[0] is not None:
                conf0 = kpt_conf[0]
                keypoints_conf = conf0.cpu().numpy() if hasattr(conf0, "cpu") else conf0

            body_coordinates: Body2DCoordinates = {}
            for idx, (x, y) in enumerate(keypoints_xy):
                # If confidence not provided, assume 1.0
                conf_val = 1.0
                if keypoints_conf is not None and idx < len(keypoints_conf):
                    conf_val = float(keypoints_conf[idx])

                if conf_val > self.min_detection_confidence:
                    body_coordinates[COCOKeypoints(idx)] = (float(x), float(y))

            if body_coordinates:
                self.logger.debug("Retrieved pose landmarks")
                return body_coordinates
            else:
                self.logger.debug("No landmarks above confidence threshold")
                return None
        except Exception as e:
            self.logger.error(f"Error parsing keypoints: {e}")
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

    def show_pose(self, img: MatLike, landmarks: Optional[Body2DCoordinates]) -> None:
        """
        Draws only the pose skeleton (landmarks and connections) on the given image.

        Args:
            img (MatLike): The input image on which the landmarks are to be drawn.
            landmarks (Optional[Dict]): A dictionary of keypoints with COCOKeypoints as keys
                                        and (x, y) coordinates as values.

        Returns:
            None
        """
        if landmarks:
            self.logger.debug("Drawing pose skeleton on the image")

            # Draw each connection in the skeleton
            for start, end in SKELETON_CONNECTIONS:
                if start in landmarks and end in landmarks:
                    x1, y1 = landmarks[start]
                    x2, y2 = landmarks[end]
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
            for keypoint, (x, y) in landmarks.items():
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
        color: tuple = (249, 210, 60),  # Light blue in BGR
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
        radius = int(max(img.shape[0], img.shape[1]) * 0.01)

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
        self.__add_text_with_pillow(
            img,
            f"{int(angle)}°",  # Unicode degree symbol
            (center[0] + radius + 5, center[1] - radius - 5),
            font_size=20,
            color=(color[2], color[1], color[0]),  # Convert BGR to RGB
        )

    # Helper moethod that adds text to an image using Pillow (private method)
    def __add_text_with_pillow(
        self, img, text, position, font_size=20, color=(255, 255, 255)
    ):
        """
        Add text with Pillow onto an OpenCV image.

        :param img: OpenCV image (numpy array).
        :param text: The text to render (supports Unicode, e.g., "45°").
        :param position: Tuple (x, y) specifying the position of the text.
        :param font_size: Size of the font.
        :param color: Text color as a tuple (R, G, B).
        """
        # Convert OpenCV image to PIL image
        pil_image = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil_image)

        # Load the font
        font = ImageFont.load_default(size=font_size)

        # Add text to the image
        draw.text(position, text, font=font, fill=color)

        # Convert the PIL image back to OpenCV format
        np.copyto(img, cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR))
