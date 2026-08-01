import os
import time
from typing import Any, Optional

import cv2
import numpy as np
import torch
from cv2.typing import MatLike
from numpy.typing import NDArray
from PIL import Image, ImageDraw, ImageFont

from badminton_analysis.core.logger import Logger
from badminton_analysis.models.types import (
    COCOKeypoints,
    Coordinate2DDict,
    CoordinateDict,
    WholeBodyCoordinateDict,
)
from badminton_analysis.models.joints import JOINTS, SKELETON_CONNECTIONS

Pose3DPrediction = dict[str, Any]
_COCO_BODY_KEYPOINT_COUNT = 17

_DEFAULT_DETECTOR_MODEL = (
    "/app/models/pose/yolox_m_8xb8-300e_humanart-c2c7a14a.onnx"
)
_DEFAULT_POSE_MODEL = (
    "/app/models/pose/rtmw3d-x_8xb64_cocktail14-384x288-b0a0eab7_20240626.onnx"
)
_DEFAULT_TENSORRT_CACHE = "/app/models/tensorrt-cache"
_DEFAULT_CAMERA_FOCAL_LENGTH = np.array(
    (1145.04940459, 1143.78109572), dtype=np.float64
)
_DEFAULT_ROOT_DEPTH = 5.14388


class PoseDetector:
    def __init__(
        self,
        model_path: str | None = None,
        min_detection_confidence: float = 0.5,
        detector_model: str | None = None,
        backend: str | None = None,
        detection_frequency: int = 7,
    ):
        self.logger = Logger(self.__class__.__name__)
        self.min_detection_confidence = min_detection_confidence
        self.landmarks: CoordinateDict = {}
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.logger.info(f"{self.device} is used")
        self.model_name = model_path or os.getenv(
            "RTMW3D_POSE_MODEL", _DEFAULT_POSE_MODEL
        )
        self.detector_model_name = detector_model or os.getenv(
            "RTMW3D_DETECTOR_MODEL", _DEFAULT_DETECTOR_MODEL
        )
        self.backend = backend or os.getenv("POSE_INFERENCE_BACKEND", "onnxruntime")
        self.execution_provider = os.getenv(
            "POSE_EXECUTION_PROVIDER", "tensorrt" if self.device == "cuda" else "cpu"
        ).lower()
        self.onnx_device_id = int(os.getenv("ONNXRUNTIME_DEVICE_ID", "0"))
        if detection_frequency < 1:
            raise ValueError("detection_frequency must be positive")
        self.detection_frequency = detection_frequency
        self.model: Any | None = None

        self.__cur_time: float = 0.0
        self.__prev_time: float = 0.0
        self._target_bbox_center: NDArray[np.float64] | None = None
        self._last_frame_shape: tuple[int, ...] | None = None
        self._last_wholebody_predictions: list[Pose3DPrediction] = []
        self._last_bboxes: list[list[float]] = []
        self._frame_index = 0
        self.active_execution_providers: dict[str, tuple[str, ...]] = {}

    def _configure_execution_providers(self, model: Any) -> None:
        if self.backend != "onnxruntime" or self.device != "cuda":
            return

        import onnxruntime as ort  # type: ignore[import-not-found]

        available = set(ort.get_available_providers())
        if self.execution_provider == "tensorrt":
            if "TensorrtExecutionProvider" not in available:
                raise RuntimeError(
                    "POSE_EXECUTION_PROVIDER=tensorrt, but ONNX Runtime does not "
                    "provide TensorrtExecutionProvider"
                )
            cache_root = os.path.abspath(
                os.getenv("POSE_TENSORRT_CACHE_DIR", _DEFAULT_TENSORRT_CACHE)
            )
            hardware_compatible = os.getenv(
                "TENSORRT_ENGINE_HW_COMPATIBLE", "true"
            ).lower() == "true"
            for name, component in (
                ("detector", model.det_model),
                ("pose", model.pose_model),
            ):
                cache_path = os.path.join(cache_root, name)
                os.makedirs(cache_path, exist_ok=True)
                options = {
                    "device_id": self.onnx_device_id,
                    "trt_engine_cache_enable": True,
                    "trt_engine_cache_path": cache_path,
                    "trt_engine_cache_prefix": f"rtmw3d_{name}",
                    "trt_fp16_enable": True,
                    "trt_engine_hw_compatible": hardware_compatible,
                    "trt_timing_cache_enable": True,
                    "trt_op_types_to_exclude": (
                        "TopK,NonMaxSuppression,NonZero,RoiAlign"
                    ),
                }
                component.session.set_providers(
                    [
                        ("TensorrtExecutionProvider", options),
                        (
                            "CUDAExecutionProvider",
                            {"device_id": self.onnx_device_id},
                        ),
                        "CPUExecutionProvider",
                    ]
                )
                active = tuple(component.session.get_providers())
                if not active or active[0] != "TensorrtExecutionProvider":
                    raise RuntimeError(
                        f"TensorRT did not activate for RTMW3D {name}: {active}"
                    )
                self.active_execution_providers[name] = active
            return

        if self.execution_provider != "cuda":
            raise ValueError(
                "POSE_EXECUTION_PROVIDER must be tensorrt, cuda, or cpu"
            )
        if "CUDAExecutionProvider" not in available:
            raise RuntimeError(
                "POSE_EXECUTION_PROVIDER=cuda, but ONNX Runtime does not provide "
                "CUDAExecutionProvider"
            )
        for name, component in (
            ("detector", model.det_model),
            ("pose", model.pose_model),
        ):
            component.session.set_providers(
                [
                    (
                        "CUDAExecutionProvider",
                        {"device_id": self.onnx_device_id},
                    ),
                    "CPUExecutionProvider",
                ]
            )
            self.active_execution_providers[name] = tuple(
                component.session.get_providers()
            )

    def _load_inferencer(self) -> Any:
        try:
            from rtmlib import Wholebody3d  # type: ignore[import-not-found]
        except ImportError as exc:
            raise RuntimeError(
                "rtmlib and onnxruntime-gpu are required for RTMW3D inference"
            ) from exc

        model = Wholebody3d(
            det=self.detector_model_name,
            det_input_size=(640, 640),
            pose=self.model_name,
            pose_input_size=(288, 384),
            backend=self.backend,
            device=self.device,
        )
        self._configure_execution_providers(model)
        return model

    @staticmethod
    def _camera_coordinates(
        keypoints_3d: NDArray[np.float64],
        keypoints_2d: NDArray[np.float64],
        image_shape: tuple[int, ...],
    ) -> NDArray[np.float64]:
        """Match the camera-space conversion used by the RTMW3D demo."""
        height, width = image_shape[:2]
        center = np.array((width / 2.0, height / 2.0), dtype=np.float64)
        depth = keypoints_3d[:, 2] + _DEFAULT_ROOT_DEPTH
        camera = np.empty((len(keypoints_3d), 3), dtype=np.float64)
        camera[:, :2] = (
            (keypoints_2d[:, :2] - center)
            / _DEFAULT_CAMERA_FOCAL_LENGTH
            * depth[:, None]
        )
        camera[:, 2] = depth

        converted = camera[:, [0, 2, 1]]
        converted[:, 0] = -converted[:, 0]
        converted[:, 2] = -converted[:, 2]
        converted[:, 2] -= np.min(converted[:, 2])
        return converted

    @property
    def fps(self) -> float:
        self.__cur_time = time.time()
        time_diff = self.__cur_time - self.__prev_time
        if time_diff == 0:
            time_diff = 1e-6
        cur_fps: float = 1.0 / time_diff
        self.__prev_time = self.__cur_time
        return cur_fps

    @staticmethod
    def compute_angle(
        point_a: NDArray[np.float64],
        point_b: NDArray[np.float64],
        point_c: NDArray[np.float64],
    ) -> Optional[float]:
        a = np.asarray(point_a, dtype=np.float64)
        b = np.asarray(point_b, dtype=np.float64)
        c = np.asarray(point_c, dtype=np.float64)
        if (
            a.ndim != 1
            or b.ndim != 1
            or c.ndim != 1
            or a.shape[0] < 2
            or a.shape != b.shape
            or a.shape != c.shape
        ):
            return None
        vector_ba = a - b
        vector_bc = c - b
        norm_ba = np.linalg.norm(vector_ba)
        norm_bc = np.linalg.norm(vector_bc)
        if norm_ba == 0 or norm_bc == 0:
            return None
        cos_theta = (vector_ba @ vector_bc) / (norm_ba * norm_bc)
        cos_theta = np.clip(cos_theta, -1, 1)
        angle_radian = np.arccos(cos_theta)
        return float(np.rad2deg(angle_radian))

    def reset_tracking(self) -> None:
        self._target_bbox_center = None
        self._last_wholebody_predictions = []
        self._last_bboxes = []
        self._frame_index = 0

    def get_pose(self, img: MatLike) -> list[Pose3DPrediction]:
        if self.model is None:
            self.model = self._load_inferencer()

        self._last_frame_shape = tuple(img.shape)
        if self._frame_index % self.detection_frequency == 0 or not self._last_bboxes:
            detected = self.model.det_model(img)
            self._last_bboxes = [
                np.asarray(value, dtype=np.float64).reshape(-1)[:4].tolist()
                for value in detected
                if np.asarray(value).size >= 4
            ]
        bboxes = self._last_bboxes or [[0.0, 0.0, img.shape[1], img.shape[0]]]
        keypoints_3d, scores, _, keypoints_2d = self.model.pose_model(
            img, bboxes=bboxes
        )
        self._frame_index += 1

        predictions_3d: list[Pose3DPrediction] = []
        predictions_2d: list[Pose3DPrediction] = []
        for raw_3d, raw_scores, raw_2d, bbox in zip(
            keypoints_3d, scores, keypoints_2d, bboxes, strict=False
        ):
            pose_3d = np.asarray(raw_3d, dtype=np.float64)
            pose_2d = np.asarray(raw_2d, dtype=np.float64)
            pose_scores = np.asarray(raw_scores, dtype=np.float64)
            if len(pose_3d) < _COCO_BODY_KEYPOINT_COUNT:
                continue
            camera_pose = self._camera_coordinates(pose_3d, pose_2d, tuple(img.shape))
            common = {
                "keypoint_scores": pose_scores.tolist(),
                "bbox": list(bbox),
            }
            predictions_3d.append(
                {**common, "keypoints": camera_pose[:_COCO_BODY_KEYPOINT_COUNT].tolist()}
            )
            predictions_2d.append({**common, "keypoints": pose_2d.tolist()})

        self._last_wholebody_predictions = predictions_2d
        return predictions_3d

    def get_3d_landmarks(
        self, results: list[Pose3DPrediction]
    ) -> CoordinateDict | None:
        if not results:
            return None

        target = self._select_target(results)
        if target is None:
            return None

        keypoints = self._as_keypoint_array(target.get("keypoints"))
        if keypoints is None:
            return None

        scores = self._as_score_array(target.get("keypoint_scores"), len(keypoints))
        body_coords: CoordinateDict = {}
        for i in range(min(_COCO_BODY_KEYPOINT_COUNT, len(keypoints))):
            if scores[i] <= self.min_detection_confidence:
                continue
            x, y, z = self._as_3d_coordinate(keypoints[i])
            body_coords[COCOKeypoints(i)] = np.array((x, y, z), dtype=np.float64)

        if not body_coords:
            return None

        bbox = self._bbox_from_prediction(target, keypoints)
        if bbox is not None:
            self._target_bbox_center = np.array(
                ((bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0),
                dtype=np.float64,
            )
        return body_coords

    def get_2d_landmarks(
        self, results: list[Pose3DPrediction] | None = None
    ) -> Coordinate2DDict | None:
        predictions = self._last_wholebody_predictions or (results or [])
        target = self._select_target(predictions)
        if target is None:
            return None

        keypoints = self._as_keypoint_array(target.get("keypoints"))
        if keypoints is None:
            return None

        scores = self._as_score_array(target.get("keypoint_scores"), len(keypoints))
        body_coords: Coordinate2DDict = {}
        for i in range(min(_COCO_BODY_KEYPOINT_COUNT, len(keypoints))):
            if scores[i] <= self.min_detection_confidence:
                continue
            x, y = keypoints[i, :2]
            body_coords[COCOKeypoints(i)] = np.array((x, y), dtype=np.float64)

        bbox = self._bbox_from_prediction(target, keypoints)
        if bbox is not None:
            self._target_bbox_center = np.array(
                ((bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0),
                dtype=np.float64,
            )
        return body_coords or None

    def get_wholebody_2d_landmarks(self) -> WholeBodyCoordinateDict | None:
        target = self._select_target(self._last_wholebody_predictions)
        if target is None:
            return None

        keypoints = self._as_keypoint_array(target.get("keypoints"))
        if keypoints is None:
            return None

        scores = self._as_score_array(target.get("keypoint_scores"), len(keypoints))
        coords: WholeBodyCoordinateDict = {}
        for i in range(len(keypoints)):
            if scores[i] <= self.min_detection_confidence:
                continue
            x, y = keypoints[i, :2]
            coords[i] = np.array((x, y), dtype=np.float64)
        return coords or None

    def _select_target(
        self, predictions: list[Pose3DPrediction]
    ) -> Pose3DPrediction | None:
        if not predictions:
            return None
        if len(predictions) == 1:
            return predictions[0]

        scored_predictions: list[tuple[float, Pose3DPrediction]] = []
        for prediction in predictions:
            keypoints = self._as_keypoint_array(prediction.get("keypoints"))
            bbox = self._bbox_from_prediction(prediction, keypoints)
            if bbox is None:
                continue

            width = max(0.0, float(bbox[2] - bbox[0]))
            height = max(0.0, float(bbox[3] - bbox[1]))
            scored_predictions.append((width * height, prediction))

        if not scored_predictions:
            return predictions[0]
        scored_predictions.sort(key=lambda item: item[0], reverse=True)
        return scored_predictions[0][1]

    def _as_keypoint_array(self, value: Any) -> NDArray[np.float64] | None:
        if value is None:
            return None
        keypoints = np.asarray(value, dtype=np.float64)
        if keypoints.ndim == 3 and keypoints.shape[0] == 1:
            keypoints = keypoints[0]
        if keypoints.ndim != 2 or keypoints.shape[1] < 2:
            return None
        return keypoints

    def _as_score_array(self, value: Any, keypoint_count: int) -> NDArray[np.float64]:
        if value is None:
            return np.ones(keypoint_count, dtype=np.float64)
        scores = np.asarray(value, dtype=np.float64)
        if scores.ndim == 2 and scores.shape[0] == 1:
            scores = scores[0]
        if scores.ndim != 1 or len(scores) != keypoint_count:
            return np.ones(keypoint_count, dtype=np.float64)
        return scores

    def _as_3d_coordinate(self, value: NDArray[np.float64]) -> NDArray[np.float64]:
        coord = np.asarray(value, dtype=np.float64)
        if coord.shape[0] >= 3:
            return coord[:3]
        return np.array((coord[0], coord[1], 0.0), dtype=np.float64)

    def _bbox_from_prediction(
        self,
        prediction: Pose3DPrediction,
        keypoints: NDArray[np.floating[Any]] | None,
    ) -> NDArray[np.float64] | None:
        bbox_value = prediction.get("bbox")
        if bbox_value is not None:
            bbox = np.asarray(bbox_value, dtype=np.float64)
            if bbox.size >= 4:
                return np.asarray(bbox.reshape(-1, 4)[0], dtype=np.float64)

        if keypoints is None or len(keypoints) == 0:
            return None

        return np.array(
            (
                float(np.min(keypoints[:, 0])),
                float(np.min(keypoints[:, 1])),
                float(np.max(keypoints[:, 0])),
                float(np.max(keypoints[:, 1])),
            ),
            dtype=np.float64,
        )

    def show_pose(self, img: MatLike, landmarks: Optional[CoordinateDict]) -> None:
        if landmarks:
            for start, end in SKELETON_CONNECTIONS:
                if start in landmarks and end in landmarks:
                    c1 = landmarks[start]
                    c2 = landmarks[end]
                    x1, y1 = float(c1[0]), float(c1[1])
                    x2, y2 = float(c2[0]), float(c2[1])
                    if x1 >= 0 and y1 >= 0 and x2 >= 0 and y2 >= 0:
                        cv2.line(
                            img,
                            (int(x1), int(y1)),
                            (int(x2), int(y2)),
                            (255, 255, 255),
                            2,
                        )
            for _, coord in landmarks.items():
                x, y = float(coord[0]), float(coord[1])
                if x >= 0 and y >= 0:
                    cv2.circle(img, (int(x), int(y)), 3, (249, 210, 60), 1)

    def show_fps(self, img: MatLike) -> None:
        fps = int(self.fps)
        cv2.putText(
            img, f"FPS: {fps}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2
        )

    def show_angle_arc(
        self,
        img: MatLike,
        point_a: NDArray[np.float64],
        point_b: NDArray[np.float64],
        point_c: NDArray[np.float64],
        angle: float,
        color: tuple[int, int, int] = (249, 210, 60),
        thickness: int = 2,
    ) -> None:
        a = np.asarray(point_a, dtype=np.float64)
        b = np.asarray(point_b, dtype=np.float64)
        c = np.asarray(point_c, dtype=np.float64)
        if (
            a.ndim != 1
            or b.ndim != 1
            or c.ndim != 1
            or a.shape[0] < 2
            or b.shape[0] < 2
            or c.shape[0] < 2
        ):
            return None
        ba = a[:2] - b[:2]
        bc = c[:2] - b[:2]
        angle_ba = np.degrees(np.arctan2(ba[1], ba[0]))
        angle_bc = np.degrees(np.arctan2(bc[1], bc[0]))
        start_angle = (angle_ba + 360) % 360
        end_angle = (angle_bc + 360) % 360
        if end_angle < start_angle:
            end_angle += 360
        arc_span = end_angle - start_angle
        if arc_span > 180:
            start_angle, end_angle = end_angle, start_angle
            start_angle -= 360
        radius = int(max(img.shape[0], img.shape[1]) * 0.01)
        center = (int(b[0]), int(b[1]))
        axes = (radius, radius)
        cv2.ellipse(
            img, center, axes, 0, start_angle, end_angle, color, thickness, cv2.LINE_AA
        )
        self.__add_text_with_pillow(
            img,
            f"{int(angle)}°",
            (center[0] + radius + 5, center[1] - radius - 5),
            20,
            (color[2], color[1], color[0]),
        )

    def show_angles(self, frame: MatLike, landmarks: CoordinateDict) -> None:
        self.show_pose(frame, landmarks)
        orig_lm = landmarks
        for key, (point_a_id, point_b_id, point_c_id) in JOINTS.items():
            if key in (
                "Nose Right Shoulder Elbow Angle",
                "Nose Left Shoulder Elbow Angle",
            ):
                continue
            if all(kp in orig_lm for kp in (point_a_id, point_b_id, point_c_id)):
                point_a = orig_lm[point_a_id]
                point_b = orig_lm[point_b_id]
                point_c = orig_lm[point_c_id]
                angle = self.compute_angle(point_a, point_b, point_c)
                if angle is not None and isinstance(angle, float):
                    self.show_angle_arc(frame, point_a, point_b, point_c, angle)

    def __add_text_with_pillow(
        self,
        img: MatLike,
        text: str,
        position: tuple[int, int],
        font_size: int = 20,
        color: tuple[int, int, int] = (255, 255, 255),
    ) -> None:
        pil_image = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil_image)
        font = ImageFont.load_default()
        draw.text(position, text, font=font, fill=color)
        np.copyto(img, cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR))
