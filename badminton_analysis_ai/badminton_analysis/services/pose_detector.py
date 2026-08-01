import time
from typing import Any, Optional, cast

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

_H36M_TO_COCO_BODY = {
    0: 9,  # nose/head proxy
    1: 10,  # left eye/head proxy
    2: 10,  # right eye/head proxy
    3: 10,  # left ear/head proxy
    4: 10,  # right ear/head proxy
    5: 11,  # left shoulder
    6: 14,  # right shoulder
    7: 12,  # left elbow
    8: 15,  # right elbow
    9: 13,  # left wrist
    10: 16,  # right wrist
    11: 4,  # left hip
    12: 1,  # right hip
    13: 5,  # left knee
    14: 2,  # right knee
    15: 6,  # left ankle
    16: 3,  # right ankle
}


class PoseDetector:
    def __init__(
        self,
        model_path: str = "image-pose-lift_tcn_8xb64-200e_h36m",
        min_detection_confidence: float = 0.5,
        pose2d_model: str = "rtmpose-m_8xb256-420e_coco-256x192",
        wholebody_model: str = "rtmw-m_8xb1024-270e_cocktail14-256x192",
    ):
        self.logger = Logger(self.__class__.__name__)
        self.min_detection_confidence = min_detection_confidence
        self.landmarks: CoordinateDict = {}

        # MMPose supports CUDA and CPU reliably. Keep MPS out of the default
        # path because the OpenMMLab stack commonly lacks MPS kernels.
        if torch.cuda.is_available():
            self.device = "cuda:0"
        else:
            self.device = "cpu"

        self.logger.info(f"{self.device} is used")
        self.model_name = model_path
        self.pose2d_model = pose2d_model
        self.wholebody_model_name = wholebody_model
        self.model: Any | None = None
        self.wholebody_model: Any | None = None

        self.__cur_time: float = 0.0
        self.__prev_time: float = 0.0
        self._target_bbox_center: NDArray[np.float64] | None = None
        self._last_frame_shape: tuple[int, ...] | None = None
        self._last_wholebody_predictions: list[Pose3DPrediction] = []

    def _load_inferencer(self, model_path: str) -> Any:
        try:
            from mmpose.apis import MMPoseInferencer  # type: ignore[import-not-found]
        except ImportError as exc:
            raise RuntimeError(
                "MMPose is required for the 3D pose backend. Install mmpose, "
                "mmcv, mmdet, and mmengine in the active environment."
            ) from exc

        return MMPoseInferencer(
            pose2d=self.pose2d_model,
            pose3d=model_path,
            device=self.device,
            show_progress=False,
        )

    def _load_2d_inferencer(self, model_path: str) -> Any:
        try:
            from mmpose.apis import MMPoseInferencer
        except ImportError as exc:
            raise RuntimeError(
                "MMPose is required for the whole-body 2D backend. Install mmpose, "
                "mmcv, mmdet, and mmengine in the active environment."
            ) from exc

        return MMPoseInferencer(
            pose2d=model_path,
            device=self.device,
            show_progress=False,
        )

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

    def get_pose(self, img: MatLike) -> list[Pose3DPrediction]:
        if self.model is None:
            self.model = self._load_inferencer(self.model_name)
        if self.wholebody_model is None:
            self.wholebody_model = self._load_2d_inferencer(self.wholebody_model_name)

        self._last_frame_shape = tuple(img.shape)
        wholebody_result = next(
            self.wholebody_model(
                img,
                return_datasamples=False,
                show=False,
                draw_bbox=False,
            )
        )
        self._last_wholebody_predictions = self._flatten_predictions(
            wholebody_result.get("predictions", [])
        )
        return self._lift_wholebody_predictions_to_3d(self._last_wholebody_predictions)

    def _lift_wholebody_predictions_to_3d(
        self, predictions: list[Pose3DPrediction]
    ) -> list[Pose3DPrediction]:
        if not predictions:
            return []

        try:
            from mmengine.structures import InstanceData  # type: ignore[import-untyped]
            from mmpose.apis import (
                convert_keypoint_definition,
                inference_pose_lifter_model,
            )
            from mmpose.structures import PoseDataSample  # type: ignore[import-not-found]
        except ImportError as exc:
            raise RuntimeError(
                "MMPose is required for lifting RTMW 2D keypoints to 3D."
            ) from exc

        lifter_model = self._pose_lifter_model()
        pose_lift_dataset = lifter_model.dataset_meta.get("dataset_name", "h36m")
        pose_samples: list[Any] = []
        source_predictions: list[Pose3DPrediction] = []

        for track_id, prediction in enumerate(predictions):
            keypoints = self._as_keypoint_array(prediction.get("keypoints"))
            if keypoints is None or len(keypoints) < _COCO_BODY_KEYPOINT_COUNT:
                continue

            scores = self._as_score_array(
                prediction.get("keypoint_scores"), len(keypoints)
            )
            body_keypoints = keypoints[:_COCO_BODY_KEYPOINT_COUNT, :2].astype(
                np.float32
            )
            body_scores = scores[:_COCO_BODY_KEYPOINT_COUNT].astype(np.float32)
            converted_keypoints = convert_keypoint_definition(
                body_keypoints[None, :, :],
                pose_det_dataset="coco",
                pose_lift_dataset=pose_lift_dataset,
            )
            converted_scores = convert_keypoint_definition(
                body_scores[None, :, None],
                pose_det_dataset="coco",
                pose_lift_dataset=pose_lift_dataset,
            ).squeeze(-1)

            bbox = self._bbox_from_prediction(prediction, keypoints)
            if bbox is None:
                bbox = self._bbox_from_prediction(prediction, body_keypoints)
            if bbox is None:
                continue

            pred_instances = InstanceData()
            pred_instances.keypoints = converted_keypoints.astype(np.float32)
            pred_instances.keypoint_scores = converted_scores.astype(np.float32)
            pred_instances.bboxes = np.asarray(bbox, dtype=np.float32).reshape(1, 4)
            pred_instances.areas = (
                pred_instances.bboxes[..., 2:] - pred_instances.bboxes[..., :2]
            ).prod(-1)

            data_sample = PoseDataSample()
            data_sample.pred_instances = pred_instances
            data_sample.gt_instances = InstanceData()
            data_sample.set_field(track_id, "track_id")
            pose_samples.append(data_sample)
            source_predictions.append(prediction)

        if not pose_samples:
            return []

        image_size = None
        if self._last_frame_shape is not None and len(self._last_frame_shape) >= 2:
            height, width = self._last_frame_shape[:2]
            image_size = (width, height)

        lift_results = inference_pose_lifter_model(
            lifter_model,
            [pose_samples],
            with_track_id=True,
            image_size=image_size,
            norm_pose_2d=True,
        )
        return self._pose_lift_results_to_predictions(lift_results, source_predictions)

    def _pose_lifter_model(self) -> Any:
        inferencer = getattr(self.model, "inferencer", self.model)
        lifter_model = getattr(inferencer, "model", None)
        if lifter_model is None:
            raise RuntimeError(
                "The configured MMPose inferencer has no pose lifter model."
            )
        return lifter_model

    def _pose_lift_results_to_predictions(
        self,
        lift_results: list[Any],
        source_predictions: list[Pose3DPrediction],
    ) -> list[Pose3DPrediction]:
        predictions: list[Pose3DPrediction] = []
        for result, source in zip(lift_results, source_predictions):
            keypoints = np.asarray(result.pred_instances.keypoints, dtype=np.float64)
            scores = np.asarray(result.pred_instances.keypoint_scores, dtype=np.float64)
            if keypoints.ndim == 4:
                keypoints = np.squeeze(keypoints, axis=1)
            if keypoints.ndim == 3 and keypoints.shape[0] == 1:
                keypoints = keypoints[0]
            if scores.ndim == 3:
                scores = np.squeeze(scores, axis=1)
            if scores.ndim == 2 and scores.shape[0] == 1:
                scores = scores[0]

            keypoints = keypoints[..., [0, 2, 1]]
            keypoints[..., 0] = -keypoints[..., 0]
            keypoints[..., 2] = -keypoints[..., 2]
            keypoints[..., 2] -= np.min(keypoints[..., 2], axis=-1, keepdims=True)

            predictions.append(
                {
                    "keypoints": self._h36m_keypoints_to_coco(keypoints).tolist(),
                    "keypoint_scores": self._h36m_scores_to_coco(scores).tolist(),
                    "bbox": source.get("bbox"),
                }
            )
        return predictions

    def _h36m_keypoints_to_coco(
        self, keypoints: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        if keypoints.ndim != 2 or keypoints.shape[0] != 17:
            return keypoints
        coco_keypoints = np.zeros((17, keypoints.shape[1]), dtype=np.float64)
        for coco_idx, h36m_idx in _H36M_TO_COCO_BODY.items():
            coco_keypoints[coco_idx] = keypoints[h36m_idx]
        return coco_keypoints

    def _h36m_scores_to_coco(self, scores: NDArray[np.float64]) -> NDArray[np.float64]:
        if scores.ndim != 1 or scores.shape[0] != 17:
            return scores
        coco_scores = np.zeros(17, dtype=np.float64)
        for coco_idx, h36m_idx in _H36M_TO_COCO_BODY.items():
            coco_scores[coco_idx] = scores[h36m_idx]
        return coco_scores

    def _flatten_predictions(self, value: Any) -> list[Pose3DPrediction]:
        predictions = cast(list[Any], value)
        if len(predictions) == 1 and isinstance(predictions[0], list):
            predictions = predictions[0]
        return [pred for pred in predictions if isinstance(pred, dict)]

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
