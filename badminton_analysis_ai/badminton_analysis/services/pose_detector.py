import os
import time
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np
import torch
import torchvision.transforms.v2.functional as tv_functional
from cv2.typing import MatLike
from numpy.typing import NDArray

from badminton_analysis.core.logger import Logger
from badminton_analysis.models.types import (
    COCOKeypoints,
    CoordinateDict,
    WholeBodyCoordinateDict,
)

PosePrediction = dict[str, Any]

# RF-DETR's COCO class id for "person" (COCO_CLASSES[1] == "person").
_PERSON_CLASS_ID = 1

# COCO-WholeBody-133 total column count that `wrist_flick.py` and
# `WholeBodyCoordinateDict` expect. RFDETRKeypointPreview only predicts the
# 17 COCO body joints (no hands/face/feet), so only slots 0-16 are ever
# populated here; the rest stay zero-confidence, which the wrist-flick
# evidence gathering already treats as "no hand detail available" and
# gracefully falls back to a coarser body-only distance metric for.
_WHOLEBODY_KEYPOINTS = 133

# Fixed batch size the cached TensorRT engine is built for. Callers that hold
# every frame already -- offline extraction, and the analysis service, which is
# handed a complete upload -- pad/chunk to this size; the interactive
# frame-at-a-time path stays on the unbatched, non-TensorRT get_pose().
BATCH_SIZE = 16

# Elbows are frequently self-occluded during a badminton swing. RF-DETR still
# tracks them coherently below the general body-joint cutoff, so accept these
# two joints at a lower confidence instead of synthesizing their coordinates
# later in the renderer.
_ELBOW_KEYPOINT_INDICES = frozenset(
    (int(COCOKeypoints.LEFT_ELBOW), int(COCOKeypoints.RIGHT_ELBOW))
)

_TRT_CACHE_ROOT = Path(
    os.getenv("BADMINTON_TRT_CACHE_DIR", str(Path.home() / ".cache" / "badminton_analysis" / "trt_engines"))
)


class PoseDetector:
    """Single-stage 2D pose detector: RF-DETR Keypoint Preview.

    RFDETRKeypointPreview predicts person detection and 17 COCO-order body
    keypoints in one forward pass, in the same index order as this repo's
    `COCOKeypoints` enum, so no schema adapter is needed (unlike the prior
    RF-DETR-detect + Sapiens2-pose two-stage pipeline, which needed one for
    Sapiens2's differently-ordered 308-keypoint output). It has no hand
    keypoints, so `wholebody_keypoints`/`wholebody_scores` only ever carry
    real data in their first 17 slots.
    """

    def __init__(
        self,
        # General body joints remain conservative enough to exclude near-zero
        # noise. Elbows have a separate lower cutoff because self-occlusion
        # during a swing depresses their confidence more than other joints.
        min_detection_confidence: float = 0.15,
        elbow_detection_confidence: float = 0.05,
        person_detection_threshold: float = 0.5,
    ):
        if not 0.0 <= elbow_detection_confidence <= min_detection_confidence:
            raise ValueError(
                "elbow_detection_confidence must be between zero and "
                "min_detection_confidence"
            )
        self.logger = Logger(self.__class__.__name__)
        self.min_detection_confidence = min_detection_confidence
        self.elbow_detection_confidence = elbow_detection_confidence
        self.person_detection_threshold = person_detection_threshold
        self.device = (
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )
        self.logger.info(f"{self.device} is used")

        self._model: Any | None = None
        self._batched_engine: Any | None = None

        self.__cur_time: float = 0.0
        self.__prev_time: float = 0.0
        self._target_bbox_center: NDArray[np.float64] | None = None
        self._last_predictions: list[PosePrediction] = []
        # Diagnostic only; nothing in this repo reads this attribute today.
        self.active_execution_providers: dict[str, tuple[str, ...]] = {
            "pose": ("torch", self.device),
        }

    def keypoint_detection_threshold(self, index: int) -> float:
        """Return the acceptance threshold for one COCO body joint.

        Both the native PyTorch/MPS path and TensorRT path produce the same
        ``PosePrediction`` structure, so applying this threshold while reading
        predictions keeps their joint filtering identical.
        """
        if int(index) in _ELBOW_KEYPOINT_INDICES:
            return self.elbow_detection_confidence
        return self.min_detection_confidence

    # Added for this deployment: the analysis response reports which provider
    # served a request, and the release gate asserts TensorRT was actually
    # used. Kept as properties so re-syncing this file from the analysis repo
    # shows up as a conflict here rather than silently dropping the signal.
    @property
    def execution_provider(self) -> str:
        return "tensorrt" if self._batched_engine is not None else "torch"

    @property
    def tensorrt_active(self) -> bool:
        return self._batched_engine is not None

    def _load_inferencer(self) -> None:
        try:
            from rfdetr import RFDETRKeypointPreview
        except ImportError as exc:
            raise RuntimeError(
                "rfdetr is required for person detection and pose estimation"
            ) from exc
        self._model = RFDETRKeypointPreview(device=self.device)
        if (
            self.device == "mps"
            and os.getenv("BADMINTON_RFDETR_MPS_FP16", "0") == "1"
        ):
            # Match the production TensorRT engine's FP16 precision while
            # avoiding a second full-precision model copy on unified memory.
            # Compilation is deliberately disabled because audit batches have
            # a variable-size final chunk.
            self._model.inference(
                compile=False,
                dtype=torch.float16,
                inplace=True,
            )

    def _load_or_build_batched_engine(self) -> None:
        """Load the cached fixed-batch TensorRT engine, building it once per
        GPU on first use (~2 minutes) and reusing the cached `.trt` file on
        every run after that. TensorRT engines are tied to the exact GPU/
        driver/TensorRT version they were built on, so the cache is keyed by
        GPU name to avoid loading an incompatible engine on different
        hardware.
        """
        if self._model is None:
            self._load_inferencer()
        if self._batched_engine is not None:
            return
        from badminton_analysis.services.trt_engine import TorchTRTEngine

        gpu_name = (
            torch.cuda.get_device_name(0).replace(" ", "_")
            if torch.cuda.is_available()
            else "cpu"
        )
        cache_dir = _TRT_CACHE_ROOT / gpu_name / f"batch{BATCH_SIZE}"
        engine_path = cache_dir / "rfdetr-keypoint-preview.trt"
        if not engine_path.exists():
            self.logger.info(
                f"Building TensorRT engine (batch={BATCH_SIZE}); this happens "
                f"once per GPU and is cached at {engine_path}"
            )
            cache_dir.mkdir(parents=True, exist_ok=True)
            self._model.export(
                output_dir=str(cache_dir),
                format="tensorrt",
                fp16=True,
                batch_size=BATCH_SIZE,
                dynamic_batch=False,
                verbose=False,
            )
        self._batched_engine = TorchTRTEngine(str(engine_path))

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
        self._last_predictions = []

    @staticmethod
    def _build_prediction(
        bbox: tuple[float, float, float, float],
        keypoints: NDArray[np.float64],
        scores: NDArray[np.float64],
    ) -> PosePrediction:
        wholebody_keypoints = np.zeros((_WHOLEBODY_KEYPOINTS, 2), dtype=np.float64)
        wholebody_scores = np.zeros(_WHOLEBODY_KEYPOINTS, dtype=np.float64)
        wholebody_keypoints[:17] = keypoints
        wholebody_scores[:17] = scores
        return {
            "bbox": list(bbox),
            "keypoints": keypoints,
            "keypoint_scores": scores,
            "wholebody_keypoints": wholebody_keypoints,
            "wholebody_scores": wholebody_scores,
        }

    def get_pose(self, img: MatLike) -> list[PosePrediction]:
        """Detect the largest person and their pose in one frame."""
        if self._model is None:
            self._load_inferencer()

        image_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        result = self._model.predict(
            image_rgb,
            threshold=self.person_detection_threshold,
            include_source_image=False,
        )
        predictions = self._largest_person_prediction(result)
        if not predictions:
            self._last_predictions = []
            return []
        prediction = predictions[0]
        x1, y1, x2, y2 = prediction["bbox"]
        self._last_predictions = predictions
        self._target_bbox_center = np.array(
            ((x1 + x2) / 2.0, (y1 + y2) / 2.0), dtype=np.float64
        )
        return self._last_predictions

    def _largest_person_prediction(self, result: Any) -> list[PosePrediction]:
        """Convert one RF-DETR result into this repository's selected person."""
        is_person = result.class_id == _PERSON_CLASS_ID
        if not np.any(is_person):
            return []

        boxes = result.data["xyxy"][is_person]
        areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        local_best = int(np.argmax(areas))
        best = np.where(is_person)[0][local_best]

        x1, y1, x2, y2 = (float(value) for value in boxes[local_best])
        coco17_keypoints = np.asarray(result.xy[best], dtype=np.float64)
        coco17_scores = np.asarray(result.keypoint_confidence[best], dtype=np.float64)

        prediction = self._build_prediction(
            (x1, y1, x2, y2), coco17_keypoints, coco17_scores
        )
        return [prediction]

    def get_poses_batch(
        self, images: list[MatLike]
    ) -> list[list[PosePrediction]]:
        """Detect the largest person and their pose across a batch of frames.

        Uses the cached fixed-batch TensorRT engine (`BATCH_SIZE` frames per
        call, ~4ms/frame) instead of the single-frame PyTorch path `get_pose`
        uses (~50ms/frame): only the offline extraction pipeline calls this,
        since it can buffer frames ahead of time, unlike live grading. Does
        not touch `_last_predictions`/`_target_bbox_center`, since those exist
        for the single-frame streaming API's own state tracking.

        A short final chunk is padded internally with a repeated last frame
        to satisfy the engine's fixed batch shape, then the padding is
        truncated back off before returning.
        """
        if not images:
            return []
        if len(images) > BATCH_SIZE:
            raise ValueError(
                f"batch of {len(images)} frames exceeds the fixed engine "
                f"batch size of {BATCH_SIZE}"
            )
        if self.device != "cuda":
            if self._model is None:
                self._load_inferencer()
            image_rgbs = [cv2.cvtColor(image, cv2.COLOR_BGR2RGB) for image in images]
            results = self._model.predict(
                image_rgbs,
                threshold=self.person_detection_threshold,
                include_source_image=False,
            )
            return [self._largest_person_prediction(result) for result in results]
        self._load_or_build_batched_engine()
        ctx = self._model.model

        padded = list(images) + [images[-1]] * (BATCH_SIZE - len(images))
        resolution = ctx.resolution
        tensors = []
        heights = []
        widths = []
        for img in padded:
            image_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            heights.append(image_rgb.shape[0])
            widths.append(image_rgb.shape[1])
            tensor = (
                torch.from_numpy(image_rgb).permute(2, 0, 1).float().to(ctx.device)
                / 255.0
            )
            tensors.append(tensor)
        batch_tensor = torch.stack(
            [
                tv_functional.resize(t, [resolution, resolution], antialias=False)
                for t in tensors
            ]
        )
        batch_tensor = tv_functional.normalize(
            batch_tensor, self._model.means, self._model.stds
        )

        assert self._batched_engine is not None
        trt_out = self._batched_engine({"input": batch_tensor})
        raw_predictions = {
            "pred_logits": trt_out["labels"],
            "pred_boxes": trt_out["dets"],
            "pred_keypoints": trt_out["keypoints"],
        }
        target_sizes = torch.tensor(
            [[h, w] for h, w in zip(heights, widths)],
            device=ctx.device,
            dtype=torch.float32,
        )
        results = ctx.postprocess(
            raw_predictions,
            target_sizes=target_sizes,
            score_threshold=self.person_detection_threshold,
        )

        batch_predictions: list[list[PosePrediction]] = []
        for index in range(len(images)):
            result = results[index]
            scores = result["scores"].detach().cpu().numpy()
            is_valid = scores > self.person_detection_threshold
            if not np.any(is_valid):
                batch_predictions.append([])
                continue

            boxes = result["boxes"].detach().cpu().numpy()[is_valid]
            keypoints = result["keypoints"].detach().cpu().numpy()[is_valid]
            areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
            best = int(np.argmax(areas))
            x1, y1, x2, y2 = (float(value) for value in boxes[best])
            coco17_keypoints = keypoints[best, :, :2].astype(np.float64)
            coco17_scores = keypoints[best, :, 2].astype(np.float64)

            prediction = self._build_prediction(
                (x1, y1, x2, y2), coco17_keypoints, coco17_scores
            )
            batch_predictions.append([prediction])
        return batch_predictions

    def get_2d_landmarks(
        self, results: list[PosePrediction] | None = None
    ) -> CoordinateDict | None:
        predictions = results if results is not None else self._last_predictions
        if not predictions:
            return None
        target = predictions[0]
        keypoints = np.asarray(target["keypoints"], dtype=np.float64)
        scores = np.asarray(target["keypoint_scores"], dtype=np.float64)

        body_coords: CoordinateDict = {}
        for i in range(len(keypoints)):
            if scores[i] <= self.keypoint_detection_threshold(i):
                continue
            body_coords[COCOKeypoints(i)] = keypoints[i]
        return body_coords or None

    def get_wholebody_2d_landmarks(self) -> WholeBodyCoordinateDict | None:
        if not self._last_predictions:
            return None
        target = self._last_predictions[0]
        keypoints = np.asarray(target["wholebody_keypoints"], dtype=np.float64)
        scores = np.asarray(target["wholebody_scores"], dtype=np.float64)

        coords: WholeBodyCoordinateDict = {}
        for i in range(len(keypoints)):
            if scores[i] <= self.keypoint_detection_threshold(i):
                continue
            coords[i] = keypoints[i]
        return coords or None

    def get_wholebody_2d_keypoints(
        self,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]] | None:
        """Return the selected person's aligned WholeBody coordinates and scores."""
        if not self._last_predictions:
            return None
        target = self._last_predictions[0]
        keypoints = np.asarray(target["wholebody_keypoints"], dtype=np.float64)
        scores = np.asarray(target["wholebody_scores"], dtype=np.float64)
        return keypoints, np.clip(scores, 0.0, 1.0)
