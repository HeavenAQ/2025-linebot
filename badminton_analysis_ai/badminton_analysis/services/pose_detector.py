import os
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import torchvision.transforms.v2.functional as tv_functional
from cv2.typing import MatLike
from numpy.typing import NDArray

from badminton_analysis.core.logger import Logger
from badminton_analysis.models.types import COCOKeypoints, CoordinateDict

PosePrediction = dict[str, Any]

# RF-DETR's COCO class id for "person" (COCO_CLASSES[1] == "person").
_PERSON_CLASS_ID = 1


def _largest_person_index(
    class_ids: NDArray[np.integer],
    boxes: NDArray[np.floating],
    scores: NDArray[np.floating] | None = None,
    threshold: float = 0.0,
) -> int | None:
    """Index of the biggest detected person, or None when there is no person.

    The CUDA TensorRT path reads raw postprocess tensors and the MPS/CPU
    ``predict()`` path reads a supervision result. Both must select the same
    person, or one video could grade differently by device, so the rule lives
    once here and both paths call it.
    """
    keep = class_ids == _PERSON_CLASS_ID
    if scores is not None:
        keep = keep & (scores > threshold)
    if not np.any(keep):
        return None
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    return int(np.argmax(np.where(keep, areas, -np.inf)))


# Fixed batch size the cached TensorRT engine is built for. Callers chunk
# whole videos to this size; a short final chunk is padded internally.
BATCH_SIZE = 16

# Elbows are frequently self-occluded during a badminton swing. RF-DETR still
# tracks them coherently below the general body-joint cutoff, so accept these
# two joints at a lower confidence instead of synthesizing their coordinates
# later in the renderer.
_ELBOW_KEYPOINT_INDICES = frozenset(
    (int(COCOKeypoints.LEFT_ELBOW), int(COCOKeypoints.RIGHT_ELBOW))
)

_TRT_CACHE_ROOT = Path(
    os.getenv(
        "BADMINTON_TRT_CACHE_DIR",
        str(Path.home() / ".cache" / "badminton_analysis" / "trt_engines"),
    )
)


class PoseDetector:
    """Single-stage 2D pose detector: RF-DETR Keypoint Preview.

    RFDETRKeypointPreview predicts person detection and 17 COCO-order body
    keypoints in one forward pass, in the same index order as this repo's
    `COCOKeypoints` enum, so no schema adapter is needed.
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
            else "mps" if torch.backends.mps.is_available() else "cpu"
        )
        self.logger.info(f"{self.device} is used")

        self._model: Any | None = None
        self._batched_engine: Any | None = None
        self._last_predictions: list[PosePrediction] = []

    def keypoint_detection_threshold(self, index: int) -> float:
        """Return the acceptance threshold for one COCO body joint.

        The TensorRT and ``predict()`` paths produce the same ``PosePrediction``
        structure, so applying this threshold while reading predictions keeps
        their joint filtering identical.
        """
        if int(index) in _ELBOW_KEYPOINT_INDICES:
            return self.elbow_detection_confidence
        return self.min_detection_confidence

    # The grade log names the pose backend, and the ``pose_tensorrt_active``
    # diagnostic reports whether the TensorRT engine served the request.
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
        if self.device == "mps" and os.getenv("BADMINTON_RFDETR_MPS_FP16", "0") == "1":
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

    def reset_tracking(self) -> None:
        self._last_predictions = []

    @staticmethod
    def _build_prediction(
        bbox: tuple[float, float, float, float],
        keypoints: NDArray[np.float64],
        scores: NDArray[np.float64],
    ) -> PosePrediction:
        return {
            "bbox": list(bbox),
            "keypoints": keypoints,
            "keypoint_scores": scores,
        }

    def _largest_person_prediction(self, result: Any) -> list[PosePrediction]:
        """Convert one RF-DETR result into this repository's selected person."""
        boxes = np.asarray(result.data["xyxy"], dtype=np.float64)
        # postprocess already dropped anything below the score threshold on
        # this path, so class and size are all that is left to decide.
        best = _largest_person_index(np.asarray(result.class_id), boxes)
        if best is None:
            return []

        x1, y1, x2, y2 = (float(value) for value in boxes[best])
        coco17_keypoints = np.asarray(result.xy[best], dtype=np.float64)
        coco17_scores = np.asarray(result.keypoint_confidence[best], dtype=np.float64)

        prediction = self._build_prediction(
            (x1, y1, x2, y2), coco17_keypoints, coco17_scores
        )
        return [prediction]

    def get_poses_batch(self, images: list[MatLike]) -> list[list[PosePrediction]]:
        """Detect the largest person and their pose across a batch of frames.

        On CUDA this runs the cached fixed-batch TensorRT engine (`BATCH_SIZE`
        frames per call); on MPS/CPU it falls back to RF-DETR's batched
        ``predict()``. It does not touch `_last_predictions`, which callers set
        per frame before reading landmarks.

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
            boxes = result["boxes"].detach().cpu().numpy()
            keypoints = result["keypoints"].detach().cpu().numpy()
            best = _largest_person_index(
                result["labels"].detach().cpu().numpy(),
                boxes,
                scores,
                self.person_detection_threshold,
            )
            if best is None:
                batch_predictions.append([])
                continue

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

    def get_dense_2d_keypoints(
        self,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]] | None:
        """The selected person's 17 body keypoints and their scores.

        Unlike ``get_2d_landmarks`` this keeps every joint, including those
        below the detection threshold, with its continuous score, so callers
        can weigh a weak keypoint rather than lose it.
        """
        if not self._last_predictions:
            return None
        target = self._last_predictions[0]
        keypoints = np.asarray(target["keypoints"], dtype=np.float64)
        scores = np.asarray(target["keypoint_scores"], dtype=np.float64)
        return keypoints, np.clip(scores, 0.0, 1.0)
