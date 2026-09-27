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


# Fixed batch size the cached TensorRT engines are built for. Callers chunk
# whole videos to this size; a short final chunk is padded internally.
BATCH_SIZE = 16

# Elbows are frequently self-occluded during a badminton swing. The pose model
# still tracks them coherently below the general body-joint cutoff, so accept
# these two joints at a lower confidence instead of synthesizing their
# coordinates later in the renderer.
_ELBOW_KEYPOINT_INDICES = frozenset(
    (int(COCOKeypoints.LEFT_ELBOW), int(COCOKeypoints.RIGHT_ELBOW))
)

_TRT_CACHE_ROOT = Path(
    os.getenv(
        "BADMINTON_TRT_CACHE_DIR",
        str(Path.home() / ".cache" / "badminton_analysis" / "trt_engines"),
    )
)

DETECTOR_ENGINE = "rfdetr-medium.trt"
POSE_ENGINE = "vitpose-plus-large.trt"
POSE_CHECKPOINT = "usyd-community/vitpose-plus-large"
# ViTPose++ is a mixture of experts over its training datasets; this one reads
# the COCO 17-keypoint schema, which is the order `COCOKeypoints` uses.
POSE_DATASET_INDEX = 0
POSE_INPUT_HEIGHT, POSE_INPUT_WIDTH = 256, 192


def _cv2_warp_affine(
    src: NDArray[np.uint8], M: NDArray[np.floating], size: tuple[int, int]
) -> NDArray[np.uint8]:
    """The warp the original ViTPose uses.

    `transformers` falls back to `scipy.ndimage`, which pushes every channel of
    the full frame through an inverse affine per crop -- about 1.3s a frame
    against 0.13s here, for keypoints that agree to 0.008px.
    """
    return cv2.warpAffine(
        src, np.asarray(M, dtype=np.float32), (size[1], size[0]), flags=cv2.INTER_LINEAR
    )


class PoseDetector:
    """Two-stage 2D pose estimation.

    RF-DETR Medium finds the player and ViTPose++-L reads the joints from a
    crop of them. The previous single-stage RF-DETR keypoint model saw the
    whole frame squashed into a square, which spent most of its resolution on
    an empty court; cropping to the player and reading the pose at 256x192
    agrees with the raters measurably better (serve checklist ICC 0.839 to
    0.863 over the fifty rated learners), and Medium is cheaper than the
    keypoint model it replaces because it only has to return a box.

    ViTPose emits COCO-17 in the same index order as `COCOKeypoints`, so no
    schema adapter is needed. It has no hand or foot keypoints, so
    `wholebody_keypoints`/`wholebody_scores` only ever carry real data in their
    first 17 slots.
    """

    def __init__(
        self,
        # 0.5 dropped real, stable elbow/wrist detections during a swing --
        # the non-dominant arm (partially self-occluded, held close to the
        # body from most camera angles) commonly scores 0.18-0.45 while still
        # tracking the correct position frame to frame; 0.15 keeps those and
        # still excludes near-zero noise.
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

        self._detector: Any | None = None
        self._pose_model: Any | None = None
        self._pose_processor: Any | None = None
        self._detector_engine: Any | None = None
        self._pose_engine: Any | None = None
        self._last_predictions: list[PosePrediction] = []

    def keypoint_detection_threshold(self, index: int) -> float:
        """Return the acceptance threshold for one COCO body joint.

        The TensorRT and torch paths produce the same ``PosePrediction``
        structure, so applying this threshold while reading predictions keeps
        their joint filtering identical.
        """
        if int(index) in _ELBOW_KEYPOINT_INDICES:
            return self.elbow_detection_confidence
        return self.min_detection_confidence

    # The grade log names the pose backend, and the ``pose_tensorrt_active``
    # diagnostic reports whether the TensorRT engines served the request.
    @property
    def execution_provider(self) -> str:
        return "tensorrt" if self.tensorrt_active else "torch"

    @property
    def tensorrt_active(self) -> bool:
        return self._detector_engine is not None and self._pose_engine is not None

    def _load_detector(self) -> None:
        try:
            from rfdetr import RFDETRMedium
        except ImportError as exc:
            raise RuntimeError("rfdetr is required for person detection") from exc
        self._detector = RFDETRMedium(device=self.device)

    def _load_pose_model(self) -> None:
        try:
            import transformers.models.vitpose.image_processing_vitpose as vitpose_processing
            from transformers import VitPoseForPoseEstimation, VitPoseImageProcessor
        except ImportError as exc:
            raise RuntimeError("transformers is required for pose estimation") from exc
        vitpose_processing.scipy_warp_affine = _cv2_warp_affine
        # Built from explicit parameters rather than the hub, so a cold start
        # never depends on network access.
        self._pose_processor = VitPoseImageProcessor(
            size={"height": POSE_INPUT_HEIGHT, "width": POSE_INPUT_WIDTH}
        )
        if self._pose_model is None and self._pose_engine is None:
            self._pose_model = (
                VitPoseForPoseEstimation.from_pretrained(
                    POSE_CHECKPOINT, dtype=torch.float32
                )
                .to(self.device)
                .eval()
            )

    def _engine_dir(self) -> Path:
        gpu_name = (
            torch.cuda.get_device_name(0).replace(" ", "_")
            if torch.cuda.is_available()
            else "cpu"
        )
        return _TRT_CACHE_ROOT / gpu_name / f"batch{BATCH_SIZE}"

    def _load_engines(self) -> None:
        """Load the two prebuilt TensorRT engines baked into the image.

        Unlike the single-stage detector this replaces, nothing is built at
        run time: `build_pose_engines.py` produces both engines on the serving
        GPU once and the deploy workflow bakes them in, because a cold start
        that compiles an engine costs minutes on a service that scales to zero.
        """
        from badminton_analysis.services.trt_engine import TorchTRTEngine

        if self._detector is None:
            self._load_detector()
        cache_dir = self._engine_dir()
        for name, attribute in (
            (DETECTOR_ENGINE, "_detector_engine"),
            (POSE_ENGINE, "_pose_engine"),
        ):
            if getattr(self, attribute) is not None:
                continue
            path = cache_dir / name
            if not path.exists():
                raise RuntimeError(
                    f"missing TensorRT engine {path}; build it with "
                    f"build_pose_engines.py on this GPU and bake it into the image"
                )
            setattr(self, attribute, TorchTRTEngine(str(path)))
        self._load_pose_model()

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

    def _person_boxes(self, images: list[MatLike]) -> list[tuple[float, ...] | None]:
        """One box per frame: the largest person, or None when there is none."""
        if self.device != "cuda":
            if self._detector is None:
                self._load_detector()
            results = self._detector.predict(
                [cv2.cvtColor(image, cv2.COLOR_BGR2RGB) for image in images],
                threshold=self.person_detection_threshold,
                include_source_image=False,
            )
            boxes = []
            for result in results:
                found = np.asarray(result.xyxy, dtype=np.float64)
                best = _largest_person_index(np.asarray(result.class_id), found)
                boxes.append(None if best is None else tuple(found[best]))
            return boxes

        ctx = self._detector.model
        padded = list(images) + [images[-1]] * (BATCH_SIZE - len(images))
        resolution = ctx.resolution
        tensors, heights, widths = [], [], []
        for image in padded:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            heights.append(image_rgb.shape[0])
            widths.append(image_rgb.shape[1])
            tensors.append(
                torch.from_numpy(image_rgb).permute(2, 0, 1).float().to(ctx.device)
                / 255.0
            )
        batch_tensor = torch.stack(
            [
                tv_functional.resize(t, [resolution, resolution], antialias=False)
                for t in tensors
            ]
        )
        batch_tensor = tv_functional.normalize(
            batch_tensor, self._detector.means, self._detector.stds
        )
        assert self._detector_engine is not None
        trt_out = self._detector_engine({"input": batch_tensor})
        target_sizes = torch.tensor(
            [[h, w] for h, w in zip(heights, widths)],
            device=ctx.device,
            dtype=torch.float32,
        )
        results = ctx.postprocess(
            {"pred_logits": trt_out["labels"], "pred_boxes": trt_out["dets"]},
            target_sizes=target_sizes,
        )
        boxes: list[tuple[float, ...] | None] = []
        for result in results[: len(images)]:
            found = result["boxes"].detach().cpu().numpy().astype(np.float64)
            labels = result["labels"].detach().cpu().numpy()
            scores = result["scores"].detach().cpu().numpy()
            best = _largest_person_index(
                labels, found, scores, self.person_detection_threshold
            )
            boxes.append(None if best is None else tuple(found[best]))
        return boxes

    def _keypoints(
        self, images: list[MatLike], boxes: list[tuple[float, ...] | None]
    ) -> list[tuple[NDArray[np.float64], NDArray[np.float64]] | None]:
        """Read every found person's joints in one batched pose call."""
        found = [index for index, box in enumerate(boxes) if box is not None]
        if not found:
            return [None] * len(images)
        if self._pose_processor is None:
            self._load_pose_model()
        crops, per_frame_boxes = [], []
        for index in found:
            x1, y1, x2, y2 = boxes[index]
            xywh = np.asarray([[x1, y1, x2 - x1, y2 - y1]], dtype=np.float32)
            per_frame_boxes.append(xywh)
            crops.append(
                self._pose_processor(
                    cv2.cvtColor(images[index], cv2.COLOR_BGR2RGB),
                    boxes=[xywh],
                    return_tensors="pt",
                )["pixel_values"]
            )
        pixel_values = torch.cat(crops)
        if self._pose_engine is not None:
            padded = pixel_values
            if len(padded) < BATCH_SIZE:
                padded = torch.cat(
                    [padded, padded[-1:].repeat(BATCH_SIZE - len(padded), 1, 1, 1)]
                )
            heatmaps = self._pose_engine(
                {"pixel_values": padded.to("cuda")}
            )["heatmaps"][: len(pixel_values)]
        else:
            with torch.no_grad():
                heatmaps = self._pose_model(
                    pixel_values=pixel_values.to(self.device),
                    dataset_index=torch.full(
                        (len(pixel_values),),
                        POSE_DATASET_INDEX,
                        dtype=torch.long,
                        device=self.device,
                    ),
                ).heatmaps
        decoded = self._pose_processor.post_process_pose_estimation(
            type("Output", (), {"heatmaps": heatmaps.detach().float().cpu()})(),
            boxes=[np.concatenate(per_frame_boxes)],
        )[0]
        out: list[tuple[NDArray[np.float64], NDArray[np.float64]] | None] = [None] * len(images)
        for position, index in enumerate(found):
            person = decoded[position]
            out[index] = (
                np.asarray(person["keypoints"], dtype=np.float64),
                np.asarray(person["scores"], dtype=np.float64),
            )
        return out

    def get_poses_batch(self, images: list[MatLike]) -> list[list[PosePrediction]]:
        """Detect the largest person and their pose across a batch of frames.

        On CUDA both stages run their cached fixed-batch TensorRT engines
        (`BATCH_SIZE` frames per call); on MPS/CPU both fall back to torch. It
        does not touch `_last_predictions`, which callers set per frame before
        reading landmarks.

        A short final chunk is padded internally with a repeated last frame to
        satisfy the engines' fixed batch shape, then the padding is truncated
        back off before returning.
        """
        if not images:
            return []
        if len(images) > BATCH_SIZE:
            raise ValueError(
                f"batch of {len(images)} frames exceeds the fixed engine "
                f"batch size of {BATCH_SIZE}"
            )
        if self.device == "cuda":
            self._load_engines()
        elif self._detector is None:
            self._load_detector()
        boxes = self._person_boxes(images)
        poses = self._keypoints(images, boxes)
        batch: list[list[PosePrediction]] = []
        for box, pose in zip(boxes, poses, strict=True):
            if box is None or pose is None:
                batch.append([])
                continue
            keypoints, scores = pose
            batch.append([self._build_prediction(tuple(box), keypoints, scores)])
        return batch

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
