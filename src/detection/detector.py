"""YOLO object detector wrapper with swappable weights and configurable resolution."""

from pathlib import Path
from typing import Any, List, Optional, Union
import numpy as np
from ultralytics import YOLO


class YOLODetector:
    """Wrapper around Ultralytics YOLO models for object detection."""

    def __init__(
        self,
        weights: Union[str, Path] = "weights/yolov8n.pt",
        imgsz: int = 640,
        conf: float = 0.25,
        classes: Optional[List[int]] = None,
    ) -> None:
        """Initialize YOLO detector with weights, image size, and confidence threshold.

        Args:
            weights: Path to model weights or model name (e.g. 'weights/yolov8n.pt').
            imgsz: Inference image resolution.
            conf: Minimum confidence threshold.
            classes: Optional list of class IDs to filter detections.
        """
        self.weights_path = Path(weights)
        self.imgsz = imgsz
        self.conf = conf
        self.classes = classes if classes and len(classes) > 0 else None

        # Ensure model weights are available, auto-downloading if necessary
        self._ensure_weights()
        self.model = YOLO(str(self.weights_path))

    def _ensure_weights(self) -> None:
        """Ensure weight files exist locally, downloading via Ultralytics if missing."""
        if self.weights_path.is_file():
            return

        self.weights_path.parent.mkdir(parents=True, exist_ok=True)
        filename = self.weights_path.name
        # If it's a standard YOLO model (e.g. yolov8n.pt), download it
        if filename.startswith("yolo"):
            temp_model = YOLO(filename)
            downloaded = Path(filename)
            if downloaded.is_file() and downloaded.resolve() != self.weights_path.resolve():
                downloaded.rename(self.weights_path)

    def detect(self, frame: np.ndarray) -> Any:
        """Run object detection on an input frame.

        Args:
            frame: Input image/frame in BGR format (NumPy ndarray).

        Returns:
            Ultralytics Boxes object for the frame.
        """
        results = self.model.predict(
            source=frame,
            imgsz=self.imgsz,
            conf=self.conf,
            classes=self.classes,
            verbose=False,
        )
        return results[0].boxes


class AerialBirdFilter:
    """Lightweight COCO aerial check that quietly filters out birds from single-class drone detections."""

    def __init__(
        self,
        weights: Union[str, Path] = "weights/yolov8n.pt",
        imgsz: int = 640,
        conf: float = 0.25,
        bird_class_id: int = 14,
        airplane_class_id: int = 4,
        iou_threshold: float = 0.30,
    ) -> None:
        self.weights_path = Path(weights)
        self.imgsz = imgsz
        self.conf = conf
        self.bird_class_id = bird_class_id
        self.airplane_class_id = airplane_class_id
        self.iou_threshold = iou_threshold
        self.model = YOLO(str(self.weights_path))

    @staticmethod
    def _box_iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        x1 = np.maximum(a[:, None, 0], b[None, :, 0])
        y1 = np.maximum(a[:, None, 1], b[None, :, 1])
        x2 = np.minimum(a[:, None, 2], b[None, :, 2])
        y2 = np.minimum(a[:, None, 3], b[None, :, 3])
        inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
        area_a = np.maximum(0.0, a[:, 2] - a[:, 0]) * np.maximum(0.0, a[:, 3] - a[:, 1])
        area_b = np.maximum(0.0, b[:, 2] - b[:, 0]) * np.maximum(0.0, b[:, 3] - b[:, 1])
        return inter / (area_a[:, None] + area_b[None, :] - inter + 1e-9)

    def filter_boxes(self, frame: np.ndarray, boxes: Any) -> tuple:
        """Return (filtered_boxes, birds_removed, airplanes_flagged)."""
        if boxes is None or len(boxes) == 0:
            return boxes, 0, 0

        res = self.model.predict(
            source=frame,
            imgsz=self.imgsz,
            conf=self.conf,
            classes=[self.bird_class_id, self.airplane_class_id],
            verbose=False,
        )
        coco_boxes = res[0].boxes
        if coco_boxes is None or len(coco_boxes) == 0:
            return boxes, 0, 0

        c_xyxy = coco_boxes.xyxy.cpu().numpy()
        c_cls = coco_boxes.cls.cpu().numpy().astype(int)
        c_conf = coco_boxes.conf.cpu().numpy()

        d_xyxy = boxes.xyxy.cpu().numpy()
        d_conf = boxes.conf.cpu().numpy()

        bird_mask = c_cls == self.bird_class_id
        plane_mask = c_cls == self.airplane_class_id

        keep = np.ones(len(d_xyxy), dtype=bool)
        birds_removed = 0
        airplanes_flagged = 0

        if np.any(bird_mask):
            iou_b = self._box_iou(d_xyxy, c_xyxy[bird_mask])
            b_confs = c_conf[bird_mask]
            for i in range(len(d_xyxy)):
                j = int(np.argmax(iou_b[i]))
                if iou_b[i, j] >= self.iou_threshold and b_confs[j] >= d_conf[i] * 0.8:
                    keep[i] = False
                    birds_removed += 1

        if np.any(plane_mask):
            iou_p = self._box_iou(d_xyxy, c_xyxy[plane_mask])
            for i in range(len(d_xyxy)):
                if keep[i] and np.max(iou_p[i]) >= self.iou_threshold:
                    airplanes_flagged += 1

        if birds_removed > 0:
            boxes = boxes[keep]
        return boxes, birds_removed, airplanes_flagged

