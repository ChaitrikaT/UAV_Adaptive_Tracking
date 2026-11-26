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

