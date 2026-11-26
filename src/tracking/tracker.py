"""ByteTrack tracker wrapper with motion trail history."""

from collections import defaultdict, deque
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
from ultralytics.trackers.byte_tracker import BYTETracker, STrack
from ultralytics.utils.checks import check_yaml
import yaml


class ByteTrackerWrapper:
    """Wrapper around Ultralytics ByteTrack algorithm maintaining track IDs and trajectory history."""

    def __init__(
        self,
        config_path: str = "bytetrack.yaml",
        trail_length: int = 30,
    ) -> None:
        """Initialize ByteTrack wrapper.

        Args:
            config_path: Name or path of tracker config (defaults to 'bytetrack.yaml').
            trail_length: Number of historical centroid points to keep for motion trail visualization.
        """
        self.config_path = config_path
        self.trail_length = trail_length
        self.track_history: Dict[int, deque] = defaultdict(
            lambda: deque(maxlen=self.trail_length)
        )
        self.tracker = self._init_tracker()

    def _init_tracker(self) -> BYTETracker:
        """Create and configure a fresh BYTETracker instance."""
        yaml_file = check_yaml(self.config_path)
        with open(yaml_file, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        args = SimpleNamespace(**cfg)
        return BYTETracker(args)

    def update(
        self, boxes: Optional[Any], frame: np.ndarray
    ) -> List[Dict[str, Any]]:
        """Update tracker state with new detections or advance Kalman filter on skipped frames.

        Args:
            boxes: Ultralytics Boxes object from detector (or None if detection is skipped).
            frame: Current video frame (BGR NumPy array).

        Returns:
            List of track dictionaries: {'id': int, 'box': (x1, y1, x2, y2), 'conf': float, 'cls': int, 'trail': list}
        """
        if boxes is not None:
            boxes_cpu = boxes.cpu()
            _ = self.tracker.update(boxes_cpu, img=frame)
        else:
            # Keep ByteTrack's internal frame counter synchronized and advance Kalman state
            self.tracker.frame_id += 1
            if len(self.tracker.tracked_stracks) > 0:
                STrack.multi_predict(self.tracker.tracked_stracks)

        active_tracks: List[Dict[str, Any]] = []
        for track in self.tracker.tracked_stracks:
            if not track.is_activated:
                continue

            track_id = int(track.track_id)
            # Ultralytics 8.1+ calls the corner box `xyxy`; older releases called it `tlbr`.
            corners = track.xyxy if hasattr(track, "xyxy") else track.tlbr
            x1, y1, x2, y2 = (float(v) for v in corners[:4])
            conf = float(track.score)
            cls_id = int(getattr(track, "cls", 0))

            cx = int((x1 + x2) / 2.0)
            cy = int((y1 + y2) / 2.0)
            w = round(max(0.0, x2 - x1), 1)
            h = round(max(0.0, y2 - y1), 1)
            self.track_history[track_id].append((cx, cy))

            active_tracks.append(
                {
                    "id": track_id,
                    "box": (x1, y1, x2, y2),
                    "center": (cx, cy),
                    "width": w,
                    "height": h,
                    "conf": conf,
                    "cls": cls_id,
                    "trail": list(self.track_history[track_id]),
                }
            )

        return active_tracks

    def reset(self) -> None:
        """Reset internal tracker state and trajectory history between video sequences."""
        self.tracker = self._init_tracker()
        self.track_history.clear()

