"""Causal condition feature extraction (10 visual, motion, and tracking-state features)."""

from typing import Any, Dict, List, Optional, Tuple
import cv2
import numpy as np


FEATURE_NAMES: List[str] = [
    "blur",
    "mean_brightness",
    "contrast",
    "frame_change",
    "box_area_ratio",
    "mean_confidence",
    "camera_motion",
    "has_track",
    "frames_since_seen",
    "target_speed",
]


class ConditionFeatureExtractor:
    """Extracts 10 lightweight causal condition features from video frames and prior tracks.

    Combines visual quality metrics (blur, brightness, contrast), ego-motion proxies
    (frame_change, DIS optical flow camera_motion), and target/tracker state
    (box_area_ratio, mean_confidence, has_track, frames_since_seen, target_speed).
    """

    def __init__(self, downscale_flow_width: int = 320) -> None:
        self.downscale_width = downscale_flow_width
        self.prev_gray_small: Optional[np.ndarray] = None
        self.prev_centers: Dict[int, Tuple[float, float]] = {}
        self.frames_since_seen: int = 0
        self.dis_flow = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_FAST)

    def extract(
        self,
        frame: np.ndarray,
        tracks: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, float]:
        """Compute 10 causal condition features for the current frame."""
        h, w = frame.shape[:2]
        frame_area = float(h * w) if h > 0 and w > 0 else 1.0
        diag = float(np.hypot(w, h)) if h > 0 and w > 0 else 1.0

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 1. Sharpness / blur (variance of Laplacian)
        laplacian_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())

        # 2. Mean brightness (0-255) & 3. Contrast (std dev of grayscale)
        mean_brightness = float(np.mean(gray))
        contrast = float(np.std(gray))

        # 4. Downscaled frame difference & 5. Camera motion (DIS optical flow)
        target_w = self.downscale_width
        target_h = max(16, int(h * (target_w / float(max(1, w)))))
        curr_gray_small = cv2.resize(gray, (target_w, target_h), interpolation=cv2.INTER_AREA)

        frame_change = 0.0
        camera_motion = 0.0
        if self.prev_gray_small is not None:
            frame_change = float(
                np.mean(np.abs(curr_gray_small.astype(np.float32) - self.prev_gray_small.astype(np.float32)))
            )
            flow = self.dis_flow.calc(self.prev_gray_small, curr_gray_small, None)
            mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
            camera_motion = float(np.mean(mag))

        self.prev_gray_small = curr_gray_small

        # 6-10. Tracker-state features (strictly causal: from previous frame's tracks)
        box_area_ratio = 0.0
        mean_confidence = 0.0
        has_track = 0.0
        target_speed = 0.0
        curr_centers: Dict[int, Tuple[float, float]] = {}

        if tracks and len(tracks) > 0:
            has_track = 1.0
            self.frames_since_seen = 0
            areas: List[float] = []
            confs: List[float] = []
            speeds: List[float] = []

            for t in tracks:
                box = t.get("box")
                if box:
                    bw = max(0.0, float(box[2] - box[0]))
                    bh = max(0.0, float(box[3] - box[1]))
                    areas.append(bw * bh)
                    cx = 0.5 * (float(box[0]) + float(box[2]))
                    cy = 0.5 * (float(box[1]) + float(box[3]))
                    tid = t.get("id")
                    if tid is not None:
                        tid_int = int(tid)
                        curr_centers[tid_int] = (cx, cy)
                        if tid_int in self.prev_centers:
                            px, py = self.prev_centers[tid_int]
                            speeds.append(float(np.hypot(cx - px, cy - py)) / diag)
                if "conf" in t:
                    confs.append(float(t["conf"]))

            if areas:
                box_area_ratio = float(np.mean(areas) / frame_area)
            if confs:
                mean_confidence = float(np.mean(confs))
            if speeds:
                target_speed = float(np.mean(speeds))
        else:
            self.frames_since_seen += 1

        self.prev_centers = curr_centers

        return {
            "blur": round(laplacian_var, 2),
            "mean_brightness": round(mean_brightness, 2),
            "contrast": round(contrast, 2),
            "frame_change": round(frame_change, 3),
            "box_area_ratio": round(box_area_ratio, 6),
            "mean_confidence": round(mean_confidence, 4),
            "camera_motion": round(camera_motion, 4),
            "has_track": has_track,
            "frames_since_seen": float(self.frames_since_seen),
            "target_speed": round(target_speed, 6),
        }

    def reset(self) -> None:
        """Reset historical optical flow and track velocity state."""
        self.prev_gray_small = None
        self.prev_centers.clear()
        self.frames_since_seen = 0
