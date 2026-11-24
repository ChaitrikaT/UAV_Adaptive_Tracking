"""Per-frame condition features available BEFORE running the detector on frame t.

Whole-frame: brightness, contrast, sharpness (Laplacian variance), change vs the
previous *consecutive* frame. Track-state: taken from the tracker after frame t-1.
No ground truth is ever used here.
Note: the older src/features.py computed 'change' between frames sampled 20 apart;
here it is between consecutive frames, so the old decision tree cannot be reused.
"""
from __future__ import annotations

import cv2
import numpy as np

FEATURE_NAMES = ["brightness", "contrast", "sharpness", "frame_change",
                 "has_track", "frames_since_seen", "target_size", "target_speed",
                 "last_conf", "prev_robust"]


def small_gray(frame_bgr) -> np.ndarray:
    return cv2.cvtColor(cv2.resize(frame_bgr, (320, 180)), cv2.COLOR_BGR2GRAY)


def global_features(small: np.ndarray, prev_small: np.ndarray | None) -> dict:
    return {"brightness": float(small.mean()),
            "contrast": float(small.std()),
            "sharpness": float(cv2.Laplacian(small, cv2.CV_64F).var()),
            "frame_change": float(cv2.absdiff(small, prev_small).mean()) if prev_small is not None else 0.0}


def feature_vector(feats: dict) -> list[float]:
    return [float(feats[name]) for name in FEATURE_NAMES]


def build_features(gfeats: dict, tracker_state: dict, prev_mode: str) -> dict:
    """THE single place where the policy's input vector is assembled.

    Used by the online pipeline AND by training rollouts, so both see identical features.
    Inputs are only: this frame's global image statistics, the tracker state after frame t-1,
    and the mode actually used on frame t-1. No reference annotations, no future frames.
    """
    feats = dict(gfeats)
    feats.update(tracker_state)
    feats["prev_robust"] = int(prev_mode == "ROBUST")
    return feats
