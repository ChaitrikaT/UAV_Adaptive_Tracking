"""Offline, whole-frame descriptors from previously saved video frames.

No annotations, object boxes, detector outputs, or future frames are inputs.
These descriptors describe sampled frames, NOT a target-specific tracker.
"""
from __future__ import annotations

import re
from pathlib import Path
import cv2
import numpy as np

PATTERN = re.compile(r"^Clip_(\d+)_frame_(\d+)\.jpg$", re.IGNORECASE)
FEATURE_COLUMNS = ("brightness", "contrast", "sharpness", "sampled_frame_change")


def clip_number(filename: str) -> int:
    match = PATTERN.fullmatch(Path(filename).name)
    if not match:
        raise ValueError(f"Unexpected frame filename: {filename}")
    return int(match.group(1))


def extract_image_features(image_path: Path, previous_small: np.ndarray | None):
    """Descriptors of whole image; frame change is relative to previous *sampled* image."""
    frame = cv2.imread(str(image_path))
    if frame is None:
        raise FileNotFoundError(f"Unable to read image: {image_path}")
    small = cv2.cvtColor(cv2.resize(frame, (320, 180)), cv2.COLOR_BGR2GRAY)
    brightness = float(small.mean())
    contrast = float(small.std())
    sharpness = float(cv2.Laplacian(small, cv2.CV_64F).var())
    sampled_change = (float(cv2.absdiff(small, previous_small).mean())
                      if previous_small is not None else 0.0)
    return (brightness, contrast, sharpness, sampled_change), small


def compute_features(image_dir: Path, filenames: list[str]) -> dict[str, tuple[float, ...]]:
    """Read frames chronologically inside each clip; never carry state across clips."""
    ordered = sorted(set(filenames), key=lambda f: (clip_number(f), f))
    previous_clip = None
    previous_small = None
    output = {}
    for name in ordered:
        current_clip = clip_number(name)
        if current_clip != previous_clip:
            previous_small = None
        output[name], previous_small = extract_image_features(image_dir / name, previous_small)
        previous_clip = current_clip
    return output
