"""Synthetic clip generator shared by tests (a white square moving over a grey frame)."""
from pathlib import Path

import cv2
import numpy as np


def make_clip(dirpath: Path, n=60, w=320, h=180):
    video = dirpath / "Clip_1.avi"
    wr = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"MJPG"), 10.0, (w, h))
    ref = dirpath / "Clip_1_refined.txt"
    lines = []
    for f in range(1, n + 1):
        img = np.full((h, w, 3), 40, np.uint8)
        x, y = 20 + 4 * f, 90
        cv2.rectangle(img, (x, y), (x + 11, y + 11), (255, 255, 255), -1)
        wr.write(img)
        lines.append(f"{f},1,{x},{y},12,12,1,1,1")
    wr.release()
    ref.write_text("\n".join(lines), encoding="utf-8")
    return video, ref
