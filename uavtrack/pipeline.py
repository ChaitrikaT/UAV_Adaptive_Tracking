"""Frame-by-frame online pipeline: decode -> features -> policy -> detector(imgsz) -> tracker.

The mode chosen by the policy for frame t changes the detector's input size for that frame
(FAST=640, ROBUST=960), i.e. it changes the inference that is actually executed.
Reference annotations are NOT read here.
"""
from __future__ import annotations

import time
from pathlib import Path

import cv2
import numpy as np

from .features import build_features, global_features, small_gray
from .policy import MODES
from .tracker import Tracker, TrackerConfig


class YoloDetector:
    """Ultralytics YOLO wrapper. Returns an (N, 5) array of [x1, y1, x2, y2, conf]."""

    def __init__(self, weights: Path, conf: float = 0.10, device=None):
        import torch
        from ultralytics import YOLO
        self.model = YOLO(str(weights))
        self.conf = conf
        self.device = device if device is not None else (0 if torch.cuda.is_available() else "cpu")
        self._cuda = torch.cuda.is_available()
        self._torch = torch

    def sync(self):
        if self._cuda:
            self._torch.cuda.synchronize()

    def warmup(self, frame):
        for size in MODES.values():
            for _ in range(3):
                self.detect(frame, size)
        self.sync()

    def detect(self, frame, imgsz: int) -> np.ndarray:
        r = self.model.predict(frame, imgsz=imgsz, conf=self.conf, device=self.device, verbose=False)[0]
        if r.boxes is None or len(r.boxes) == 0:
            return np.zeros((0, 5))
        return np.hstack([r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy()[:, None]])


def draw(frame, tracks, mode, frame_no, label="predicted IDs (no GT shown)"):
    out = frame.copy()
    for t in tracks:
        p1, p2 = (int(t["x1"]), int(t["y1"])), (int(t["x2"]), int(t["y2"]))
        color = (0, 200, 255) if t["observed"] else (160, 160, 160)
        cv2.rectangle(out, p1, p2, color, 3)
        cv2.putText(out, f"ID {t['track_id']} {t['conf']:.2f}", (p1[0], max(20, p1[1] - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    cv2.rectangle(out, (0, 0), (out.shape[1], 40), (20, 20, 20), -1)
    cv2.putText(out, f"Frame {frame_no} | mode {mode} ({MODES[mode]}px) | {label}",
                (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
    return out


def run_video(video_path: Path, detector, policy, start=1, end=-1, tracker_cfg: TrackerConfig | None = None,
              video_out: Path | None = None, log_every=100, banner_label=None, draw_fn=None):
    """Process frames start..end (1-based, consecutive). Returns (frame_rows, track_rows)."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open {video_path}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps_in = cap.get(cv2.CAP_PROP_FPS) or 30.0
    if start > 1:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start - 1)
    last = end if end > 0 else (total if total > 0 else 10 ** 9)
    writer = None
    tracker, prev_small, prev_mode = Tracker(tracker_cfg), None, "FAST"
    policy.reset()
    frame_rows, track_rows = [], []
    for frame_no in range(start, last + 1):
        t0 = time.perf_counter()
        ok, frame = cap.read()
        if not ok:
            break
        t_dec = time.perf_counter()
        small = small_gray(frame)
        feats = build_features(global_features(small, prev_small), tracker.primary_state(), prev_mode)
        mode = policy.decide(feats)
        t_feat = time.perf_counter()
        if hasattr(detector, "sync"):
            detector.sync()
        dets = detector.detect(frame, MODES[mode])
        if hasattr(detector, "sync"):
            detector.sync()
        t_det = time.perf_counter()
        tracks = tracker.update(dets)
        t_trk = time.perf_counter()
        for t in tracks:
            track_rows.append({"frame": frame_no, **t})
        frame_rows.append({
            "frame": frame_no, "mode": mode, "imgsz": MODES[mode], "n_det": len(dets), "n_tracks": len(tracks),
            "decode_ms": round((t_dec - t0) * 1e3, 3), "feature_policy_ms": round((t_feat - t_dec) * 1e3, 3),
            "detect_ms": round((t_det - t_feat) * 1e3, 3), "track_ms": round((t_trk - t_det) * 1e3, 3),
            "total_ms": round((t_trk - t0) * 1e3, 3), **{k: round(float(v), 4) for k, v in feats.items()}})
        if video_out is not None:
            drawn = (draw_fn(frame, tracks, mode, frame_no) if draw_fn
                     else draw(frame, tracks, mode, frame_no, banner_label or "predicted IDs (no GT shown)"))
            if writer is None:   # FIX: size must come from the DRAWN frame (a draw_fn may change the size)
                h, w = drawn.shape[:2]
                video_out.parent.mkdir(parents=True, exist_ok=True)
                writer = cv2.VideoWriter(str(video_out), cv2.VideoWriter_fourcc(*"mp4v"), min(fps_in, 30.0), (w, h))
                if not writer.isOpened():
                    raise RuntimeError(f"Could not open video writer for {video_out} ({w}x{h})")
            writer.write(drawn)
        prev_small, prev_mode = small, mode
        if log_every and len(frame_rows) % log_every == 0:
            print(f"  processed {len(frame_rows)} frames (last mode {mode})", flush=True)
    cap.release()
    if writer is not None:
        writer.release()
    return frame_rows, track_rows


def summarize_timing(frame_rows) -> dict:
    if not frame_rows:
        return {}
    n = len(frame_rows)
    tot = sum(r["total_ms"] for r in frame_rows) / n
    return {"frames": n,
            "robust_fraction": round(sum(r["mode"] == "ROBUST" for r in frame_rows) / n, 4),
            "mean_detect_ms": round(sum(r["detect_ms"] for r in frame_rows) / n, 3),
            "mean_total_ms_incl_decode": round(tot, 3),
            "end_to_end_fps": round(1000.0 / tot, 2) if tot else 0.0}
