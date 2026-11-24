"""Archived-results replay: the existing detector-review video + plots of the SAVED per-frame CSVs.

Nothing is computed by a model here. Detections and condition values were measured earlier by
src/review_detector_video.py and src/condition_analysis.py; this script only draws them next to
the recorded video. It does not track, does not switch modes, and uses no new measurements.

  python demo_archived_review.py
Inputs : demo/videos/clip_7_detector_review_h264.mp4
         demo/data/clip_7_detector_review.csv, demo/data/clip_7_condition_analysis.csv
         (or results/videos|conditions/ with the same names; results/ wins if present)
Output : demo/videos/clip_7_archived_results_replay_h264.mp4
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np

from uavtrack.video_io import H264Writer

ROOT = Path(__file__).resolve().parent
W, VH, PH = 1280, 720, 330          # canvas width, video height, panel height
L, R = 70, 1210                      # plot x-range
GREEN, GREY, ORANGE, BLUE, RED = (60, 190, 60), (150, 150, 150), (30, 140, 240), (230, 140, 40), (60, 60, 220)


def find(name_dir: str, name: str) -> Path:
    for base in (ROOT / "results" / name_dir, ROOT / "demo" / "data"):
        if (base / name).is_file():
            return base / name
    raise SystemExit(f"Missing {name}: looked in results/{name_dir}/ and demo/data/")


def load(det_csv: Path, cond_csv: Path):
    det = list(csv.DictReader(det_csv.open(encoding="utf-8-sig")))
    cond = list(csv.DictReader(cond_csv.open(encoding="utf-8-sig")))
    if [r["frame"] for r in det] != [r["frame"] for r in cond]:
        raise SystemExit("Detector CSV and condition CSV frame lists differ.")
    return det, cond


def summarize(det):
    pred = sum(int(r["predicted_boxes"]) > 0 for r in det)
    matched = sum(float(r["max_iou"]) >= 0.5 for r in det)
    best = cur = 0; start = bs = be = None
    for r in det:
        if float(r["max_iou"]) < 0.5:
            if cur == 0:
                start = int(r["frame"])
            cur += 1
            if cur > best:
                best, bs, be = cur, start, int(r["frame"])
        else:
            cur = 0
    return {"frames": len(det), "frames_with_detections": pred, "frames_with_iou_at_least_05": matched,
            "longest_unmatched_run": {"start": bs, "end": be, "frames": best}}


def verify(summary) -> bool:
    path = ROOT / "demo" / "final" / "verified_metrics.json"
    if not path.is_file():
        print("verified_metrics.json not found; skipping cross-check.")
        return True
    ref = json.loads(path.read_text(encoding="utf-8"))
    ok = all(ref[k] == summary[k] for k in ("frames", "frames_with_detections", "frames_with_iou_at_least_05",
                                            "longest_unmatched_run"))
    print("CSV numbers vs demo/final/verified_metrics.json:", "MATCH" if ok else f"MISMATCH {summary} vs {ref}")
    return ok


def text(img, s, org, scale=0.5, color=(255, 255, 255), thick=1):
    cv2.putText(img, s, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)


def build_panel(det, cond, run, t):
    n = len(det)
    panel = np.full((PH, W, 3), 28, np.uint8)
    x_of = lambda i: int(L + (R - L) * (i + 0.5) / n)
    bw = max(2, (R - L) // n - 1)
    # ---- plot A: max IoU of detector vs reference, per frame (saved values)
    ay0, ay1 = 70, 160
    text(panel, "A  max IoU between detector box and reference box (saved CSV). Bars >= 0.5 = matched", (L, 20), 0.5)
    text(panel, "ARCHIVED RESULTS REPLAY - not online processing, no tracking IDs, nothing re-measured", (L, 42), 0.5, (80, 200, 255))
    lr = run["longest_unmatched_run"]
    if lr["frames"]:
        x0, x1 = x_of(lr["start"] - 1) - bw, x_of(lr["end"] - 1) + bw
        cv2.rectangle(panel, (x0, ay0), (x1, ay1), (45, 40, 60), -1)
        text(panel, f"longest unmatched run: frames {lr['start']}-{lr['end']} ({lr['frames']})", (x0 + 6, ay0 + 14), 0.42, (170, 170, 230))
    for i, r in enumerate(det):
        v = min(1.0, float(r["max_iou"]))
        h = int((ay1 - ay0) * v)
        cv2.rectangle(panel, (x_of(i) - bw // 2, ay1 - h), (x_of(i) + bw // 2, ay1), GREEN if v >= 0.5 else GREY, -1)
        if int(r["predicted_boxes"]) > 0:
            cv2.circle(panel, (x_of(i), ay1 + 12), 2, (220, 0, 220), -1)
    yth = ay1 - (ay1 - ay0) // 2
    cv2.line(panel, (L, yth), (R, yth), (200, 200, 60), 1)
    text(panel, "0.5", (L - 30, yth + 4), 0.4, (200, 200, 60)); text(panel, "1.0", (L - 30, ay0 + 4), 0.4); text(panel, "0", (L - 20, ay1), 0.4)
    text(panel, "magenta dots = frames where the detector produced >= 1 box", (L, ay1 + 32), 0.4, (220, 120, 220))
    # ---- plot B: condition features (min-max normalised for drawing; true ranges printed)
    by0, by1 = 232, 302
    text(panel, "B  whole-frame condition values (saved CSV, each line scaled to its own min-max)", (L, 214), 0.5)
    for key, col, label in (("brightness", BLUE, "brightness"), ("sharpness", ORANGE, "sharpness")):
        vals = np.array([float(r[key]) for r in cond])
        lo, hi = vals.min(), vals.max()
        pts = [(x_of(i), int(by1 - (by1 - by0) * (v - lo) / (hi - lo + 1e-9))) for i, v in enumerate(vals)]
        cv2.polylines(panel, [np.array(pts, np.int32)], False, col, 1, cv2.LINE_AA)
        text(panel, f"{label} {lo:.0f}-{hi:.0f}", (R - 330 if key == "brightness" else R - 150, 214), 0.4, col)
    # ---- cursor + numbers for the current frame
    cx = x_of(t)
    cv2.line(panel, (cx, 55), (cx, by1 + 5), (255, 255, 255), 1)
    d, c = det[t], cond[t]
    status = "MATCHED" if float(d["max_iou"]) >= 0.5 else "not matched"
    text(panel, f"Frame {d['frame']}/{n} | maxIoU {float(d['max_iou']):.2f} ({status}) | detector boxes {d['predicted_boxes']} | "
                f"reference boxes {d['reference_boxes']} | brightness {float(c['brightness']):.1f} | sharpness {float(c['sharpness']):.1f}",
         (L, 322), 0.45, (255, 255, 255))
    return panel


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", type=Path, default=ROOT / "demo/videos/clip_7_detector_review_h264.mp4")
    ap.add_argument("--out", type=Path, default=ROOT / "demo/videos/clip_7_archived_results_replay_h264.mp4")
    a = ap.parse_args()
    det, cond = load(find("videos", "clip_7_detector_review.csv"), find("conditions", "clip_7_condition_analysis.csv"))
    run = summarize(det)
    print("Recomputed from CSV:", json.dumps(run))
    if not verify(run):
        raise SystemExit("Saved CSVs do not match the committed verified metrics; refusing to render.")
    cap = cv2.VideoCapture(str(a.video))
    if not cap.isOpened():
        raise SystemExit(f"Cannot open {a.video}")
    n_video = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if n_video != len(det):
        raise SystemExit(f"Video has {n_video} frames but CSV has {len(det)} rows; alignment unsafe.")
    out = H264Writer(a.out, 10.0, (W, VH + PH))
    for t in range(len(det)):
        ok, frame = cap.read()
        if not ok:
            raise SystemExit(f"Video ended early at frame {t + 1}")
        canvas = np.vstack([cv2.resize(frame, (W, VH), interpolation=cv2.INTER_AREA), build_panel(det, cond, run, t)])
        out.write(canvas)
    cap.release()
    print("Encoding:", out.close(), "->", a.out)


if __name__ == "__main__":
    main()
