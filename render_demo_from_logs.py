"""Re-render the integrated demo video from the SAVED logs. No detector, no GPU, no re-evaluation.

  python render_demo_from_logs.py --clip 8 --policy difficulty --model results\\policy\\difficulty_train7_default.joblib
Reads  results/final/demo/clip<N>_<policy>_frames.csv and _tracks.csv (written by make_integrated_demo.py)
       and the recorded clip data/videos/Clip_<N>.mov.   Writes  results/final/demo/clip<N>_<policy>_integrated_demo.mp4
Orange boxes = tracker output with PREDICTED IDs; green thin boxes = reference annotation (evaluation overlay only).
"""
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

import run_evaluation as RE
from uavtrack.evaluate import load_reference
from uavtrack.policy import MODES
from uavtrack.video_io import transcode_h264

W, PH, HIST = 1280, 230, 300


def compose(frame, frame_no, clip, policy_name, tracks, ref_boxes, mode, score, tau, hist):
    sc = W / frame.shape[1]
    img = cv2.resize(frame, (W, int(frame.shape[0] * sc)), interpolation=cv2.INTER_AREA)
    for g in ref_boxes:
        cv2.rectangle(img, (int(g[1] * sc) - 3, int(g[2] * sc) - 3), (int(g[3] * sc) + 3, int(g[4] * sc) + 3), (60, 200, 60), 1)
    for t in tracks:
        p1, p2 = (int(t["x1"] * sc), int(t["y1"] * sc)), (int(t["x2"] * sc), int(t["y2"] * sc))
        col = (0, 160, 255) if t["observed"] else (150, 150, 150)
        cv2.rectangle(img, p1, p2, col, 2)
        cv2.putText(img, f"ID {t['track_id']} {t['conf']:.2f}", (p1[0], max(16, p1[1] - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2, cv2.LINE_AA)
    cv2.rectangle(img, (0, 0), (W, 34), (20, 20, 20), -1)
    cv2.putText(img, f"RECORDED CLIP {clip} | frame {frame_no} | policy {policy_name} | orange = tracker output (PREDICTED IDs), green = reference (evaluation only)",
                (8, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    panel = np.full((PH, W, 3), 28, np.uint8)
    h = hist[-HIST:]; x0, x1, y0, y1 = 60, W - 20, 40, 130
    cv2.putText(panel, "estimated difficulty (prob. FAST misses); dotted line = switch threshold", (x0, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (230, 230, 230), 1, cv2.LINE_AA)
    cv2.rectangle(panel, (x0, y0), (x1, y1), (70, 70, 70), 1)
    if tau is not None and tau <= 1:
        yt = int(y1 - (y1 - y0) * tau)
        for xx in range(x0, x1, 12):
            cv2.line(panel, (xx, yt), (xx + 6, yt), (200, 200, 60), 1)
    pts = [(int(x0 + (x1 - x0) * i / max(1, HIST - 1)), int(y1 - (y1 - y0) * v)) for i, (v, _) in enumerate(h) if v is not None]
    if len(pts) > 1:
        cv2.polylines(panel, [np.array(pts, np.int32)], False, (200, 120, 230), 2, cv2.LINE_AA)
    for i, (_, m) in enumerate(h):
        xx = int(x0 + (x1 - x0) * i / max(1, HIST - 1))
        cv2.line(panel, (xx, 150), (xx, 172), (60, 60, 230) if m == "ROBUST" else (190, 130, 60), 3)
    cv2.putText(panel, "mode per frame: blue = FAST (640 px), red = ROBUST (960 px)", (x0, 190), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (230, 230, 230), 1, cv2.LINE_AA)
    ids = sorted({t["track_id"] for t in tracks})
    cv2.putText(panel, f"mode {mode} ({MODES[mode]} px) | difficulty {'n/a' if score is None else f'{score:.2f}'} | active predicted IDs {ids}",
                (x0, 214), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return np.vstack([img, panel])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--clip", type=int, required=True); ap.add_argument("--policy", default="difficulty")
    ap.add_argument("--model", type=Path, default=None, help="difficulty_train*.joblib (only used to draw the threshold line)")
    ap.add_argument("--max-frames", type=int, default=None, help="render only the first N logged frames (quick check)")
    ap.add_argument("--no-reference", action="store_true"); ap.add_argument("--allow-test", action="store_true")
    a = ap.parse_args()
    if a.clip in RE.TEST_CLIPS and not a.allow_test:
        raise SystemExit("Clips 9/10 are the reserved test set: add --allow-test (rendering logs only, no new evaluation).")
    d = RE.ROOT / "results" / "final" / "demo"; stem = f"clip{a.clip}_{a.policy}"
    fcsv, tcsv = d / f"{stem}_frames.csv", d / f"{stem}_tracks.csv"
    if not fcsv.is_file():
        raise SystemExit(f"Missing {fcsv}")
    frames = list(csv.DictReader(fcsv.open(encoding="utf-8-sig")))
    tracks = defaultdict(list)
    if tcsv.is_file():
        for r in csv.DictReader(tcsv.open(encoding="utf-8-sig")):
            tracks[int(r["frame"])].append({"track_id": int(r["track_id"]), "x1": float(r["x1"]), "y1": float(r["y1"]), "x2": float(r["x2"]),
                                            "y2": float(r["y2"]), "conf": float(r["conf"]), "observed": int(float(r["observed"]))})
    tau = None
    if a.model and Path(a.model).is_file():
        import joblib
        tau = float(joblib.load(a.model)["tau"])
    video = RE.find_video(a.clip)
    cap = cv2.VideoCapture(str(video)); W0 = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); H0 = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = min(cap.get(cv2.CAP_PROP_FPS) or 30.0, 30.0)
    ref = {} if a.no_reference else load_reference(RE.find_ref(a.clip), W0, H0)
    first = int(frames[0]["frame"])
    if first > 1:
        cap.set(cv2.CAP_PROP_POS_FRAMES, first - 1)
    if a.max_frames:
        frames = frames[:a.max_frames]
    tmp, out = d / f"{stem}.render.tmp.mp4", d / f"{stem}_integrated_demo.mp4"
    writer, hist, n = None, [], 0
    for row in frames:
        ok, frame = cap.read()
        if not ok:
            break
        fno = int(row["frame"]); score = float(row["difficulty"]) if row.get("difficulty") not in (None, "") else None
        hist.append((score, row["mode"]))
        img = compose(frame, fno, a.clip, a.policy, tracks.get(fno, []), ref.get(fno, []), row["mode"], score, tau, hist)
        if writer is None:
            writer = cv2.VideoWriter(str(tmp), cv2.VideoWriter_fourcc(*"mp4v"), fps, (img.shape[1], img.shape[0]))
            if not writer.isOpened():
                raise SystemExit(f"Could not open video writer ({img.shape[1]}x{img.shape[0]})")
        writer.write(img); n += 1
        if n % 100 == 0:
            print(f"  rendered {n} frames", flush=True)
    cap.release()
    if writer is None:
        raise SystemExit("No frames were rendered (could not read the video).")
    writer.release()
    status = transcode_h264(tmp, out); tmp.unlink(missing_ok=True)
    chk = cv2.VideoCapture(str(out)); got = int(chk.get(cv2.CAP_PROP_FRAME_COUNT)); chk.release()
    print(f"Rendered {n} frames -> {out} ({status}); file reports {got} frames, {out.stat().st_size / 1e6:.1f} MB")
    if got < n * 0.95:
        print("WARNING: the output has fewer frames than rendered. Open it with VLC; if it still fails send me this line.")


if __name__ == "__main__":
    main()
