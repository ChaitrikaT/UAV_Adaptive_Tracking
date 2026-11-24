"""End-to-end integrated demonstration on a RECORDED clip (offline research; no live following, no control output).

Per frame:  decode -> condition features -> difficulty estimate -> FAST/ROBUST decision -> YOLO (640 or 960)
            -> tracker (own predicted IDs) -> annotated video + per-frame logs -> tracking metrics vs reference.
  python make_integrated_demo.py --clip 8 --end 900 --policy difficulty --model results/policy/difficulty_train7_default.joblib
Outputs in results/final/demo/: clip<N>_<policy>_integrated_demo.mp4, frames.csv, tracks.csv, metrics.json, id_summary.csv
Green thin boxes = reference annotation (evaluation overlay only); orange boxes = tracker output with PREDICTED IDs.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np

import run_evaluation as RE
from run_final_evaluation import load_settings
from uavtrack.difficulty import DifficultyPolicy
from uavtrack.evaluate import evaluate, load_reference
from uavtrack.pipeline import run_video, summarize_timing
from uavtrack.policy import MODES, make_policy
from uavtrack.tracker import TrackerConfig
from uavtrack.video_io import transcode_h264

W, PH, HIST = 1280, 230, 300


def write_csv(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        if rows:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--clip", type=int, default=8)
    ap.add_argument("--start", type=int, default=1); ap.add_argument("--end", type=int, default=900)
    ap.add_argument("--policy", choices=["difficulty", "rule", "fast", "robust", "learned"], default="difficulty")
    ap.add_argument("--model", type=Path, default=None, help="difficulty_train*.joblib (or online_policy.joblib for --policy learned)")
    ap.add_argument("--tag", default="default"); ap.add_argument("--conf", type=float, default=0.10)
    ap.add_argument("--weights", type=Path, default=RE.ROOT / "results/training/uav_detector_v1/weights/best.pt")
    ap.add_argument("--device", default="auto"); ap.add_argument("--no-reference", action="store_true", help="do not draw reference boxes")
    ap.add_argument("--allow-test", action="store_true")
    ap.add_argument("--high-conf", type=float, default=None); ap.add_argument("--max-age", type=int, default=None); ap.add_argument("--emit-coasting", action="store_true")
    a = ap.parse_args()
    if a.clip in RE.TEST_CLIPS and not a.allow_test:
        raise SystemExit("Clips 9/10 are the reserved test set: add --allow-test only for the final run.")
    st = load_settings(a.tag, a); cfg = TrackerConfig(high_conf=st["high_conf"], max_age=st["max_age"], emit_coasting=st["emit_coasting"])
    video, ref_path = RE.find_video(a.clip), RE.find_ref(a.clip)
    if a.policy == "difficulty":
        model = a.model or (RE.policy_dir() / f"difficulty_train7_{a.tag}.joblib")
        if not Path(model).is_file():
            raise SystemExit(f"Missing {model}. Run run_final_evaluation.py --train 7 --eval 8 first.")
        policy, _ = DifficultyPolicy.load(model)
    else:
        policy = make_policy(a.policy, a.model or RE.policy_dir() / "online_policy.joblib")
    cap = cv2.VideoCapture(str(video)); ok, first = cap.read(); cap.release()
    if not ok:
        raise SystemExit(f"Cannot read {video}")
    H0, W0 = first.shape[:2]
    ref = load_reference(ref_path, W0, H0)
    device = None if a.device == "auto" else (a.device if a.device == "cpu" else int(a.device))
    det = RE.make_detector(a.weights, a.conf, device); det.warmup(first)
    out_dir = RE.ROOT / "results" / "final" / "demo"; out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"clip{a.clip}_{a.policy}"
    hist = []

    def draw(frame, tracks, mode, frame_no):
        s = getattr(policy, "last_score", None)
        hist.append((np.nan if s is None else s, mode))
        sc = W / frame.shape[1]; img = cv2.resize(frame, (W, int(frame.shape[0] * sc)), interpolation=cv2.INTER_AREA)
        if not a.no_reference:
            for g in ref.get(frame_no, []):
                cv2.rectangle(img, (int(g[1] * sc) - 3, int(g[2] * sc) - 3), (int(g[3] * sc) + 3, int(g[4] * sc) + 3), (60, 200, 60), 1)
        for t in tracks:
            p1, p2 = (int(t["x1"] * sc), int(t["y1"] * sc)), (int(t["x2"] * sc), int(t["y2"] * sc))
            col = (0, 160, 255) if t["observed"] else (150, 150, 150)
            cv2.rectangle(img, p1, p2, col, 2)
            cv2.putText(img, f"ID {t['track_id']} {t['conf']:.2f}", (p1[0], max(16, p1[1] - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2, cv2.LINE_AA)
        cv2.rectangle(img, (0, 0), (W, 34), (20, 20, 20), -1)
        cv2.putText(img, f"RECORDED CLIP {a.clip} | frame {frame_no} | policy {a.policy} | orange = tracker output with PREDICTED IDs, green = reference (evaluation only)",
                    (8, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        panel = np.full((PH, W, 3), 28, np.uint8)
        h = hist[-HIST:]; x0, x1, y0, y1 = 60, W - 20, 40, 130
        cv2.putText(panel, "estimated difficulty (prob. FAST misses) - dotted = switch threshold", (x0, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (230, 230, 230), 1, cv2.LINE_AA)
        cv2.rectangle(panel, (x0, y0), (x1, y1), (70, 70, 70), 1)
        tau = getattr(policy, "tau", None)
        if tau is not None and tau <= 1:
            yt = int(y1 - (y1 - y0) * tau)
            for xx in range(x0, x1, 12):
                cv2.line(panel, (xx, yt), (xx + 6, yt), (200, 200, 60), 1)
        pts = [(int(x0 + (x1 - x0) * i / max(1, HIST - 1)), int(y1 - (y1 - y0) * v)) for i, (v, _) in enumerate(h) if not np.isnan(v)]
        if len(pts) > 1:
            cv2.polylines(panel, [np.array(pts, np.int32)], False, (200, 120, 230), 2, cv2.LINE_AA)
        for i, (_, m) in enumerate(h):
            xx = int(x0 + (x1 - x0) * i / max(1, HIST - 1))
            cv2.line(panel, (xx, 150), (xx, 172), (60, 60, 230) if m == "ROBUST" else (190, 130, 60), 3)
        cv2.putText(panel, "mode per frame: blue = FAST (640 px), red = ROBUST (960 px)", (x0, 190), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (230, 230, 230), 1, cv2.LINE_AA)
        ids = sorted({t["track_id"] for t in tracks})
        cv2.putText(panel, f"mode {mode} ({MODES[mode]} px) | difficulty {'n/a' if s is None else f'{s:.2f}'} | active predicted IDs {ids}",
                    (x0, 214), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        return np.vstack([img, panel])

    tmp = out_dir / f"{stem}.tmp.mp4"
    frame_rows, track_rows = run_video(video, det, policy, a.start, a.end, cfg, video_out=tmp, log_every=100, draw_fn=draw)
    status = transcode_h264(tmp, out_dir / f"{stem}_integrated_demo.mp4"); tmp.unlink(missing_ok=True)
    if hasattr(policy, "est"):                       # add the difficulty score per frame (recomputed from the logged features)
        from uavtrack.features import FEATURE_NAMES
        for r in frame_rows:
            r["difficulty"] = round(policy.est.score({k: r[k] for k in FEATURE_NAMES}), 4)
    write_csv(out_dir / f"{stem}_frames.csv", frame_rows); write_csv(out_dir / f"{stem}_tracks.csv", track_rows)
    pred = {}
    for t in track_rows:
        pred.setdefault(t["frame"], []).append((t["track_id"], t["x1"], t["y1"], t["x2"], t["y2"]))
    metrics = evaluate(ref, pred, [r["frame"] for r in frame_rows])
    life = {}
    for t in track_rows:
        d = life.setdefault(t["track_id"], {"track_id": t["track_id"], "first_frame": t["frame"], "last_frame": t["frame"], "frames": 0})
        d["last_frame"] = t["frame"]; d["frames"] += 1
    write_csv(out_dir / f"{stem}_id_summary.csv", sorted(life.values(), key=lambda d: d["track_id"]))
    out = {"clip": a.clip, "policy": a.policy, "frames": [a.start, a.end], "tracker_settings": st, "detector_conf": a.conf, "video_encoding": status,
           "timing_end_to_end_incl_decode_excl_video_writing": summarize_timing(frame_rows), "tracking_vs_reference": metrics,
           "reference_ids_in_clip": len({g[0] for f in ref for g in ref[f]})}
    (out_dir / f"{stem}_metrics.json").write_text(json.dumps(out, indent=2, default=float), encoding="utf-8")
    print(json.dumps(out, indent=2, default=float)); print(f"\nVideo: {out_dir / (stem + '_integrated_demo.mp4')}")


if __name__ == "__main__":
    main()
