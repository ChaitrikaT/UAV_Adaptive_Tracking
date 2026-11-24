"""Run the online tracker with FAST / ROBUST / rule / learned policies on a recorded Purdue clip.

Examples (from the project root, venv active):
  python run_tracking.py --clip 7 --end 300 --policy all
  python run_tracking.py --clip 8 --policy learned --save-video
Clips 9 and 10 are the reserved TEST clips: they need --allow-test. Run them once, at the end.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import cv2

from uavtrack.evaluate import evaluate, load_predictions, load_reference
from uavtrack.pipeline import YoloDetector, run_video, summarize_timing
from uavtrack.policy import make_policy
from uavtrack.tracker import TrackerConfig

ROOT = Path(__file__).resolve().parent


def write_csv(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip", type=int, required=True)
    ap.add_argument("--policy", default="all", choices=["fast", "robust", "rule", "learned", "all"])
    ap.add_argument("--start", type=int, default=1)
    ap.add_argument("--end", type=int, default=-1, help="last frame (1-based); -1 = whole clip")
    ap.add_argument("--conf", type=float, default=0.10, help="detector confidence floor")
    ap.add_argument("--high-conf", type=float, default=0.30)
    ap.add_argument("--max-age", type=int, default=30)
    ap.add_argument("--emit-coasting", action="store_true")
    ap.add_argument("--state-mode", choices=["primary", "worst"], default="primary")
    ap.add_argument("--weights", type=Path, default=ROOT / "results/training/uav_detector_v1/weights/best.pt")
    ap.add_argument("--policy-model", type=Path, default=ROOT / "results/policy/online_policy.joblib")
    ap.add_argument("--save-video", action="store_true")
    ap.add_argument("--out", type=Path, default=ROOT / "results/tracking")
    ap.add_argument("--allow-test", action="store_true", help="required for reserved test clips 9 and 10")
    a = ap.parse_args()
    if a.clip in (9, 10) and not a.allow_test:
        ap.error("Clips 9 and 10 are reserved test clips. Re-run with --allow-test only for the final evaluation.")

    video = ROOT / "data/videos" / f"Clip_{a.clip}.mov"
    ref_path = ROOT / "data/annotations" / f"Clip_{a.clip}_refined.txt"
    for p in (video, ref_path, a.weights):
        if not p.exists():
            raise SystemExit(f"Missing file: {p}")
    names = ["fast", "robust", "rule", "learned"] if a.policy == "all" else [a.policy]
    if "learned" in names and not a.policy_model.is_file():
        if a.policy == "all":
            print("Skipping 'learned': no model yet (run train_online_policy.py first).")
            names.remove("learned")
    cfg = TrackerConfig(high_conf=a.high_conf, max_age=a.max_age, emit_coasting=a.emit_coasting, state_mode=a.state_mode)
    detector = YoloDetector(a.weights, conf=a.conf)
    cap = cv2.VideoCapture(str(video)); ok, first = cap.read(); cap.release()
    if not ok:
        raise SystemExit(f"Cannot read {video}")
    detector.warmup(first)
    H, W = first.shape[:2]
    ref = load_reference(ref_path, W, H)

    summary = []
    for name in names:
        print(f"== clip {a.clip} | policy {name}", flush=True)
        policy = make_policy(name, a.policy_model)
        meta = getattr(policy, "meta", {})
        for key, now in (("detector_conf", a.conf), ("high_conf", a.high_conf), ("max_age", a.max_age)):
            if key in meta and meta[key] != now:
                print(f"WARNING: learned policy was trained with {key}={meta[key]} but this run uses {now}.")
        out_dir = a.out / f"clip_{a.clip}" / name
        frame_rows, track_rows = run_video(video, detector, policy, a.start, a.end, cfg,
                                           out_dir / "tracked.mp4" if a.save_video else None)
        write_csv(out_dir / "frames.csv", frame_rows)
        write_csv(out_dir / "tracks.csv", track_rows)
        pred = load_predictions(out_dir / "tracks.csv") if track_rows else {}
        metrics = evaluate(ref, pred, [r["frame"] for r in frame_rows])
        result = {"policy": name, "clip": a.clip, **summarize_timing(frame_rows), **metrics}
        (out_dir / "summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        summary.append(result)
    write_csv(a.out / f"clip_{a.clip}" / "comparison.csv", summary)
    cols = ["policy", "frames_scored", "recall", "precision", "IDF1", "MOTA", "ID_switches",
            "distinct_pred_ids", "robust_fraction", "mean_total_ms_incl_decode", "end_to_end_fps"]
    print("\n" + " | ".join(cols))
    for r in summary:
        print(" | ".join(str(r.get(c, "")) for c in cols))
    print(f"\nSaved to {a.out / f'clip_{a.clip}'}")
    print("Timing = decode + features + policy + detector + tracker (video writing excluded).")


if __name__ == "__main__":
    main()
