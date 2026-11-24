"""Train the FAST/ROBUST decision tree from ONE clip, with features identical to deployment.

  python train_online_policy.py --clip 7 --end 600
Step 1 runs the detector in BOTH modes on every frame (cached to results/policy/cache_clip_N.pkl).
Step 2 rolls policies out over the cache (rule policy, then the learned tree, DAgger-style) so the
        tracker-state features come from the modes the policy itself chose, as in deployment.
Frame labels use reference boxes only (see uavtrack/collect.py::label_frame). Frames where BOTH modes
find nothing are excluded by default (--tie-both-fail exclude); ties where both find the target -> FAST.
Evaluate the result on an unseen clip:  python run_tracking.py --clip 8 --policy all
Never train on clips 9/10 (reserved test).
"""
from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

from uavtrack.collect import cache_clip, dagger_train
from uavtrack.features import FEATURE_NAMES
from uavtrack.tracker import TrackerConfig

ROOT = Path(__file__).resolve().parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip", type=int, default=7)
    ap.add_argument("--start", type=int, default=1)
    ap.add_argument("--end", type=int, default=-1)
    ap.add_argument("--conf", type=float, default=0.10)
    ap.add_argument("--high-conf", type=float, default=0.30, help="tracker setting; must match run_tracking.py")
    ap.add_argument("--max-age", type=int, default=30)
    ap.add_argument("--weights", type=Path, default=ROOT / "results/training/uav_detector_v1/weights/best.pt")
    ap.add_argument("--max-depth", type=int, default=3)
    ap.add_argument("--iterations", type=int, default=2)
    ap.add_argument("--tie-both-fail", choices=["exclude", "fast", "robust"], default="exclude")
    ap.add_argument("--tie-equal-hit", choices=["fast", "robust"], default="fast")
    ap.add_argument("--tag", default="default")
    ap.add_argument("--reuse-cache", action="store_true", help="skip detector runs; reuse saved cache")
    a = ap.parse_args()
    if a.clip in (9, 10):
        ap.error("Clips 9 and 10 are reserved test clips and must not be used for training.")
    out = ROOT / "results/policy"
    out.mkdir(parents=True, exist_ok=True)
    cache_path = out / f"cache_clip_{a.clip}_{a.tag}.pkl"
    if a.reuse_cache and cache_path.is_file():
        cache = pickle.loads(cache_path.read_bytes())
        print(f"Reusing {cache_path} ({len(cache['frames'])} frames)")
    else:
        from uavtrack.pipeline import YoloDetector
        import cv2
        video = ROOT / "data/videos" / f"Clip_{a.clip}.mov"
        ref = ROOT / "data/annotations" / f"Clip_{a.clip}_refined.txt"
        for p in (video, ref, a.weights):
            if not p.exists():
                raise SystemExit(f"Missing file: {p}")
        det = YoloDetector(a.weights, conf=a.conf)
        cap = cv2.VideoCapture(str(video)); ok, first = cap.read(); cap.release()
        if not ok:
            raise SystemExit(f"Cannot read {video}")
        det.warmup(first)
        cache = cache_clip(video, ref, det, a.start, a.end)
        cache_path.write_bytes(pickle.dumps(cache))
    cfg = TrackerConfig(high_conf=a.high_conf, max_age=a.max_age)
    try:
        clf, report = dagger_train(cache, cfg, a.iterations, a.max_depth, a.tie_both_fail, a.tie_equal_hit)
    except ValueError as exc:
        raise SystemExit(f"{exc}\nUse more frames or report that ROBUST never helps on this clip.")
    import joblib
    from sklearn.tree import export_text
    bundle = {"model": clf, "feature_names": FEATURE_NAMES, "detector_conf": a.conf, "high_conf": a.high_conf,
              "max_age": a.max_age, "train_clip": a.clip, "tie_rules": report["tie_rules"]}
    joblib.dump(bundle, out / "online_policy.joblib")
    text = export_text(clf, feature_names=FEATURE_NAMES)
    (out / "online_policy_tree.txt").write_text(text, encoding="utf-8")
    (out / "training_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(text)
    print(f"Saved: {out / 'online_policy.joblib'}")
    print("Next: python run_tracking.py --clip 8 --policy all   (true online evaluation on an unseen clip)")


if __name__ == "__main__":
    main()
