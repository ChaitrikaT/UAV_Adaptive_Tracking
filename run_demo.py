"""Run the condition-aware pipeline on ANY video and write annotated mp4s + a comparison table.

Put this file in the SAME folder that contains the `uavtrack` package (your project root), then e.g.:

  # one policy, one video
  python run_demo.py --video path\\to\\clip.mp4 --weights path\\to\\uav_detector_v1.pt --policy robust

  # all policies on the same video, in one go (warms up first so run order does not bias timing)
  python run_demo.py --video path\\to\\clip.mp4 --weights path\\to\\uav_detector_v1.pt --compare \
         --difficulty-model path\\to\\difficulty_policy.joblib

Policies: fast | robust | rule | difficulty | learned
  difficulty needs --difficulty-model (a joblib bundle with keys estimator, tau, hold)
  learned    needs --learned-model    (the decision-tree joblib from your training run)
Tracker settings are the frozen ones from the paper: high_conf=0.4, max_age=60, emit_coasting=False.
There is NO ground truth here: you get boxes, IDs, mode timeline and timing, not recall/IDF1.
"""
import argparse
import csv
import sys
from pathlib import Path

from uavtrack.pipeline import YoloDetector, run_video, summarize_timing
from uavtrack.policy import make_policy
from uavtrack.tracker import TrackerConfig

FROZEN = dict(high_conf=0.4, max_age=60, emit_coasting=False)


def build_policy(name, args):
    if name == "difficulty":
        from uavtrack.difficulty import DifficultyPolicy
        if not args.difficulty_model:
            sys.exit("--difficulty-model is required for the difficulty policy")
        return DifficultyPolicy.load(Path(args.difficulty_model))[0]
    return make_policy(name, Path(args.learned_model) if args.learned_model else None)


def run_one(name, args, detector):
    policy = build_policy(name, args)
    out = Path(args.out) / f"{Path(args.video).stem}_{name}.mp4"
    print(f"\n== {name} -> {out}")
    frames, tracks = run_video(Path(args.video), detector, policy, start=args.start, end=args.end,
                               tracker_cfg=TrackerConfig(**FROZEN), video_out=out,
                               banner_label=f"policy={name} (no ground truth)")
    s = summarize_timing(frames)
    s.update(policy=name, distinct_track_ids=len({t["track_id"] for t in tracks}),
             frames_with_a_track=len({t["frame"] for t in tracks}))
    return s


def main(argv=None, detector=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--video", required=True)
    ap.add_argument("--weights", help="YOLO .pt file (your trained detector)")
    ap.add_argument("--policy", default="rule", choices=["fast", "robust", "rule", "difficulty", "learned"])
    ap.add_argument("--compare", action="store_true", help="run fast, robust, rule (+difficulty/learned if model given)")
    ap.add_argument("--difficulty-model")
    ap.add_argument("--learned-model")
    ap.add_argument("--start", type=int, default=1)
    ap.add_argument("--end", type=int, default=-1, help="-1 = until the end")
    ap.add_argument("--conf", type=float, default=0.10, help="detector confidence floor (paper: 0.10)")
    ap.add_argument("--out", default="demo_out")
    args = ap.parse_args(argv)

    if detector is None:
        if not args.weights:
            sys.exit("--weights is required")
        detector = YoloDetector(Path(args.weights), conf=args.conf)
    import cv2
    cap = cv2.VideoCapture(args.video)
    ok, first = cap.read()
    cap.release()
    if not ok:
        sys.exit(f"cannot read {args.video}")
    if hasattr(detector, "warmup"):
        detector.warmup(first)          # avoid the cold-start effect on the first run

    names = [args.policy]
    if args.compare:
        names = ["fast", "robust", "rule"] + (["difficulty"] if args.difficulty_model else []) \
                + (["learned"] if args.learned_model else [])
    rows = [run_one(n, args, detector) for n in names]

    Path(args.out).mkdir(parents=True, exist_ok=True)
    table = Path(args.out) / f"{Path(args.video).stem}_summary.csv"
    keys = ["policy", "frames", "robust_fraction", "mean_detect_ms", "mean_total_ms_incl_decode",
            "end_to_end_fps", "distinct_track_ids", "frames_with_a_track"]
    with open(table, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows({k: r.get(k) for k in keys} for r in rows)
    print("\n" + "  ".join(f"{k[:14]:>14}" for k in keys))
    for r in rows:
        print("  ".join(f"{str(r.get(k)):>14}" for k in keys))
    print(f"\nSaved: {table}  and annotated videos in {args.out}/")
    return rows


if __name__ == "__main__":
    main()