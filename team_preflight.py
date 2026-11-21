"""Read-only status overview for a teammate's local checkout.

Standard-library only: prints clear separation between committed demo,
local experiment inputs and retrospective research add-on.
"""
from __future__ import annotations

from pathlib import Path
import argparse
import sys

ROOT = Path(__file__).resolve().parent

GROUPS = {
    "Committed saved demo": [
        "run_demo.py", "demo/final/index.html",
        "demo/final/verified_metrics.json",
        "demo/videos/clip_7_detector_review_h264.mp4",
    ],
    "Offline research add-on": [
        "run_research_review.py", "research_addon/benchmark.py",
        "research_addon/policy_audit.py", "research_addon/report.py",
        "research_addon/track_audit.py", "research_addon/timeline.py",
        "tests/test_research_addon.py",
    ],
    "Local experiment data (optional in clean clone)": [
        "results/offline_benchmark/per_frame.csv",
        "results/selector/mode_decisions.csv",
        "results/selector/experiment.json",
        "results/videos/clip_7_detector_review.csv",
        "results/conditions/clip_7_condition_analysis.csv",
        "results/training/uav_detector_v1/weights/best.pt",
    ],
}


def inspect(root: Path) -> tuple[dict[str, dict[str, bool]], bool]:
    status = {
        group: {name: (root / name).is_file() for name in paths}
        for group, paths in GROUPS.items()
    }
    healthy_demo = all(status["Committed saved demo"].values())
    return status, healthy_demo


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=ROOT)
    args = p.parse_args(argv)
    root = args.root.resolve()
    status, healthy_demo = inspect(root)
    print("TEAM CHECKPOINT — offline perception mini-project")
    print("Project folder:", root)
    for group, checks in status.items():
        print("\n" + group)
        for name, exists in checks.items():
            print(f"  {'OK     ' if exists else 'ABSENT '} {name}")
    print("\nSAVED DEMO:", "READY" if healthy_demo else "INCOMPLETE")
    print("NOTE: Local results/weights are intentionally absent in a clean Git clone.")
    print("NOTE: This check does not validate UAV tracking or live mode switching.")
    return 0 if healthy_demo else 1


if __name__ == "__main__":
    sys.exit(main())
