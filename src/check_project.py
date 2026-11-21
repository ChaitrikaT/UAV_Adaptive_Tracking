"""Read-only project / demo preflight. Standard-library only.

Run from any directory: python src/check_project.py
"""
from __future__ import annotations

import json
from html.parser import HTMLParser
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
DEMO = ROOT / "demo"


class Assets(HTMLParser):
    def __init__(self):
        super().__init__()
        self.sources = []

    def handle_starttag(self, tag, attrs):
        if tag.lower() not in {"video", "source", "img"}:
            return
        d = dict(attrs)
        if d.get("src"):
            self.sources.append(d["src"])


def check() -> int:
    errors, warnings = [], []
    page = DEMO / "final" / "index.html"
    metrics = DEMO / "final" / "verified_metrics.json"
    for path in (page, metrics, ROOT / "run_demo.py", ROOT / "README.md"):
        if not path.is_file() or path.stat().st_size == 0:
            errors.append(f"Missing or empty: {path.relative_to(ROOT)}")

    if page.exists():
        parser = Assets()
        parser.feed(page.read_text(encoding="utf-8"))
        for source in parser.sources:
            if "://" in source or source.startswith("data:"):
                warnings.append(f"External/dashboard asset, not verified: {source}")
                continue
            path = (page.parent / source.split("?")[0].split("#")[0]).resolve()
            if not path.is_relative_to(DEMO.resolve()):
                errors.append(f"Asset escapes demo folder: {source}")
            elif not path.is_file() or path.stat().st_size == 0:
                errors.append(f"Missing HTML asset: {source}")
        if not any(s.lower().endswith(".mp4") for s in parser.sources):
            warnings.append("No MP4 was referenced by the dashboard")

    if metrics.exists():
        try:
            m = json.loads(metrics.read_text(encoding="utf-8"))
            frames, detected, matched = (int(m[k]) for k in
                ("frames", "frames_with_detections", "frames_with_iou_at_least_05"))
            if not 0 <= matched <= detected <= frames:
                errors.append("Video frame counts are internally inconsistent")
            if abs(100 * matched / frames - float(m["frame_match_rate_pct"])) > 0.02:
                errors.append("Reported video match rate does not match counts")
            for r in m["selector_comparison"]:
                tp, fp, fn = int(r["tp"]), int(r["fp"]), int(r["fn"])
                if min(tp, fp, fn) < 0:
                    errors.append(f"Negative TP/FP/FN: {r['policy']}")
                recall = 100 * tp / (tp + fn) if tp + fn else 0
                if abs(recall - float(r["recall_pct"])) > 0.02:
                    errors.append(f"Invalid saved recall: {r['policy']}")
        except (ValueError, ZeroDivisionError, KeyError, TypeError) as e:
            errors.append(f"Invalid verified_metrics.json: {e}")

    for name in ("src/benchmark_modes.py", "src/train_selector.py", "src/adaptive.py",
                 "src/review_detector_video.py", "src/evaluate_saved_track_ids.py"):
        if not (ROOT / name).is_file():
            warnings.append(f"Source missing from checkout: {name}")
    if not (ROOT / "data").exists():
        warnings.append("Purdue data not present (expected for a demo-only clone)")
    if not (ROOT / "results" / "training").exists():
        warnings.append("Trained checkpoint not present (expected for a demo-only clone)")

    print("PROJECT DEMO PRE-FLIGHT")
    print("=======================")
    for issue in errors:
        print("ERROR:", issue)
    for issue in warnings:
        print("NOTE:", issue)
    print("RESULT:", "DEMO NOT READY" if errors else "SAVED DEMO READY")
    print("SCOPE: Saved detection/selector results only; no predicted UAV track IDs or live adaptive video.")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(check())
