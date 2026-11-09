
from pathlib import Path
import csv

ROOT = Path(__file__).resolve().parents[1]

CSV_PATH = ROOT / "results" / "videos" / "clip_7_detector_review.csv"
OUTPUT = ROOT / "results" / "videos" / "clip_7_baseline_report.txt"

if not CSV_PATH.exists():
    raise FileNotFoundError(f"Missing: {CSV_PATH}")

with CSV_PATH.open(newline="", encoding="utf-8") as f:
    rows = list(csv.DictReader(f))

frames = len(rows)
predicted = sum(int(r["predicted_boxes"]) > 0 for r in rows)
matched = sum(float(r["max_iou"]) >= 0.5 for r in rows)

# Find the longest continuous period without a matched detection.
longest = 0
current = 0
start = None
worst_start = None
worst_end = None

for r in rows:
    frame = int(r["frame"])
    success = float(r["max_iou"]) >= 0.5

    if not success:
        if current == 0:
            start = frame
        current += 1

        if current > longest:
            longest = current
            worst_start = start
            worst_end = frame
    else:
        current = 0

report = f"""
OFFLINE DETECTION BASELINE — CLIP 7

Frames evaluated: {frames}
Frames with model predictions: {predicted}
Frames with IoU >= 0.5: {matched}
Frame match rate: {matched / frames:.2%}
Longest consecutive interval without IoU >= 0.5:
  Frames {worst_start} to {worst_end}
  Length: {longest} frames

NOTE:
This measures frame-level detection continuity.
It does NOT measure tracking identity accuracy.
"""

print(report)
OUTPUT.write_text(report, encoding="utf-8")
print(f"Saved report: {OUTPUT}")
