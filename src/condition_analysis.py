"""Offline, descriptive condition analysis for an annotated video review.

Reads a local prerecorded video and an existing detector-review CSV.
Does not train a detector, produce target coordinates, or perform tracking.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
from statistics import mean, median

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def report_stats(title: str, rows: list[dict]) -> str:
    if not rows:
        return f"{title}: no frames"
    keys = ("brightness", "contrast", "sharpness", "frame_change")
    parts = [f"{title}: {len(rows)} frames"]
    for key in keys:
        nums = [float(r[key]) for r in rows if r[key] != ""]
        if nums:
            parts.append(f"  {key}: mean={mean(nums):.2f}, median={median(nums):.2f}")
    match_count = sum(int(r["matched_iou50"]) for r in rows)
    parts.append(f"  matched IoU>=0.5: {match_count}/{len(rows)} ({match_count / len(rows):.1%})")
    return "\n".join(parts)


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze visual conditions in a recorded video")
    parser.add_argument("--clip", type=int, default=7)
    args = parser.parse_args()

    video = ROOT / "data" / "videos" / f"Clip_{args.clip}.mov"
    review = ROOT / "results" / "videos" / f"clip_{args.clip}_detector_review.csv"
    outdir = ROOT / "results" / "conditions"
    outdir.mkdir(parents=True, exist_ok=True)

    for file in (video, review):
        if not file.is_file():
            raise FileNotFoundError(f"Missing input: {file}")

    with review.open("r", newline="", encoding="utf-8-sig") as f:
        raw = list(csv.DictReader(f))
    expected = {"frame", "predicted_boxes", "max_iou"}
    if not raw or not expected.issubset(raw[0]):
        raise ValueError(f"Review CSV needs columns: {sorted(expected)}")

    outcomes = {int(r["frame"]): r for r in raw}
    last_frame = max(outcomes)
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open video: {video}")

    previous_gray = None
    results = []
    try:
        for frame_number in range(1, last_frame + 1):
            ok, frame = cap.read()
            if not ok:
                raise RuntimeError(f"Video ended before frame {frame_number}")

            # Downsample a whole frame for consistent descriptive statistics.
            small = cv2.resize(frame, (480, 270), interpolation=cv2.INTER_AREA)
            gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)

            if frame_number in outcomes:
                item = outcomes[frame_number]
                max_iou = float(item["max_iou"])
                change = (
                    float(np.mean(cv2.absdiff(gray, previous_gray)))
                    if previous_gray is not None else None
                )
                results.append({
                    "frame": frame_number,
                    "brightness": round(float(gray.mean()), 3),
                    "contrast": round(float(gray.std()), 3),
                    "sharpness": round(float(cv2.Laplacian(gray, cv2.CV_64F).var()), 3),
                    "frame_change": "" if change is None else round(change, 3),
                    "has_prediction": int(int(item["predicted_boxes"]) > 0),
                    "matched_iou50": int(max_iou >= 0.5),
                    "max_iou": max_iou,
                })
            previous_gray = gray
    finally:
        cap.release()

    output_csv = outdir / f"clip_{args.clip}_condition_analysis.csv"
    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "frame", "brightness", "contrast", "sharpness", "frame_change",
            "has_prediction", "matched_iou50", "max_iou"
        ])
        writer.writeheader()
        writer.writerows(results)

    matched = [r for r in results if r["matched_iou50"]]
    missed = [r for r in results if not r["matched_iou50"]]
    text = [
        f"CLIP {args.clip} — OFFLINE VISUAL CONDITION ANALYSIS",
        f"Reviewed frames: {len(results)}",
        "", report_stats("Frames with IoU>=0.5 match", matched),
        "", report_stats("Frames without IoU>=0.5 match", missed),
    ]
    if args.clip == 7 and min(outcomes) <= 1 and last_frame >= 120:
        for label, a, b in [
            ("Early: 1-38", 1, 38),
            ("Failure interval: 39-106", 39, 106),
            ("Recovery: 107-120", 107, 120),
        ]:
            segment = [r for r in results if a <= r["frame"] <= b]
            text.extend(["", report_stats(label, segment)])
    text += [
        "", "Interpretation notes:",
        "- Metrics describe the entire frame, not specifically the UAV.",
        "- Frame change also includes camera motion and scene movement.",
        "- IoU matches use reference labels AFTER prediction for analysis only.",
        "- Correlations do not establish the cause of detection failure.",
        "- This does not produce track IDs or adaptive mode decisions.",
    ]
    report = "\n".join(text) + "\n"
    output_txt = outdir / f"clip_{args.clip}_condition_summary.txt"
    output_txt.write_text(report, encoding="utf-8")
    print(report)
    print(f"Saved CSV: {output_csv}")
    print(f"Saved summary: {output_txt}")


if __name__ == "__main__":
    main()
