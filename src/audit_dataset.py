
from pathlib import Path
import csv
import cv2
from collections import Counter

ROOT = Path(__file__).resolve().parents[1]

VIDEO_DIR = ROOT / "data" / "videos"
ANNO_DIR = ROOT / "data" / "annotations"
RESULT_DIR = ROOT / "results"

RESULT_DIR.mkdir(exist_ok=True)

# Provisional sequence-level split.
SPLITS = {
    "train": [1, 2, 3, 4, 5, 6],
    "val": [7, 8],
    "test": [9, 10],
}

report = []
manifest = []

for split, clip_numbers in SPLITS.items():

    for clip_no in clip_numbers:

        video = VIDEO_DIR / f"Clip_{clip_no}.mov"
        label = ANNO_DIR / f"Clip_{clip_no}_refined.txt"

        print(f"\nChecking Clip {clip_no} ({split})")

        manifest.append({
            "clip": clip_no,
            "split": split,
            "video": str(video),
            "annotation": str(label),
        })

        if not video.exists() or not label.exists():
            print("MISSING VIDEO OR ANNOTATION")
            continue

        cap = cv2.VideoCapture(str(video))

        if not cap.isOpened():
            print("ERROR: Video cannot be opened")
            continue

        frame_count = int(
            cap.get(cv2.CAP_PROP_FRAME_COUNT)
        )
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(
            cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        )
        height = int(
            cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        )

        cap.release()

        annotated_frames = set()
        target_ids = set()
        boxes_per_frame = Counter()

        total_boxes = 0
        invalid_boxes = 0
        out_of_range_frames = 0

        with open(
            label, "r", encoding="utf-8-sig",
            errors="replace"
        ) as f:

            reader = csv.reader(f)

            for row in reader:

                if len(row) < 6:
                    continue

                try:
                    frame = int(float(row[0]))
                    target_id = int(float(row[1]))
                    x, y, w, h = map(
                        float, row[2:6]
                    )
                except ValueError:
                    invalid_boxes += 1
                    continue

                total_boxes += 1

                if frame < 1 or frame > frame_count:
                    out_of_range_frames += 1
                    continue

                annotated_frames.add(frame)
                target_ids.add(target_id)
                boxes_per_frame[frame] += 1

                if not (
                    w > 0 and h > 0
                    and x >= 0 and y >= 0
                    and x + w <= width
                    and y + h <= height
                ):
                    invalid_boxes += 1

        coverage = (
            len(annotated_frames) / frame_count
            if frame_count > 0 else 0
        )

        result = {
            "clip": clip_no,
            "split": split,
            "frames": frame_count,
            "fps": round(fps, 2),
            "width": width,
            "height": height,
            "annotated_frames": len(annotated_frames),
            "annotation_coverage": round(
                coverage, 4
            ),
            "total_boxes": total_boxes,
            "unique_target_ids": len(target_ids),
            "max_boxes_in_frame": max(
                boxes_per_frame.values(), default=0
            ),
            "invalid_boxes": invalid_boxes,
            "out_of_range_frames":
                out_of_range_frames,
        }

        report.append(result)

        print("Video frames:", frame_count)
        print("Resolution:", width, "x", height)
        print("FPS:", round(fps, 2))
        print("Annotated frames:", len(annotated_frames))
        print("Total boxes:", total_boxes)
        print("Unique IDs:", len(target_ids))
        print("Invalid boxes:", invalid_boxes)
        print(
            "Out-of-range annotations:",
            out_of_range_frames
        )

if report:
    output = RESULT_DIR / "dataset_audit.csv"

    with open(output, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=report[0].keys()
        )
        writer.writeheader()
        writer.writerows(report)

    print("\nSaved:", output)

manifest_output = RESULT_DIR / "split_manifest.csv"

with open(manifest_output, "w", newline="") as f:
    writer = csv.DictWriter(
        f, fieldnames=[
            "clip", "split", "video", "annotation"
        ]
    )
    writer.writeheader()
    writer.writerows(manifest)

print("Saved:", manifest_output)
print("\nDataset audit finished.")
