
from pathlib import Path
from collections import defaultdict
import csv
import cv2

ROOT = Path(__file__).resolve().parents[1]

VIDEO = ROOT / "data/videos/Clip_1.mov"
LABELS = ROOT / "data/annotations/Clip_1_refined.txt"
OUTPUT = ROOT / "results/gt_sequence_clip1.mp4"

OUTPUT.parent.mkdir(exist_ok=True)

# Load frame-by-frame ground truth
annotations = defaultdict(list)

with open(LABELS, "r", encoding="utf-8-sig") as f:
    for row in csv.reader(f):
        if len(row) < 6:
            continue

        try:
            frame_no = int(float(row[0]))
            target_id = int(float(row[1]))
            x, y, w, h = map(float, row[2:6])
        except ValueError:
            continue

        if frame_no >= 1 and w > 0 and h > 0:
            annotations[frame_no].append(
                (target_id, x, y, w, h)
            )

cap = cv2.VideoCapture(str(VIDEO))

if not cap.isOpened():
    raise RuntimeError("Cannot open Clip_1.mov")

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

start_frame = 130
end_frame = 180

writer = cv2.VideoWriter(
    str(OUTPUT),
    cv2.VideoWriter_fourcc(*"mp4v"),
    10,
    (width, height)
)

if not writer.isOpened():
    cap.release()
    raise RuntimeError("Could not create output video.")

cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame - 1)

written = 0

for frame_no in range(start_frame, end_frame + 1):
    success, frame = cap.read()

    if not success:
        break

    for target_id, x, y, w, h in annotations.get(frame_no, []):
        x1 = max(0, int(round(x)))
        y1 = max(0, int(round(y)))
        x2 = min(width - 1, int(round(x + w)))
        y2 = min(height - 1, int(round(y + h)))

        if x2 <= x1 or y2 <= y1:
            continue

        cv2.rectangle(
            frame, (x1, y1), (x2, y2),
            (0, 255, 0), 3
        )

        cv2.putText(
            frame,
            f"ID {target_id}",
            (x1, max(20, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6, (0, 255, 0), 2
        )

    cv2.putText(
        frame,
        f"Frame {frame_no}",
        (30, 45),
        cv2.FONT_HERSHEY_SIMPLEX,
        1, (255, 255, 255), 2
    )

    writer.write(frame)
    written += 1

cap.release()
writer.release()

print("Frames saved:", written)
print("Output:", OUTPUT)
print("Done!")
