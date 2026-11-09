
from pathlib import Path
from collections import defaultdict
import csv
import cv2

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "inspection"
OUT.mkdir(parents=True, exist_ok=True)

for clip_no in [1, 2, 3]:

    video = ROOT / "data" / "videos" / f"Clip_{clip_no}.mov"
    label = ROOT / "data" / "annotations" / f"Clip_{clip_no}_refined.txt"

    if not video.exists() or not label.exists():
        print(f"Clip {clip_no}: Missing file")
        continue

    boxes = defaultdict(list)

    with open(label, encoding="utf-8-sig") as f:
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
                boxes[frame_no].append(
                    (target_id, x, y, w, h)
                )

    frames = sorted(boxes)

    if not frames:
        print(f"Clip {clip_no}: No annotations")
        continue

    # Inspect early, middle and late portions
    samples = sorted(set([
        frames[0],
        frames[len(frames) // 2],
        frames[-1]
    ]))

    cap = cv2.VideoCapture(str(video))

    if not cap.isOpened():
        print(f"Clip {clip_no}: Cannot open video")
        continue

    for frame_no in samples:

        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no - 1)
        ok, image = cap.read()

        if not ok:
            continue

        H, W = image.shape[:2]

        # Inspect the first labelled target in this frame
        target_id, x, y, w, h = boxes[frame_no][0]

        # Small crop around the original bounding box
        padding = 25

        left = max(0, int(x) - padding)
        top = max(0, int(y) - padding)
        right = min(W, int(x + w) + padding)
        bottom = min(H, int(y + h) + padding)

        crop = image[top:bottom, left:right]

        if crop.size == 0:
            continue

        # 10x enlargement without inventing image details
        raw_zoom = cv2.resize(
            crop, None, fx=10, fy=10,
            interpolation=cv2.INTER_NEAREST
        )

        marked = crop.copy()

        cv2.rectangle(
            marked,
            (int(x) - left, int(y) - top),
            (int(x + w) - left, int(y + h) - top),
            (0, 255, 0),
            1
        )

        marked_zoom = cv2.resize(
            marked, None, fx=10, fy=10,
            interpolation=cv2.INTER_NEAREST
        )

        name = f"clip_{clip_no}_frame_{frame_no}"

        cv2.imwrite(
            str(OUT / f"{name}_RAW.jpg"),
            raw_zoom
        )

        cv2.imwrite(
            str(OUT / f"{name}_BOX.jpg"),
            marked_zoom
        )

        print(
            f"Clip {clip_no}, frame {frame_no}, "
            f"ID {target_id}, box {w:.1f}x{h:.1f}px"
        )

    cap.release()

print("\nInspection images saved to results/inspection/")
