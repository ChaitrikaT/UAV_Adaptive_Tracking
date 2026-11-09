
from pathlib import Path
from collections import defaultdict
import csv
import cv2

ROOT = Path(__file__).resolve().parents[1]
VIDEO_DIR = ROOT / "data" / "videos"
LABEL_DIR = ROOT / "data" / "annotations"
RESULT_DIR = ROOT / "results"

RESULT_DIR.mkdir(exist_ok=True)


def load_annotations(path):
    annotations = defaultdict(list)

    with open(path, "r", encoding="utf-8-sig") as file:
        reader = csv.reader(file)

        for row in reader:
            if len(row) < 6:
                continue

            try:
                frame_no = int(float(row[0]))
                track_id = int(float(row[1]))

                x = float(row[2])
                y = float(row[3])
                width = float(row[4])
                height = float(row[5])

            except ValueError:
                continue

            if frame_no < 1 or width <= 0 or height <= 0:
                continue

            annotations[frame_no].append(
                (track_id, x, y, width, height)
            )

    return annotations


for clip_no in [1, 2, 3]:

    video_path = VIDEO_DIR / f"Clip_{clip_no}.mov"
    label_path = LABEL_DIR / f"Clip_{clip_no}_refined.txt"

    print(f"\n========== CLIP {clip_no} ==========")

    if not video_path.exists() or not label_path.exists():
        print("Missing video or annotation file.")
        continue

    annotations = load_annotations(label_path)

    print("Annotated frames:", len(annotations))

    unique_ids = sorted({
        target[0]
        for boxes in annotations.values()
        for target in boxes
    })

    print("Target IDs:", unique_ids)

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        print("Could not open video.")
        continue

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print("Video frames:", total_frames)

    # Select first, middle and last annotated frames
    frame_numbers = sorted(annotations.keys())

    if not frame_numbers:
        print("No valid annotations found.")
        cap.release()
        continue

    selected_frames = sorted(set([
        frame_numbers[0],
        frame_numbers[len(frame_numbers) // 2],
        frame_numbers[-1]
    ]))

    for frame_no in selected_frames:

        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no - 1)

        success, frame = cap.read()

        if not success:
            print("Could not read frame:", frame_no)
            continue

        image_height, image_width = frame.shape[:2]

        valid_boxes = []

        for track_id, x, y, width, height in annotations[frame_no]:

            x1 = max(0, int(round(x)))
            y1 = max(0, int(round(y)))

            x2 = min(image_width - 1, int(round(x + width)))
            y2 = min(image_height - 1, int(round(y + height)))

            if x2 <= x1 or y2 <= y1:
                continue

            valid_boxes.append((track_id, x1, y1, x2, y2))

            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                3
            )

            cv2.putText(
                frame,
                f"ID {track_id}",
                (x1, max(25, y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 255, 0),
                2
            )

        output_path = (
            RESULT_DIR /
            f"gt_clip_{clip_no}_frame_{frame_no}.jpg"
        )

        cv2.imwrite(str(output_path), frame)

        print(
            f"Frame {frame_no}: "
            f"{len(valid_boxes)} boxes -> {output_path.name}"
        )

        # Save an enlarged view of the first target
        if valid_boxes:

            _, x1, y1, x2, y2 = valid_boxes[0]

            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2

            crop_x1 = max(0, center_x - 120)
            crop_y1 = max(0, center_y - 120)
            crop_x2 = min(image_width, center_x + 120)
            crop_y2 = min(image_height, center_y + 120)

            crop = frame[
                crop_y1:crop_y2,
                crop_x1:crop_x2
            ]

            if crop.size > 0:
                zoom = cv2.resize(
                    crop,
                    None,
                    fx=2,
                    fy=2,
                    interpolation=cv2.INTER_CUBIC
                )

                zoom_path = (
                    RESULT_DIR /
                    f"gt_clip_{clip_no}_frame_{frame_no}_zoom.jpg"
                )

                cv2.imwrite(str(zoom_path), zoom)

    cap.release()

print("\nGround-truth visualization complete!")
