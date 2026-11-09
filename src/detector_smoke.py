
from pathlib import Path
import cv2
import torch
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]

VIDEO = ROOT / "data" / "videos" / "Clip_1.mov"
OUTPUT = ROOT / "results" / "detector_smoke"

OUTPUT.mkdir(parents=True, exist_ok=True)

# Official COCO-pretrained YOLO11n:
# Used only to verify that inference works.
model = YOLO("yolo11n.pt")

device = 0 if torch.cuda.is_available() else "cpu"

print("Using device:", device)
print("Video:", VIDEO)

cap = cv2.VideoCapture(str(VIDEO))

if not cap.isOpened():
    raise RuntimeError("Could not open Clip_1.mov")

# Inspect a few frames from the recorded clip
sample_frames = [1, 75, 150, 225, 299]

for frame_no in sample_frames:

    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no - 1)

    success, frame = cap.read()

    if not success:
        print("Could not read frame", frame_no)
        continue

    result = model.predict(
        source=frame,
        imgsz=960,
        conf=0.15,
        device=device,
        verbose=False
    )[0]

    annotated = result.plot()

    output_path = (
        OUTPUT / f"frame_{frame_no}_prediction.jpg"
    )

    cv2.imwrite(str(output_path), annotated)

    count = len(result.boxes) if result.boxes is not None else 0

    print(
        f"Frame {frame_no}: "
        f"{count} COCO detections, "
        f"saved {output_path.name}"
    )

cap.release()

print("\nDetector smoke test complete.")
