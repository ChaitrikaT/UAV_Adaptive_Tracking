"""Check model metadata and installation. This is NOT an accuracy test."""
from pathlib import Path
import argparse
import torch
from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", default="yolo11n.pt", help="Path to model weights")
    args = parser.parse_args()
    model = YOLO(args.weights)
    names = {int(k): str(v) for k, v in model.names.items()}
    labels = [value.casefold().strip() for value in names.values()]
    has_target_class = any(name in ("uav", "drone") for name in labels)
    print("Loaded model:", args.weights)
    print("PyTorch:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())
    print("Classes:", names)
    print("Dedicated drone/UAV class:", has_target_class)
    if not has_target_class:
        print("RESULT: Not suitable for UAV-specific detection quality claims.")
    else:
        print("RESULT: UAV class found; accuracy is STILL UNVERIFIED.")
        print("Next: compare saved detections with held-out reference annotations.")

if __name__ == "__main__":
    main()
