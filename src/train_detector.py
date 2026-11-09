from pathlib import Path
import torch
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]

DATA_YAML = ROOT / "datasets" / "prepared" / "data.yaml"
OUTPUT_DIR = ROOT / "results" / "training"


def main():
    if not DATA_YAML.exists():
        raise FileNotFoundError(
            f"Dataset configuration not found: {DATA_YAML}"
        )

    print("Dataset:", DATA_YAML)
    print("CUDA available:", torch.cuda.is_available())

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA unavailable. Check your PyTorch environment."
        )

    # Pretrained weights are the starting point.
    model = YOLO("yolo11n.pt")

    model.train(
        data=str(DATA_YAML),
        epochs=12,
        imgsz=960,
        batch=2,
        device=0,
        workers=0,
        patience=6,
        seed=42,
        plots=True,
        project=str(OUTPUT_DIR),
        name="uav_detector_v1",
        exist_ok=False,
    )

    print("\nTraining finished.")
    print("Check:", OUTPUT_DIR)


if __name__ == "__main__":
    main()