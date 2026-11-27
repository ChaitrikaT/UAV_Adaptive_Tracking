"""Detector fine-tuning module for custom UAV / Purdue datasets.

Usage:
  1. Prepare YOLO dataset from Videos/ + data/gt/ (MOT .txt) and fine-tune:
       python -m src.training.finetune_detector --videos Videos --gt-dir data/gt --epochs 30
  2. Or pass an existing YOLO data.yaml directly:
       python -m src.training.finetune_detector --data-yaml data/yolo_drone/drone_dataset.yaml --epochs 50
"""

import argparse
from pathlib import Path
import shutil
from typing import Any, Dict, List, Optional, Union
import cv2
import yaml

from src.evaluation.ground_truth import discover_gt_for_video, load_mot_gt


def prepare_yolo_dataset(
    videos_dir: Union[str, Path] = "Videos",
    gt_dir: Union[str, Path] = "data/gt",
    output_dir: Union[str, Path] = "data/yolo_drone",
    sample_every: int = 10,
    val_fraction: float = 0.2,
    pseudo_weights: Optional[str] = None,
    pseudo_conf: float = 0.35,
) -> Path:
    """Extract frames from videos and convert MOT GT (or pseudo-labels) into YOLO format."""
    v_root = Path(videos_dir)
    out_root = Path(output_dir)
    for split in ("train", "val"):
        (out_root / "images" / split).mkdir(parents=True, exist_ok=True)
        (out_root / "labels" / split).mkdir(parents=True, exist_ok=True)

    videos = sorted(
        p for p in v_root.rglob("*")
        if p.is_file() and p.suffix.lower() in (".mp4", ".mov", ".avi", ".mkv") and not p.name.startswith("._")
    )
    if not videos:
        raise FileNotFoundError(f"No videos found under '{v_root}'.")

    pseudo_model = None
    if pseudo_weights:
        from ultralytics import YOLO
        pseudo_model = YOLO(pseudo_weights)

    n_val = max(1, int(round(len(videos) * val_fraction))) if len(videos) > 1 else 0
    val_set = set(v.stem for v in videos[-n_val:]) if n_val > 0 else set()

    written = {"train": 0, "val": 0}
    for vid in videos:
        gt_path = discover_gt_for_video(vid, gt_dir=gt_dir)
        gt_by_frame: Dict[int, List[tuple]] = {}
        if gt_path is not None:
            gt_df = load_mot_gt(gt_path)
            for f, grp in gt_df.groupby("frame"):
                gt_by_frame[int(f)] = [
                    (float(r.bbox_x1), float(r.bbox_y1), float(r.bbox_x2), float(r.bbox_y2))
                    for r in grp.itertuples()
                ]
        elif pseudo_model is None:
            continue  # skip unlabelled videos unless pseudo-labelling is enabled

        split = "val" if vid.stem in val_set else "train"
        cap = cv2.VideoCapture(str(vid))
        idx = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if idx % sample_every == 0:
                h, w = frame.shape[:2]
                boxes = gt_by_frame.get(idx, [])
                if not boxes and pseudo_model is not None:
                    res = pseudo_model.predict(frame, conf=pseudo_conf, verbose=False)
                    if res and res[0].boxes is not None:
                        boxes = [tuple(map(float, b)) for b in res[0].boxes.xyxy.cpu().numpy()]

                if boxes:
                    stem = f"{vid.stem}_f{idx:06d}"
                    img_path = out_root / "images" / split / f"{stem}.jpg"
                    lbl_path = out_root / "labels" / split / f"{stem}.txt"
                    cv2.imwrite(str(img_path), frame)
                    lines = []
                    for x1, y1, x2, y2 in boxes:
                        xc = max(0.0, min(1.0, ((x1 + x2) * 0.5) / w))
                        yc = max(0.0, min(1.0, ((y1 + y2) * 0.5) / h))
                        bw = max(0.0, min(1.0, (x2 - x1) / w))
                        bh = max(0.0, min(1.0, (y2 - y1) / h))
                        if bw > 0 and bh > 0:
                            lines.append(f"0 {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}")
                    lbl_path.write_text("\n".join(lines), encoding="utf-8")
                    written[split] += 1
            idx += 1
        cap.release()

    if written["train"] == 0:
        raise RuntimeError(
            "No labelled training frames were generated. Place MOT ground truth files in "
            f"'{gt_dir}' or pass --pseudo-weights weights/yolov8s.pt."
        )

    yaml_path = out_root / "drone_dataset.yaml"
    cfg = {
        "path": str(out_root.resolve()),
        "train": "images/train",
        "val": "images/val" if written["val"] > 0 else "images/train",
        "names": {0: "drone"},
    }
    yaml_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    print(f"Prepared YOLO dataset at {yaml_path} (train={written['train']}, val={written['val']})")
    return yaml_path


def finetune_detector(
    data_yaml: Union[str, Path] = "data/yolo_drone/drone_dataset.yaml",
    base_weights: str = "weights/yolov8n.pt",
    epochs: int = 30,
    imgsz: int = 640,
    batch: int = 16,
    out_weights: Union[str, Path] = "weights/drone_yolov8n.pt",
) -> Any:
    """Fine-tune a YOLO detector on a custom drone dataset and save best weights."""
    from ultralytics import YOLO

    yaml_file = Path(data_yaml)
    if not yaml_file.is_file():
        raise FileNotFoundError(f"Dataset YAML not found: {yaml_file}")

    model = YOLO(base_weights)
    results = model.train(
        data=str(yaml_file),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        workers=0,
        project="outputs/yolo_train",
        name=Path(out_weights).stem,
        exist_ok=True,
    )

    best_pt = Path(results.save_dir) / "weights" / "best.pt"
    if best_pt.is_file():
        dst = Path(out_weights)
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(best_pt, dst)
        print(f"Saved fine-tuned weights -> {dst}")
    return results


def main() -> None:
    p = argparse.ArgumentParser(description="Prepare Purdue/drone dataset and fine-tune YOLO.")
    p.add_argument("--data-yaml", type=str, default=None, help="Existing YOLO data.yaml (skips extraction)")
    p.add_argument("--videos", type=str, default="Videos", help="Folder with training videos")
    p.add_argument("--gt-dir", type=str, default="data/gt", help="Folder with MOT .txt ground truth")
    p.add_argument("--sample-every", type=int, default=10, help="Extract 1 frame every N frames")
    p.add_argument("--pseudo-weights", type=str, default=None, help="Optional YOLO weights for pseudo-labelling")
    p.add_argument("--base-weights", type=str, default="weights/yolov8n.pt")
    p.add_argument("--out-weights", type=str, default="weights/drone_yolov8n.pt")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch", type=int, default=16)
    args = p.parse_args()

    data_yaml = args.data_yaml
    if not data_yaml:
        data_yaml = prepare_yolo_dataset(
            videos_dir=args.videos,
            gt_dir=args.gt_dir,
            sample_every=args.sample_every,
            pseudo_weights=args.pseudo_weights,
        )
    finetune_detector(
        data_yaml=data_yaml,
        base_weights=args.base_weights,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        out_weights=args.out_weights,
    )


if __name__ == "__main__":
    main()
