"""Create leak-free 32/8/10 Purdue split (Dogfight 40/10 precedent) and prepare YOLO dataset.

Usage:
  python -m src.training.prepare_splits
"""

import json
from pathlib import Path
import shutil
import sys
from typing import Dict, List
import cv2
import yaml

from src.evaluation.ground_truth import discover_gt_for_video, load_mot_gt


def create_purdue_split(
    videos_dir: Path = Path("Videos"),
    split_path: Path = Path("data/splits/purdue_split.json"),
) -> Dict[str, List[str]]:
    """Create 32 train / 8 val / 10 test split following Dogfight 40/10 precedent."""
    train = [f"Clip_{i}" for i in range(1, 33)]
    val = [f"Clip_{i}" for i in range(33, 41)]
    test = [f"Clip_{i}" for i in range(41, 51)]

    # Verify all 50 videos exist
    missing = [
        name for name in (train + val + test)
        if not (videos_dir / f"{name}.mov").is_file()
    ]
    if missing:
        raise FileNotFoundError(f"Missing Purdue videos in {videos_dir}: {missing[:5]}")

    verify_no_leakage(train, val, test)

    split = {
        "protocol": "Dogfight 40/10 (32 train, 8 val, 10 held-out test)",
        "train": train,
        "val": val,
        "test": test,
    }
    split_path.parent.mkdir(parents=True, exist_ok=True)
    split_path.write_text(json.dumps(split, indent=2), encoding="utf-8")
    print(f"Saved split -> {split_path} (train={len(train)}, val={len(val)}, test={len(test)})")
    return split


def verify_no_leakage(train: List[str], val: List[str], test: List[str]) -> None:
    """Exit with error if any test video appears in train or val, or train/val overlap."""
    s_train, s_val, s_test = set(train), set(val), set(test)
    leak_tr = s_test & s_train
    leak_va = s_test & s_val
    overlap = s_train & s_val
    if leak_tr or leak_va or overlap:
        print(
            f"LEAKAGE DETECTED! test&train={sorted(leak_tr)}, "
            f"test&val={sorted(leak_va)}, train&val={sorted(overlap)}",
            file=sys.stderr,
        )
        sys.exit(1)
    print("Leakage check PASSED: 0 overlap between train, val, and test.")


def build_split_yolo_dataset(
    split: Dict[str, List[str]],
    videos_dir: Path = Path("Videos"),
    gt_dir: Path = Path("data/gt"),
    existing_yolo_dir: Path = Path("data/yolo_drone"),
    out_dir: Path = Path("data/yolo_drone_split"),
    sample_every: int = 10,
) -> Path:
    """Build leak-free YOLO dataset for train & val splits only."""
    for s in ("train", "val"):
        img_dir = out_dir / "images" / s
        lbl_dir = out_dir / "labels" / s
        if img_dir.exists():
            shutil.rmtree(img_dir)
        if lbl_dir.exists():
            shutil.rmtree(lbl_dir)
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)

    # Index existing extracted frames if available so we avoid slow video re-decoding
    existing_imgs = {}
    for sub in ("train", "val"):
        p_img = existing_yolo_dir / "images" / sub
        p_lbl = existing_yolo_dir / "labels" / sub
        if p_img.is_dir():
            for jpg in p_img.glob("*.jpg"):
                lbl = p_lbl / f"{jpg.stem}.txt"
                if lbl.is_file():
                    clip_name = jpg.stem.rsplit("_f", 1)[0]
                    existing_imgs.setdefault(clip_name, []).append((jpg, lbl))

    counts = {"train": 0, "val": 0}
    for s in ("train", "val"):
        for clip_name in split[s]:
            if clip_name in existing_imgs and len(existing_imgs[clip_name]) > 0:
                for jpg, lbl in existing_imgs[clip_name]:
                    shutil.copy2(jpg, out_dir / "images" / s / jpg.name)
                    shutil.copy2(lbl, out_dir / "labels" / s / lbl.name)
                    counts[s] += 1
                continue

            # Extract from video + MOT GT
            vid_path = videos_dir / f"{clip_name}.mov"
            gt_path = discover_gt_for_video(vid_path, gt_dir=gt_dir)
            if gt_path is None:
                continue
            gt_df = load_mot_gt(gt_path)
            gt_by_frame = {}
            for f, grp in gt_df.groupby("frame"):
                gt_by_frame[int(f)] = [
                    (float(r.bbox_x1), float(r.bbox_y1), float(r.bbox_x2), float(r.bbox_y2))
                    for r in grp.itertuples()
                ]

            cap = cv2.VideoCapture(str(vid_path))
            idx = 0
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                if idx % sample_every == 0:
                    boxes = gt_by_frame.get(idx, [])
                    if boxes:
                        h, w = frame.shape[:2]
                        stem = f"{clip_name}_f{idx:06d}"
                        cv2.imwrite(str(out_dir / "images" / s / f"{stem}.jpg"), frame)
                        lines = []
                        for x1, y1, x2, y2 in boxes:
                            xc = max(0.0, min(1.0, ((x1 + x2) * 0.5) / w))
                            yc = max(0.0, min(1.0, ((y1 + y2) * 0.5) / h))
                            bw = max(0.0, min(1.0, (x2 - x1) / w))
                            bh = max(0.0, min(1.0, (y2 - y1) / h))
                            if bw > 0 and bh > 0:
                                lines.append(f"0 {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}")
                        (out_dir / "labels" / s / f"{stem}.txt").write_text("\n".join(lines), encoding="utf-8")
                        counts[s] += 1
                idx += 1
            cap.release()

    # Final safety check: ensure no test clip landed in train/val YOLO dirs
    test_set = set(split["test"])
    for s in ("train", "val"):
        found_clips = {p.stem.rsplit("_f", 1)[0] for p in (out_dir / "images" / s).glob("*.jpg")}
        leaked = found_clips & test_set
        if leaked:
            raise RuntimeError(f"Test clips leaked into YOLO {s}: {sorted(leaked)}")

    yaml_path = out_dir / "drone_dataset.yaml"
    cfg = {
        "path": str(out_dir.resolve()),
        "train": "images/train",
        "val": "images/val",
        "names": {0: "drone"},
    }
    yaml_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    print(f"Prepared leak-free YOLO dataset at {yaml_path} (train={counts['train']}, val={counts['val']})")
    return yaml_path


if __name__ == "__main__":
    sp = create_purdue_split()
    build_split_yolo_dataset(sp)
