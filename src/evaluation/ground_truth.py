"""Ground-truth annotation loaders for MOT, CSV, and Purdue UAV formats."""

from pathlib import Path
from typing import Optional, Union
import pandas as pd


GT_COLUMNS = ["frame", "track_id", "bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2"]


def load_mot_gt(path: Union[str, Path], one_indexed_frames: Optional[bool] = None) -> pd.DataFrame:
    """Load ground-truth bounding boxes from MOT-format (.txt) or CSV.

    Standard MOT format: ``frame, id, bb_left, bb_top, bb_width, bb_height, conf, class, vis``
    Also supports headered CSV with ``frame, track_id, bbox_x1, bbox_y1, bbox_x2, bbox_y2``
    or ``frame, id, x, y, w, h``.

    Returns:
        DataFrame with columns: frame (0-indexed int), track_id (int),
        bbox_x1, bbox_y1, bbox_x2, bbox_y2 (float).
    """
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"Ground truth file not found: {p}")

    # Peek first line to detect whether a header row exists
    first_line = p.read_text(encoding="utf-8", errors="ignore").splitlines()[0].strip()
    has_header = any(tok in first_line.lower() for tok in ("frame", "track_id", "bbox_x1", "bb_left"))

    if has_header:
        df = pd.read_csv(p)
        cols_lower = {c.lower().strip(): c for c in df.columns}
        if {"bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2"}.issubset(cols_lower):
            out = pd.DataFrame({
                "frame": df[cols_lower["frame"]].astype(int),
                "track_id": df[cols_lower.get("track_id", cols_lower.get("id", "track_id"))].astype(int),
                "bbox_x1": df[cols_lower["bbox_x1"]].astype(float),
                "bbox_y1": df[cols_lower["bbox_y1"]].astype(float),
                "bbox_x2": df[cols_lower["bbox_x2"]].astype(float),
                "bbox_y2": df[cols_lower["bbox_y2"]].astype(float),
            })
        else:
            # Headered x, y, w, h
            x = df[cols_lower.get("bb_left", cols_lower.get("x", df.columns[2]))].astype(float)
            y = df[cols_lower.get("bb_top", cols_lower.get("y", df.columns[3]))].astype(float)
            w = df[cols_lower.get("bb_width", cols_lower.get("w", df.columns[4]))].astype(float)
            h = df[cols_lower.get("bb_height", cols_lower.get("h", df.columns[5]))].astype(float)
            tid_col = cols_lower.get("track_id", cols_lower.get("id", df.columns[1]))
            out = pd.DataFrame({
                "frame": df[cols_lower.get("frame", df.columns[0])].astype(int),
                "track_id": df[tid_col].astype(int),
                "bbox_x1": x,
                "bbox_y1": y,
                "bbox_x2": x + w,
                "bbox_y2": y + h,
            })
    else:
        raw = pd.read_csv(p, header=None)
        if raw.shape[1] < 6:
            raise ValueError(f"Expected >= 6 columns in MOT ground truth '{p}', got {raw.shape[1]}")
        # Filter out ignore/zero-conf rows if MOT conf column exists (col 6 == 0)
        if raw.shape[1] >= 7:
            raw = raw[raw.iloc[:, 6].astype(float) != 0].copy()
        x = raw.iloc[:, 2].astype(float)
        y = raw.iloc[:, 3].astype(float)
        w = raw.iloc[:, 4].astype(float)
        h = raw.iloc[:, 5].astype(float)
        out = pd.DataFrame({
            "frame": raw.iloc[:, 0].astype(int),
            "track_id": raw.iloc[:, 1].astype(int),
            "bbox_x1": x,
            "bbox_y1": y,
            "bbox_x2": x + w,
            "bbox_y2": y + h,
        })

    if not out.empty:
        if one_indexed_frames is True or (one_indexed_frames is None and int(out["frame"].min()) >= 1 and not has_header):
            out["frame"] = out["frame"] - 1

    return out.reset_index(drop=True)


def discover_gt_for_video(
    video_path: Union[str, Path],
    gt_dir: Union[str, Path] = "data/gt",
) -> Optional[Path]:
    """Find a matching ground-truth annotation file for a video if one exists."""
    v = Path(video_path)
    stem = v.stem
    gt_root = Path(gt_dir)
    candidates = [
        gt_root / f"{stem}.txt",
        gt_root / f"{stem}.csv",
        gt_root / f"{stem}_gt.txt",
        gt_root / stem / "gt" / "gt.txt",
        v.parent / f"{stem}_gt.txt",
        v.parent / f"{stem}.txt",
    ]
    for cand in candidates:
        if cand.is_file():
            return cand
    return None
