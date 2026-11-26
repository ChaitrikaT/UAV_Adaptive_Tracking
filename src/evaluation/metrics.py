"""Evaluation metrics: agreement, CLEAR MOT (MOTA/IDSW), IDF1, HOTA, and TrackEval runner."""

from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

from src.evaluation.ground_truth import load_mot_gt


def compute_fps(frame_count: int, elapsed_time_seconds: float) -> float:
    if elapsed_time_seconds <= 0:
        return 0.0
    return float(frame_count / elapsed_time_seconds)


def compute_detector_call_ratio(detector_calls: int, total_frames: int) -> float:
    if total_frames <= 0:
        return 0.0
    return float(detector_calls / total_frames)


def _iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """IoU matrix between boxes a (N,4) and b (M,4) in x1,y1,x2,y2."""
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area_a = np.maximum(0.0, a[:, 2] - a[:, 0]) * np.maximum(0.0, a[:, 3] - a[:, 1])
    area_b = np.maximum(0.0, b[:, 2] - b[:, 0]) * np.maximum(0.0, b[:, 3] - b[:, 1])
    return inter / (area_a[:, None] + area_b[None, :] - inter + 1e-9)


def frame_agreement(
    tracks: pd.DataFrame,
    reference_tracks: pd.DataFrame,
    n_frames: int,
    reference_conf: float = 0.5,
    iou_thr: float = 0.5,
) -> np.ndarray:
    """Per frame: share of the reference's confident targets that ``tracks`` also found."""
    cols = ["bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2"]
    found = {f: g[cols].to_numpy(float) for f, g in tracks.groupby("frame")} if len(tracks) else {}
    ref = (
        reference_tracks[reference_tracks["confidence"] >= reference_conf]
        if len(reference_tracks) and "confidence" in reference_tracks.columns
        else reference_tracks
    )
    ref_by = {f: g[cols].to_numpy(float) for f, g in ref.groupby("frame")} if len(ref) else {}

    agree = np.ones(n_frames, dtype=float)
    for f, rb in ref_by.items():
        if f >= n_frames:
            continue
        fb = found.get(f)
        if fb is None or len(fb) == 0:
            agree[f] = 0.0
            continue
        iou = _iou(rb, fb)
        matched, free = 0, np.ones(len(fb), dtype=bool)
        for r in np.argsort(-iou.max(axis=1)):
            cand = np.where(free, iou[r], -1.0)
            j = int(cand.argmax())
            if cand[j] >= iou_thr:
                free[j] = False
                matched += 1
        agree[f] = matched / len(rb)
    return agree


def _match_frame(
    gt_boxes: np.ndarray,
    pr_boxes: np.ndarray,
    iou_thr: float,
) -> List[Tuple[int, int, float]]:
    """Optimal 1-to-1 Hungarian matching for one frame above ``iou_thr``."""
    if len(gt_boxes) == 0 or len(pr_boxes) == 0:
        return []
    iou = _iou(gt_boxes, pr_boxes)
    row_ind, col_ind = linear_sum_assignment(-iou)
    matches = []
    for r, c in zip(row_ind, col_ind):
        if iou[r, c] >= iou_thr:
            matches.append((int(r), int(c), float(iou[r, c])))
    return matches


def evaluate_against_ground_truth(
    tracks_df: pd.DataFrame,
    gt_df: pd.DataFrame,
    iou_thr: float = 0.5,
) -> Dict[str, Any]:
    """Compute CLEAR MOT (MOTA, IDSW, Recall, Precision), IDF1, and HOTA.

    Uses Hungarian assignment per frame for MOTA/IDSW, global ID co-occurrence
    assignment for IDF1, and multi-threshold alpha sweep [0.05..0.95] for HOTA.
    """
    cols = ["bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2"]
    gt_by = {
        int(f): (g["track_id"].to_numpy(int), g[cols].to_numpy(float))
        for f, g in gt_df.groupby("frame")
    } if len(gt_df) else {}
    pr_by = {
        int(f): (g["track_id"].to_numpy(int), g[cols].to_numpy(float))
        for f, g in tracks_df.groupby("frame")
    } if len(tracks_df) else {}

    all_frames = sorted(set(gt_by.keys()) | set(pr_by.keys()))
    total_gt = int(len(gt_df))
    total_pr = int(len(tracks_df))

    if total_gt == 0:
        return {
            "HOTA": 0.0,
            "IDF1": 0.0,
            "MOTA": 0.0,
            "IDSW": 0,
            "TP": 0,
            "FP": total_pr,
            "FN": 0,
            "Recall": 0.0,
            "Precision": 0.0,
            "DetA": 0.0,
            "AssA": 0.0,
            "gt_objects": 0,
        }

    # --- 1. CLEAR MOT + IDF1 co-occurrence at iou_thr ---
    tp = 0
    fp = 0
    fn = 0
    idsw = 0
    last_pred_for_gt: Dict[int, int] = {}
    id_matches: Dict[Tuple[int, int], int] = defaultdict(int)
    gt_id_counts: Dict[int, int] = defaultdict(int)
    pr_id_counts: Dict[int, int] = defaultdict(int)

    for f in all_frames:
        g_ids, g_boxes = gt_by.get(f, (np.empty(0, int), np.empty((0, 4), float)))
        p_ids, p_boxes = pr_by.get(f, (np.empty(0, int), np.empty((0, 4), float)))

        for gid in g_ids:
            gt_id_counts[int(gid)] += 1
        for pid in p_ids:
            pr_id_counts[int(pid)] += 1

        matches = _match_frame(g_boxes, p_boxes, iou_thr)
        tp += len(matches)
        fn += len(g_ids) - len(matches)
        fp += len(p_ids) - len(matches)

        for r, c, _ in matches:
            gid = int(g_ids[r])
            pid = int(p_ids[c])
            if gid in last_pred_for_gt and last_pred_for_gt[gid] != pid:
                idsw += 1
            last_pred_for_gt[gid] = pid
            id_matches[(gid, pid)] += 1

    mota = 1.0 - (fn + fp + idsw) / max(1, total_gt)
    recall = tp / max(1, total_gt)
    precision = tp / max(1, tp + fp)

    # Global ID assignment for IDF1
    uniq_g = sorted(gt_id_counts.keys())
    uniq_p = sorted(pr_id_counts.keys())
    idtp = 0
    if uniq_g and uniq_p:
        cost = np.zeros((len(uniq_g), len(uniq_p)), dtype=float)
        g_idx = {g: i for i, g in enumerate(uniq_g)}
        p_idx = {p: j for j, p in enumerate(uniq_p)}
        for (g, p), cnt in id_matches.items():
            cost[g_idx[g], p_idx[p]] = cnt
        ri, ci = linear_sum_assignment(-cost)
        idtp = int(sum(cost[r, c] for r, c in zip(ri, ci)))
    idf1 = (2.0 * idtp) / max(1, total_gt + total_pr)

    # --- 2. HOTA across alpha thresholds 0.05..0.95 ---
    alphas = np.linspace(0.05, 0.95, 19)
    deta_list: List[float] = []
    assa_list: List[float] = []
    hota_list: List[float] = []

    for alpha in alphas:
        a_tp = 0
        a_fp = 0
        a_fn = 0
        a_id_matches: Dict[Tuple[int, int], int] = defaultdict(int)
        matched_pairs: List[Tuple[int, int]] = []

        for f in all_frames:
            g_ids, g_boxes = gt_by.get(f, (np.empty(0, int), np.empty((0, 4), float)))
            p_ids, p_boxes = pr_by.get(f, (np.empty(0, int), np.empty((0, 4), float)))
            matches = _match_frame(g_boxes, p_boxes, float(alpha))
            a_tp += len(matches)
            a_fn += len(g_ids) - len(matches)
            a_fp += len(p_ids) - len(matches)
            for r, c, _ in matches:
                gid, pid = int(g_ids[r]), int(p_ids[c])
                a_id_matches[(gid, pid)] += 1
                matched_pairs.append((gid, pid))

        deta = a_tp / max(1, a_tp + a_fn + a_fp)
        if matched_pairs:
            ass_scores = []
            for gid, pid in matched_pairs:
                tpa = a_id_matches[(gid, pid)]
                fna = gt_id_counts[gid] - tpa
                fpa = pr_id_counts[pid] - tpa
                ass_scores.append(tpa / max(1, tpa + fna + fpa))
            assa = float(np.mean(ass_scores))
        else:
            assa = 0.0

        deta_list.append(deta)
        assa_list.append(assa)
        hota_list.append(float(np.sqrt(deta * assa)))

    return {
        "HOTA": round(float(np.mean(hota_list)) * 100.0, 2),
        "IDF1": round(idf1 * 100.0, 2),
        "MOTA": round(mota * 100.0, 2),
        "IDSW": int(idsw),
        "TP": int(tp),
        "FP": int(fp),
        "FN": int(fn),
        "Recall": round(recall * 100.0, 2),
        "Precision": round(precision * 100.0, 2),
        "DetA": round(float(np.mean(deta_list)) * 100.0, 2),
        "AssA": round(float(np.mean(assa_list)) * 100.0, 2),
        "gt_objects": total_gt,
    }


def run_metrics(
    summary: Dict[str, Any],
    reference: Optional[Dict[str, Any]] = None,
    gt_df: Optional[pd.DataFrame] = None,
) -> Dict[str, Any]:
    """Headline numbers for one TrackingPipeline.run_video() summary."""
    frames, tracks = summary["records_df"], summary["tracks_df"]
    n = max(1, summary["total_frames"])
    out = {
        "frames": summary["total_frames"],
        "seconds": summary["total_time_seconds"],
        "fps": summary["avg_fps"],
        "detector_calls": summary["detector_calls"],
        "detector_pct": round(summary["detector_call_ratio"] * 100, 1),
        "robust_pct": round(float((frames["mode"] == "ROBUST").mean() * 100), 1) if len(frames) else 0.0,
        "switches": summary["mode_switches"],
        "targets": summary.get("total_unique_targets", 0),
        "frames_with_target_pct": round(tracks["frame"].nunique() / n * 100, 1) if len(tracks) else 0.0,
        "mean_targets_per_frame": round(float(frames["num_tracks"].mean()), 2) if len(frames) else 0.0,
    }
    if reference is not None:
        agree = frame_agreement(tracks, reference["tracks_df"], len(reference["records_df"]))
        out["agreement_pct"] = round(float(agree.mean() * 100), 1)
    if gt_df is not None and not gt_df.empty:
        gt_scores = evaluate_against_ground_truth(tracks, gt_df)
        out.update(gt_scores)
    return out


def summarize_run_metrics(metrics_df: pd.DataFrame) -> Dict[str, Any]:
    if metrics_df.empty:
        return {}

    total_frames = len(metrics_df)
    fast_count = int((metrics_df["mode"] == "FAST").sum())
    robust_count = int((metrics_df["mode"] == "ROBUST").sum())
    detector_calls = int(metrics_df["detector_invoked"].sum()) if "detector_invoked" in metrics_df else total_frames

    return {
        "total_frames": total_frames,
        "avg_fps": round(float(metrics_df["fps"].mean()), 1) if "fps" in metrics_df else 0.0,
        "detector_calls": detector_calls,
        "detector_call_ratio": round(detector_calls / max(1, total_frames), 3),
        "fast_mode_ratio": round(fast_count / max(1, total_frames), 3),
        "robust_mode_ratio": round(robust_count / max(1, total_frames), 3),
        "mean_tracks": round(float(metrics_df["num_tracks"].mean()), 2) if "num_tracks" in metrics_df else 0.0,
    }


def run_trackeval(
    predicted_tracks_dir: Union[str, Path],
    ground_truth_dir: Union[str, Path],
    iou_thr: float = 0.5,
) -> Dict[str, Any]:
    """Evaluate all predicted track CSV/MOT files against ground-truth files."""
    pred_dir = Path(predicted_tracks_dir)
    gt_dir = Path(ground_truth_dir)
    if not pred_dir.is_dir():
        raise FileNotFoundError(f"Predicted tracks directory not found: {pred_dir}")
    if not gt_dir.is_dir():
        raise FileNotFoundError(f"Ground truth directory not found: {gt_dir}")

    results: Dict[str, Any] = {}
    for pred_file in sorted(pred_dir.glob("*_tracks.csv")):
        # Match <clip>_<mode>_tracks.csv -> <clip>.txt or <clip>_gt.txt
        stem_parts = pred_file.stem.replace("_tracks", "")
        gt_candidates = list(gt_dir.glob(f"{stem_parts.split('_')[0]}*.txt")) + list(
            gt_dir.glob(f"{stem_parts.split('_')[0]}*.csv")
        )
        if not gt_candidates:
            continue
        pr_df = pd.read_csv(pred_file)
        gt_df = load_mot_gt(gt_candidates[0])
        results[pred_file.name] = evaluate_against_ground_truth(pr_df, gt_df, iou_thr=iou_thr)

    return results
