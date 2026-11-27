"""Leak-free evaluation on Purdue val/test splits with HOTA, IDF1, MOTA, IDSW, shuffled FPS, and 95% CIs.

Usage:
  python -m src.evaluation.eval_split --split test --max-frames 300 --repeats 5
"""

import argparse
import json
from pathlib import Path
import random
import time
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd

from src.evaluation.ground_truth import discover_gt_for_video, load_mot_gt
from src.evaluation.metrics import evaluate_against_ground_truth, frame_agreement
from src.evaluation.protocol import block_bootstrap_ci, verify_online_replay_equivalence
from src.pipeline.pipeline import Mode, TrackingPipeline
from src.training.prepare_splits import verify_no_leakage
from src.utils.config import load_config


CONFIGS = [
    ("Always FAST", Mode.FAST, None),
    ("Always ROBUST", Mode.ROBUST, None),
    ("Adaptive (Rules)", Mode.ADAPTIVE, "rules"),
    ("Adaptive (Decision Tree)", Mode.ADAPTIVE, "tree"),
    ("Adaptive (Random Forest)", Mode.ADAPTIVE, "forest"),
]


def _make_pipe(mode: Mode, sel: Optional[str], config_path: str, threshold: Optional[float]) -> TrackingPipeline:
    return TrackingPipeline(
        mode=mode,
        config_path=config_path,
        selector_type=sel or "rules",
        selector_threshold=threshold,
    )


def tune_threshold_on_val(
    val_videos: List[Path],
    gt_dir: str,
    config_path: str,
    selector_kind: str,
    candidates: List[float] = (0.35, 0.45, 0.50, 0.55, 0.65),
    max_frames: int = 150,
) -> float:
    """Pick threshold on validation split maximizing IDF1 - 0.1 * detector_call_ratio."""
    best_thr, best_score = 0.50, -1e9
    for thr in candidates:
        pipe = _make_pipe(Mode.ADAPTIVE, selector_kind, config_path, thr)
        idf1s, ratios = [], []
        for v in val_videos:
            gt_path = discover_gt_for_video(v, gt_dir=gt_dir)
            if gt_path is None:
                continue
            gt_df = load_mot_gt(gt_path)
            summ = pipe.run_video(v, max_frames=max_frames, save_outputs=False)
            gt_sub = gt_df[gt_df["frame"] < summ["total_frames"]]
            m = evaluate_against_ground_truth(summ["tracks_df"], gt_sub, iou_thr=0.3)
            idf1s.append(m["IDF1"])
            ratios.append(summ["detector_call_ratio"])
        score = float(np.mean(idf1s)) - 10.0 * float(np.mean(ratios)) if idf1s else -1e9
        print(f"  [val tune {selector_kind}] thr={thr:.2f} -> IDF1={np.mean(idf1s):.2f}, det_ratio={np.mean(ratios):.3f}, score={score:.2f}")
        if score > best_score:
            best_score, best_thr = score, thr
    print(f"Selected threshold for {selector_kind}: {best_thr:.2f}")
    return best_thr


def evaluate_split(
    split_file: str = "data/splits/purdue_split.json",
    split_name: str = "test",
    gt_dir: str = "data/gt",
    videos_dir: str = "Videos",
    config_path: str = "config.yaml",
    max_frames: Optional[int] = 300,
    repeats: int = 5,
    seed: int = 42,
) -> Dict[str, Any]:
    sp = json.loads(Path(split_file).read_text(encoding="utf-8"))
    verify_no_leakage(sp["train"], sp["val"], sp["test"])
    cfg = load_config(config_path)

    videos = [Path(videos_dir) / f"{name}.mov" for name in sp[split_name]]
    val_videos = [Path(videos_dir) / f"{name}.mov" for name in sp["val"]]

    # Tune thresholds on val only
    thresholds = {
        "tree": tune_threshold_on_val(val_videos, gt_dir, config_path, "tree", max_frames=120),
        "forest": tune_threshold_on_val(val_videos, gt_dir, config_path, "forest", max_frames=120),
    }

    # Build pipelines once (warm-up on first clip)
    pipes = {
        label: _make_pipe(mode, sel, config_path, thresholds.get(sel) if sel else None)
        for label, mode, sel in CONFIGS
    }
    print("Warm-up run on first clip...")
    for label, pipe in pipes.items():
        pipe.run_video(videos[0], max_frames=30, save_outputs=False)

    rng = random.Random(seed)
    all_rows = []
    per_frame_agree: Dict[str, List[np.ndarray]] = {label: [] for label, _, _ in CONFIGS}

    for v in videos:
        gt_path = discover_gt_for_video(v, gt_dir=gt_dir)
        gt_df = load_mot_gt(gt_path) if gt_path else pd.DataFrame()
        order = list(CONFIGS)
        rng.shuffle(order)

        for label, mode, sel in order:
            pipe = pipes[label]
            summ = pipe.run_video(v, max_frames=max_frames, save_outputs=False)
            n_f = summ["total_frames"]
            gt_sub = gt_df[gt_df["frame"] < n_f] if not gt_df.empty else gt_df
            m = evaluate_against_ground_truth(summ["tracks_df"], gt_sub, iou_thr=0.3) if not gt_sub.empty else {}

            if not gt_sub.empty:
                ag = frame_agreement(summ["tracks_df"], gt_sub, n_f, reference_conf=0.0, iou_thr=0.3)
                per_frame_agree[label].append(ag)

            robust_pct = round(float((summ["records_df"]["mode"] == "ROBUST").mean() * 100.0), 2)
            replay_ok = True
            if mode == Mode.ADAPTIVE:
                rep = verify_online_replay_equivalence(
                    summ["records_df"], sel or "rules", cfg, threshold=thresholds.get(sel)
                )
                replay_ok = rep["equivalent"]

            all_rows.append({
                "clip": v.stem,
                "config": label,
                "frames": n_f,
                "HOTA": m.get("HOTA", np.nan),
                "IDF1": m.get("IDF1", np.nan),
                "MOTA": m.get("MOTA", np.nan),
                "IDSW": m.get("IDSW", np.nan),
                "Recall": m.get("Recall", np.nan),
                "Precision": m.get("Precision", np.nan),
                "FPS": summ["avg_fps"],
                "detector_pct": round(summ["detector_call_ratio"] * 100.0, 2),
                "robust_pct": robust_pct,
                "birds_filtered": summ.get("birds_filtered", 0),
                "replay_ok": replay_ok,
            })

    # Shuffled-order timing benchmark on first 2 test clips for median/range FPS
    timing_fps: Dict[str, List[float]] = {label: [] for label, _, _ in CONFIGS}
    bench_clips = videos[:2]
    for r in range(repeats):
        order = list(CONFIGS)
        rng.shuffle(order)
        for label, _, _ in order:
            t0 = time.perf_counter()
            frames_done = 0
            for bc in bench_clips:
                s = pipes[label].run_video(bc, max_frames=120, save_outputs=False)
                frames_done += s["total_frames"]
            dt = time.perf_counter() - t0
            timing_fps[label].append(round(frames_done / max(dt, 1e-6), 2))

    df = pd.DataFrame(all_rows)
    out_dir = Path("outputs/metrics")
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f"purdue_{split_name}_eval.csv"
    df.to_csv(csv_path, index=False)

    # Aggregate summary + 95% block-bootstrap CI vs Always FAST
    fast_ag = np.concatenate(per_frame_agree["Always FAST"]) if per_frame_agree["Always FAST"] else np.array([])
    summary_rows = []
    for label, _, _ in CONFIGS:
        sub = df[df["config"] == label]
        fps_list = timing_fps[label]
        ag_arr = np.concatenate(per_frame_agree[label]) if per_frame_agree[label] else np.array([])
        if len(fast_ag) > 0 and len(ag_arr) == len(fast_ag) and label != "Always FAST":
            mean_d, lo_d, hi_d = block_bootstrap_ci(ag_arr * 100.0, fast_ag * 100.0, block_size=15, n_boot=500)
        else:
            mean_d, lo_d, hi_d = 0.0, 0.0, 0.0

        summary_rows.append({
            "config": label,
            "HOTA": round(float(sub["HOTA"].mean()), 2),
            "IDF1": round(float(sub["IDF1"].mean()), 2),
            "MOTA": round(float(sub["MOTA"].mean()), 2),
            "IDSW_sum": int(sub["IDSW"].sum()),
            "Recall": round(float(sub["Recall"].mean()), 2),
            "Precision": round(float(sub["Precision"].mean()), 2),
            "detector_pct": round(float(sub["detector_pct"].mean()), 2),
            "robust_pct": round(float(sub["robust_pct"].mean()), 2),
            "fps_median": round(float(np.median(fps_list)), 2),
            "fps_min": round(float(min(fps_list)), 2),
            "fps_max": round(float(max(fps_list)), 2),
            "gt_recall_delta_vs_fast": round(mean_d, 2),
            "ci95_lo": round(lo_d, 2),
            "ci95_hi": round(hi_d, 2),
            "birds_filtered": int(sub["birds_filtered"].sum()),
            "replay_all_ok": bool(sub["replay_ok"].all()),
        })

    summary_df = pd.DataFrame(summary_rows)
    summary_csv = out_dir / f"purdue_{split_name}_summary.csv"
    summary_df.to_csv(summary_csv, index=False)

    payload = {
        "split": split_name,
        "thresholds": thresholds,
        "summary": summary_rows,
        "timing_repeats": timing_fps,
    }
    json_path = out_dir / f"purdue_{split_name}_eval.json"
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print("\n=== HELD-OUT TEST SUMMARY ===")
    print(summary_df.to_string(index=False))
    print(f"\nSaved -> {csv_path}, {summary_csv}, {json_path}")
    return payload


def main() -> None:
    p = argparse.ArgumentParser(description="Evaluate FAST/ROBUST/Adaptive on Purdue split.")
    p.add_argument("--split-file", default="data/splits/purdue_split.json")
    p.add_argument("--split", default="test")
    p.add_argument("--gt-dir", default="data/gt")
    p.add_argument("--videos", default="Videos")
    p.add_argument("--config", default="config.yaml")
    p.add_argument("--max-frames", type=int, default=300)
    p.add_argument("--repeats", type=int, default=5)
    args = p.parse_args()
    evaluate_split(
        split_file=args.split_file,
        split_name=args.split,
        gt_dir=args.gt_dir,
        videos_dir=args.videos,
        config_path=args.config,
        max_frames=args.max_frames,
        repeats=args.repeats,
    )


if __name__ == "__main__":
    main()
