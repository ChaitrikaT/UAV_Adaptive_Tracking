"""Train all learned mode selectors with leak-free GroupKFold cross-validation.

Run:
  python -m src.training.train_selector                  # generate labels + train all models
  python -m src.training.train_selector --skip-labels    # retrain on existing labels CSV
  python -m src.training.train_selector --models tree forest gbm logreg svm mlp
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Union

import cv2
import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupKFold
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.utils.class_weight import compute_sample_weight

from src.evaluation.metrics import frame_agreement
from src.selector.selector import BASE_FEATURES, ROLLING_WINDOW, RollingFeatures, RuleBasedSelector
from src.utils.config import load_config

VIDEO_TYPES = (".mp4", ".avi", ".mov", ".mkv")
CONDITIONS = ["clean", "blur", "shake", "tiny", "dark", "noise"]


def _apply_condition(
    frame: np.ndarray,
    condition: str,
    strength: float,
    rng: np.random.Generator,
    state: Dict[str, Any],
) -> np.ndarray:
    h, w = frame.shape[:2]
    if condition == "blur":
        k = int(7 + strength * 18) | 1
        return cv2.GaussianBlur(frame, (k, k), 0)
    if condition == "shake":
        amp = 10 + strength * 30
        state["offset"] = 0.7 * state.get("offset", np.zeros(2)) + rng.normal(0, amp * 0.5, 2)
        dx, dy = np.clip(state["offset"], -amp, amp)
        m = np.float32([[1, 0, dx], [0, 1, dy]])
        return cv2.warpAffine(frame, m, (w, h), borderMode=cv2.BORDER_REFLECT)
    if condition == "tiny":
        s = 0.55 - strength * 0.25
        small = cv2.resize(frame, (max(16, int(w * s)), max(16, int(h * s))), interpolation=cv2.INTER_AREA)
        canvas = np.empty_like(frame)
        canvas[:] = frame.reshape(-1, 3).mean(axis=0).astype(np.uint8)
        y0, x0 = (h - small.shape[0]) // 2, (w - small.shape[1]) // 2
        canvas[y0 : y0 + small.shape[0], x0 : x0 + small.shape[1]] = small
        return canvas
    if condition == "dark":
        gamma = 2.0 + strength * 1.5
        lut = (np.linspace(0, 1, 256) ** gamma * 255).astype(np.uint8)
        dark = cv2.LUT(frame, lut).astype(np.int16)
        return np.clip(dark + rng.normal(0, 6, frame.shape), 0, 255).astype(np.uint8)
    if condition == "noise":
        sigma = 12 + strength * 25
        return np.clip(frame.astype(np.int16) + rng.normal(0, sigma, frame.shape), 0, 255).astype(np.uint8)
    return frame


def make_condition_variant(
    src: Union[str, Path],
    dst: Union[str, Path],
    seed: int,
    segment_frames: int = 90,
    max_frames: Optional[int] = None,
) -> List[str]:
    rng = np.random.default_rng(seed)
    cap = cv2.VideoCapture(str(src))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    Path(dst).parent.mkdir(parents=True, exist_ok=True)
    out = cv2.VideoWriter(str(dst), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    schedule: List[str] = []
    i, condition, strength, state = 0, "clean", 0.0, {}
    while max_frames is None or i < max_frames:
        ok, frame = cap.read()
        if not ok:
            break
        if i % segment_frames == 0:
            condition, strength, state = str(rng.choice(CONDITIONS)), float(rng.uniform()), {}
        schedule.append(condition)
        out.write(_apply_condition(frame, condition, strength, rng, state))
        i += 1
    cap.release()
    out.release()
    return schedule


def forward_window_mean(values: np.ndarray, window: int) -> np.ndarray:
    csum = np.concatenate([[0.0], np.cumsum(values)])
    idx = np.arange(len(values))
    end = np.minimum(idx + window, len(values))
    return (csum[end] - csum[idx]) / (end - idx)


def _rolling_rows(records: pd.DataFrame) -> np.ndarray:
    roll = RollingFeatures(ROLLING_WINDOW)
    return np.asarray([roll.update(r) for r in records.to_dict("records")], dtype=float)


def find_videos(roots: Iterable[Union[str, Path]]) -> List[Path]:
    """Video files under the given files/folders, skipping macOS metadata and duplicates."""
    seen, found = set(), []
    for root in roots:
        root = Path(root)
        if not root.exists():
            continue
        files = [root] if root.is_file() else sorted(p for p in root.rglob("*") if p.is_file())
        for p in files:
            if p.suffix.lower() not in VIDEO_TYPES:
                continue
            if p.name.startswith("temp_raw_") or p.name.startswith("._") or "__MACOSX" in p.parts:
                continue
            key = (p.name, p.stat().st_size)
            if key not in seen:
                seen.add(key)
                found.append(p)
    return found


def build_labels(
    videos: Optional[List[Union[str, Path]]] = None,
    output_path: Union[str, Path] = "data/processed/training_labels.csv",
    variants_dir: Union[str, Path] = "data/processed/variants",
    variants_per_video: int = 2,
    window: int = 15,
    tau: float = 0.8,
    teacher_conf: float = 0.25,
    max_frames: Optional[int] = None,
    config_path: Union[str, Path] = "config.yaml",
    gt_dir: Optional[Union[str, Path]] = None,
) -> pd.DataFrame:
    """Generate training rows with all 10 causal features.

    When ``gt_dir`` is provided, labels are derived directly from ground truth:
    a frame is labelled ROBUST if FAST misses GT targets in the next ``window`` frames.
    Otherwise falls back to ROBUST-teacher agreement.
    """
    from src.evaluation.ground_truth import discover_gt_for_video, load_mot_gt
    from src.pipeline.pipeline import Mode, TrackingPipeline

    videos = [
        Path(v)
        for v in (videos or find_videos(["data/uploads", "data/raw", "Videos", "data/sample_test.mp4"]))
    ]
    if not videos:
        raise FileNotFoundError("No videos found. Put flight videos in Videos/, data/raw/, or data/uploads/.")

    clips: List[Path] = []
    for v in videos:
        clips.append(v)
        for k in range(variants_per_video):
            dst = Path(variants_dir) / f"{v.stem}__var{k}.mp4"
            if not dst.is_file():
                print(f"  making condition-varied copy {dst.name}")
                make_condition_variant(v, dst, seed=1000 + k, max_frames=max_frames)
            clips.append(dst)

    fast = TrackingPipeline(Mode.FAST, config_path=config_path)
    robust = None if gt_dir else TrackingPipeline(Mode.ROBUST, config_path=config_path)
    names = RollingFeatures.names()
    tables = []
    for n, clip in enumerate(clips, 1):
        src_stem = clip.stem.split("__var")[0]
        print(f"[{n}/{len(clips)}] {clip.name}: FAST pass...", flush=True)
        f_sum = fast.run_video(clip, max_frames=max_frames, save_outputs=False)

        if gt_dir:
            gt_path = discover_gt_for_video(clip.parent / f"{src_stem}.mov", gt_dir=gt_dir)
            if gt_path is None:
                gt_path = discover_gt_for_video(clip, gt_dir=gt_dir)
            if gt_path is None:
                print(f"    skipping {clip.name}: no GT found in {gt_dir}")
                continue
            gt_df = load_mot_gt(gt_path)
            frames = len(f_sum["records_df"])
            agree = frame_agreement(f_sum["tracks_df"], gt_df, frames, reference_conf=0.0, iou_thr=0.3)
            future = forward_window_mean(agree, window)
            label = np.where(future >= tau, "FAST", "ROBUST")
            passes = (("fast_pass", f_sum),)
        else:
            print(f"[{n}/{len(clips)}] {clip.name}: ROBUST pass...", flush=True)
            r_sum = robust.run_video(clip, max_frames=max_frames, save_outputs=False)
            frames = min(len(f_sum["records_df"]), len(r_sum["records_df"]))
            agree = frame_agreement(f_sum["tracks_df"], r_sum["tracks_df"], frames, reference_conf=teacher_conf)
            future = forward_window_mean(agree, window)
            label = np.where(future >= tau, "FAST", "ROBUST")
            passes = (("fast_pass", f_sum), ("robust_pass", r_sum))

        for source, summ in passes:
            rows = pd.DataFrame(_rolling_rows(summ["records_df"].iloc[:frames]), columns=names)
            rows["frame"] = np.arange(frames)
            rows["agreement"] = agree.round(3)
            rows["future_agreement"] = future.round(3)
            rows["label"] = label
            rows["clip"] = clip.stem
            rows["source_video"] = src_stem
            rows["pass"] = source
            tables.append(rows)
        print(f"    {frames} frames, ROBUST needed on {np.mean(label == 'ROBUST') * 100:.0f}% of them")

    table = pd.concat(tables, ignore_index=True)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(output_path, index=False)
    print(f"Saved {len(table)} labelled rows to {output_path}")
    return table


def _scores(y_true, y_pred, y_prob_robust: Optional[np.ndarray] = None) -> Dict[str, float]:
    out = {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "balanced_accuracy": round(float(balanced_accuracy_score(y_true, y_pred)), 4),
        "f1_robust": round(float(f1_score(y_true, y_pred, pos_label="ROBUST", zero_division=0)), 4),
        "predicted_robust_share": round(float(np.mean(np.asarray(y_pred) == "ROBUST")), 4),
    }
    if y_prob_robust is not None and len(set(y_true)) == 2:
        y_bin = (np.asarray(y_true) == "ROBUST").astype(int)
        out["roc_auc"] = round(float(roc_auc_score(y_bin, y_prob_robust)), 4)
    return out


def _build_models(random_state: int, n_estimators: int) -> Dict[str, Any]:
    models: Dict[str, Any] = {
        "tree": DecisionTreeClassifier(
            max_depth=5, min_samples_leaf=20, class_weight="balanced", random_state=random_state
        ),
        "forest": RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=10,
            min_samples_leaf=10,
            class_weight="balanced",
            n_jobs=-1,
            random_state=random_state,
        ),
        "gbm": GradientBoostingClassifier(
            n_estimators=min(150, n_estimators),
            max_depth=4,
            learning_rate=0.08,
            random_state=random_state,
        ),
        "logreg": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=random_state)),
        ]),
        "svm": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", CalibratedClassifierCV(SVC(kernel="rbf", C=1.0, class_weight="balanced", random_state=random_state), cv=3)),
        ]),
        "mlp": Pipeline([
            ("scaler", StandardScaler()),
            (
                "clf",
                MLPClassifier(
                    hidden_layer_sizes=(64, 32),
                    max_iter=300,
                    early_stopping=True,
                    random_state=random_state,
                ),
            ),
        ]),
    }

    try:
        from xgboost import XGBClassifier  # type: ignore

        models["xgb"] = XGBClassifier(
            n_estimators=n_estimators,
            max_depth=5,
            learning_rate=0.08,
            eval_metric="logloss",
            random_state=random_state,
            n_jobs=-1,
        )
    except ImportError:
        pass

    try:
        from lightgbm import LGBMClassifier  # type: ignore

        models["lgbm"] = LGBMClassifier(
            n_estimators=n_estimators,
            max_depth=6,
            learning_rate=0.08,
            class_weight="balanced",
            random_state=random_state,
            verbose=-1,
        )
    except ImportError:
        pass

    return models


def _feature_importance(model: Any, names: List[str]) -> Dict[str, float]:
    est = model.steps[-1][1] if hasattr(model, "steps") else model
    if hasattr(est, "feature_importances_"):
        vals = np.asarray(est.feature_importances_, dtype=float)
    elif hasattr(est, "coef_"):
        vals = np.abs(np.ravel(est.coef_))
        s = vals.sum()
        if s > 0:
            vals = vals / s
    else:
        return {}
    return dict(sorted(zip(names, np.round(vals, 4).tolist()), key=lambda kv: -kv[1]))


def train_selector(
    training_data_path: Union[str, Path] = "data/processed/training_labels.csv",
    weights_dir: Union[str, Path] = "weights",
    report_path: Union[str, Path] = "outputs/selector/report.json",
    n_estimators: int = 100,
    random_state: int = 42,
    config_path: Union[str, Path] = "config.yaml",
    selected_models: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Fit all selector models with leak-free source_video GroupKFold CV."""
    data = pd.read_csv(training_data_path)
    # Support older CSVs with 5 features or new CSVs with 10 features
    available_base = [f for f in BASE_FEATURES if f in data.columns]
    names = RollingFeatures.names(available_base)
    missing = [c for c in names if c not in data.columns]
    if missing:
        raise ValueError(f"Training CSV is missing columns: {missing}")

    X = data[names].to_numpy(float)
    y = data["label"].to_numpy()
    # Leak-free grouping by source_video (not clip!) so synthetic variants never leak across folds
    group_col = "source_video" if "source_video" in data.columns else "clip"
    groups = data[group_col].to_numpy()

    if len(set(y)) < 2:
        raise ValueError(
            f"Every label is '{y[0]}'. The selector needs both FAST and ROBUST examples: "
            "add more varied videos or adjust --tau."
        )

    cfg = load_config(config_path)
    rules = RuleBasedSelector((cfg.get("selector") or {}).get("rule_thresholds", {}))
    rule_pred = np.array([rules.predict(r) for r in data[available_base].to_dict("records")])

    n_groups = len(set(groups))
    folds = list(GroupKFold(n_splits=min(5, n_groups)).split(X, y, groups)) if n_groups >= 2 else []
    report: Dict[str, Any] = {
        "rows": int(len(data)),
        "source_videos": int(n_groups),
        "group_by": group_col,
        "label_share_robust": round(float(np.mean(y == "ROBUST")), 4),
        "features": names,
        "cv_folds": len(folds),
        "rules_baseline": _scores(y, rule_pred),
        "models": {},
    }

    all_models = _build_models(random_state, n_estimators)
    if selected_models:
        all_models = {k: v for k, v in all_models.items() if k in selected_models}

    weights_dir = Path(weights_dir)
    weights_dir.mkdir(parents=True, exist_ok=True)

    for kind, model in all_models.items():
        uses_binary = kind in ("xgb",)
        y_fit = (y == "ROBUST").astype(int) if uses_binary else y

        cv_pred = np.empty(len(y), dtype=object)
        cv_prob = np.zeros(len(y), dtype=float)
        for train_idx, test_idx in folds:
            fold_model = clone(model)
            if kind == "gbm":
                sw = compute_sample_weight("balanced", y_fit[train_idx])
                fold_model.fit(X[train_idx], y_fit[train_idx], sample_weight=sw)
            else:
                fold_model.fit(X[train_idx], y_fit[train_idx])
            raw_pred = fold_model.predict(X[test_idx])
            if uses_binary:
                cv_pred[test_idx] = np.where(raw_pred == 1, "ROBUST", "FAST")
                cv_prob[test_idx] = fold_model.predict_proba(X[test_idx])[:, 1]
            else:
                cv_pred[test_idx] = raw_pred
                if hasattr(fold_model, "predict_proba"):
                    r_col = list(fold_model.classes_).index("ROBUST")
                    cv_prob[test_idx] = fold_model.predict_proba(X[test_idx])[:, r_col]

        cv_scores = _scores(y, cv_pred, cv_prob if folds else None) if folds else None

        if kind == "gbm":
            model.fit(X, y_fit, sample_weight=compute_sample_weight("balanced", y_fit))
        else:
            model.fit(X, y_fit)
        train_raw = model.predict(X)
        train_pred = np.where(train_raw == 1, "ROBUST", "FAST") if uses_binary else train_raw
        train_prob = None
        if hasattr(model, "predict_proba"):
            r_col = 1 if uses_binary else list(model.classes_).index("ROBUST")
            train_prob = model.predict_proba(X)[:, r_col]

        entry = {
            "cv": cv_scores,
            "cv_confusion": (
                confusion_matrix(y, cv_pred, labels=["FAST", "ROBUST"]).tolist() if folds else None
            ),
            "train": _scores(y, train_pred, train_prob),
            "feature_importance": _feature_importance(model, names),
        }
        if kind == "tree":
            entry["rules_text"] = export_text(model, feature_names=names, max_depth=5)
        report["models"][kind] = entry

        bundle = {
            "kind": kind,
            "model": model,
            "features": names,
            "threshold": 0.5,
            "window": ROLLING_WINDOW,
            "decide_every": 3,
            "report": entry,
            "trained_on": {"rows": int(len(data)), "source_videos": int(n_groups)},
        }
        path = weights_dir / f"selector_{kind}.joblib"
        joblib.dump(bundle, path)
        print(f"Saved {kind} -> {path}")

    Path(report_path).parent.mkdir(parents=True, exist_ok=True)
    Path(report_path).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Report -> {report_path}")
    return report


def _print_summary(report: Dict[str, Any]) -> None:
    print(
        f"\n{report['rows']} rows from {report['source_videos']} source videos "
        f"(grouped by {report['group_by']}), ROBUST label on {report['label_share_robust'] * 100:.0f}% of rows"
    )
    print(f"{'selector':<10}{'accuracy':>10}{'bal. acc':>10}{'F1 robust':>11}{'ROC-AUC':>10}{'robust %':>10}")
    rows = [("rules", report["rules_baseline"])] + [
        (k, v["cv"] or v["train"]) for k, v in report["models"].items()
    ]
    for name, s in rows:
        auc = f"{s['roc_auc']:.3f}" if "roc_auc" in s else "   n/a"
        print(
            f"{name:<10}{s['accuracy']:>10.3f}{s['balanced_accuracy']:>10.3f}"
            f"{s['f1_robust']:>11.3f}{auc:>10}{s['predicted_robust_share'] * 100:>9.0f}%"
        )


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--videos", nargs="*", help="video files or folders")
    p.add_argument("--variants", type=int, default=2, help="condition-varied copies per video")
    p.add_argument("--max-frames", type=int, default=None, help="cap frames per clip")
    p.add_argument("--window", type=int, default=15, help="frames ahead the label looks at")
    p.add_argument("--tau", type=float, default=0.8, help="share of ROBUST's targets FAST must find")
    p.add_argument("--skip-labels", action="store_true", help="reuse data/processed/training_labels.csv")
    p.add_argument("--models", nargs="*", default=None, help="subset of models to train")
    p.add_argument("--gt-dir", type=str, default=None, help="MOT GT dir for GT-based labels instead of ROBUST teacher")
    p.add_argument("--split-file", type=str, default=None, help="JSON split file (e.g. data/splits/purdue_split.json)")
    p.add_argument("--split", type=str, default="train", help="Split key to use from --split-file")
    args = p.parse_args()

    if not args.skip_labels:
        if args.split_file:
            sp = json.loads(Path(args.split_file).read_text(encoding="utf-8"))
            videos = [Path("Videos") / f"{name}.mov" for name in sp[args.split]]
        else:
            videos = find_videos(args.videos) if args.videos else None
        build_labels(
            videos,
            variants_per_video=args.variants,
            window=args.window,
            tau=args.tau,
            max_frames=args.max_frames,
            gt_dir=args.gt_dir,
        )
    _print_summary(train_selector(selected_models=args.models))


if __name__ == "__main__":
    main()
