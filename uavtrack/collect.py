"""Training data for the online policy: cache both modes once, then roll policies out on the cache.

Why a cache + rollout instead of the first version (tracker fed by ROBUST detections only):
  * CONFIRMED BUG: with ROBUST-fed state, `prev_robust` was constant (1) in training but varies at
    deployment, and tracker-state features (frames_since_seen, size, confidence) followed ROBUST
    behaviour that the deployed policy does not reproduce. Training and deployment inputs differed.
  * The detector is deterministic per (frame, imgsz), so detections of BOTH modes can be stored once.
    A rollout then steps the tracker with the detections of the mode the *policy under training*
    chose, exactly as the online pipeline would, with no further inference.
  * DAgger-style iterations (rule policy -> learned policy -> ...) aggregate data from the states
    the learned policy itself visits.
Reference boxes are used ONLY to label frames (which mode found the target), never as features.
"""
from __future__ import annotations

import time
from pathlib import Path

import cv2
import numpy as np

from .evaluate import count_matches, load_reference
from .features import FEATURE_NAMES, build_features, global_features, small_gray
from .labels import label_frame  # noqa: F401  (re-exported: single label rule lives in labels.py)
from .policy import MODES, FixedPolicy, LearnedPolicy, RulePolicy
from .tracker import Tracker, TrackerConfig


def count_hits(ref_boxes, dets, thr=0.5) -> int:
    """Reference boxes matched one-to-one at IoU >= thr (optimal, IoU-gated assignment)."""
    return count_matches([g[1:] for g in ref_boxes], [d[:4] for d in dets], thr)


_hits = count_hits  # name used by the correctness-patch tests


def target_characteristics(gray, box) -> dict:
    """Reference-derived target descriptors for ANALYSIS only (never fed to the online policy)."""
    h, w = gray.shape
    x1, y1, x2, y2 = [int(round(v)) for v in box]
    x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)
    if x2 - x1 < 2 or y2 - y1 < 2:
        return {}
    inner = gray[y1:y2, x1:x2].astype(float)
    pad = max(4, int(0.5 * max(x2 - x1, y2 - y1)))
    X1, Y1, X2, Y2 = max(0, x1 - pad), max(0, y1 - pad), min(w, x2 + pad), min(h, y2 + pad)
    ring = gray[Y1:Y2, X1:X2].astype(float)
    ring_mean = (ring.sum() - inner.sum()) / max(1, ring.size - inner.size)
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    return {"target_size_px": float(np.sqrt((x2 - x1) * (y2 - y1))),
            "local_contrast": float(abs(inner.mean() - ring_mean)),
            "edge_distance_frac": float(min(cx, cy, w - cx, h - cy) / min(w, h)),
            "cx": float(cx), "cy": float(cy)}


def cache_clip(video_path: Path, ref_path: Path, detector, start=1, end=-1, log_every=100) -> dict:
    """Run BOTH modes on every frame once; store detections, timings, global features, reference boxes.

    Timings are measured here (decode, global-feature extraction, detector per mode, GPU-synchronised)
    so a policy replay can compose end-to-end cost from real measurements.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open {video_path}")
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    ref = load_reference(ref_path, w, h)
    if start > 1:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start - 1)
    last = end if end > 0 else (total if total > 0 else 10 ** 9)
    frames, prev_small = [], None
    for frame_no in range(start, last + 1):
        t0 = time.perf_counter()
        ok, frame = cap.read()
        if not ok:
            break
        t1 = time.perf_counter()
        small = small_gray(frame)
        g = global_features(small, prev_small)
        t2 = time.perf_counter()
        rec = {"frame": frame_no, "g": g, "ref": ref.get(frame_no),
               "decode_ms": (t1 - t0) * 1e3, "feature_ms": (t2 - t1) * 1e3}
        for mode, size in MODES.items():
            if hasattr(detector, "sync"):
                detector.sync()
            t0 = time.perf_counter()
            rec[f"dets_{mode.lower()}"] = np.asarray(detector.detect(frame, size), dtype=float).reshape(-1, 5)
            if hasattr(detector, "sync"):
                detector.sync()
            rec[f"{mode.lower()}_ms"] = (time.perf_counter() - t0) * 1e3
        if rec["ref"]:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            rec["target"] = target_characteristics(gray, rec["ref"][0][1:])   # first reference object
        frames.append(rec)
        prev_small = small
        if log_every and frame_no % log_every == 0:
            print(f"  cached up to frame {frame_no}", flush=True)
    cap.release()
    return {"frames": frames, "width": w, "height": h, "video": str(video_path), "conf_floor": None}


def rollout(cache: dict, policy, tracker_cfg: TrackerConfig | None = None, tracks_out: list | None = None) -> list[dict]:
    """Replay a policy over cached detections exactly as pipeline.run_video would.

    Each row records the CHOSEN mode, the features the policy saw, labels (reference-derived, never
    features) and a composed per-frame cost: decode + global features (measured in cache_clip) +
    chosen-mode detector time (measured in cache_clip) + policy/state/tracker overhead (measured here).
    If tracks_out is a list, tracker output rows are appended to it.
    """
    tracker, prev_mode = Tracker(tracker_cfg), "FAST"
    policy.reset()
    rows = []
    for rec in cache["frames"]:
        t0 = time.perf_counter()
        feats = build_features(rec["g"], tracker.primary_state(), prev_mode)
        mode = policy.decide(feats)
        t1 = time.perf_counter()
        dets = rec["dets_robust"] if mode == "ROBUST" else rec["dets_fast"]
        out = tracker.update(dets)
        t2 = time.perf_counter()
        if tracks_out is not None:
            tracks_out.extend({"frame": rec["frame"], **o} for o in out)
        det_ms = rec["robust_ms"] if mode == "ROBUST" else rec["fast_ms"]
        overhead = (t1 - t0 + t2 - t1) * 1e3
        row = {"frame": rec["frame"], "mode": mode, **{k: float(feats[k]) for k in FEATURE_NAMES},
               "labelled": rec["ref"] is not None, "ref_boxes": len(rec["ref"]) if rec["ref"] else 0,
               "fast_tp": count_hits(rec["ref"], rec["dets_fast"]) if rec["ref"] else None,
               "robust_tp": count_hits(rec["ref"], rec["dets_robust"]) if rec["ref"] else None,
               "fast_ms": rec["fast_ms"], "robust_ms": rec["robust_ms"], "detect_ms": det_ms,
               "decode_ms": rec["decode_ms"], "feature_ms": rec["feature_ms"], "overhead_ms": overhead,
               "total_ms": rec["decode_ms"] + rec["feature_ms"] + det_ms + overhead,
               "n_tracks": len(out)}
        rows.append(row)
        prev_mode = mode
    return rows


def _resolve_driver(driver):
    if driver is None:
        return RulePolicy()
    if isinstance(driver, str):
        if driver.lower() in ("fast", "robust"):
            return FixedPolicy(driver.upper())
        if driver.lower() == "rule":
            return RulePolicy()
        raise ValueError(f"Unknown driver '{driver}'")
    return driver


def collect_clip(video_path: Path, ref_path: Path, detector, start=1, end=-1, tracker_cfg=None, log_every=100,
                 driver=None):
    """Labelled rows for one clip, with tracker state/prev_robust following a DRIVER policy (default rule).

    Thin wrapper over cache_clip + rollout, so the collected features are the ones run_video would see
    online for that driver (see tests). Only frames with reference annotations are returned.
    """
    cache = cache_clip(video_path, ref_path, detector, start, end, log_every)
    rows = rollout(cache, _resolve_driver(driver), tracker_cfg)
    keep = ("frame", *FEATURE_NAMES, "ref_boxes", "fast_tp", "robust_tp", "fast_ms", "robust_ms")
    return [{**{k: r[k] for k in keep}, "driver_robust": int(r["mode"] == "ROBUST")} for r in rows if r["labelled"]]


def build_xy(rows, both_fail="exclude", equal_hit="fast"):
    X, y, dropped = [], [], 0
    for r in rows:
        if not r["labelled"]:
            continue
        lab = label_frame(r["fast_tp"], r["robust_tp"], both_fail, equal_hit)
        if lab is None:
            dropped += 1
            continue
        X.append([r[k] for k in FEATURE_NAMES]); y.append(lab)
    return X, y, dropped


def fit_tree(X, y, max_depth=3):
    from collections import Counter
    from sklearn.tree import DecisionTreeClassifier
    if len(Counter(y)) < 2:
        raise ValueError(f"Only one label class {dict(Counter(y))}; a learned policy would be constant.")
    return DecisionTreeClassifier(max_depth=max_depth, min_samples_leaf=10, class_weight="balanced",
                                  random_state=42).fit(X, y)


def blocked_cv(X, y, folds=5, max_depth=3) -> dict:
    """Contiguous (not shuffled) folds: neighbouring frames are near-duplicates, shuffling would leak."""
    from sklearn.metrics import balanced_accuracy_score
    n = len(X)
    scores = []
    for k in range(folds):
        lo, hi = k * n // folds, (k + 1) * n // folds
        tr_x, tr_y = X[:lo] + X[hi:], y[:lo] + y[hi:]
        if hi <= lo or len(set(tr_y)) < 2 or len(set(y[lo:hi])) < 2:
            continue  # balanced accuracy is undefined for a one-class test fold
        clf = fit_tree(tr_x, tr_y, max_depth)
        scores.append(balanced_accuracy_score(y[lo:hi], clf.predict(X[lo:hi])))
    return {"folds_used": len(scores), "mean_balanced_accuracy": round(float(np.mean(scores)), 4) if scores else None,
            "note": "0.5 = no better than chance on balanced classes"}


def dagger_train(cache: dict, tracker_cfg: TrackerConfig | None = None, iterations=2, max_depth=3,
                 both_fail="exclude", equal_hit="fast"):
    """Iteration 0 rolls out the rule policy; later iterations roll out the previously learned tree
    and aggregate all rollouts. Returns (classifier, report dict)."""
    caches = list(cache) if isinstance(cache, (list, tuple)) else [cache]
    policy, data, history = RulePolicy(), [], []
    clf = None
    for it in range(iterations):
        rows = [r for c in caches for r in rollout(c, policy, tracker_cfg)]
        data += rows
        X, y, dropped = build_xy(data, both_fail, equal_hit)
        clf = fit_tree(X, y, max_depth)
        from collections import Counter
        history.append({"iteration": it, "rollout_policy": policy.name,
                        "rollout_robust_fraction": round(sum(r["mode"] == "ROBUST" for r in rows) / len(rows), 4),
                        "aggregated_rows": len(data), "training_rows": len(X), "dropped_no_evidence": dropped,
                        "labels": dict(Counter(y))})
        policy = LearnedPolicy(clf)
    X, y, _ = build_xy([r for c in caches for r in rollout(c, policy, tracker_cfg)], both_fail, equal_hit)
    cv = blocked_cv(X, y, max_depth=max_depth) if len(set(y)) > 1 else {"folds_used": 0}
    return clf, {"history": history, "blocked_cv_final_policy_rollout": cv,
                 "tie_rules": {"both_fail": both_fail, "equal_hit": equal_hit}}
