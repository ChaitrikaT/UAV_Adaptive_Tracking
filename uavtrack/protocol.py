"""Offline research-evaluation protocol on cached per-frame detections of BOTH modes.

Everything here consumes the cache written by collect.cache_clip (real detector, real video).
Reference annotations are used only for scoring/labels/analysis, never as policy inputs.
Protocol note: like uavtrack.evaluate, frames WITHOUT reference rows are not scored (project variant).

Pieces
  detection_report   : precision / recall / F1 at fixed confidence thresholds + AP50, per mode
  condition_report   : per-frame scene + target descriptors vs whether each mode found the target
  difficulty_auc     : can condition features predict that FAST will miss? (cross-clip AUC)
  replay_eval        : run a policy over the cache -> tracker output, tracking metrics, composed cost
  paired_block_bootstrap / verdict : is a policy better than FAST / ROBUST, at what cost, with CIs
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np

from .collect import count_hits, rollout
from .evaluate import evaluate, match_frame
from .features import FEATURE_NAMES
from .policy import FixedPolicy
from .tracker import iou_xyxy

CONF_THRESHOLDS = (0.10, 0.25, 0.50)


# ------------------------------------------------------------------ detection
def ap50(cache: dict, mode: str, thr=0.5) -> float:
    """All-point interpolated AP at IoU>=thr over annotated frames (detections have conf >= detector floor,
    so this is NOT directly comparable with Ultralytics' mAP50, which uses a lower floor)."""
    key = f"dets_{mode.lower()}"
    gts, entries, n_gt = {}, [], 0
    for i, rec in enumerate(cache["frames"]):
        if rec["ref"]:
            gts[i] = rec["ref"]; n_gt += len(rec["ref"])
            entries += [(float(d[4]), i, d[:4]) for d in rec[key]]
    if not n_gt or not entries:
        return 0.0
    entries.sort(key=lambda e: -e[0])
    used = {i: [False] * len(g) for i, g in gts.items()}
    hit = []
    for _, i, box in entries:
        best, bj = 0.0, -1
        for j, g in enumerate(gts[i]):
            v = iou_xyxy(g[1:], box)
            if not used[i][j] and v > best:
                best, bj = v, j
        if best >= thr:
            used[i][bj] = True
        hit.append(1 if best >= thr else 0)
    hit = np.array(hit)
    tp, fp = np.cumsum(hit), np.cumsum(1 - hit)
    rec, prec = tp / n_gt, tp / (tp + fp)
    mrec, mpre = np.concatenate([[0], rec, [1]]), np.concatenate([[0], prec, [0]])
    for k in range(len(mpre) - 2, -1, -1):
        mpre[k] = max(mpre[k], mpre[k + 1])
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))


def detection_report(cache: dict, thr=0.5) -> dict:
    out = {"annotated_frames": sum(1 for r in cache["frames"] if r["ref"]),
           "total_frames": len(cache["frames"]), "iou_thr": thr, "modes": {}}
    for mode in ("FAST", "ROBUST"):
        key, per = f"dets_{mode.lower()}", {}
        for ct in CONF_THRESHOLDS:
            tp = fp = fn = hit_frames = 0
            for rec in cache["frames"]:
                if not rec["ref"]:
                    continue
                dets = [d for d in rec[key] if d[4] >= ct]
                m = len(match_frame([(0, *g[1:]) for g in rec["ref"]], [(0, *d[:4]) for d in dets], thr))
                tp += m; fp += len(dets) - m; fn += len(rec["ref"]) - m; hit_frames += m > 0
            p = tp / (tp + fp) if tp + fp else 0.0
            r = tp / (tp + fn) if tp + fn else 0.0
            per[f"conf>={ct:.2f}"] = {"TP": tp, "FP": fp, "FN": fn, "precision": round(p, 4), "recall": round(r, 4),
                                      "F1": round(2 * p * r / (p + r), 4) if p + r else 0.0,
                                      "frame_hit_rate": round(hit_frames / out["annotated_frames"], 4)
                                      if out["annotated_frames"] else 0.0}
        per["AP50"] = round(ap50(cache, mode, thr), 4)
        per["mean_detect_ms"] = round(float(np.mean([r[f"{mode.lower()}_ms"] for r in cache["frames"]])), 3)
        out["modes"][mode] = per
    return out


# ------------------------------------------------------------------ conditions
def condition_rows(cache: dict, thr=0.5) -> list[dict]:
    rows, prev = [], None
    for rec in cache["frames"]:
        if not rec["ref"] or "target" not in rec or not rec["target"]:
            continue
        t = rec["target"]
        speed = None
        if prev is not None and rec["frame"] > prev[0]:
            speed = float(np.hypot(t["cx"] - prev[1], t["cy"] - prev[2]) / (rec["frame"] - prev[0]))
        prev = (rec["frame"], t["cx"], t["cy"])
        rows.append({"frame": rec["frame"], **rec["g"], "target_size_px": t["target_size_px"],
                     "local_contrast": t["local_contrast"], "edge_distance_frac": t["edge_distance_frac"],
                     "target_speed_px_per_frame": speed,
                     "fast_hit": int(count_hits(rec["ref"], rec["dets_fast"], thr) > 0),
                     "robust_hit": int(count_hits(rec["ref"], rec["dets_robust"], thr) > 0)})
    return rows


DESCRIPTORS = ["brightness", "contrast", "sharpness", "frame_change", "target_size_px", "local_contrast",
               "edge_distance_frac", "target_speed_px_per_frame"]


def condition_report(rows: list[dict], bins=4) -> dict:
    """Hit rate per quantile bin and Spearman correlation of each descriptor with detection success.
    Exploratory: correlation is not causation; neighbouring frames are not independent samples."""
    from scipy.stats import spearmanr
    out = {"n_frames": len(rows), "descriptors": {}}
    for d in DESCRIPTORS:
        pts = [(r[d], r["fast_hit"], r["robust_hit"]) for r in rows if r.get(d) is not None]
        if len(pts) < 8:
            continue
        x = np.array([p[0] for p in pts]); fh = np.array([p[1] for p in pts]); rh = np.array([p[2] for p in pts])
        edges = np.unique(np.quantile(x, np.linspace(0, 1, bins + 1)))
        b = np.clip(np.digitize(x, edges[1:-1]), 0, len(edges) - 2) if len(edges) > 2 else np.zeros(len(x), int)
        table = [{"range": [round(float(x[b == k].min()), 3), round(float(x[b == k].max()), 3)], "n": int((b == k).sum()),
                  "fast_hit_rate": round(float(fh[b == k].mean()), 3), "robust_hit_rate": round(float(rh[b == k].mean()), 3)}
                 for k in range(len(edges) - 1) if (b == k).any()]

        def rho(y):
            if y.std() == 0 or x.std() == 0:
                return None
            r, p = spearmanr(x, y)
            return {"rho": round(float(r), 3), "p": round(float(p), 4)}
        out["descriptors"][d] = {"bins": table, "spearman_fast": rho(fh), "spearman_robust": rho(rh)}
    return out


def _auc_fit_eval(Xtr, ytr, Xte, yte):
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.preprocessing import StandardScaler
    if len(set(ytr)) < 2 or len(set(yte)) < 2:
        return None
    sc = StandardScaler().fit(Xtr)
    clf = LogisticRegression(class_weight="balanced", max_iter=1000).fit(sc.transform(Xtr), ytr)
    return round(float(roc_auc_score(yte, clf.predict_proba(sc.transform(Xte))[:, 1])), 3)


def difficulty_auc(train_cache: dict, test_cache: dict, cfg=None) -> dict:
    """AUC (0.5 = chance) of predicting 'FAST misses the target' on test_cache from a model fitted on train_cache.
    scene   : whole-frame brightness/contrast/sharpness/change (online-available)
    online  : all policy features (scene + tracker state), tracker driven by FAST (online-available)
    target  : scene + reference-derived target size/contrast/edge/speed (ANALYSIS ONLY, not available online)"""
    res = {}
    ctr, cte = condition_rows(train_cache), condition_rows(test_cache)
    for name, cols in (("scene", DESCRIPTORS[:4]), ("target_plus_scene", [d for d in DESCRIPTORS if d != "target_speed_px_per_frame"])):
        f = lambda rows: ([[r[c] for c in cols] for r in rows], [1 - r["fast_hit"] for r in rows])
        (xa, ya), (xb, yb) = f(ctr), f(cte)
        res[name] = _auc_fit_eval(xa, ya, xb, yb) if xa and xb else None
    def online(cache):
        rows = [r for r in rollout(cache, FixedPolicy("FAST"), cfg) if r["labelled"]]
        return [[r[k] for k in FEATURE_NAMES] for r in rows], [int(r["fast_tp"] == 0) for r in rows]
    (xa, ya), (xb, yb) = online(train_cache), online(test_cache)
    res["online_policy_features"] = _auc_fit_eval(xa, ya, xb, yb) if xa and xb else None
    return res


# ------------------------------------------------------------------ policy replay + comparison
def replay_eval(cache: dict, policy, cfg=None, iou_thr=0.5) -> dict:
    tracks: list = []
    rows = rollout(cache, policy, cfg, tracks)
    pred = defaultdict(list)
    for t in tracks:
        pred[t["frame"]].append((t["track_id"], t["x1"], t["y1"], t["x2"], t["y2"]))
    ref = {r["frame"]: r["ref"] for r in cache["frames"] if r["ref"]}
    frames = [r["frame"] for r in cache["frames"]]
    metrics = evaluate(ref, pred, frames, iou_thr)
    n = len(rows)
    chosen_tp = np.array([(r["robust_tp"] if r["mode"] == "ROBUST" else r["fast_tp"]) or 0 for r in rows], float)
    ref_n = np.array([r["ref_boxes"] for r in rows], float)
    total_ms = np.array([r["total_ms"] for r in rows], float)
    return {"policy": getattr(policy, "name", "policy"), "tracking": metrics,
            "detection_recall_chosen_mode": round(float(chosen_tp.sum() / ref_n.sum()), 4) if ref_n.sum() else 0.0,
            "robust_fraction": round(sum(r["mode"] == "ROBUST" for r in rows) / n, 4),
            "mean_detect_ms": round(float(np.mean([r["detect_ms"] for r in rows])), 3),
            "mean_overhead_ms": round(float(np.mean([r["overhead_ms"] for r in rows])), 3),
            "mean_total_ms": round(float(total_ms.mean()), 3),
            "composed_fps": round(1000.0 / float(total_ms.mean()), 2),
            "_arrays": {"tp": chosen_tp, "ref": ref_n, "cost": total_ms}}


def paired_block_bootstrap(per_clip: list[dict], block=30, n_boot=2000, seed=0) -> dict:
    """per_clip: [{'tp_a','tp_b','ref','cost_a','cost_b'}] aligned per frame. Resamples contiguous blocks.
    Returns [point, lo95, hi95] for d_recall = recall_a - recall_b and d_cost_ms = mean cost_a - mean cost_b."""
    blocks = []
    for c in per_clip:
        for s in range(0, len(c["ref"]), block):
            sl = slice(s, s + block)
            blocks.append([c["tp_a"][sl].sum(), c["tp_b"][sl].sum(), c["ref"][sl].sum(),
                           c["cost_a"][sl].sum(), c["cost_b"][sl].sum(), len(c["ref"][sl])])
    B = np.array(blocks, float)
    S = B.sum(axis=0)
    point = ((S[0] - S[1]) / max(S[2], 1), (S[3] - S[4]) / max(S[5], 1))
    idx = np.random.default_rng(seed).integers(0, len(B), size=(n_boot, len(B)))
    T = B[idx].sum(axis=1)
    dr, dc = (T[:, 0] - T[:, 1]) / np.maximum(T[:, 2], 1), (T[:, 3] - T[:, 4]) / np.maximum(T[:, 5], 1)
    ci = lambda v, p: [round(float(p), 4), round(float(np.percentile(v, 2.5)), 4), round(float(np.percentile(v, 97.5)), 4)]
    return {"d_recall": ci(dr, point[0]), "d_cost_ms": ci(dc, point[1]), "blocks": len(B), "block_frames": block}


def verdict(vs_fast: dict, vs_robust: dict, margin=0.02) -> str:
    """Decision rule (stated in the report). d_* = learned - baseline; CIs are 95% block-bootstrap.
    SUPPORTED   : recall better than FAST (CI lower > 0) AND cheaper than ROBUST (cost CI upper < 0)
                  AND not worse than ROBUST by more than `margin` recall (CI lower > -margin).
    NOT SUPPORTED: not better than FAST (recall CI upper <= 0), OR clearly worse than ROBUST (recall CI upper < 0)
                  without being cheaper (cost CI lower >= 0).
    otherwise INCONCLUSIVE."""
    _, f_lo, f_hi = vs_fast["d_recall"]; _, r_lo, r_hi = vs_robust["d_recall"]; _, c_lo, c_hi = vs_robust["d_cost_ms"]
    if f_lo > 0 and c_hi < 0 and r_lo > -margin:
        return "SUPPORTED"
    if f_hi <= 0 or (r_hi < 0 and c_lo >= 0):
        return "NOT SUPPORTED"
    return "INCONCLUSIVE"


def compare_learned(results: dict, block=30, n_boot=2000, margin=0.02) -> dict:
    """results: {policy: [replay_eval dict per eval clip]} with equal clip order. Paired over frames."""
    def pair(a, b):
        per = [{"tp_a": ra["_arrays"]["tp"], "tp_b": rb["_arrays"]["tp"], "ref": ra["_arrays"]["ref"],
                "cost_a": ra["_arrays"]["cost"], "cost_b": rb["_arrays"]["cost"]}
               for ra, rb in zip(results[a], results[b])]
        return paired_block_bootstrap(per, block, n_boot)
    out = {"learned_vs_fast": pair("learned", "fast"), "learned_vs_robust": pair("learned", "robust"),
           "robust_vs_fast": pair("robust", "fast"), "rule_vs_fast": pair("rule", "fast")}
    out["verdict_learned"] = verdict(out["learned_vs_fast"], out["learned_vs_robust"], margin)
    out["margin"] = margin
    return out


def strip_arrays(res: dict) -> dict:
    return {k: v for k, v in res.items() if k != "_arrays"}
