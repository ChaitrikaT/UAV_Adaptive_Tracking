"""Tracking metrics of PREDICTED tracks against reference annotations.

Reference IDs are used only here, for scoring. Reports detection P/R at IoU>=thr,
ID switches, fragmentations, MOTA (CLEAR-MOT, Bernardin & Stiefelhagen 2008) and IDF1
(Ristani et al. 2016), with IoU>=thr as the match criterion.

Matching per frame follows CLEAR-MOT: (1) a reference object keeps its previous prediction ID
if that prediction is still present and overlaps (IoU>=thr); (2) the remaining objects are
matched by an assignment that is GATED by IoU>=thr BEFORE it is solved, so the number of valid
matches is maximised first and IoU is the secondary criterion.

PROTOCOL NOTE. With annotated_only=True (default) frames that have no reference rows are not
scored at all, so predictions there are NOT counted as false positives. That is a project
choice (the dataset does not confirm those frames are empty), NOT the official MOTChallenge
protocol. Only annotated_only=False scores every listed frame; the result dict reports which
protocol was used in "protocol". Do not present these numbers as official benchmark scores.
Prediction IDs must be unique within a frame (ValueError otherwise). Duplicate reference ids in a
frame only raise a warning and are counted in 'ref_duplicate_id_rows'.
"""
from __future__ import annotations

import csv
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

from .tracker import iou_xyxy


def load_reference(path: Path, width=None, height=None):
    """-> {frame: [(id, x1, y1, x2, y2)]}  (frames 1-based, boxes clipped to image if size given)"""
    out = defaultdict(list)
    with Path(path).open("r", encoding="utf-8-sig", errors="replace", newline="") as f:
        for row in csv.reader(f):
            if len(row) < 6:
                continue
            try:
                fr, tid = int(float(row[0])), int(float(row[1]))
                x, y, w, h = map(float, row[2:6])
            except (ValueError, OverflowError):
                continue
            if fr < 1 or w <= 0 or h <= 0:
                continue
            x1, y1, x2, y2 = x, y, x + w, y + h
            if width and height:
                x1, y1, x2, y2 = max(0, x1), max(0, y1), min(width, x2), min(height, y2)
            if x2 > x1 and y2 > y1:
                out[fr].append((tid, x1, y1, x2, y2))
    return out


def load_predictions(path: Path):
    """tracks.csv -> {frame: [(id, x1, y1, x2, y2)]}"""
    out = defaultdict(list)
    with Path(path).open(newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            out[int(r["frame"])].append((int(r["track_id"]), float(r["x1"]), float(r["y1"]),
                                         float(r["x2"]), float(r["y2"])))
    return out


BIG = 1e6  # cost of an invalid (IoU < thr) pair in the gated assignment


def _duplicate_id_rows(rows_by_frame, frames) -> int:
    """Number of rows whose id already occurred earlier in the same frame."""
    return sum(len(rows) - len({r[0] for r in rows}) for rows in (rows_by_frame.get(fr, []) for fr in frames))


def match_frame(g, p, thr, prev=None):
    """One-frame CLEAR-MOT matching. g, p: lists of (id, x1, y1, x2, y2).

    prev: {ref id -> pred id} from earlier frames (continuity). Returns [(gi, pj)].
    """
    prev = prev or {}
    pairs, used_g, used_p = [], set(), set()
    pid_index = {pr[0]: pj for pj, pr in enumerate(p)}
    for gi, gt in enumerate(g):                      # 1) continuity
        pj = pid_index.get(prev.get(gt[0]))
        if pj is not None and pj not in used_p and iou_xyxy(gt[1:], p[pj][1:]) >= thr:
            pairs.append((gi, pj)); used_g.add(gi); used_p.add(pj)
    rest_g = [i for i in range(len(g)) if i not in used_g]
    rest_p = [j for j in range(len(p)) if j not in used_p]
    if rest_g and rest_p:                            # 2) gated optimal assignment
        iou = np.array([[iou_xyxy(g[i][1:], p[j][1:]) for j in rest_p] for i in rest_g])
        cost = np.where(iou >= thr, 1.0 - iou, BIG)
        for r, c in zip(*linear_sum_assignment(cost)):
            if iou[r, c] >= thr:
                pairs.append((rest_g[r], rest_p[c]))
    return pairs


def count_matches(ref_boxes, det_boxes, thr=0.5) -> int:
    """Maximum number of one-to-one matches at IoU>=thr (ids ignored). Used for frame labels."""
    g = [(0, *b[:4]) for b in ref_boxes]
    p = [(0, *b[:4]) for b in det_boxes]
    return len(match_frame(g, p, thr))


def evaluate(ref, pred, frames, iou_thr=0.5, annotated_only=True) -> dict:
    frames = sorted({f for f in frames if (f in ref) or not annotated_only})
    if _duplicate_id_rows(pred, frames):
        bad = next(fr for fr in frames if len(pred.get(fr, [])) != len({r[0] for r in pred[fr]}))
        raise ValueError(f"prediction: duplicate track id inside frame {bad}; MOT metrics are undefined")
    ref_dups = _duplicate_id_rows(ref, frames)
    if ref_dups:
        warnings.warn(f"reference has {ref_dups} row(s) repeating an id inside one frame; "
                      "ID-based metrics for those ids are unreliable (see src/audit_reference_ids.py)")
    tp = fp = fn = idsw = n_ref = n_pred = 0
    last_match: dict[int, int] = {}          # ref id -> pred id (most recent match, kept across gaps)
    ref_tracked: dict[int, list[bool]] = defaultdict(list)

    for fr in frames:
        g, p = ref.get(fr, []), pred.get(fr, [])
        n_ref += len(g); n_pred += len(p)
        pairs = match_frame(g, p, iou_thr, last_match)
        tp += len(pairs); fp += len(p) - len(pairs); fn += len(g) - len(pairs)
        got = {g[gi][0]: p[pj][0] for gi, pj in pairs}
        for gt in g:
            ref_tracked[gt[0]].append(gt[0] in got)
            if gt[0] in got:
                if gt[0] in last_match and last_match[gt[0]] != got[gt[0]]:
                    idsw += 1
                last_match[gt[0]] = got[gt[0]]
    # IDF1: count frames where (ref id, pred id) overlap at IoU>=thr, then take the single
    # best one-to-one identity assignment over the whole sequence.
    overlap: dict[tuple[int, int], int] = defaultdict(int)
    for fr in frames:
        for gt in ref.get(fr, []):
            for pr in pred.get(fr, []):
                if iou_xyxy(gt[1:], pr[1:]) >= iou_thr:
                    overlap[(gt[0], pr[0])] += 1
    idtp = 0
    if overlap:
        gids = sorted({k[0] for k in overlap}); pids = sorted({k[1] for k in overlap})
        gi_ = {v: i for i, v in enumerate(gids)}; pi_ = {v: i for i, v in enumerate(pids)}
        m = np.zeros((len(gids), len(pids)))
        for (a, b), v in overlap.items():
            m[gi_[a], pi_[b]] = v
        r, c = linear_sum_assignment(-m)
        idtp = int(m[r, c].sum())
    idfn, idfp = n_ref - idtp, n_pred - idtp
    frag = 0  # tracked -> lost -> tracked again (MOTChallenge "Frag"); a late FIRST pickup is not one
    for flags in ref_tracked.values():
        ever = False
        for prev, cur in zip(flags, flags[1:]):
            ever = ever or prev
            if ever and (not prev) and cur:
                frag += 1
    return {
        "frames_scored": len(frames), "ref_boxes": n_ref, "pred_boxes": n_pred,
        "TP": tp, "FP": fp, "FN": fn, "ID_switches": idsw, "fragmentations": frag,
        "ref_duplicate_id_rows": ref_dups, "IDTP": idtp, "IDFP": idfp, "IDFN": idfn,
        "precision": round(tp / (tp + fp), 4) if tp + fp else 0.0,
        "recall": round(tp / (tp + fn), 4) if tp + fn else 0.0,
        "MOTA": round(1 - (fn + fp + idsw) / n_ref, 4) if n_ref else 0.0,
        "IDF1": round(2 * idtp / (2 * idtp + idfp + idfn), 4) if (2 * idtp + idfp + idfn) else 0.0,
        "distinct_pred_ids": len({q[0] for fr in frames for q in pred.get(fr, [])}),
        "iou_thr": iou_thr, "annotated_only": annotated_only,
        "protocol": ("annotated-frames-only (project variant, not official MOTChallenge)"
                     if annotated_only else "all listed frames scored (CLEAR-MOT/IDF1 at IoU threshold)"),
    }
