"""Training labels for the FAST/ROBUST selector (reference boxes used ONLY here, never as features).

Rule, per annotated frame (fast_tp / robust_tp = number of reference targets matched at IoU>=0.5):
  robust_tp >  fast_tp                 -> "ROBUST"  (ROBUST found strictly more targets)
  fast_tp   >  robust_tp               -> "FAST"    (ROBUST gained nothing and cost more)
  equal and > 0                        -> "FAST"    (same accuracy; FAST is cheaper. Cost-based tie-break)
  equal and == 0 (BOTH modes failed)   -> None      (no evidence for either mode: EXCLUDED from training)

The older rule `ROBUST if robust_tp > fast_tp else FAST` also labelled the both-failed frames
FAST, which injected an unjustified FAST preference (and, being hardest frames, exactly the
ones where a selector should learn to ask for more effort).
"""
from __future__ import annotations

from collections import Counter


def label_frame(fast_tp, robust_tp, both_fail="exclude", equal_hit="fast"):
    """Preferred mode, or None when the frame carries no evidence (default for both-modes-failed)."""
    fast_tp, robust_tp = int(fast_tp), int(robust_tp)
    if robust_tp > fast_tp:
        return "ROBUST"
    if fast_tp > robust_tp:
        return "FAST"
    if fast_tp > 0:
        return equal_hit.upper()
    return None if both_fail == "exclude" else both_fail.upper()


def build_training_set(rows, feature_names, both_fail="exclude", equal_hit="fast"):
    """-> X, y, report. Rows whose label is None (both modes failed) are dropped and counted."""
    X, y, n_excluded = [], [], 0
    for r in rows:
        lab = label_frame(r["fast_tp"], r["robust_tp"], both_fail, equal_hit)
        if lab is None:
            n_excluded += 1
            continue
        X.append([float(r[k]) for k in feature_names]); y.append(lab)
    report = {"rows": len(rows), "used": len(y), "excluded_both_failed": n_excluded,
              "labels": dict(Counter(y))}
    return X, y, report
