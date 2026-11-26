"""Verification protocol: online-replay equivalence, block-bootstrap CIs, and formal verdict."""

from typing import Any, Dict, Optional, Tuple
import numpy as np
import pandas as pd

from src.selector.selector import BASE_FEATURES, HysteresisSmoother, make_selector


def verify_online_replay_equivalence(
    records_df: pd.DataFrame,
    selector_kind: str,
    config: Dict[str, Any],
    threshold: Optional[float] = None,
) -> Dict[str, Any]:
    """Verify that replaying causal frame features reproduces the exact live mode sequence.

    Returns:
        Dictionary with ``equivalent`` (bool), ``total_frames``, ``mismatches``,
        and ``agreement_pct``.
    """
    if records_df.empty or "mode" not in records_df.columns:
        return {"equivalent": True, "total_frames": 0, "mismatches": 0, "agreement_pct": 100.0}

    selector, actual_kind = make_selector(selector_kind, config, threshold=threshold)
    hyst_cfg = config.get("hysteresis", {})
    smoother = HysteresisSmoother(
        min_mode_duration=int(hyst_cfg.get("min_mode_duration", 15)),
        persistence_frames=int(hyst_cfg.get("persistence_frames", 5)),
        initial_mode="FAST",
    )
    selector.reset()
    smoother.reset(initial_mode="FAST")

    replayed = []
    for row in records_df.to_dict("records"):
        feats = {k: float(row.get(k, 0.0)) for k in BASE_FEATURES}
        raw_mode = selector.predict(feats)
        replayed.append(smoother.update(raw_mode))

    recorded = records_df["mode"].tolist()
    mismatches = sum(1 for a, b in zip(recorded, replayed) if a != b)
    total = len(recorded)
    return {
        "equivalent": mismatches == 0,
        "selector_used": actual_kind,
        "total_frames": total,
        "mismatches": mismatches,
        "agreement_pct": round((total - mismatches) / max(1, total) * 100.0, 2),
    }


def block_bootstrap_ci(
    values_a: np.ndarray,
    values_b: np.ndarray,
    block_size: int = 15,
    n_boot: int = 500,
    alpha: float = 0.05,
    seed: int = 42,
) -> Tuple[float, float, float]:
    """Paired block-bootstrap CI for mean(values_a - values_b) preserving temporal correlation."""
    diff = np.asarray(values_a, dtype=float) - np.asarray(values_b, dtype=float)
    n = len(diff)
    if n == 0:
        return 0.0, 0.0, 0.0

    rng = np.random.default_rng(seed)
    bsize = max(1, min(block_size, n))
    n_blocks = int(np.ceil(n / bsize))
    boot_means = np.empty(n_boot, dtype=float)

    for i in range(n_boot):
        starts = rng.integers(0, max(1, n - bsize + 1), size=n_blocks)
        sample = np.concatenate([diff[s : s + bsize] for s in starts])[:n]
        boot_means[i] = float(np.mean(sample))

    lo = float(np.percentile(boot_means, 100.0 * (alpha / 2.0)))
    hi = float(np.percentile(boot_means, 100.0 * (1.0 - alpha / 2.0)))
    return float(np.mean(diff)), lo, hi


def evaluate_verdict(
    adaptive_metrics: Dict[str, Any],
    fast_metrics: Dict[str, Any],
    robust_metrics: Dict[str, Any],
) -> Dict[str, Any]:
    """Produce a formal research verdict comparing ADAPTIVE vs FAST and ROBUST."""
    ad_det = float(adaptive_metrics.get("detector_pct", 100.0))
    rb_det = float(robust_metrics.get("detector_pct", 100.0))
    compute_saved_pct = round(rb_det - ad_det, 1)

    has_gt = "IDF1" in adaptive_metrics and "IDF1" in robust_metrics
    if has_gt:
        quality_key = "IDF1"
        ad_q = float(adaptive_metrics["IDF1"])
        fa_q = float(fast_metrics["IDF1"])
        rb_q = float(robust_metrics["IDF1"])
    else:
        quality_key = "agreement_pct"
        ad_q = float(adaptive_metrics.get("agreement_pct", 0.0))
        fa_q = float(fast_metrics.get("agreement_pct", 0.0))
        rb_q = 100.0

    beats_fast = ad_q > fa_q
    near_robust = ad_q >= (rb_q - 5.0)
    saves_compute = compute_saved_pct >= 5.0

    if beats_fast and near_robust and saves_compute:
        verdict = "PASS"
        reason = (
            f"ADAPTIVE improves {quality_key} over FAST ({ad_q:.1f} vs {fa_q:.1f}), "
            f"stays within 5 pts of ROBUST ({rb_q:.1f}), and saves {compute_saved_pct:.1f}% detector calls."
        )
    elif near_robust and saves_compute:
        verdict = "MARGINAL"
        reason = (
            f"ADAPTIVE saves {compute_saved_pct:.1f}% detector calls and matches ROBUST ({ad_q:.1f} vs {rb_q:.1f}), "
            f"but FAST already achieved {fa_q:.1f} on this clip."
        )
    else:
        verdict = "NEGATIVE"
        reason = (
            f"ADAPTIVE achieved {quality_key}={ad_q:.1f} (FAST={fa_q:.1f}, ROBUST={rb_q:.1f}) "
            f"with {compute_saved_pct:.1f}% detector savings."
        )

    return {
        "verdict": verdict,
        "quality_metric": quality_key,
        "adaptive_quality": ad_q,
        "fast_quality": fa_q,
        "robust_quality": rb_q,
        "detector_savings_pct": compute_saved_pct,
        "reason": reason,
    }
