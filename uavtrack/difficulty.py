"""Explicit, condition-aware difficulty estimator and the policy that uses it (Objectives 2 and 3).

difficulty score s in [0,1] = estimated probability that the FAST detector will miss the target on this frame,
computed BEFORE the detector runs, from the policy features (see features.FEATURE_NAMES):
  scene      : brightness, contrast, sharpness (blur), frame_change (global motion)
  temporal   : has_track, frames_since_seen (track stability), last_conf (detector confidence),
               target_size (scale), target_speed (motion), prev_robust
Policy: use ROBUST when s > tau (tau chosen on the TRAINING clip only, see choose_tau), keep ROBUST for `hold` frames.
Reference boxes are used only to create labels; they are never inputs.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .collect import rollout
from .features import FEATURE_NAMES, feature_vector
from .policy import RulePolicy


class DifficultyEstimator:
    """Standardised logistic regression stored as plain arrays (no sklearn needed at inference)."""

    def __init__(self, mean, scale, coef, intercept):
        self.mean, self.scale = np.asarray(mean, float), np.asarray(scale, float)
        self.coef, self.intercept = np.asarray(coef, float), float(intercept)

    def score_matrix(self, X) -> np.ndarray:
        z = (np.asarray(X, float) - self.mean) / self.scale
        return 1.0 / (1.0 + np.exp(-(z @ self.coef + self.intercept)))

    def score(self, feats) -> float:
        vec = feature_vector(feats) if isinstance(feats, dict) else feats
        return float(self.score_matrix([vec])[0])

    def importance(self) -> dict:
        """Standardised coefficients: sign and size show which conditions raise the difficulty score."""
        return {n: round(float(c), 4) for n, c in zip(FEATURE_NAMES, self.coef)}


class DifficultyPolicy:
    name = "difficulty"

    def __init__(self, estimator: DifficultyEstimator, tau: float, hold: int = 3):
        self.est, self.tau, self.hold = estimator, float(tau), int(hold)
        self.last_score = None
        self.reset()

    def reset(self):
        self._left = 0
        self.last_score = None

    def decide(self, feats: dict) -> str:
        self.last_score = self.est.score(feats)
        if self.last_score > self.tau:
            self._left = self.hold
        if self._left > 0:
            self._left -= 1
            return "ROBUST"
        return "FAST"

    @classmethod
    def load(cls, path: Path):
        import joblib
        b = joblib.load(path)
        return cls(b["estimator"], b["tau"], b.get("hold", 3)), b


def _xy(rows):
    X = [[r[k] for k in FEATURE_NAMES] for r in rows]
    y = [int(r["fast_tp"] == 0) for r in rows]          # "hard" = FAST finds no reference target
    return np.array(X, float), np.array(y, int)


def fit_estimator(rows) -> DifficultyEstimator:
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    X, y = _xy(rows)
    if len(set(y)) < 2:
        raise ValueError(f"Only one difficulty class present ({dict(zip(*np.unique(y, return_counts=True)))}); cannot fit an estimator.")
    sc = StandardScaler().fit(X)
    clf = LogisticRegression(class_weight="balanced", max_iter=2000).fit(sc.transform(X), y)
    scale = np.where(sc.scale_ == 0, 1.0, sc.scale_)
    return DifficultyEstimator(sc.mean_, scale, clf.coef_[0], clf.intercept_[0])


def choose_tau(est: DifficultyEstimator, rows, lam: float):
    """tau maximising  sum_{s>tau}(robust_tp - fast_tp) - lam * #{s>tau}.
    lam = how many extra target hits one ROBUST call must buy to be worth its cost (explicit cost weight).
    If no tau has positive gain the policy never switches (tau = 1.01) and that is reported, not hidden."""
    X, _ = _xy(rows)
    s = est.score_matrix(X)
    delta = np.array([(r["robust_tp"] or 0) - (r["fast_tp"] or 0) for r in rows], float)
    best_tau, best_gain = 1.01, 0.0
    for tau in np.unique(np.concatenate([s, [1.01]]))[::-1]:       # high -> low: ties prefer the cheaper (higher) tau
        m = s > tau
        gain = float(delta[m].sum() - lam * m.sum())
        if gain > best_gain + 1e-12:
            best_tau, best_gain = float(tau), gain
    return best_tau, best_gain


def blocked_auc(scores, labels, folds=5):
    from sklearn.metrics import roc_auc_score
    n, vals = len(scores), []
    for k in range(folds):
        lo, hi = k * n // folds, (k + 1) * n // folds
        if hi > lo and len(set(labels[lo:hi])) == 2:
            vals.append(roc_auc_score(labels[lo:hi], scores[lo:hi]))
    return float(np.mean(vals)) if vals else None


def train_difficulty(caches, cfg=None, iterations=2, lam=0.1, hold=3):
    """On-policy training: iteration 0 rolls out the rule policy, later iterations roll out the current
    difficulty policy, and all labelled rows are aggregated (same scheme as collect.dagger_train)."""
    policy, data, history = RulePolicy(), [], []
    est, tau = None, 1.01
    for it in range(iterations):
        rows = [r for c in caches for r in rollout(c, policy, cfg)]
        data += [r for r in rows if r["labelled"]]
        est = fit_estimator(data)
        tau, gain = choose_tau(est, data, lam)
        X, y = _xy(data)
        history.append({"iteration": it, "rollout_policy": policy.name, "rows": len(data), "hard_fraction": round(float(y.mean()), 4),
                        "tau": round(tau, 4), "training_gain_tp": round(gain, 3),
                        "rollout_robust_fraction": round(sum(r["mode"] == "ROBUST" for r in rows) / len(rows), 4)})
        policy = DifficultyPolicy(est, tau, hold)
    X, y = _xy(data)
    report = {"history": history, "lam": lam, "hold": hold, "tau": tau,
              "standardised_coefficients": est.importance(),
              "blocked_cv_auc_on_training_rows": blocked_auc(est.score_matrix(X), y),
              "never_switches": tau > 1.0}
    return est, tau, report
