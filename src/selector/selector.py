"""Multi-model perception mode selectors and hysteresis smoothing.

Supports rule-based baseline plus 8 learned classifiers:
  * ``rules``  - Hand-set threshold baseline
  * ``tree``   - DecisionTreeClassifier
  * ``forest`` - RandomForestClassifier
  * ``gbm``    - GradientBoostingClassifier
  * ``logreg`` - Calibrated LogisticRegression (with StandardScaler)
  * ``svm``    - RBF-kernel SVC with probability calibration
  * ``mlp``    - Multi-Layer Perceptron neural network
  * ``xgb``    - XGBoost classifier (when xgboost is installed)
  * ``lgbm``   - LightGBM classifier (when lightgbm is installed)
"""

from collections import deque
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import warnings
import joblib
import numpy as np

from src.features.features import FEATURE_NAMES

BASE_FEATURES: List[str] = list(FEATURE_NAMES)
ROLLING_WINDOW = 15

SELECTOR_LABELS: Dict[str, str] = {
    "rules": "Rules (Baseline)",
    "tree": "Decision Tree",
    "forest": "Random Forest",
    "gbm": "Gradient Boosting",
    "logreg": "Logistic Regression",
    "svm": "SVM (RBF)",
    "mlp": "MLP Neural Net",
    "xgb": "XGBoost",
    "lgbm": "LightGBM",
}

LEARNED_SELECTOR_KINDS: tuple[str, ...] = (
    "tree",
    "forest",
    "gbm",
    "logreg",
    "svm",
    "mlp",
    "xgb",
    "lgbm",
)

_ALIASES: Dict[str, str] = {
    "rule_based": "rules",
    "rule": "rules",
    "ml": "forest",
    "rf": "forest",
    "random_forest": "forest",
    "decision_tree": "tree",
    "gradient_boosting": "gbm",
    "gb": "gbm",
    "logistic": "logreg",
    "logistic_regression": "logreg",
    "lr": "logreg",
    "svc": "svm",
    "nn": "mlp",
    "neural_net": "mlp",
    "xgboost": "xgb",
    "lightgbm": "lgbm",
}


class RollingFeatures:
    """Turns per-frame base features into a causal vector [current + rolling mean]."""

    def __init__(self, window: int = ROLLING_WINDOW, base_features: Optional[List[str]] = None) -> None:
        self.window = window
        self.base_features = list(base_features or BASE_FEATURES)
        self.history: deque = deque(maxlen=window)

    @staticmethod
    def names(base_features: Optional[List[str]] = None) -> List[str]:
        base = list(base_features or BASE_FEATURES)
        return base + [f"{n}_avg" for n in base]

    def update(self, features: Dict[str, float]) -> List[float]:
        row = [float(features.get(n, 0.0)) for n in self.base_features]
        self.history.append(row)
        return row + np.mean(self.history, axis=0).tolist()

    def reset(self) -> None:
        self.history.clear()


class RuleBasedSelector:
    """Heuristic mode selector deciding between FAST and ROBUST perception modes."""

    def __init__(self, thresholds: Optional[Dict[str, float]] = None) -> None:
        defaults = {
            "blur_threshold": 100.0,
            "area_ratio_threshold": 0.003,
            "conf_threshold": 0.45,
            "camera_motion_threshold": 3.5,
            "contrast_threshold": 22.0,
        }
        self.thresholds = {**defaults, **(thresholds or {})}

    def predict(self, features: Dict[str, float]) -> str:
        blur = features.get("blur", 200.0)
        box_area = features.get("box_area_ratio", 0.01)
        conf = features.get("mean_confidence", 0.8)
        motion = features.get("camera_motion", 0.0)
        contrast = features.get("contrast", 50.0)

        is_blurry = blur < self.thresholds["blur_threshold"]
        is_tiny_target = 0.0 < box_area < self.thresholds["area_ratio_threshold"]
        is_low_conf = 0.0 < conf < self.thresholds["conf_threshold"]
        is_fast_motion = motion > self.thresholds["camera_motion_threshold"]
        is_low_contrast = contrast < self.thresholds["contrast_threshold"]

        if is_blurry or is_tiny_target or is_low_conf or is_fast_motion or is_low_contrast:
            return "ROBUST"
        return "FAST"

    def reset(self) -> None:
        """Rules keep no state between frames."""


class HysteresisSmoother:
    """Smoother preventing frequent rapid oscillations between perception modes."""

    def __init__(
        self,
        min_mode_duration: int = 15,
        persistence_frames: int = 5,
        initial_mode: str = "FAST",
    ) -> None:
        self.min_mode_duration = min_mode_duration
        self.persistence_frames = persistence_frames
        self.current_mode = initial_mode
        self.frames_in_current_mode = 0
        self.candidate_mode: Optional[str] = None
        self.candidate_count = 0

    def update(self, proposed_mode: str) -> str:
        self.frames_in_current_mode += 1

        if proposed_mode != self.current_mode:
            if proposed_mode == self.candidate_mode:
                self.candidate_count += 1
            else:
                self.candidate_mode = proposed_mode
                self.candidate_count = 1

            if (
                self.frames_in_current_mode >= self.min_mode_duration
                and self.candidate_count >= self.persistence_frames
            ):
                self.current_mode = proposed_mode
                self.frames_in_current_mode = 0
                self.candidate_mode = None
                self.candidate_count = 0
        else:
            self.candidate_mode = None
            self.candidate_count = 0

        return self.current_mode

    def reset(self, initial_mode: str = "FAST") -> None:
        self.current_mode = initial_mode
        self.frames_in_current_mode = 0
        self.candidate_mode = None
        self.candidate_count = 0


class MLSelector:
    """Learned mode selector loaded from a joblib bundle.

    Supports both tree ensembles (fast raw-tree path for tree/forest) and general
    probabilistic classifiers (GBM, LogisticRegression, SVM, MLP, XGBoost, LightGBM).
    """

    def __init__(
        self,
        model_path: Union[str, Path],
        decide_every: Optional[int] = None,
        threshold: Optional[float] = None,
    ) -> None:
        self.model_path = Path(model_path)
        if not self.model_path.is_file():
            raise FileNotFoundError(
                f"No trained selector at '{self.model_path}'. "
                "Train selectors with: python -m src.training.train_selector"
            )
        bundle: Dict[str, Any] = joblib.load(self.model_path)
        self.bundle = bundle
        self.kind: str = bundle.get("kind", "forest")
        self.model = bundle["model"]
        if hasattr(self.model, "n_jobs"):
            self.model.n_jobs = 1

        self.feature_names: List[str] = bundle["features"]
        n_base = len(self.feature_names) // 2
        base_features = self.feature_names[:n_base] if n_base > 0 else BASE_FEATURES
        self.threshold: float = float(threshold if threshold is not None else bundle.get("threshold", 0.5))

        # Fast raw-tree path only for DecisionTree / RandomForest
        self._trees = None
        if self.kind in ("tree", "forest") and hasattr(self.model, "tree_"):
            self._trees = [self.model.tree_]
        elif self.kind == "forest" and hasattr(self.model, "estimators_"):
            ests = self.model.estimators_
            if ests and hasattr(ests[0], "tree_"):
                self._trees = [est.tree_ for est in ests]

        self.decide_every = int(decide_every or bundle.get("decide_every", 3))
        self.rolling = RollingFeatures(
            int(bundle.get("window", ROLLING_WINDOW)),
            base_features=base_features,
        )

        classes = list(getattr(self.model, "classes_", ["FAST", "ROBUST"]))
        if "ROBUST" in classes:
            self._robust_col = classes.index("ROBUST")
        elif 1 in classes:
            self._robust_col = classes.index(1)
        else:
            self._robust_col = len(classes) - 1

        self._frame = 0
        self._last_mode = "FAST"
        self.last_robust_prob = 0.0

    def predict(self, features: Dict[str, float]) -> str:
        """Return 'FAST' or 'ROBUST' for the current frame."""
        vec = self.rolling.update(features)
        if self._frame % self.decide_every == 0:
            self.last_robust_prob = float(self.robust_probability(vec))
            self._last_mode = "ROBUST" if self.last_robust_prob >= self.threshold else "FAST"
        self._frame += 1
        return self._last_mode

    def robust_probability(self, vec: List[float]) -> float:
        """Compute P(ROBUST needed) for a single feature vector."""
        x = np.asarray([vec], dtype=np.float32)
        if self._trees is not None:
            total = 0.0
            for tree in self._trees:
                counts = np.ravel(tree.predict(x)[0])
                total += counts[self._robust_col] / max(counts.sum(), 1e-12)
            return float(total / len(self._trees))

        if hasattr(self.model, "predict_proba"):
            probs = self.model.predict_proba(x)[0]
            return float(probs[self._robust_col])

        pred = self.model.predict(x)[0]
        return 1.0 if pred in ("ROBUST", 1) else 0.0

    def reset(self) -> None:
        self.rolling.reset()
        self._frame = 0
        self._last_mode = "FAST"
        self.last_robust_prob = 0.0


def normalize_selector_kind(kind: Optional[str]) -> str:
    kind = (kind or "rules").strip().lower()
    kind = _ALIASES.get(kind, kind)
    if kind not in SELECTOR_LABELS:
        raise ValueError(f"Unknown selector '{kind}'. Use one of: {', '.join(SELECTOR_LABELS)}")
    return kind


def selector_model_path(kind: str, config: Dict[str, Any]) -> Path:
    models = (config.get("selector") or {}).get("models") or {}
    return Path(models.get(kind, f"weights/selector_{kind}.joblib"))


def available_selectors(config: Dict[str, Any]) -> List[str]:
    """Selectors that can run right now: rules, plus every learned model that has been trained."""
    return ["rules"] + [k for k in LEARNED_SELECTOR_KINDS if selector_model_path(k, config).is_file()]


def make_selector(kind: Optional[str], config: Dict[str, Any], threshold: Optional[float] = None):
    """Build the requested selector. Returns (selector, kind actually used)."""
    kind = normalize_selector_kind(kind)
    sel_cfg = config.get("selector") or {}
    if kind == "rules":
        return RuleBasedSelector(thresholds=sel_cfg.get("rule_thresholds", {})), "rules"
    path = selector_model_path(kind, config)
    if not path.is_file():
        warnings.warn(
            f"{SELECTOR_LABELS[kind]} selector not trained yet ({path} missing); "
            "using rules. Train with: python -m src.training.train_selector"
        )
        return RuleBasedSelector(thresholds=sel_cfg.get("rule_thresholds", {})), "rules"
    if threshold is None:
        threshold = sel_cfg.get("threshold")
    return MLSelector(path, decide_every=sel_cfg.get("decide_every"), threshold=threshold), kind
