"""FAST/ROBUST mode policies. decide() sees only features available before frame t."""
from __future__ import annotations

from pathlib import Path

from .features import feature_vector

MODES = {"FAST": 640, "ROBUST": 960}


class FixedPolicy:
    def __init__(self, mode: str):
        assert mode in MODES
        self.mode = mode
        self.name = f"always_{mode.lower()}"

    def reset(self):
        pass

    def decide(self, feats: dict) -> str:
        return self.mode


class RulePolicy:
    """Transparent hand-written baseline: search harder when the target is hard to hold.

    ROBUST if: no confirmed track, track unseen for >= lost_frames, target smaller than
    small_px, or last detection confidence below low_conf. Once ROBUST, stay for >= hold frames.
    """
    name = "rule"

    def __init__(self, lost_frames=1, small_px=24.0, low_conf=0.35, hold=5):
        self.lost_frames, self.small_px, self.low_conf, self.hold = lost_frames, small_px, low_conf, hold
        self.reset()

    def reset(self):
        self._robust_left = 0

    def decide(self, feats: dict) -> str:
        hard = (not feats["has_track"] or feats["frames_since_seen"] >= self.lost_frames
                or feats["target_size"] < self.small_px or feats["last_conf"] < self.low_conf)
        if hard:
            self._robust_left = self.hold
        if self._robust_left > 0:
            self._robust_left -= 1
            return "ROBUST"
        return "FAST"


class LearnedPolicy:
    """Decision tree from train_online_policy.py. Accepts a model, a saved bundle, or a path."""
    name = "learned"

    def __init__(self, source):
        self.meta = {}
        if isinstance(source, (str, Path)):
            import joblib
            source = joblib.load(source)
        if isinstance(source, dict):
            self.meta = {k: v for k, v in source.items() if k != "model"}
            source = source["model"]
        self.model = source
        self.reset()

    def reset(self):
        pass

    def decide(self, feats: dict) -> str:
        return str(self.model.predict([feature_vector(feats)])[0])


def make_policy(name: str, model_path: Path | None = None):
    if name in ("fast", "robust"):
        return FixedPolicy(name.upper())
    if name == "rule":
        return RulePolicy()
    if name == "learned":
        if model_path is None or not Path(model_path).is_file():
            raise FileNotFoundError(f"Learned policy file not found: {model_path}. Run train_online_policy.py first.")
        return LearnedPolicy(Path(model_path))
    raise ValueError(f"Unknown policy: {name}")
