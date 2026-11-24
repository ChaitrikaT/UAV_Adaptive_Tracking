"""Tests for training/deployment consistency, leakage and label handling. All inputs are SYNTHETIC."""
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from uavtrack.collect import (blocked_cv, build_xy, cache_clip, count_hits, dagger_train, label_frame,
                              rollout)
from uavtrack.features import FEATURE_NAMES
from uavtrack.pipeline import run_video
from uavtrack.policy import LearnedPolicy, RulePolicy
from uavtrack.tracker import TrackerConfig
try:
    from _synth import make_clip
except ImportError:
    from tests._synth import make_clip


class DeterministicDetector:
    """Output depends only on (frame content, imgsz): FAST misses every 3rd frame, ROBUST never."""
    def detect(self, frame, imgsz):
        ys, xs = np.where(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) > 200)
        if len(xs) == 0:
            return np.zeros((0, 5))
        frame_idx = round((xs.min() - 20) / 4)
        if imgsz == 640 and frame_idx % 3 == 0:
            return np.zeros((0, 5))
        return np.array([[xs.min(), ys.min(), xs.max() + 1, ys.max() + 1, 0.8]])


RULE = dict(hold=2, small_px=5.0)


class LabelTests(unittest.TestCase):
    def test_label_table(self):
        self.assertEqual(label_frame(0, 1), "ROBUST")
        self.assertEqual(label_frame(1, 0), "FAST")
        self.assertEqual(label_frame(1, 1), "FAST")                       # both hit: cheaper mode
        self.assertEqual(label_frame(1, 1, equal_hit="robust"), "ROBUST")
        self.assertIsNone(label_frame(0, 0))                              # both fail: no evidence
        self.assertEqual(label_frame(0, 0, both_fail="fast"), "FAST")
        self.assertEqual(label_frame(0, 0, both_fail="robust"), "ROBUST")

    def test_both_fail_frames_are_excluded_not_defaulted_to_fast(self):
        rows = [{"labelled": True, "fast_tp": 0, "robust_tp": 0, **{k: 0.0 for k in FEATURE_NAMES}}] * 5
        rows += [{"labelled": True, "fast_tp": 0, "robust_tp": 1, **{k: 1.0 for k in FEATURE_NAMES}}] * 2
        rows += [{"labelled": False, "fast_tp": None, "robust_tp": None, **{k: 2.0 for k in FEATURE_NAMES}}]
        X, y, dropped = build_xy(rows)
        self.assertEqual((len(y), dropped, set(y)), (2, 5, {"ROBUST"}))
        X, y, dropped = build_xy(rows, both_fail="fast")
        self.assertEqual((len(y), dropped), (7, 0))

    def test_count_hits_is_one_to_one(self):
        ref = [(1, 0, 0, 20, 20)]
        self.assertEqual(count_hits(ref, np.array([[0, 0, 20, 20, .9], [0, 0, 20, 20, .8]])), 1)


class ConsistencyTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.video, self.ref = make_clip(Path(self.tmp.name))
        self.det = DeterministicDetector()
        self.cache = cache_clip(self.video, self.ref, self.det, log_every=0)

    def tearDown(self):
        self.tmp.cleanup()

    def test_training_rollout_matches_online_pipeline(self):
        """Same features, same chosen modes, same tracker output => no train/deploy mismatch."""
        cfg = TrackerConfig()
        rows = rollout(self.cache, RulePolicy(**RULE), cfg)
        online, tracks = run_video(self.video, self.det, RulePolicy(**RULE), tracker_cfg=cfg, log_every=0)
        self.assertEqual([r["mode"] for r in rows], [r["mode"] for r in online])
        for a, b in zip(rows, online):
            for k in FEATURE_NAMES:
                self.assertAlmostEqual(a[k], b[k], places=3, msg=f"frame {a['frame']} feature {k}")
        self.assertEqual(sum(r["n_tracks"] for r in online), len(tracks))

    def test_prev_robust_is_not_constant_in_training_rows(self):
        """Regression: first version trained with prev_robust always 1."""
        rows = rollout(self.cache, RulePolicy(**RULE))
        self.assertEqual({r["prev_robust"] for r in rows}, {0.0, 1.0})

    def test_features_do_not_depend_on_reference_annotations(self):
        shifted = {"frames": [dict(r, ref=[(1, 0, 0, 5, 5)] if r["ref"] else None) for r in self.cache["frames"]]}
        a = rollout(self.cache, RulePolicy(**RULE))
        b = rollout(shifted, RulePolicy(**RULE))
        self.assertEqual([r["mode"] for r in a], [r["mode"] for r in b])
        for x, y in zip(a, b):
            for k in FEATURE_NAMES:
                self.assertEqual(x[k], y[k])
        self.assertNotEqual([r["robust_tp"] for r in a], [r["robust_tp"] for r in b])  # only labels change

    def test_dagger_training_and_bundle_roundtrip(self):
        clf, report = dagger_train(self.cache, TrackerConfig(), iterations=2)
        self.assertEqual(len(report["history"]), 2)
        self.assertEqual(report["tie_rules"]["both_fail"], "exclude")
        import joblib
        path = Path(self.tmp.name) / "p.joblib"
        joblib.dump({"model": clf, "detector_conf": 0.1}, path)
        pol = LearnedPolicy(path)
        self.assertEqual(pol.meta["detector_conf"], 0.1)
        run, _ = run_video(self.video, self.det, pol, log_every=0)
        self.assertEqual(len(run), 60)

    def test_blocked_cv_uses_contiguous_folds(self):
        X = [[float(i)] for i in range(60)]
        y = ["FAST"] * 30 + ["ROBUST"] * 30
        res = blocked_cv(X, y, folds=5)
        self.assertIn("mean_balanced_accuracy", res)


if __name__ == "__main__":
    unittest.main()
