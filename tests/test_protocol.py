"""Tests of the evaluation protocol and CLI wiring. ALL data is SYNTHETIC (white squares, a fake detector)."""
import argparse
import json
import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np

import run_evaluation as RE
from uavtrack.collect import cache_clip
from uavtrack.protocol import (ap50, compare_learned, detection_report, condition_report, condition_rows,
                               paired_block_bootstrap, replay_eval, verdict)
from uavtrack.policy import FixedPolicy
try:
    from _synth import make_clip
    from test_policy_training import DeterministicDetector
except ImportError:
    from tests._synth import make_clip
    from tests.test_policy_training import DeterministicDetector


class FakeYolo(DeterministicDetector):
    device = "fake"

    def warmup(self, frame):
        pass


def fake_cache(frames):
    """frames: list of (ref_boxes or None, dets_fast, dets_robust)."""
    out = []
    for i, (ref, df, dr) in enumerate(frames, 1):
        out.append({"frame": i, "g": dict(brightness=100.0, contrast=10.0, sharpness=50.0, frame_change=1.0),
                    "ref": ref, "dets_fast": np.array(df, float).reshape(-1, 5), "dets_robust": np.array(dr, float).reshape(-1, 5),
                    "fast_ms": 10.0, "robust_ms": 12.0, "decode_ms": 5.0, "feature_ms": 1.0})
    return {"frames": out, "width": 100, "height": 100}


R = [(1, 10, 10, 30, 30)]
HIT, MISS = [10, 10, 30, 30, 0.9], [60, 60, 80, 80, 0.8]


class DetectionTests(unittest.TestCase):
    def test_precision_recall_and_ap(self):
        cache = fake_cache([(R, [HIT], [HIT, MISS]), (R, [MISS], [HIT]), (None, [HIT], [HIT]), (R, [], [HIT])])
        rep = detection_report(cache)
        self.assertEqual(rep["annotated_frames"], 3)
        f, r = rep["modes"]["FAST"]["conf>=0.10"], rep["modes"]["ROBUST"]["conf>=0.10"]
        self.assertEqual((f["TP"], f["FP"], f["FN"]), (1, 1, 2))            # unannotated frame 3 is not scored
        self.assertEqual((r["TP"], r["FP"], r["FN"]), (3, 1, 0))
        self.assertAlmostEqual(ap50(cache, "ROBUST"), 1.0, places=3)         # all 3 GT found, one FP ranked lower
        self.assertLess(ap50(cache, "FAST"), 0.5)

    def test_confidence_threshold_changes_counts(self):
        cache = fake_cache([(R, [[10, 10, 30, 30, 0.2]], [[10, 10, 30, 30, 0.2]])])
        m = detection_report(cache)["modes"]["FAST"]
        self.assertEqual(m["conf>=0.10"]["TP"], 1)
        self.assertEqual(m["conf>=0.25"]["TP"], 0)


class StatsTests(unittest.TestCase):
    def per_clip(self, tp_a, tp_b, cost_a, cost_b, n=120):
        return [{"tp_a": np.full(n, tp_a), "tp_b": np.full(n, tp_b), "ref": np.ones(n),
                 "cost_a": np.full(n, cost_a), "cost_b": np.full(n, cost_b)}]

    def test_bootstrap_identical_policies_gives_zero_interval(self):
        r = paired_block_bootstrap(self.per_clip(0.5, 0.5, 10.0, 10.0))
        self.assertEqual(r["d_recall"][1:], [0.0, 0.0])
        self.assertEqual(r["d_cost_ms"][1:], [0.0, 0.0])

    def test_bootstrap_detects_a_constant_difference(self):
        r = paired_block_bootstrap(self.per_clip(0.8, 0.5, 11.0, 10.0))
        self.assertAlmostEqual(r["d_recall"][0], 0.3, places=4)
        self.assertGreater(r["d_recall"][1], 0.29)
        self.assertAlmostEqual(r["d_cost_ms"][0], 1.0, places=4)

    def test_verdict_rules(self):
        ci = lambda p, lo, hi: [p, lo, hi]
        better_fast = {"d_recall": ci(.1, .05, .15), "d_cost_ms": ci(0, 0, 0)}
        close_cheaper = {"d_recall": ci(-.005, -.01, .0), "d_cost_ms": ci(-1.0, -1.5, -.5)}
        self.assertEqual(verdict(better_fast, close_cheaper), "SUPPORTED")
        no_gain = {"d_recall": ci(-.01, -.03, .0), "d_cost_ms": ci(0, 0, 0)}   # CI upper <= 0: not better than FAST
        undecided = {"d_recall": ci(.0, -.02, .02), "d_cost_ms": ci(0, 0, 0)}
        self.assertEqual(verdict(undecided, close_cheaper), "INCONCLUSIVE")
        self.assertEqual(verdict(no_gain, close_cheaper), "NOT SUPPORTED")
        worse_not_cheaper = {"d_recall": ci(-.2, -.3, -.1), "d_cost_ms": ci(.5, .2, .8)}
        self.assertEqual(verdict(better_fast, worse_not_cheaper), "NOT SUPPORTED")
        vague = {"d_recall": ci(.0, -.1, .1), "d_cost_ms": ci(-.1, -1, .8)}
        self.assertEqual(verdict(better_fast, vague), "INCONCLUSIVE")


class WiredBase(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        vids = self.tmp / "data" / "videos"; anns = self.tmp / "data" / "annotations"
        vids.mkdir(parents=True); anns.mkdir(parents=True)
        src_v, src_r = make_clip(self.tmp)
        for c in (7, 8):
            shutil.copy(src_v, vids / f"Clip_{c}.avi"); shutil.copy(src_r, anns / f"Clip_{c}_refined.txt")
        self.weights = self.tmp / "best.pt"; self.weights.write_bytes(b"fake")
        self.old_root, self.old_make = RE.ROOT, RE.make_detector
        RE.ROOT = self.tmp
        RE.make_detector = lambda weights, conf, device: FakeYolo()

    def tearDown(self):
        RE.ROOT, RE.make_detector = self.old_root, self.old_make
        shutil.rmtree(self.tmp, ignore_errors=True)

    def run_cli(self, *argv):
        RE.main(list(argv))


class CliWiringTests(WiredBase):
    def test_full_pipeline_cache_analyze_policy(self):
        self.run_cli("cache", "--clips", "7", "8", "--weights", str(self.weights))
        self.assertTrue((self.tmp / "results/policy/cache_clip_7_default.pkl").is_file())
        self.run_cli("analyze", "--clips", "7", "8")
        a = json.loads((self.tmp / "results/evaluation/analysis_default.json").read_text())
        modes = a["clips"]["8"]["detection"]["modes"]
        self.assertGreater(modes["ROBUST"]["conf>=0.10"]["recall"], modes["FAST"]["conf>=0.10"]["recall"])
        self.assertIn("scene", a["difficulty_cross_clip_auc"]["7->8"])
        self.run_cli("policy", "--train", "7", "--eval", "8", "--n-boot", "200")
        res = json.loads((self.tmp / "results/evaluation/policy_train7_eval8_default.json").read_text())
        per = res["per_clip"]["8"]
        self.assertEqual(set(per), {"fast", "robust", "rule", "learned"})
        self.assertEqual(per["fast"]["robust_fraction"], 0.0)
        self.assertEqual(per["robust"]["robust_fraction"], 1.0)
        self.assertGreater(per["robust"]["tracking"]["recall"], per["fast"]["tracking"]["recall"])
        self.assertIn(res["comparison"]["verdict_learned"], {"SUPPORTED", "NOT SUPPORTED", "INCONCLUSIVE"})
        self.assertTrue((self.tmp / "results/policy/online_policy.joblib").is_file())

    def test_split_guards(self):
        self.run_cli("cache", "--clips", "7", "8", "--weights", str(self.weights))
        with self.assertRaises(SystemExit):
            self.run_cli("cache", "--clips", "9", "--weights", str(self.weights))          # needs --allow-test
        with self.assertRaises(SystemExit):
            self.run_cli("policy", "--train", "7", "--eval", "9")                           # needs --final
        with self.assertRaises(SystemExit):
            self.run_cli("policy", "--train", "7", "--eval", "7")                           # not held out
        with self.assertRaises(SystemExit):
            self.run_cli("policy", "--train", "9", "--eval", "8")                           # never train on test

    def test_replay_matches_online_in_the_wiring(self):
        self.run_cli("cache", "--clips", "7", "--weights", str(self.weights))
        self.run_cli("verify-online", "--clip", "7", "--end", "40", "--policy", "rule", "--weights", str(self.weights))
        out = json.loads((self.tmp / "results/evaluation/verify_online_clip7_rule_default.json").read_text())
        self.assertTrue(out["replay_equals_online"], out)

    def test_condition_report_on_cached_synthetic_clip(self):
        vids = self.tmp / "data/videos/Clip_7.avi"
        cache = cache_clip(vids, self.tmp / "data/annotations/Clip_7_refined.txt", FakeYolo(), log_every=0)
        rows = condition_rows(cache)
        self.assertEqual(len(rows), 60)
        rep = condition_report(rows)
        self.assertIn("target_size_px", rep["descriptors"])
        r = replay_eval(cache, FixedPolicy("ROBUST"))
        self.assertGreater(r["tracking"]["recall"], 0.9)
        self.assertGreater(r["composed_fps"], 0)


if __name__ == "__main__":
    unittest.main()


class TuneTests(WiredBase):
    def test_tune_runs_and_refuses_test_clips(self):
        self.run_cli("cache", "--clips", "7", "--weights", str(self.weights))
        self.run_cli("tune", "--clip", "7", "--high-conf-grid", "0.3", "--max-age-grid", "10", "30")
        self.assertTrue((self.tmp / "results/evaluation/tune_clip7_default.csv").is_file())
        with self.assertRaises(SystemExit):
            self.run_cli("tune", "--clip", "10")
