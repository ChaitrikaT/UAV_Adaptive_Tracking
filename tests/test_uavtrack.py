import csv
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from uavtrack.evaluate import evaluate
from uavtrack.pipeline import run_video, summarize_timing
from uavtrack.policy import RulePolicy, FixedPolicy
from uavtrack.tracker import Tracker, TrackerConfig

try:
    from _synth import make_clip
except ImportError:  # run with a different top-level directory
    from tests._synth import make_clip


def box(cx, cy, s=12):
    return [cx - s / 2, cy - s / 2, cx + s / 2, cy + s / 2]


class TrackerTests(unittest.TestCase):
    def test_id_survives_short_gap_and_resets_after_long_gap(self):
        t = Tracker(TrackerConfig(max_age=8, min_hits=2))
        ids = []
        for f in range(40):
            if f in (10, 11, 12, 13, 14):          # 5-frame gap < max_age
                dets = []
            elif 25 <= f < 36:                      # 11-frame gap > max_age
                dets = []
            else:
                dets = [box(100 + 6 * f, 200, 14) + [0.8]]
            out = t.update(dets)
            ids.extend(o["track_id"] for o in out)
        self.assertEqual(ids[0], 1)
        self.assertEqual(set(ids[:19]), {1})        # same ID before and after the short gap
        self.assertEqual(ids[-1], 2)                # new ID after a gap longer than max_age

    def test_two_targets_keep_separate_ids(self):
        t = Tracker()
        seen = {}
        for f in range(30):
            dets = [box(50 + 5 * f, 100) + [0.9], box(600 - 5 * f, 300) + [0.9]]
            for o in t.update(dets):
                cy = (o["y1"] + o["y2"]) / 2
                seen.setdefault("top" if cy < 200 else "bottom", set()).add(o["track_id"])
        self.assertEqual(len(seen["top"]), 1)
        self.assertEqual(len(seen["bottom"]), 1)
        self.assertNotEqual(seen["top"], seen["bottom"])

    def test_single_frame_false_positive_gets_no_id(self):
        t = Tracker()
        self.assertEqual(t.update([box(10, 10) + [0.9]]), [])
        self.assertEqual(t.update([]), [])
        self.assertEqual(t.primary_state()["has_track"], 0)

    def test_low_confidence_detection_extends_existing_track_only(self):
        t = Tracker()
        for f in range(4):
            t.update([box(100 + 5 * f, 100) + [0.9]])
        out = t.update([box(120, 100) + [0.15]])
        self.assertEqual(len(out), 1)
        t2 = Tracker()
        self.assertEqual(t2.update([box(100, 100) + [0.15]]), [])  # cannot start a track


class StateModeTests(unittest.TestCase):
    def _two_targets_one_vanishes(self, mode):
        t = Tracker(TrackerConfig(state_mode=mode))
        for f in range(12):
            t.update([box(100 + 5 * f, 100) + [0.9], box(500 - 5 * f, 250, 8) + [0.6]])
        for f in range(12, 15):                     # second target no longer detected
            t.update([box(100 + 5 * f, 100) + [0.9]])
        return t.primary_state()

    def test_worst_state_sees_the_struggling_track_primary_does_not(self):
        self.assertEqual(self._two_targets_one_vanishes("primary")["frames_since_seen"], 0)
        worst = self._two_targets_one_vanishes("worst")
        self.assertEqual(worst["frames_since_seen"], 3)
        self.assertLess(worst["target_size"], 10)


class EvaluateTests(unittest.TestCase):
    def _ref(self, n=10):
        return {f: [(1, 10 + f, 10, 30 + f, 30)] for f in range(1, n + 1)}

    def test_perfect_tracking(self):
        ref = self._ref()
        pred = {f: [(7, *g[1:])] for f, gs in ref.items() for g in gs}
        m = evaluate(ref, pred, list(ref))
        self.assertEqual((m["TP"], m["FP"], m["FN"], m["ID_switches"]), (10, 0, 0, 0))
        self.assertEqual(m["IDF1"], 1.0)
        self.assertEqual(m["MOTA"], 1.0)

    def test_id_switch_is_counted_and_idf1_drops(self):
        ref = self._ref()
        pred = {f: [(1 if f <= 5 else 2, *g[1:])] for f, gs in ref.items() for g in gs}
        m = evaluate(ref, pred, list(ref))
        self.assertEqual(m["ID_switches"], 1)
        self.assertAlmostEqual(m["IDF1"], 0.5)
        self.assertEqual(m["distinct_pred_ids"], 2)

    def test_misses_reduce_recall_and_unannotated_frames_are_skipped(self):
        ref = self._ref()
        pred = {f: [(1, *g[1:])] for f, gs in ref.items() for g in gs if f <= 4}
        pred[99] = [(1, 0, 0, 5, 5)]                      # prediction in an unannotated frame
        m = evaluate(ref, pred, list(range(1, 11)) + [99])
        self.assertEqual((m["TP"], m["FN"], m["FP"]), (4, 6, 0))
        self.assertAlmostEqual(m["recall"], 0.4)


class PolicyTests(unittest.TestCase):
    def feats(self, **kw):
        base = dict(has_track=1, frames_since_seen=0, target_size=40.0, last_conf=0.8)
        base.update(kw)
        return base

    def test_rule_policy_goes_robust_when_lost_and_holds(self):
        p = RulePolicy(hold=3)
        self.assertEqual(p.decide(self.feats()), "FAST")
        self.assertEqual(p.decide(self.feats(frames_since_seen=2)), "ROBUST")
        self.assertEqual([p.decide(self.feats()) for _ in range(3)], ["ROBUST", "ROBUST", "FAST"])
        self.assertEqual(p.decide(self.feats(has_track=0)), "ROBUST")


class FakeDetector:
    """Finds the bright dot. FAST (640) misses every 3rd frame; ROBUST (960) never does."""
    def __init__(self):
        self.calls = []

    def detect(self, frame, imgsz):
        self.calls.append(imgsz)
        g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        ys, xs = np.where(g > 200)
        if len(xs) == 0:
            return np.zeros((0, 5))
        if imgsz == 640 and (len(self.calls) % 3 == 0):
            return np.zeros((0, 5))
        return np.array([[xs.min(), ys.min(), xs.max() + 1, ys.max() + 1, 0.8]])


class PipelineTests(unittest.TestCase):
    def test_end_to_end_on_synthetic_video(self):
        with tempfile.TemporaryDirectory() as d:
            video, ref_path = make_clip(Path(d))
            det = FakeDetector()
            rows, tracks = run_video(video, det, RulePolicy(hold=2, small_px=5.0), log_every=0)
            self.assertEqual(len(rows), 60)
            self.assertTrue({r["mode"] for r in rows} >= {"FAST", "ROBUST"})   # mode really switched
            self.assertEqual(set(det.calls), {640, 960})                       # ...and changed inference size
            from uavtrack.evaluate import load_reference
            pred = {}
            for t in tracks:
                pred.setdefault(t["frame"], []).append((t["track_id"], t["x1"], t["y1"], t["x2"], t["y2"]))
            m = evaluate(load_reference(ref_path), pred, [r["frame"] for r in rows])
            self.assertEqual(m["distinct_pred_ids"], 1)
            self.assertEqual(m["ID_switches"], 0)
            # An online policy decides BEFORE seeing frame t, so it cannot prevent the miss that
            # triggers it: recall is only expected to be >= fixed FAST here (the fake FAST detector
            # misses 1 call in 3). Gap filling is the tracker's job (emit_coasting), tested below.
            fast_rows, fast_tracks = run_video(video, FakeDetector(), FixedPolicy("FAST"), log_every=0)
            fast_pred = {}
            for t in fast_tracks:
                fast_pred.setdefault(t["frame"], []).append((t["track_id"], t["x1"], t["y1"], t["x2"], t["y2"]))
            fm = evaluate(load_reference(ref_path), fast_pred, [r["frame"] for r in fast_rows])
            self.assertGreaterEqual(m["recall"], fm["recall"])
            s = summarize_timing(rows)
            self.assertIn("end_to_end_fps", s)

    def test_coasting_fills_short_detector_misses(self):
        with tempfile.TemporaryDirectory() as d:
            video, ref_path = make_clip(Path(d))
            rows, tracks = run_video(video, FakeDetector(), RulePolicy(hold=2, small_px=5.0),
                                     tracker_cfg=TrackerConfig(emit_coasting=True), log_every=0)
            from uavtrack.evaluate import load_reference
            pred = {}
            for t in tracks:
                pred.setdefault(t["frame"], []).append((t["track_id"], t["x1"], t["y1"], t["x2"], t["y2"]))
            m = evaluate(load_reference(ref_path), pred, [r["frame"] for r in rows])
            self.assertGreater(m["recall"], 0.9)
            self.assertEqual(m["distinct_pred_ids"], 1)

    def test_fixed_fast_misses_more_than_fixed_robust(self):
        with tempfile.TemporaryDirectory() as d:
            video, _ = make_clip(Path(d))
            fast, _ = run_video(video, FakeDetector(), FixedPolicy("FAST"), log_every=0)
            rob, _ = run_video(video, FakeDetector(), FixedPolicy("ROBUST"), log_every=0)
            self.assertLess(sum(r["n_det"] for r in fast), sum(r["n_det"] for r in rob))


if __name__ == "__main__":
    unittest.main()
