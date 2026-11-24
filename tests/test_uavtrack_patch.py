"""Regression tests for the evaluation + selector correctness patch.

ALL data here is synthetic (hand-built tracks, drawn white squares, a fake detector).
Nothing in this file is a real-dataset or real-model result.
"""
import random
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from uavtrack.collect import _hits, collect_clip
from uavtrack.evaluate import count_matches, evaluate
from uavtrack.features import FEATURE_NAMES
from uavtrack.labels import build_training_set, label_frame
from uavtrack.pipeline import run_video
from uavtrack.policy import FixedPolicy, RulePolicy


def B(x, y=10, w=12, h=12):
    return (x, y, x + w, y + h)


def seq(ids_by_frame, x0=10, step=1):
    """{frame: [(id, box)]} helper -> {frame: [(id, x1,y1,x2,y2)]} moving right."""
    return {f: [(i, *B(x0 + step * f + 30 * k)) for k, i in enumerate(ids)] for f, ids in ids_by_frame.items()}


class MetricDefinitionTests(unittest.TestCase):
    def test_id_switch_counts_once_and_matches_hand_values(self):
        ref = seq({f: [1] for f in range(1, 11)})
        pred = {f: [((1 if f <= 5 else 2), *ref[f][0][1:])] for f in ref}
        m = evaluate(ref, pred, list(ref))
        self.assertEqual((m["TP"], m["FP"], m["FN"], m["ID_switches"]), (10, 0, 0, 1))
        self.assertAlmostEqual(m["MOTA"], 1 - 1 / 10)        # (FN+FP+IDSW)/GT = 1/10
        self.assertAlmostEqual(m["IDF1"], 0.5)               # IDTP=5, IDFP=5, IDFN=5

    def test_gap_same_id_is_not_a_switch_but_is_a_fragmentation(self):
        ref = seq({f: [1] for f in range(1, 11)})
        pred = {f: [(1, *ref[f][0][1:])] for f in ref if f not in (4, 5, 6)}
        m = evaluate(ref, pred, list(ref))
        self.assertEqual((m["TP"], m["FP"], m["FN"], m["ID_switches"], m["fragmentations"]), (7, 0, 3, 0, 1))
        self.assertAlmostEqual(m["MOTA"], 1 - 3 / 10)
        self.assertAlmostEqual(m["IDF1"], 2 * 7 / (2 * 7 + 0 + 3), delta=1e-3)

    def test_id_change_across_a_gap_is_still_a_switch(self):
        # CLEAR-MOT compares with the LAST matched id, not only the previous frame.
        ref = seq({f: [1] for f in range(1, 11)})
        pred = {f: [((1 if f < 4 else 2), *ref[f][0][1:])] for f in ref if f not in (4, 5)}
        m = evaluate(ref, pred, list(ref))
        self.assertEqual(m["ID_switches"], 1)
        self.assertEqual(m["FN"], 2)

    def test_crossing_swap_gives_two_switches(self):
        ref = {1: [(1, *B(10)), (2, *B(40))], 2: [(1, *B(10)), (2, *B(40))]}
        pred = {1: [(5, *B(10)), (6, *B(40))], 2: [(6, *B(10)), (5, *B(40))]}
        m = evaluate(ref, pred, [1, 2])
        self.assertEqual((m["TP"], m["ID_switches"]), (4, 2))

    def test_duplicate_prediction_on_one_target_is_one_false_positive(self):
        ref = seq({f: [1] for f in range(1, 6)})
        pred = {f: [(1, *ref[f][0][1:]), (9, *ref[f][0][1:])] for f in ref}   # same box, 2 ids
        m = evaluate(ref, pred, list(ref))
        self.assertEqual((m["TP"], m["FP"], m["FN"], m["ID_switches"]), (5, 5, 0, 0))
        self.assertAlmostEqual(m["MOTA"], 1 - 5 / 5)
        self.assertAlmostEqual(m["precision"], 0.5)
        self.assertAlmostEqual(m["IDF1"], 2 * 5 / (2 * 5 + 5 + 0), delta=1e-3)   # IDFP=5

    def test_duplicate_track_id_inside_one_frame_is_rejected(self):
        ref = {1: [(1, *B(10))]}
        with self.assertRaises(ValueError):
            evaluate(ref, {1: [(3, *B(10)), (3, *B(11))]}, [1])
        with self.assertWarns(UserWarning):                      # reference duplicates: warn + count
            m = evaluate({1: [(1, *B(10)), (1, *B(50))]}, {}, [1])
        self.assertEqual(m["ref_duplicate_id_rows"], 1)

    def test_all_missed_observations(self):
        ref = seq({f: [1] for f in range(1, 6)})
        m = evaluate(ref, {}, list(ref))
        self.assertEqual((m["TP"], m["FN"], m["FP"]), (0, 5, 0))
        self.assertEqual((m["MOTA"], m["IDF1"], m["recall"]), (0.0, 0.0, 0.0))

    def test_only_false_positives_gives_negative_mota(self):
        ref = seq({f: [1] for f in range(1, 4)})
        pred = {f: [(1, *B(200))] for f in ref}
        m = evaluate(ref, pred, list(ref))
        self.assertEqual((m["TP"], m["FP"], m["FN"]), (0, 3, 3))
        self.assertAlmostEqual(m["MOTA"], 1 - 6 / 3)

    def test_iou_exactly_at_threshold_is_a_match(self):
        ref = {1: [(1, 0, 0, 10, 10)]}
        pred = {1: [(1, 0, 0, 10, 20)]}                       # IoU = 100/200 = 0.5
        self.assertEqual(evaluate(ref, pred, [1])["TP"], 1)
        self.assertEqual(evaluate(ref, {1: [(1, 0, 0, 10, 21)]}, [1])["TP"], 0)

    def test_gated_assignment_keeps_maximum_number_of_valid_matches(self):
        """BUG regression. ref1=[32,44], ref2=[28,40]; pred A=[32,44] (IoU 1 / 0.5 with ref1/ref2),
        pred B=[33,45] (0.846 / 0.41). Un-gated min-cost assignment picks ref1-A + ref2-B and then
        discards the invalid ref2-B (TP=1). Correct CLEAR-MOT matching gives ref1-B + ref2-A (TP=2)."""
        ref = {1: [(1, *B(32)), (2, *B(28))]}
        pred = {1: [(10, *B(32)), (11, *B(33))]}
        m = evaluate(ref, pred, [1])
        self.assertEqual((m["TP"], m["FP"], m["FN"]), (2, 0, 0))
        self.assertEqual(count_matches([g[1:] for g in ref[1]], [p[1:] for p in pred[1]]), 2)
        self.assertEqual(_hits(ref[1], np.array([[*p[1:], 0.9] for p in pred[1]])), 2)

    def test_protocol_flag_and_unannotated_frames(self):
        ref = seq({f: [1] for f in range(1, 4)})
        pred = {f: [(1, *ref[f][0][1:])] for f in ref}
        pred[50] = [(1, *B(0))]
        a = evaluate(ref, pred, [1, 2, 3, 50])                  # project variant: frame 50 unscored
        b = evaluate(ref, pred, [1, 2, 3, 50], annotated_only=False)
        self.assertEqual((a["FP"], b["FP"]), (0, 1))
        self.assertIn("not official", a["protocol"])
        self.assertNotIn("not official", b["protocol"])

    def test_repeated_frames_are_scored_once(self):
        ref = seq({f: [1] for f in range(1, 4)})
        pred = {f: [(1, *ref[f][0][1:])] for f in ref}
        self.assertEqual(evaluate(ref, pred, [1, 2, 3, 3, 2, 1])["ref_boxes"], 3)


class OracleAgreementTests(unittest.TestCase):
    """Cross-check against py-motmetrics (optional dev dependency; skipped if not installed)."""

    @staticmethod
    def _scenario(rng):
        nid, n = rng.randint(1, 4), rng.randint(5, 40)
        ref, pred = {}, {}
        for f in range(1, n + 1):
            for i in range(nid):
                if rng.random() < 0.9:
                    x = 20 * i + f * rng.choice([1, 1, 1, 4])
                    ref.setdefault(f, []).append((i + 1, x, 10, x + 12, 22))
        for f in range(1, n + 1):
            used, ps = set(), []
            for (i, x1, y1, x2, y2) in ref.get(f, []):
                if rng.random() < 0.15:
                    continue
                pid = rng.choice([i, i, i, 10 + i, 20 + rng.randint(0, 2)])
                j = rng.randint(-4, 4) if rng.random() < 0.4 else 0
                if pid not in used:
                    used.add(pid); ps.append((pid, x1 + j, y1, x2 + j, y2))
                if rng.random() < 0.1 and pid + 100 not in used:
                    used.add(pid + 100); ps.append((pid + 100, x1 + 1, y1, x2 + 1, y2))
            if ps:
                pred[f] = ps
        return ref, pred

    def test_agrees_with_motmetrics_on_random_synthetic_scenarios(self):
        try:
            import motmetrics as mm
        except ImportError:
            self.skipTest("py-motmetrics not installed (optional): pip install motmetrics")
        if not hasattr(np, "asfarray"):                        # motmetrics 1.4 vs NumPy 2 shim
            np.asfarray = lambda a, dtype=float: np.asarray(a, dtype=dtype)
        rng, hard, soft, n = random.Random(7), [], [], 0
        for _ in range(500):
            ref, pred = self._scenario(rng)
            frames = sorted(ref)
            if not frames:
                continue
            n += 1
            acc = mm.MOTAccumulator(auto_id=False)
            for f in frames:
                g, p = ref.get(f, []), pred.get(f, [])
                gb = [(a[1], a[2], a[3] - a[1], a[4] - a[2]) for a in g]
                pb = [(a[1], a[2], a[3] - a[1], a[4] - a[2]) for a in p]
                d = mm.distances.iou_matrix(gb, pb, max_iou=0.5) if g and p else np.empty((len(g), len(p)))
                acc.update([a[0] for a in g], [a[0] for a in p], d, frameid=f)
            o = mm.metrics.create().compute(
                acc, metrics=["num_switches", "mota", "idf1", "num_misses", "num_false_positives", "num_matches"],
                name="x").iloc[0]
            m = evaluate(ref, pred, frames)
            if not (m["TP"] == o.num_matches + o.num_switches and m["FN"] == o.num_misses
                    and m["FP"] == o.num_false_positives):
                hard.append((ref, pred))
            elif not (m["ID_switches"] == o.num_switches and abs(m["MOTA"] - o.mota) < 1e-3
                      and abs(m["IDF1"] - o.idf1) < 1e-3):
                soft.append((ref, pred))
        # TP/FP/FN must always agree. ID switches/MOTA/IDF1 may differ only when two assignments
        # have exactly equal cost (library tie-breaking); allow at most 1% of scenarios.
        self.assertEqual(len(hard), 0, f"{len(hard)} scenarios disagree on TP/FP/FN with py-motmetrics")
        self.assertLessEqual(len(soft), n // 100, f"{len(soft)}/{n} scenarios differ in IDSW/MOTA/IDF1")


class LabelTests(unittest.TestCase):
    def test_label_table(self):
        self.assertEqual(label_frame(0, 1), "ROBUST")
        self.assertEqual(label_frame(1, 2), "ROBUST")
        self.assertEqual(label_frame(1, 0), "FAST")
        self.assertEqual(label_frame(2, 1), "FAST")
        self.assertEqual(label_frame(1, 1), "FAST")            # equal, non-zero: FAST is cheaper
        self.assertIsNone(label_frame(0, 0))                    # both failed: no evidence

    def test_both_failed_frames_are_excluded_not_labelled_fast(self):
        rows = [{"fast_tp": 0, "robust_tp": 0, "x": 1.0}] * 7 + [
            {"fast_tp": 0, "robust_tp": 1, "x": 2.0}, {"fast_tp": 1, "robust_tp": 1, "x": 3.0}]
        X, y, rep = build_training_set(rows, ["x"])
        self.assertEqual(y, ["ROBUST", "FAST"])
        self.assertEqual((rep["rows"], rep["used"], rep["excluded_both_failed"]), (9, 2, 7))
        # old rule would have produced 8 FAST vs 1 ROBUST
        old = ["ROBUST" if r["robust_tp"] > r["fast_tp"] else "FAST" for r in rows]
        self.assertEqual(old.count("FAST"), 8)

    def test_only_failed_frames_leaves_nothing_to_learn(self):
        X, y, rep = build_training_set([{"fast_tp": 0, "robust_tp": 0}] * 3, [])
        self.assertEqual((y, rep["used"]), ([], 0))


# ---------------------------------------------------------------- selector / collection ----
class FrameFake:
    """Deterministic per FRAME (not per call): FAST (640) misses frames with f % 3 == 0."""
    def detect(self, frame, imgsz):
        g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        ys, xs = np.where(g > 200)
        if len(xs) == 0:
            return np.zeros((0, 5))
        f = int(round((xs.min() - 20) / 4))
        if imgsz == 640 and f % 3 == 0:
            return np.zeros((0, 5))
        return np.array([[xs.min(), ys.min(), xs.max() + 1, ys.max() + 1, 0.8]])


def make_clip(dirpath: Path, n=60, w=320, h=180, tail_variant=False, name="Clip_1"):
    video, ref = dirpath / f"{name}.avi", dirpath / f"{name}_refined.txt"
    wr = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"MJPG"), 10.0, (w, h))
    lines = []
    for f in range(1, n + 1):
        img = np.full((h, w, 3), 40, np.uint8)
        x, y = 20 + 4 * f, 90
        if tail_variant and f > 30:                 # different FUTURE: other brightness, other motion
            img[:] = 90
            y = 40
        cv2.rectangle(img, (x, y), (x + 11, y + 11), (255, 255, 255), -1)
        wr.write(img)
        lines.append(f"{f},1,{x},{y},12,12,1,1,1")
    wr.release()
    ref.write_text("\n".join(lines), encoding="utf-8")
    return video, ref


class SelectorMethodologyTests(unittest.TestCase):
    def test_label_columns_are_never_features(self):
        for banned in ("ref_boxes", "fast_tp", "robust_tp", "fast_ms", "robust_ms", "driver_robust", "frame"):
            self.assertNotIn(banned, FEATURE_NAMES)

    def test_collected_features_equal_the_features_run_video_sees_online(self):
        """Same policy, same detector -> training features == deployment features, row by row."""
        with tempfile.TemporaryDirectory() as d:
            video, ref = make_clip(Path(d))
            rows = collect_clip(video, ref, FrameFake(), log_every=0, driver=RulePolicy())
            online, _ = run_video(video, FrameFake(), RulePolicy(), log_every=0)
            by_frame = {r["frame"]: r for r in online}
            self.assertEqual(len(rows), 60)
            for r in rows:
                for k in FEATURE_NAMES:
                    self.assertAlmostEqual(r[k], by_frame[r["frame"]][k], delta=2e-3,
                                           msg=f"frame {r['frame']} feature {k}")

    def test_fast_driver_state_is_fast_state_not_robust_state(self):
        with tempfile.TemporaryDirectory() as d:
            video, ref = make_clip(Path(d))
            fast = collect_clip(video, ref, FrameFake(), log_every=0, driver="fast")
            rob = collect_clip(video, ref, FrameFake(), log_every=0, driver="robust")
            self.assertEqual({r["prev_robust"] for r in fast}, {0.0})          # was hard-coded 1 before
            self.assertEqual({r["prev_robust"] for r in rob[1:]}, {1.0})
            self.assertEqual(max(r["frames_since_seen"] for r in rob[5:]), 0)   # ROBUST never misses once a track exists
            self.assertGreaterEqual(max(r["frames_since_seen"] for r in fast[5:]), 1)  # FAST misses show up

    def test_features_are_causal_and_independent_of_reference_and_future(self):
        with tempfile.TemporaryDirectory() as d1, tempfile.TemporaryDirectory() as d2:
            va, ra = make_clip(Path(d1))
            vb, rb = make_clip(Path(d2), tail_variant=True)                     # identical until frame 30
            a = collect_clip(va, ra, FrameFake(), log_every=0, driver=RulePolicy())
            b = collect_clip(vb, rb, FrameFake(), log_every=0, driver=RulePolicy())
            for ra_, rb_ in zip(a, b):
                if ra_["frame"] <= 30:
                    for k in FEATURE_NAMES:
                        self.assertEqual(ra_[k], rb_[k], f"future leaked into frame {ra_['frame']} feature {k}")
            # corrupt every reference box: features unchanged, only labels may change
            ref_bad = Path(d1) / "bad.txt"
            ref_bad.write_text("\n".join(f"{f},1,5,5,12,12,1,1,1" for f in range(1, 61)), encoding="utf-8")
            c = collect_clip(va, ref_bad, FrameFake(), log_every=0, driver=RulePolicy())
            for ra_, rc_ in zip(a, c):
                for k in FEATURE_NAMES:
                    self.assertEqual(ra_[k], rc_[k])
            self.assertTrue(all(r["fast_tp"] == 0 and r["robust_tp"] == 0 for r in c))
            self.assertEqual(sum(label_frame(r["fast_tp"], r["robust_tp"]) is None for r in c), 60)

    def test_driver_names_and_unknown_driver(self):
        with tempfile.TemporaryDirectory() as d:
            video, ref = make_clip(Path(d), n=12)
            for name in ("rule", "fast", "robust", None):
                self.assertEqual(len(collect_clip(video, ref, FrameFake(), log_every=0, driver=name)), 12)
            with self.assertRaises(ValueError):
                collect_clip(video, ref, FrameFake(), log_every=0, driver="bogus")

    def test_labels_on_synthetic_clip(self):
        with tempfile.TemporaryDirectory() as d:
            video, ref = make_clip(Path(d))
            rows = collect_clip(video, ref, FrameFake(), log_every=0, driver=FixedPolicy("FAST"))
            labs = [label_frame(r["fast_tp"], r["robust_tp"]) for r in rows]
            self.assertEqual(labs.count("ROBUST"), 20)       # frames 3,6,...,60 where FAST misses
            self.assertEqual(labs.count("FAST"), 40)
            self.assertNotIn(None, labs)


if __name__ == "__main__":
    unittest.main()
