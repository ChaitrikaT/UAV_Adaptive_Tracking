"""Regression tests for uavtrack.evaluate against hand-computed CLEAR-MOT / IDF1 values.

All inputs are SYNTHETIC boxes. Definitions used:
  MOTA = 1 - (FN + FP + IDSW) / num_ref_boxes
  IDSW : a reference object is matched to a predicted ID different from its previous matched ID
  Frag : a reference trajectory is tracked, then lost, then tracked again
  IDF1 = 2*IDTP / (2*IDTP + IDFP + IDFN) with one global ref-ID <-> pred-ID assignment
"""
import itertools
import random
import unittest

from uavtrack.evaluate import evaluate

BOX = (10, 10, 30, 30)


def ref_one(n=10):
    return {f: [(1, *BOX)] for f in range(1, n + 1)}


def pred_from(spec):
    """spec: {frame: [(id, box), ...]}"""
    return {f: [(i, *b) for i, b in items] for f, items in spec.items()}


class ClearMotTests(unittest.TestCase):
    def test_missed_detections(self):
        pred = pred_from({f: [(5, BOX)] for f in range(1, 5)})
        m = evaluate(ref_one(), pred, range(1, 11))
        self.assertEqual((m["TP"], m["FP"], m["FN"], m["ID_switches"]), (4, 0, 6, 0))
        self.assertAlmostEqual(m["MOTA"], 0.4)
        self.assertAlmostEqual(m["IDF1"], 8 / 14, places=4)
        self.assertEqual(m["fragmentations"], 0)          # never resumed

    def test_identity_change(self):
        pred = pred_from({f: [(1 if f <= 5 else 2, BOX)] for f in range(1, 11)})
        m = evaluate(ref_one(), pred, range(1, 11))
        self.assertEqual((m["TP"], m["FP"], m["FN"], m["ID_switches"]), (10, 0, 0, 1))
        self.assertAlmostEqual(m["MOTA"], 0.9)
        self.assertAlmostEqual(m["IDF1"], 0.5)
        self.assertEqual(m["fragmentations"], 0)

    def test_duplicate_prediction_different_ids(self):
        pred = pred_from({f: [(1, BOX), (2, BOX)] for f in range(1, 11)})
        m = evaluate(ref_one(), pred, range(1, 11))
        self.assertEqual((m["TP"], m["FP"], m["FN"], m["ID_switches"]), (10, 10, 0, 0))
        self.assertAlmostEqual(m["MOTA"], 0.0)
        self.assertAlmostEqual(m["IDF1"], 20 / 30, places=4)

    def test_duplicate_track_id_inside_one_frame_is_rejected(self):
        """Same predicted ID twice in a frame is undefined for MOT metrics (v2 silently double counted IDTP)."""
        pred = pred_from({f: [(1, BOX), (1, BOX)] for f in range(1, 11)})
        with self.assertRaises(ValueError):
            evaluate(ref_one(), pred, range(1, 11))

    def test_duplicate_does_not_cause_switch_when_previous_match_still_valid(self):
        shifted = (14, 10, 34, 30)                          # IoU with BOX = 16*20/(2*400-320) = 0.667
        spec = {f: [(1, shifted)] for f in range(1, 11)}
        for f in range(4, 7):
            spec[f] = [(2, BOX), (1, shifted)]              # better-fitting duplicate appears
        m = evaluate(ref_one(), pred_from(spec), range(1, 11))
        self.assertEqual(m["ID_switches"], 0)
        self.assertEqual(m["FP"], 3)

    def test_gap_same_id_is_fragmentation_not_switch(self):
        pred = pred_from({f: [(1, BOX)] for f in range(1, 11) if f not in (4, 5, 6)})
        m = evaluate(ref_one(), pred, range(1, 11))
        self.assertEqual((m["FN"], m["ID_switches"], m["fragmentations"]), (3, 0, 1))
        self.assertAlmostEqual(m["IDF1"], 14 / 17, places=4)

    def test_gap_new_id_is_switch_and_fragmentation(self):
        pred = pred_from({f: [(1 if f < 4 else 2, BOX)] for f in range(1, 11) if f not in (4, 5, 6)})
        m = evaluate(ref_one(), pred, range(1, 11))
        self.assertEqual((m["FN"], m["ID_switches"], m["fragmentations"]), (3, 1, 1))

    def test_leading_miss_is_not_a_fragmentation(self):
        """Regression: the first version counted a late first pickup as a fragmentation."""
        pred = pred_from({f: [(5, BOX)] for f in range(4, 11)})
        m = evaluate(ref_one(), pred, range(1, 11))
        self.assertEqual(m["fragmentations"], 0)

    def test_two_objects_swapping_ids(self):
        A, B = (0, 0, 20, 20), (200, 0, 220, 20)
        ref = {f: [(1, *A), (2, *B)] for f in range(1, 7)}
        pred = {f: ([(10, *A), (20, *B)] if f <= 3 else [(20, *A), (10, *B)]) for f in range(1, 7)}
        m = evaluate(ref, pred, range(1, 7))
        self.assertEqual(m["ID_switches"], 2)
        self.assertAlmostEqual(m["MOTA"], 1 - 2 / 12, places=3)
        self.assertAlmostEqual(m["IDF1"], 0.5)

    def test_unannotated_frames_are_unknown_by_default(self):
        ref = ref_one(5)
        pred = pred_from({1: [(1, BOX)], 9: [(1, BOX)]})
        m = evaluate(ref, pred, list(range(1, 6)) + [9])
        self.assertEqual(m["FP"], 0)
        m2 = evaluate(ref, pred, list(range(1, 6)) + [9], annotated_only=False)
        self.assertEqual(m2["FP"], 1)

    def test_empty_inputs_do_not_crash(self):
        m = evaluate(ref_one(3), {}, [1, 2, 3])
        self.assertEqual((m["TP"], m["FN"], m["IDF1"], m["MOTA"]), (0, 3, 0.0, 0.0))


class IdF1BruteForceTests(unittest.TestCase):
    def test_matches_exhaustive_assignment_on_random_cases(self):
        rnd = random.Random(7)
        for _ in range(150):
            n_ref, n_frames = rnd.randint(1, 3), rnd.randint(3, 8)
            slot = lambda i: (100 * i, 0, 100 * i + 20, 20)      # disjoint boxes => IoU is 1 or 0
            ref, pred = {}, {}
            for f in range(1, n_frames + 1):
                ref[f] = [(g, *slot(g)) for g in range(n_ref) if rnd.random() < 0.85]
                pred[f] = [(rnd.randint(1, 3), *slot(rnd.randint(0, n_ref - 1)))
                           for _ in range(rnd.randint(0, 3))]
                seen = set()                                      # ids unique within a frame
                pred[f] = [p for p in pred[f] if not (p[0] in seen or seen.add(p[0]))]
            frames = [f for f in ref if ref[f]]
            n_r = sum(len(ref[f]) for f in frames)
            n_p = sum(len(pred[f]) for f in frames)
            gids = sorted({g[0] for f in frames for g in ref[f]})
            pids = sorted({p[0] for f in frames for p in pred[f]})
            best = 0
            for choice in itertools.product(pids + [None], repeat=len(gids)):
                used = [c for c in choice if c is not None]
                if len(used) != len(set(used)):
                    continue
                tp = sum(1 for f in frames for g in ref[f] for p in pred[f]
                         if choice[gids.index(g[0])] == p[0] and g[1:] == p[1:])
                best = max(best, tp)
            expected = 2 * best / (n_r + n_p) if (n_r + n_p) else 0.0
            got = evaluate(ref, pred, frames)["IDF1"]
            self.assertAlmostEqual(got, round(expected, 4), places=4, msg=f"ref={ref} pred={pred}")


if __name__ == "__main__":
    unittest.main()
