"""REAL saved data check: the committed Clip 7 CSVs must reproduce the committed verified metrics.

Uses demo/data/*.csv (copies of results/videos|conditions CSVs) and demo/final/verified_metrics.json.
No model, no video decoding: this checks the archived numbers only.
"""
import csv
import json
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DET = ROOT / "demo/data/clip_7_detector_review.csv"
COND = ROOT / "demo/data/clip_7_condition_analysis.csv"
METRICS = ROOT / "demo/final/verified_metrics.json"


@unittest.skipUnless(DET.is_file() and COND.is_file(), "demo/data CSVs not installed")
class SavedClip7Tests(unittest.TestCase):
    def test_csvs_are_aligned_and_consistent(self):
        det = list(csv.DictReader(DET.open(encoding="utf-8-sig")))
        cond = list(csv.DictReader(COND.open(encoding="utf-8-sig")))
        self.assertEqual([r["frame"] for r in det], [r["frame"] for r in cond])
        for a, b in zip(det, cond):
            self.assertEqual(float(a["max_iou"]) >= 0.5, b["matched_iou50"] == "1")
            self.assertEqual(int(a["predicted_boxes"]) > 0, b["has_prediction"] == "1")

    @unittest.skipUnless(METRICS.is_file(), "verified_metrics.json not present")
    def test_numbers_match_verified_metrics(self):
        from demo_archived_review import summarize
        det = list(csv.DictReader(DET.open(encoding="utf-8-sig")))
        s, ref = summarize(det), json.loads(METRICS.read_text(encoding="utf-8"))
        for key in ("frames", "frames_with_detections", "frames_with_iou_at_least_05", "longest_unmatched_run"):
            self.assertEqual(s[key], ref[key], key)


if __name__ == "__main__":
    unittest.main()
