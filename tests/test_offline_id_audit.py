"""Run: python -m unittest discover -s tests -v"""
import sys
import unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from evaluate_saved_track_ids import count_identity_changes

class TestSavedIdAudit(unittest.TestCase):
    def test_gap_reassignment_is_not_hidden(self):
        # Reference identity A matched to predicted track 1 then 4 after a gap.
        self.assertEqual(count_identity_changes({"A": [(1, 1), (2, 1), (6, 4)]}), (0, 1, 1))

    def test_adjacent_change(self):
        self.assertEqual(count_identity_changes({"A": [(1, 2), (2, 3), (3, 3)]}), (1, 0, 0))

    def test_same_id_after_gap(self):
        self.assertEqual(count_identity_changes({"A": [(1, 2), (8, 2)]}), (0, 0, 1))

if __name__ == "__main__":
    unittest.main()
