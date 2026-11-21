"""Unit tests use invented measurements; they do not run image detection/tracking."""
import csv
import json
import tempfile
import unittest
from pathlib import Path
from research_addon.benchmark import read_pairs, summarize, Outcome
from research_addon.policy_audit import audit
from research_addon.track_audit import score_tracks
from research_addon.report import write_report


class AuditTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.bench = self.root / 'per_frame.csv'
        self.dec = self.root / 'mode_decisions.csv'
        self.exp = self.root / 'experiment.json'
        self.bench.write_text('frame,mode,latency_ms,TP,FP,FN\n'
                              'Clip_8_frame_000001.jpg,FAST,10,0,0,1\n'
                              'Clip_8_frame_000001.jpg,ROBUST,12,1,0,0\n'
                              'Clip_8_frame_000021.jpg,FAST,11,1,0,0\n'
                              'Clip_8_frame_000021.jpg,ROBUST,13,1,1,0\n', encoding='utf8')
        self.dec.write_text('frame,selected_mode\nClip_8_frame_000001.jpg,ROBUST\n'
                            'Clip_8_frame_000021.jpg,FAST\n', encoding='utf8')
        self.exp.write_text('{"train_clip":7,"eval_clip":8}', encoding='utf8')

    def test_read_pairs(self):
        p = read_pairs(self.bench)
        self.assertEqual(len(p), 2)
        self.assertEqual(p['Clip_8_frame_000001.jpg']['ROBUST'].tp, 1)

    def test_aggregate(self):
        self.assertEqual(summarize([Outcome(1, 1, 0, 10)])['f1_iou50'], .66667)

    def test_selector_replay(self):
        r = audit(self.bench, self.dec, self.exp)
        self.assertEqual(r['evaluation_clip'], 8)
        self.assertEqual(r['comparison']['LEARNED_OFFLINE_REPLAY']['TP'], 2)
        self.assertEqual(r['comparison']['ALWAYS_FAST']['TP'], 1)

    def test_missing_mode_rejected(self):
        s=self.bench.read_text()
        self.bench.write_text(s.replace('Clip_8_frame_000021.jpg,ROBUST,13,1,1,0\n',''))
        with self.assertRaisesRegex(ValueError, 'FAST and ROBUST'):
            read_pairs(self.bench)

    def test_inconsistent_gt_rejected(self):
        s=self.bench.read_text()
        self.bench.write_text(s.replace('Clip_8_frame_000001.jpg,ROBUST,12,1,0,0',
                                        'Clip_8_frame_000001.jpg,ROBUST,12,2,0,0'))
        with self.assertRaisesRegex(ValueError, 'ground-truth count'):
            read_pairs(self.bench)

    def test_duplicate_decision_rejected(self):
        self.dec.write_text(self.dec.read_text()+'Clip_8_frame_000021.jpg,FAST\n')
        with self.assertRaisesRegex(ValueError, 'Duplicate decision'):
            audit(self.bench, self.dec, self.exp)

    def test_decision_coverage_rejected(self):
        self.dec.write_text('frame,selected_mode\nClip_8_frame_000001.jpg,ROBUST\n')
        with self.assertRaisesRegex(ValueError, 'coverage'):
            audit(self.bench, self.dec, self.exp)

    def test_split_overlap_rejected(self):
        self.exp.write_text('{"train_clip":8,"eval_clip":8}')
        with self.assertRaisesRegex(ValueError, 'identical'):
            audit(self.bench, self.dec, self.exp)

    def test_negative_invalid(self):
        s=self.bench.read_text()
        self.bench.write_text(s.replace('FAST,10,0,0,1', 'FAST,10,-1,0,1'))
        with self.assertRaises(ValueError):
            read_pairs(self.bench)

    def test_latency_invalid(self):
        s=self.bench.read_text()
        self.bench.write_text(s.replace('FAST,10,0,0,1', 'FAST,nan,0,0,1'))
        with self.assertRaises(ValueError):
            read_pairs(self.bench)

    def test_track_gap_change(self):
        pred = self.root / 'tracks.csv'
        ref = self.root / 'gt.txt'
        pred.write_text('clip,frame,track_id,x,y,w,h\n7,1,1,0,0,10,10\n7,2,1,1,0,10,10\n7,5,4,4,0,10,10\n')
        ref.write_text('1,50,0,0,10,10,1,-1,-1,-1\n2,50,1,0,10,10,1,-1,-1,-1\n'
                       '3,50,2,0,10,10,1,-1,-1,-1\n4,50,3,0,10,10,1,-1,-1,-1\n'
                       '5,50,4,0,10,10,1,-1,-1,-1\n')
        result = score_tracks(pred, ref, 7)
        self.assertEqual(result['matched_boxes'], 3)
        self.assertEqual(result['identity_changes_across_gaps'], 1)
        self.assertEqual(result['adjacent_matched_identity_changes'], 0)

    def test_no_reference_id_leak(self):
        pred = self.root / 'tracks.csv'
        ref = self.root / 'gt.txt'
        pred.write_text('clip,frame,track_id,x,y,w,h\n7,1,999,0,0,10,10\n')
        ref.write_text('1,50,0,0,10,10,1,-1,-1,-1\n')
        r = score_tracks(pred, ref, 7)
        self.assertEqual(r['matched_boxes'], 1)
        self.assertEqual(r['reference_ids_with_matches'], 1)

    def test_report_written(self):
        review = audit(self.bench, self.dec, self.exp)
        out = self.root / 'review'
        write_report(out, {'selector_review': review, 'track_review': None},
                     {'benchmark': self.bench, 'decisions': self.dec, 'experiment': self.exp})
        self.assertTrue((out / 'index.html').is_file())
        self.assertTrue((out / 'research_report.md').is_file())
        self.assertTrue(json.loads((out / 'research_metrics.json').read_text())['input_manifest']['benchmark']['sha256'])


class TimelineTests(unittest.TestCase):
    def test_join_and_gap(self):
        from research_addon.timeline import join_timeline
        with tempfile.TemporaryDirectory() as t:
            root=Path(t)
            a=root/'review.csv';b=root/'features.csv';dest=root/'out'/'timeline.csv'
            a.write_text('frame,reference_boxes,predicted_boxes,max_iou\n'
                         '1,1,1,0.7\n2,1,0,0.0\n3,1,0,0.0\n4,1,1,0.6\n')
            b.write_text('frame,brightness,contrast,sharpness,frame_change\n'
                         '1,120,48,200,\n2,121,48,201,2\n3,122,48,202,3\n4,123,48,203,2\n')
            result=join_timeline(a,b,dest)
            self.assertEqual(result['matched_frames'],2)
            self.assertEqual(result['longest_consecutive_unmatched_frames'],2)
            self.assertTrue(dest.is_file())


if __name__ == '__main__':
    unittest.main()
