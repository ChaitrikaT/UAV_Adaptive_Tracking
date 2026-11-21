"""Join already-saved detector diagnostic and condition summaries by frame.

A post-hoc analysis only: reference-derived IoU is a *label* for evaluation,
never a control signal or an input to an online selector.
"""
from __future__ import annotations

import csv
from pathlib import Path
from statistics import mean

FEATURES = ('brightness', 'contrast', 'sharpness', 'frame_change')


def _read_rows(path: Path, required: set[str]) -> dict[int, dict]:
    with path.open(newline='', encoding='utf-8-sig') as f:
        rdr = csv.DictReader(f)
        missing = required - set(rdr.fieldnames or [])
        if missing:
            raise ValueError(f'Missing {sorted(missing)} from {path}')
        output = {}
        for row in rdr:
            frame = int(row['frame'])
            if frame in output:
                raise ValueError(f'Duplicate frame {frame} in {path}')
            output[frame] = row
    return output


def join_timeline(detector_review: Path, condition_review: Path, dest: Path) -> dict:
    if not detector_review.is_file() or not condition_review.is_file():
        raise FileNotFoundError('Both video detector-review and condition-analysis CSVs are required')
    det = _read_rows(detector_review, {'frame', 'reference_boxes', 'predicted_boxes', 'max_iou'})
    cond = _read_rows(condition_review, {'frame', *FEATURES})
    if set(det) != set(cond):
        raise ValueError(f'Frame sets do not match: {len(det)} detector frames vs {len(cond)} condition frames')
    items = []
    longest = (0, None, None)
    active_start = None
    previous_frame = None
    for f in sorted(det):
        row = det[f]
        score = float(row['max_iou'])
        match = int(score >= 0.5)
        if not match:
            if active_start is None or previous_frame is None or f != previous_frame + 1:
                active_start = f
            length = f - active_start + 1
            if length > longest[0]:
                longest = length, active_start, f
        else:
            active_start = None
        previous_frame = f
        items.append({'frame': f, 'predicted_boxes': int(row['predicted_boxes']),
                      'reference_boxes': int(row['reference_boxes']),
                      'max_iou': score, 'matched_iou50': match,
                      **{k:cond[f][k] for k in FEATURES}})
    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open('w', newline='', encoding='utf-8') as f:
        fields = list(items[0]) if items else ['frame', 'predicted_boxes', 'reference_boxes', 'max_iou', 'matched_iou50', *FEATURES]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader(); writer.writerows(items)
    success = sum(x['matched_iou50'] for x in items)
    return {'frames': len(items), 'matched_frames': success,
            'match_rate': round(success / len(items), 4) if items else 0,
            'longest_consecutive_unmatched_frames': longest[0],
            'longest_unmatched_start_frame': longest[1],
            'longest_unmatched_end_frame': longest[2],
            'feature_mean_success_vs_failure': {
                feature: {
                    str(ok): round(mean(float(x[feature]) for x in items if x['matched_iou50'] == ok and x[feature] != ''), 3)
                    if any(x['matched_iou50']==ok and x[feature]!='' for x in items) else None
                    for ok in (0,1)}
                for feature in FEATURES},
            'warning': 'Descriptive only; IoU uses reference labels and cannot drive the selector.'}
