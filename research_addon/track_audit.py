"""Retrospective audit of externally generated object-identity CSVs.

This reads saved track records and reference records. It does not produce IDs,
associate live detections, estimate trajectories, or run a target tracker.
This is NOT a replacement for standardized IDF1/HOTA/TrackEval.
"""
from __future__ import annotations

from collections import defaultdict
import csv
from pathlib import Path
import math


def _iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[0] + a[2], b[0] + b[2]), min(a[1] + a[3], b[1] + b[3])
    intersection = max(0, x2-x1) * max(0, y2-y1)
    union = a[2]*a[3] + b[2]*b[3] - intersection
    return intersection/union if union > 0 else 0.0


def _parse_box(row):
    box = tuple(float(row[k]) for k in ('x', 'y', 'w', 'h'))
    if not all(math.isfinite(x) for x in box) or box[2] <= 0 or box[3] <= 0:
        raise ValueError(f'Invalid bounding box in row: {row}')
    return box


def read_predictions(path: Path, clip: int):
    if not path.is_file():
        raise FileNotFoundError(path)
    data = defaultdict(list)
    with path.open(newline='', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        required = {'clip', 'frame', 'track_id', 'x', 'y', 'w', 'h'}
        if required - set(reader.fieldnames or []):
            raise ValueError(f'Expected saved track columns: {sorted(required)}')
        for row in reader:
            if int(row['clip']) == clip:
                frame, track_id = int(row['frame']), int(row['track_id'])
                if frame <= 0:
                    raise ValueError('Frame IDs must be one-indexed')
                if track_id in [x[0] for x in data[frame]]:
                    raise ValueError(f'Duplicate track ID {track_id} in frame {frame}')
                data[frame].append((track_id, _parse_box(row)))
    return data


def read_reference(path: Path):
    if not path.is_file():
        raise FileNotFoundError(path)
    data = defaultdict(list)
    with path.open(newline='', encoding='utf-8-sig') as f:
        for row in csv.reader(f):
            if not row or row[0].startswith('#'):
                continue
            if len(row) < 6:
                raise ValueError(f'Expected at least 6 MOT columns, got {row}')
            frame, ref_id = int(float(row[0])), int(float(row[1]))
            box = tuple(float(x) for x in row[2:6])
            if frame < 1 or box[2] <= 0 or box[3] <= 0 or not all(math.isfinite(x) for x in box):
                raise ValueError(f'Invalid reference row {row}')
            data[frame].append((ref_id, box))
    return data


def score_tracks(saved_csv: Path, reference_csv: Path, clip: int, threshold: float = .5) -> dict:
    if not (0 < threshold <= 1):
        raise ValueError('IoU threshold must be in (0, 1]')
    pred = read_predictions(saved_csv, clip)
    ref = read_reference(reference_csv)
    all_frames = sorted(set(ref) | set(pred))
    tp = fp = fn = 0
    history = defaultdict(list)
    for frame in all_frames:
        references, predictions = ref.get(frame, []), pred.get(frame, [])
        candidates = sorted(((_iou(g[1], p[1]), i, j) for i,g in enumerate(references)
                             for j,p in enumerate(predictions)), reverse=True)
        used_ref, used_pred = set(), set()
        for score, i, j in candidates:
            if score < threshold:
                break
            if i in used_ref or j in used_pred:
                continue
            used_ref.add(i); used_pred.add(j)
            history[references[i][0]].append((frame, predictions[j][0]))
        tp += len(used_ref)
        fp += len(predictions) - len(used_pred)
        fn += len(references) - len(used_ref)
    adjacent_changes = gap_changes = 0
    for observations in history.values():
        observations.sort()
        for (f0, id0), (f1, id1) in zip(observations, observations[1:]):
            if id0 != id1:
                if f1 == f0 + 1:
                    adjacent_changes += 1
                else:
                    gap_changes += 1
    return {
        'clip': clip, 'frames_with_reference': len(ref),
        'gt_boxes': sum(len(v) for v in ref.values()),
        'predicted_boxes': sum(len(v) for v in pred.values()),
        'matched_boxes': tp, 'unmatched_predictions': fp, 'missed_reference_boxes': fn,
        'precision_iou50': round(tp/(tp+fp), 5) if tp+fp else 0,
        'recall_iou50': round(tp/(tp+fn), 5) if tp+fn else 0,
        'reference_ids_with_matches': len(history),
        'adjacent_matched_identity_changes': adjacent_changes,
        'identity_changes_across_gaps': gap_changes,
        'caution': 'Retrospective diagnostic only. NOT standardized IDF1, HOTA or a tracker.',
    }
