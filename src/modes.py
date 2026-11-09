"""Utilities for replaying previously measured FAST/ROBUST *still-frame* outcomes.

Reference labels appear only in the historical benchmark's evaluation columns.
They must never be supplied to a selector when making decisions.
"""
from __future__ import annotations

import csv
from pathlib import Path

MODES = {'FAST': 640, 'ROBUST': 960}
NEEDED = {'frame', 'mode', 'latency_ms', 'TP', 'FP', 'FN'}


def read_pairs(path: Path) -> dict[str, dict[str, dict]]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing benchmark: {path}")
    pairs = {}
    with path.open(newline='', encoding='utf-8-sig') as stream:
        reader = csv.DictReader(stream)
        if not reader.fieldnames or not NEEDED.issubset(reader.fieldnames):
            raise ValueError('Benchmark must include frame,mode,latency_ms,TP,FP,FN. '
                             'Run benchmark with --uav-model on validation clips.')
        for row in reader:
            filename = row['frame']
            mode = row['mode'].upper().strip()
            if mode not in MODES:
                raise ValueError(f"Unsupported mode: {mode}")
            if mode in pairs.setdefault(filename, {}):
                raise ValueError(f"Duplicate mode {mode} for {filename}")
            pairs[filename][mode] = {
                'TP': int(row['TP']), 'FP': int(row['FP']),
                'FN': int(row['FN']), 'latency_ms': float(row['latency_ms'])
            }
    for frame, outcomes in pairs.items():
        if set(outcomes) != set(MODES):
            raise ValueError(f"Both FAST and ROBUST required for {frame}")
        for mode, counts in outcomes.items():
            if min(counts['TP'], counts['FP'], counts['FN']) < 0:
                raise ValueError(f"Negative evaluation counts for {frame}: {mode}")
    return pairs


def preferred_mode(pair: dict[str, dict]) -> str:
    """Training-only label: prioritize recall matches, then fewer false positives.

    Tie defaults to FAST as the nominal low-resolution mode. This choice is a
    transparent *study definition*, not an experimentally proven ideal reward.
    """
    fast, robust = pair['FAST'], pair['ROBUST']
    if robust['TP'] > fast['TP']:
        return 'ROBUST'
    if robust['TP'] == fast['TP'] and robust['FP'] < fast['FP']:
        return 'ROBUST'
    return 'FAST'


def aggregate(rows: list[dict]) -> dict:
    if not rows:
        raise ValueError('Cannot aggregate zero benchmark outcomes')
    tp = sum(r['TP'] for r in rows)
    fp = sum(r['FP'] for r in rows)
    fn = sum(r['FN'] for r in rows)
    mean_ms = sum(r['latency_ms'] for r in rows) / len(rows)
    return {
        'frames': len(rows), 'TP': tp, 'FP': fp, 'FN': fn,
        'precision_iou50': round(tp / (tp + fp), 4) if tp + fp else 0,
        'recall_iou50': round(tp / (tp + fn), 4) if tp + fn else 0,
        'mean_model_call_latency_ms': round(mean_ms, 3),
        'fps_equivalent': round(1000.0 / mean_ms, 2) if mean_ms else 0,
    }
