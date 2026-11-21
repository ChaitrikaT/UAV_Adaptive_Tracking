"""Validate paired FAST/ROBUST *saved* detection measurements.

FAST and ROBUST are names for pre-recorded experimental configurations;
no inference models, cameras, or trackers are controlled here.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import csv
import math
import re
from statistics import mean, median

MODES = ('FAST', 'ROBUST')
PATTERN = re.compile(r'^Clip_(\d+)_frame_(\d+)\.(?:jpg|png)$', re.IGNORECASE)


@dataclass(frozen=True)
class Outcome:
    tp: int
    fp: int
    fn: int
    latency_ms: float


def _int(value: str, label: str) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f'Invalid integer for {label}: {value!r}') from exc
    if parsed < 0:
        raise ValueError(f'Negative {label}: {parsed}')
    return parsed


def parse_clip(filename: str) -> int:
    match = PATTERN.fullmatch(Path(filename).name)
    if not match:
        raise ValueError(f'Unrecognized frame name {filename!r}; expected Clip_N_frame_000001.jpg')
    return int(match.group(1))


def read_pairs(path: Path) -> dict[str, dict[str, Outcome]]:
    """Reject incomplete paired comparisons and mismatched reference counts."""
    if not path.is_file():
        raise FileNotFoundError(f'Missing saved benchmark CSV: {path}')
    pairs: dict[str, dict[str, Outcome]] = {}
    with path.open('r', encoding='utf-8-sig', newline='') as f:
        rdr = csv.DictReader(f)
        required = {'frame', 'mode', 'latency_ms', 'TP', 'FP', 'FN'}
        absent = required - set(rdr.fieldnames or [])
        if absent:
            raise ValueError(f'Benchmark columns missing: {sorted(absent)}')
        for lineno, row in enumerate(rdr, 2):
            name = (row['frame'] or '').strip()
            parse_clip(name)
            mode = (row['mode'] or '').strip().upper()
            if mode not in MODES:
                raise ValueError(f'Line {lineno}: invalid mode {mode!r}')
            try:
                latency = float(row['latency_ms'])
            except (TypeError, ValueError) as exc:
                raise ValueError(f'Line {lineno}: invalid latency') from exc
            if not (math.isfinite(latency) and latency > 0):
                raise ValueError(f'Line {lineno}: non-positive or nonfinite latency')
            record = Outcome(*[_int(row[k], k) for k in ('TP', 'FP', 'FN')], latency)
            if mode in pairs.setdefault(name, {}):
                raise ValueError(f'Line {lineno}: duplicate frame/mode {name}/{mode}')
            pairs[name][mode] = record
    if not pairs:
        raise ValueError('Benchmark CSV contains no rows')
    for filename, modes in pairs.items():
        if set(modes) != set(MODES):
            raise ValueError(f'{filename}: FAST and ROBUST must both be present')
        n_fast = modes['FAST'].tp + modes['FAST'].fn
        n_robust = modes['ROBUST'].tp + modes['ROBUST'].fn
        if n_fast != n_robust:
            raise ValueError(f'{filename}: modes disagree on ground-truth count ({n_fast} vs {n_robust})')
    return pairs


def summarize(outcomes: list[Outcome]) -> dict:
    if not outcomes:
        raise ValueError('Cannot summarize empty outcomes')
    tp = sum(x.tp for x in outcomes)
    fp = sum(x.fp for x in outcomes)
    fn = sum(x.fn for x in outcomes)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    latencies = sorted(x.latency_ms for x in outcomes)
    p95 = latencies[math.ceil(0.95 * len(latencies)) - 1]
    avg = mean(latencies)
    return {
        'frames': len(outcomes), 'TP': tp, 'FP': fp, 'FN': fn,
        'precision_iou50': round(precision, 5),
        'recall_iou50': round(recall, 5),
        'f1_iou50': round(f1, 5),
        'mean_model_call_latency_ms': round(avg, 3),
        'median_model_call_latency_ms': round(median(latencies), 3),
        'p95_model_call_latency_ms': round(p95, 3),
        'model_call_fps_equivalent': round(1000.0 / avg, 2),
    }
