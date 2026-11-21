"""Retrospective comparison of archived mode decisions against paired outcomes."""
from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path
from .benchmark import read_pairs, parse_clip, summarize


def read_decisions(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f'Missing selector decisions: {path}')
    with path.open(encoding='utf-8-sig', newline='') as f:
        rdr = csv.DictReader(f)
        required = {'frame', 'selected_mode'}
        missing = required - set(rdr.fieldnames or [])
        if missing:
            raise ValueError(f'Decision columns missing: {sorted(missing)}')
        rows = list(rdr)
    if not rows:
        raise ValueError('Mode decisions CSV is empty')
    return rows


def audit(benchmark_path: Path, decision_path: Path, experiment_path: Path | None) -> dict:
    pairs = read_pairs(benchmark_path)
    decisions = read_decisions(decision_path)
    experiment = None
    if experiment_path and experiment_path.is_file():
        experiment = json.loads(experiment_path.read_text(encoding='utf-8-sig'))
        if int(experiment['train_clip']) == int(experiment['eval_clip']):
            raise ValueError('Experiment train and evaluation clips are identical')
    seen = set()
    outcomes_fast, outcomes_robust, outcomes_learned = [], [], []
    selected = Counter()
    clips = set()
    for row in decisions:
        frame = (row.get('frame') or '').strip()
        if frame in seen:
            raise ValueError(f'Duplicate decision for frame {frame}')
        seen.add(frame)
        if frame not in pairs:
            raise ValueError(f'Decision frame missing from benchmark: {frame}')
        mode = (row.get('selected_mode') or '').strip().upper()
        if mode not in ('FAST', 'ROBUST'):
            raise ValueError(f'Unexpected selected mode: {mode}')
        clip = parse_clip(frame)
        clips.add(clip)
        if experiment and clip != int(experiment['eval_clip']):
            raise ValueError(f'{frame}: does not belong to declared evaluation clip {experiment["eval_clip"]}')
        # Reference preferences in the CSV are retrospective labels, never inputs.
        outcomes_fast.append(pairs[frame]['FAST'])
        outcomes_robust.append(pairs[frame]['ROBUST'])
        outcomes_learned.append(pairs[frame][mode])
        selected[mode] += 1
    if len(clips) != 1:
        raise ValueError(f'Expected a single evaluation clip, found {sorted(clips)}')
    eval_clip = next(iter(clips))
    if experiment and int(experiment['train_clip']) == eval_clip:
        raise ValueError('Selector training and evaluation clips overlap')
    coverage = len(decisions) / sum(parse_clip(name) == eval_clip for name in pairs)
    if coverage < 1.0:
        raise ValueError(f'Only {len(decisions)} decisions for evaluation clip {eval_clip} (coverage {coverage:.0%})')
    outcomes = {
        'ALWAYS_FAST': summarize(outcomes_fast),
        'ALWAYS_ROBUST': summarize(outcomes_robust),
        'LEARNED_OFFLINE_REPLAY': summarize(outcomes_learned),
    }
    robust = outcomes['ALWAYS_ROBUST']
    learned = outcomes['LEARNED_OFFLINE_REPLAY']
    beats_robust_recall = learned['recall_iou50'] > robust['recall_iou50']
    beats_robust_latency = learned['mean_model_call_latency_ms'] < robust['mean_model_call_latency_ms']
    return {
        'evaluation_clip': eval_clip,
        'train_clip': int(experiment['train_clip']) if experiment else None,
        'sampled_frames': len(decisions),
        'mode_counts': dict(selected),
        'measurement_scope': 'Saved still-frame detector calls; not video FPS, tracking, or online adaptation',
        'comparison': outcomes,
        'learned_better_recall_than_robust': beats_robust_recall,
        'learned_lower_model_latency_than_robust': beats_robust_latency,
        'remarks': [
            'The evaluation clip was held out from selector training, but may have been used for detector validation.',
            'Feature, image decoding, switching and tracking overhead are excluded from these timings.',
            'Do not infer the selector improves tracking or online perception from these replayed results.',
        ],
    }
