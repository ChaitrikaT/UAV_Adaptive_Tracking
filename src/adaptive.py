"""OFFLINE replay of saved mode decisions against premeasured detector outcomes.

This is a counterfactual comparison on archived still frames only. It is NOT an
operational selector, a tracker, or an end-to-end video throughput measurement.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
from modes import read_pairs, aggregate
from features import clip_number

ROOT = Path(__file__).resolve().parents[1]


def main():
    out = ROOT / 'results/selector'
    decisions_path = out / 'mode_decisions.csv'
    meta_path = out / 'experiment.json'
    if not decisions_path.is_file() or not meta_path.is_file():
        raise SystemExit('Run python src\\train_selector.py first.')
    meta = json.loads(meta_path.read_text(encoding='utf-8'))
    benchmark = ROOT / 'results/offline_benchmark/per_frame.csv'
    pairs = read_pairs(benchmark)
    with decisions_path.open(newline='', encoding='utf-8') as f:
        decisions = list(csv.DictReader(f))
    if not decisions:
        raise SystemExit('No saved decisions.')
    eval_clip = int(meta['eval_clip'])
    if eval_clip == int(meta['train_clip']):
        raise ValueError('Evaluation clip overlaps training clip')
    rows_fast, rows_robust, rows_selected = [], [], []
    for d in decisions:
        name = d['frame']
        if clip_number(name) != eval_clip or name not in pairs:
            raise ValueError(f'Decision frame is missing or from wrong clip: {name}')
        chosen = d['selected_mode']
        if chosen not in ('FAST', 'ROBUST'):
            raise ValueError(f'Invalid mode: {chosen}')
        rows_fast.append(pairs[name]['FAST'])
        rows_robust.append(pairs[name]['ROBUST'])
        rows_selected.append(pairs[name][chosen])
    results = [('ALWAYS_FAST', aggregate(rows_fast)),
               ('ALWAYS_ROBUST', aggregate(rows_robust)),
               ('LEARNED_OFFLINE_REPLAY', aggregate(rows_selected))]
    out.mkdir(parents=True, exist_ok=True)
    with (out / 'comparison.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['policy'] + list(results[0][1].keys()))
        writer.writeheader()
        for policy, scores in results:
            writer.writerow({'policy': policy, **scores})
    lines = [f'OFFLINE MODE COMPARISON — CLIP {eval_clip}',
             f'Trained on Clip {meta["train_clip"]}; evaluated on different Clip {eval_clip}',
             'All policies are replayed against the same already-measured still frames.', '']
    for policy, scores in results:
        lines.append(f'{policy}: {scores}')
    lines += ['', 'CAUTIONS:',
              '- GT was used to construct training labels and to score evaluation only; never to compute input features.',
              '- Recorded model-call latency EXCLUDES feature extraction, decoding, and switching overhead.',
              '- This is not ByteTrack, tracking accuracy, or real-time video FPS.',
              '- One-clip training and one-clip evaluation cannot establish generalization.',
              '- If the learned policy does not improve over both fixed modes, report that honestly.']
    (out / 'report.txt').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print('\n'.join(lines))
    print(f'\nSaved: {out / "comparison.csv"}')


if __name__ == '__main__':
    main()
