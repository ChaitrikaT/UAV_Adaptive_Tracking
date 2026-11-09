"""Build a one-page OFFLINE experimental review from saved, completed outputs.

Never runs a detector, never creates tracks, never controls any vehicle.
Usage, from repository root: python src/finalize_project.py
"""
from __future__ import annotations

from collections import Counter
from datetime import datetime
from html import escape
from pathlib import Path
import csv
import json
import re

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / 'results'
DEST = RESULTS / 'final'
DEST.mkdir(parents=True, exist_ok=True)


def csv_rows(path: Path, needed: set[str]) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f'Required experiment file missing: {path}')
    with path.open('r', newline='', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        fields = {str(s).strip() for s in (reader.fieldnames or [])}
        if not needed.issubset(fields):
            raise ValueError(f'{path.name} lacks columns: {sorted(needed-fields)}; found {sorted(fields)}')
        return [dict(r) for r in reader]


def to_number(v: str) -> float:
    return float(str(v).strip())


def clip_from_filename(name: str) -> str:
    match = re.search(r'Clip_(\d+)_frame_', name, flags=re.IGNORECASE)
    return 'Clip ' + match.group(1) if match else 'Unknown clip'


def audit_timeline(rows: list[dict[str, str]]) -> dict[str, object]:
    data = sorted((int(r['frame']), int(r['reference_boxes']), int(r['predicted_boxes']),
                   float(r['max_iou'])) for r in rows)
    if not data:
        raise ValueError('Detector video CSV is empty')
    if len({a for a, *_ in data}) != len(data):
        raise ValueError('Duplicate frame numbers in detector review CSV')
    matched = [int(iou >= .5) for _, _, _, iou in data]
    with_preds = sum(p > 0 for _, _, p, _ in data)
    longest, start, end, current, current_start, lastframe = 0, None, None, 0, None, None
    for frame, _, _, iou in data:
        if iou >= .5:
            current = 0
        else:
            if current == 0 or (lastframe is not None and frame != lastframe + 1):
                current, current_start = 0, frame
            current += 1
            if current > longest:
                longest, start, end = current, current_start, frame
        lastframe = frame
    return {'data': data, 'frames': len(data), 'matched': sum(matched),
            'with_preds': with_preds, 'longest': longest, 'gap_start': start, 'gap_end': end,
            'first': data[0][0], 'last': data[-1][0]}


def make_charts(timeline: dict[str, object], policies: list[dict[str, str]]) -> list[str]:
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print('WARNING: matplotlib not installed; skipping charts (HTML/report still generated).')
        return []
    data = timeline['data']
    names = []
    fig, ax = plt.subplots(figsize=(12, 2.4))
    xs = [f for f, _, _, _ in data]
    ys = [int(iou >= .5) for _, _, _, iou in data]
    ax.step(xs, ys, where='mid', color='#2563eb', linewidth=1.7)
    ax.set(xlim=(xs[0], xs[-1]), ylim=(-.12, 1.18), title='Clip 7: detection matches across consecutive frames',
           ylabel='IoU ≥ 0.5', xlabel='Video frame')
    ax.set_yticks([0, 1]); ax.set_yticklabels(['No', 'Yes'])
    ax.grid(axis='x', alpha=.25)
    fig.tight_layout(); fig.savefig(DEST / 'clip7_detection_timeline.png', dpi=150); plt.close(fig)
    names.append('clip7_detection_timeline.png')

    if policies:
        labels = [r['policy'].replace('LEARNED_OFFLINE_REPLAY','Learned replay').replace('ALWAYS_','Always ').title() for r in policies]
        recalls = [100 * to_number(r['recall_iou50']) for r in policies]
        latencies = [to_number(r['mean_model_call_latency_ms']) for r in policies]
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.barh(labels, recalls, color='#2563eb')
        ax.set(xlabel='Recall at IoU 0.5 (%)', title='Clip 8: offline mode-policy replay — detection recall')
        ax.set_xlim(0, max(recalls or [1]) * 1.3 + .1)
        for y, val in enumerate(recalls): ax.text(val + .2, y, f'{val:.2f}%', va='center', fontsize=9)
        fig.tight_layout(); fig.savefig(DEST / 'policy_recall.png', dpi=150); plt.close(fig)
        names.append('policy_recall.png')

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.barh(labels, latencies, color='#64748b')
        ax.set(xlabel='Model-call latency (ms)', title='Clip 8: latency excludes decoding, features, and switching')
        ax.set_xlim(0, max(latencies or [1]) * 1.30 + .1)
        for y, val in enumerate(latencies): ax.text(val + .3, y, f'{val:.2f} ms', va='center', fontsize=9)
        fig.tight_layout(); fig.savefig(DEST / 'policy_latency.png', dpi=150); plt.close(fig)
        names.append('policy_latency.png')
    return names


def main():
    video_csv = RESULTS / 'videos' / 'clip_7_detector_review.csv'
    policies_csv = RESULTS / 'selector' / 'comparison.csv'
    benchmark_csv = RESULTS / 'offline_benchmark' / 'per_frame.csv'
    conditions_txt = RESULTS / 'conditions' / 'clip_7_condition_summary.txt'
    video_mp4 = RESULTS / 'videos' / 'clip_7_detector_review.mp4'
    model_pt = RESULTS / 'training' / 'uav_detector_v1' / 'weights' / 'best.pt'

    timeline = audit_timeline(csv_rows(video_csv, {'frame', 'reference_boxes', 'predicted_boxes', 'max_iou'}))
    policies = csv_rows(policies_csv, {'policy', 'frames', 'TP', 'FP', 'FN',
                                     'precision_iou50', 'recall_iou50', 'mean_model_call_latency_ms'})
    expected_policies = {'ALWAYS_FAST', 'ALWAYS_ROBUST', 'LEARNED_OFFLINE_REPLAY'}
    present = {p['policy'] for p in policies}
    if not expected_policies.issubset(present):
        raise ValueError('Missing policies in selector comparison: ' + str(expected_policies-present))
    policies = sorted((p for p in policies if p['policy'] in expected_policies),
                      key=lambda p: ['ALWAYS_FAST', 'ALWAYS_ROBUST', 'LEARNED_OFFLINE_REPLAY'].index(p['policy']))

    warnings = []
    benchmark_clips = {}
    if benchmark_csv.exists():
        samples = csv_rows(benchmark_csv, {'frame', 'mode'})
        benchmark_clips = dict(sorted(Counter(clip_from_filename(r['frame']) for r in samples).items()))
        if set(benchmark_clips) != {'Clip 7', 'Clip 8'}:
            warnings.append('Benchmark input clips are not exactly Clips 7 and 8; confirm split before describing results as validation-only.')
    else:
        warnings.append('per_frame.csv missing; cannot independently verify the benchmark split.')
    if not conditions_txt.exists():
        warnings.append('Condition summary missing; do not claim that condition analysis was generated by this report.')
    if not video_mp4.exists():
        warnings.append('Clip 7 review MP4 missing; video player will not load.')
    if not model_pt.exists():
        warnings.append('best.pt missing at expected path; model artifact may have moved.')

    charts = make_charts(timeline, policies)
    comparison = []
    for r in policies:
        comparison.append({
            'policy': r['policy'], 'frames': int(r['frames']),
            'tp': int(r['TP']), 'fp': int(r['FP']), 'fn': int(r['FN']),
            'precision_pct': round(to_number(r['precision_iou50'])*100, 2),
            'recall_pct': round(to_number(r['recall_iou50'])*100, 2),
            'latency_ms': round(to_number(r['mean_model_call_latency_ms']), 3),
        })
    best_recall = max(comparison, key=lambda d: d['recall_pct'])['policy']
    learned_is_best = best_recall == 'LEARNED_OFFLINE_REPLAY'
    observed = {
        'video_clip': 7, 'frames': timeline['frames'], 'frames_with_detections': timeline['with_preds'],
        'frames_with_iou_at_least_05': timeline['matched'],
        'frame_match_rate_pct': round(100*timeline['matched']/timeline['frames'], 2),
        'longest_unmatched_run': {'start': timeline['gap_start'], 'end': timeline['gap_end'], 'frames': timeline['longest']},
        'selector_comparison': comparison, 'replay_training_clip': 7, 'replay_evaluation_clip': 8,
        'benchmark_clips': benchmark_clips, 'warnings': warnings,
        'capabilities_not_demonstrated': ['Independent predicted tracking IDs',
                                         'End-to-end adaptive video processing',
                                         'IDF1/HOTA/MOTA tracking scores',
                                         'Real-time video throughput'],
    }
    (DEST / 'verified_metrics.json').write_text(json.dumps(observed, indent=2), encoding='utf-8')

    lines = [
        '# Mini-project: offline experimental status', '',
        '## What works',
        '- Dataset prepared and a lightweight one-class UAV detector fine-tuned on GPU.',
        f'- Recorded Clip 7 detector review: {timeline["matched"]}/{timeline["frames"]} frames match at IoU ≥ 0.5 '
        f'({100*timeline["matched"]/timeline["frames"]:.2f}%).',
        f'- Longest contiguous period without a match: frames {timeline["gap_start"]}–{timeline["gap_end"]} '
        f'({timeline["longest"]} frames).',
        '- Global image features extracted; the observed averages did not clearly separate matched and unmatched frames.',
        '- Fixed-mode and learned mode-policy offline replay evaluated on archived sampled frames.', '',
        '## Offline policy comparison (Clip 8)', '',
        '| Policy | TP | FP | FN | Precision | Recall | Model-call latency |',
        '|---|---:|---:|---:|---:|---:|---:|',
    ]
    for d in comparison:
        lines.append(f'| {d["policy"]} | {d["tp"]} | {d["fp"]} | {d["fn"]} | '
                     f'{d["precision_pct"]:.2f}% | {d["recall_pct"]:.2f}% | {d["latency_ms"]:.3f} ms |')
    lines.extend(['',
        '**Conclusion:** ' + ('The learned replay has the highest recall among these policies.' if learned_is_best else
                              'The learned replay does not achieve the highest recall; do not claim it improves on always ROBUST.'),
        '', '## Important limitations',
        '- The demo video shows detector predictions versus reference boxes, **not predicted tracking identities**.',
        '- The selection experiment replays saved still-frame outcomes; it is **not adaptive video tracking**.',
        '- Model-call latency excludes frame decoding, feature computation, switching, and tracker cost.',
        '- Clip 7 and Clip 8 offer limited sequence-level evidence. The split and any other reserved test clips must remain documented.',
        '- These results do not demonstrate reliable real-time operation.',
        '', '## Guide demonstration (in order)',
        '1. Open `results/videos/clip_7_detector_review.mp4` (green = reference, magenta = predicted).',
        '2. Show `results/final/clip7_detection_timeline.png` to explain the long failure interval.',
        '3. Show `results/final/policy_recall.png` and `results/final/policy_latency.png` for measured comparisons.',
        '4. Explain honestly that independently predicted track IDs and integrated adaptive video evaluation are missing.',
        '', '## Source paths',
        f'- Detector review: `{video_csv.relative_to(ROOT)}`',
        f'- Selector comparison: `{policies_csv.relative_to(ROOT)}`',
    ])
    if benchmark_clips:
        lines.append(f'- Benchmark clip counts (rows across both modes): `{benchmark_clips}`')
    if warnings:
        lines.extend(['', '## Data checks / warnings'] + [f'- {w}' for w in warnings])
    (DEST / 'guide_report.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')

    policy_table = ''.join('<tr>'+''.join(f'<td>{escape(str(d[k]))}</td>' for k in
                             ['policy','tp','fp','fn','precision_pct','recall_pct','latency_ms'])+'</tr>'
                           for d in comparison)
    issues = ''.join(f'<li>{escape(w)}</li>' for w in warnings) or '<li>No additional file/split warnings.</li>'
    imgs = ''.join(f'<figure><img src="{escape(n)}" alt="Offline experiment chart"></figure>' for n in charts)
    video_html = ('<video controls preload="metadata" width="100%" src="../videos/clip_7_detector_review.mp4"></video>'
                  if video_mp4.exists() else '<p>Video was not found at expected path.</p>')
    html = f'''<!doctype html><html lang="en"><head><meta charset="utf-8">
    <title>UAV mini-project: offline results</title><style>
    body{{font:16px/1.55 system-ui,Arial,sans-serif;background:#f7fafc;color:#14213d;max-width:1020px;margin:35px auto;padding:0 20px}}
    .card{{background:white;border:1px solid #dee5ee;border-radius:14px;padding:24px;margin:20px 0}}
    h1,h2{{line-height:1.25}}h1{{font-size:30px}}h2{{font-size:21px}}
    .kpis{{display:flex;flex-wrap:wrap;gap:14px}}.kpis div{{flex:1;min-width:160px;background:#edf3fe;border-radius:10px;padding:14px}}
    .kpis b{{display:block;font-size:24px}}table{{border-collapse:collapse;width:100%;font-size:14px}}
    th,td{{padding:9px;border-bottom:1px solid #e1e7ee;text-align:left}}img{{max-width:100%;height:auto}}figure{{margin:0 0 24px}}
    .warn{{border-left:4px solid #d97706;padding-left:13px}}.note{{color:#475569}}
    </style></head><body>
    <h1>Condition-Aware Adaptive Perception — Offline Experimental Review</h1>
    <p class="note">Recorded-video detector baseline and archived still-frame mode-choice experiment. Not a deployed or complete tracking system.</p>
    <section class="card"><h2>Clip 7 detection continuity</h2>
    <div class="kpis"><div><b>{timeline['frames']}</b>frames reviewed</div>
    <div><b>{timeline['with_preds']}</b>frames with predictions</div>
    <div><b>{timeline['matched']}</b>frames with IoU ≥ 0.5</div>
    <div><b>{timeline['longest']}</b>longest unmatched run (frames)</div></div>
    <p>Longest unmatched interval: frames {timeline['gap_start']}–{timeline['gap_end']}.</p>{video_html}
    <p class="note">Green boxes are Purdue reference annotations; magenta boxes are independent detector predictions. No predicted tracking IDs are shown.</p></section>
    <section class="card"><h2>Learned mode replay (Clip 8)</h2>
    <table><thead><tr><th>Policy</th><th>TP</th><th>FP</th><th>FN</th><th>Precision %</th><th>Recall %</th><th>Latency ms</th></tr></thead>
    <tbody>{policy_table}</tbody></table>
    <p><strong>{'Learned replay leads this set on recall.' if learned_is_best else 'Always ROBUST has better recall than learned replay.'}</strong>
    These are archived still-image results, not video-tracking metrics.</p></section>
    <section class="card"><h2>Charts</h2>{imgs}</section>
    <section class="card"><h2>What remains missing</h2>
    <ul><li>Independently generated predicted track IDs</li>
    <li>Integrated frame-by-frame adaptive processing evaluated on a video sequence</li>
    <li>Formal tracking performance metrics</li></ul></section>
    <section class="card warn"><h2>Data provenance and cautions</h2>
    <p>Benchmark clip rows: {escape(str(benchmark_clips))}</p><ul>{issues}</ul>
    <p>Latency excludes frame decoding, feature extraction and switching overhead.</p></section>
    </body></html>'''
    (DEST / 'index.html').write_text(html, encoding='utf-8')

    print('\nFINAL OFFLINE PROJECT REVIEW')
    print(f"Clip 7 IoU≥0.5 matches: {timeline['matched']}/{timeline['frames']}")
    print(f"Longest gap: {timeline['gap_start']}–{timeline['gap_end']} ({timeline['longest']} frames)")
    print('Selector recall: ' + ', '.join(f"{d['policy']}={d['recall_pct']:.2f}%" for d in comparison))
    print('Recorded benchmark clip counts:', benchmark_clips)
    for w in warnings: print('WARNING:', w)
    print('Generated:', DEST / 'index.html')
    print('Generated:', DEST / 'guide_report.md')
    print('Generated:', DEST / 'verified_metrics.json')
    print('Generated charts:', ', '.join(charts) if charts else 'none')
    print('TRACKING STATUS: predicted tracking IDs NOT YET IMPLEMENTED / VERIFIED.')


if __name__ == '__main__':
    main()
