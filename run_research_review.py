"""Single command for a retrospective, auditable evaluation of saved results.

Usage:
    python run_research_review.py
    python run_research_review.py --example
    python run_research_review.py --tracks path/to/predicted_tracks.csv --reference data/annotations/Clip_7_refined.txt --clip 7
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from research_addon.policy_audit import audit
from research_addon.track_audit import score_tracks
from research_addon.report import write_report
from research_addon.timeline import join_timeline

ROOT = Path(__file__).resolve().parent


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root', type=Path, default=ROOT)
    ap.add_argument('--example', action='store_true', help='Use bundled invented toy example instead of project results')
    ap.add_argument('--benchmark', type=Path, default=None)
    ap.add_argument('--decisions', type=Path, default=None)
    ap.add_argument('--experiment', type=Path, default=None)
    ap.add_argument('--tracks', type=Path, default=None, help='Independently generated saved track boxes (does not generate tracks)')
    ap.add_argument('--reference', type=Path, default=None, help='MOT reference TXT file for the same clip')
    ap.add_argument('--clip', type=int, default=None)
    ap.add_argument('--timeline-clip', type=int, default=7, help='Join existing video diagnostic with condition CSV if both exist')
    ap.add_argument('--out', type=Path, default=None)
    args = ap.parse_args()
    root = args.root.resolve()
    if args.example:
        benchmark = ROOT / 'examples' / 'toy_per_frame.csv'
        decisions = ROOT / 'examples' / 'toy_mode_decisions.csv'
        experiment = ROOT / 'examples' / 'toy_experiment.json'
        output = args.out or root / 'results' / 'research_review_example'
    else:
        benchmark = args.benchmark or root / 'results' / 'offline_benchmark' / 'per_frame.csv'
        decisions = args.decisions or root / 'results' / 'selector' / 'mode_decisions.csv'
        experiment = args.experiment or root / 'results' / 'selector' / 'experiment.json'
        output = args.out or root / 'results' / 'research_review'
    if args.tracks is not None and (args.reference is None or args.clip is None):
        ap.error('--tracks must be accompanied by --reference and --clip')
    if (args.reference is not None or args.clip is not None) and args.tracks is None:
        ap.error('--reference and --clip are only used with --tracks')
    selector_review = None
    if benchmark.is_file() and decisions.is_file():
        selector_review = audit(benchmark, decisions, experiment)
    else:
        print('NOTE: No paired saved benchmark + decisions found. Selector audit skipped.')
        print('Expected:', benchmark, 'and', decisions)
    track_review = None
    if args.tracks is not None:
        track_review = score_tracks(args.tracks, args.reference, args.clip)
    timeline_review = None
    if not args.example:
        diagnostic = root / 'results' / 'videos' / f'clip_{args.timeline_clip}_detector_review.csv'
        condition = root / 'results' / 'conditions' / f'clip_{args.timeline_clip}_condition_analysis.csv'
        if diagnostic.is_file() and condition.is_file():
            timeline_review = join_timeline(diagnostic, condition, output / 'timeline_review.csv')
            print('Saved existing video/condition timeline:', output / 'timeline_review.csv')
    results = {'selector_review': selector_review, 'track_review': track_review, 'timeline_review': timeline_review}
    write_report(output, results, {
        'benchmark': benchmark, 'decisions': decisions, 'experiment': experiment,
        'predicted_tracks': args.tracks, 'reference_tracks': args.reference,
        'video_diagnostic': root / 'results' / 'videos' / f'clip_{args.timeline_clip}_detector_review.csv',
        'condition_analysis': root / 'results' / 'conditions' / f'clip_{args.timeline_clip}_condition_analysis.csv',
    })
    print('Research report:', output / 'research_report.md')
    print('Local HTML:', output / 'index.html')
    print('Machine-readable review:', output / 'research_metrics.json')
    if selector_review:
        print('Evaluation clip:', selector_review['evaluation_clip'])
        for policy, score in selector_review['comparison'].items():
            print(f'{policy}: recall={score["recall_iou50"]:.1%}, '
                  f'mean model-call latency={score["mean_model_call_latency_ms"]:.3f} ms')
    if track_review:
        print('Saved track ID review:', json.dumps(track_review, indent=2))


if __name__ == '__main__':
    main()
