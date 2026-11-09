"""Train a small OFFLINE mode-choice classifier on archived validation data.

Default study split: learn labels from Clip 7; evaluate on unseen Clip 8.
Does not operate on live feeds, control vehicles, or generate tracking IDs.
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

from features import FEATURE_COLUMNS, clip_number, compute_features
from modes import read_pairs, preferred_mode

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-clip', type=int, default=7)
    parser.add_argument('--eval-clip', type=int, default=8)
    parser.add_argument('--benchmark', type=Path,
                        default=ROOT / 'results/offline_benchmark/per_frame.csv')
    parser.add_argument('--images', type=Path,
                        default=ROOT / 'datasets/prepared/images/val')
    args = parser.parse_args()
    if args.train_clip == args.eval_clip:
        parser.error('Training and evaluation clips must be different')
    if args.train_clip in (9, 10) or args.eval_clip in (9, 10):
        parser.error('Clips 9 and 10 are reserved test clips; use validation clips here')
    try:
        from sklearn.dummy import DummyClassifier
        from sklearn.tree import DecisionTreeClassifier
        import joblib
    except ImportError as exc:
        raise SystemExit('Install the missing ML dependency: python -m pip install scikit-learn joblib') from exc

    pairs = read_pairs(args.benchmark)
    train_names = sorted(n for n in pairs if clip_number(n) == args.train_clip)
    eval_names = sorted(n for n in pairs if clip_number(n) == args.eval_clip)
    if not train_names or not eval_names:
        raise SystemExit(f'Benchmark needs both Clip {args.train_clip} and Clip {args.eval_clip}. '
                         'Check per_frame.csv filenames and whether your benchmark used validation images.')
    features = compute_features(args.images, train_names + eval_names)
    x_train = [features[n] for n in train_names]
    y_train = [preferred_mode(pairs[n]) for n in train_names]
    x_eval = [features[n] for n in eval_names]
    y_eval = [preferred_mode(pairs[n]) for n in eval_names]  # Reference-only score, NOT model input
    counts = Counter(y_train)

    if len(counts) < 2:
        classifier = DummyClassifier(strategy='most_frequent')
        print('NOTE: Training labels contain only one class; using constant selector.')
    else:
        classifier = DecisionTreeClassifier(max_depth=3, min_samples_leaf=8,
                                            random_state=42)
    classifier.fit(x_train, y_train)
    predictions = [str(p) for p in classifier.predict(x_eval)]
    out = ROOT / 'results/selector'
    out.mkdir(parents=True, exist_ok=True)
    joblib.dump(classifier, out / 'selector_model.joblib')
    with (out / 'mode_decisions.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=['frame', 'eval_clip', 'selected_mode', 'reference_preference'] + list(FEATURE_COLUMNS))
        writer.writeheader()
        for name, choice, reference in zip(eval_names, predictions, y_eval):
            writer.writerow({'frame': name, 'eval_clip': args.eval_clip, 'selected_mode': choice,
                             'reference_preference': reference,
                             **dict(zip(FEATURE_COLUMNS, [round(x, 5) for x in features[name]]))})

    meta = {'train_clip': args.train_clip, 'eval_clip': args.eval_clip,
            'training_frames': len(train_names), 'evaluation_frames': len(eval_names),
            'training_labels': dict(counts), 'selected_modes': dict(Counter(predictions)),
            'description': 'OFFLINE still-frame experiment; no track IDs, no streaming, no end-to-end FPS'}
    (out / 'experiment.json').write_text(json.dumps(meta, indent=2), encoding='utf-8')
    correct = sum(p == a for p, a in zip(predictions, y_eval))
    always_fast_correct = sum(a == 'FAST' for a in y_eval)
    print(f'Train: Clip {args.train_clip} ({len(train_names)} frames); labels: {dict(counts)}')
    print(f'Evaluate: Clip {args.eval_clip} ({len(eval_names)} held-out frames)')
    print(f'Selected modes: {dict(Counter(predictions))}')
    print(f'Agreement with reference preference: {correct}/{len(y_eval)} = {correct/len(y_eval):.1%}')
    print(f'Always-FAST reference agreement: {always_fast_correct}/{len(y_eval)} = {always_fast_correct/len(y_eval):.1%}')
    print('NOTE: These labels are based on offline reference comparisons, NOT tracking or deployment outcomes.')
    print(f'Saved: {out / "mode_decisions.csv"}')
    print('Next: python src\\adaptive.py')


if __name__ == '__main__':
    main()
