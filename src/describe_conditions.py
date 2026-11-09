"""Describe visual conditions on stored test frames, for offline research only.

GT target size is retrospective reference metadata, NOT a test-time ML input.
"""
from pathlib import Path
import csv
import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'datasets' / 'prepared'
OUT = ROOT / 'results' / 'condition_features.csv'
OUT.parent.mkdir(parents=True, exist_ok=True)
rows = []

for split in ('train', 'val', 'test'):
    for image_path in sorted((DATA / 'images' / split).glob('*.jpg')):
        frame = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        if frame is None:
            raise RuntimeError(f'Cannot load {image_path}')
        lap_var = float(cv2.Laplacian(frame, cv2.CV_64F).var())
        gt_path = DATA / 'labels' / split / (image_path.stem + '.txt')
        gt_sizes = []
        for line in gt_path.read_text(encoding='utf-8').splitlines():
            pieces = line.split()
            if len(pieces) != 5:
                raise ValueError(f'Bad label line in {gt_path}: {line}')
            _, _, _, bw, bh = map(float, pieces)
            gt_sizes.append(bw * bh)
        rows.append({
            'split': split, 'image': image_path.name,
            'mean_brightness': round(float(frame.mean()), 3),
            'contrast_std': round(float(frame.std()), 3),
            'blur_laplacian_variance': round(lap_var, 3),
            'gt_targets': len(gt_sizes),
            'gt_smallest_area_ratio_REFERENCE_ONLY': round(min(gt_sizes), 8) if gt_sizes else '',
        })

if not rows:
    raise RuntimeError('No prepared images. Run prepare_dataset.py first.')
with OUT.open('w', newline='', encoding='utf-8') as fh:
    w = csv.DictWriter(fh, fieldnames=rows[0].keys())
    w.writeheader(); w.writerows(rows)
print('Saved:', OUT)
print(f'Rows: {len(rows)}')
print('IMPORTANT: ground-truth size is for retrospective analysis only, not a valid test-time selector input.')
