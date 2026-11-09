"""Make simple charts from already measured FAST/ROBUST offline results."""
from pathlib import Path
import csv
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
folder = ROOT / 'results' / 'offline_benchmark'
file = folder / 'summary.csv'
if not file.is_file():
    raise FileNotFoundError(f'Run benchmark_modes.py first: {file}')
with file.open(newline='', encoding='utf-8') as fh:
    rows = list(csv.DictReader(fh))
if not rows:
    raise RuntimeError('No benchmark results found')

for column, ylabel, filename in [
    ('mean_latency_ms', 'Mean inference-call latency (ms)', 'latency_comparison.png'),
    ('fps_equivalent', 'Estimated model-call throughput (images/s)', 'throughput_comparison.png'),
]:
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar([r['mode'] for r in rows], [float(r[column]) for r in rows])
    ax.set_ylabel(ylabel)
    ax.set_title('Offline recorded-frame benchmark')
    fig.tight_layout()
    target = folder / filename
    fig.savefig(target, dpi=160)
    plt.close(fig)
    print('Saved:', target)

if 'recall_iou50' in rows[0]:
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar([r['mode'] for r in rows], [float(r['recall_iou50']) for r in rows])
    ax.set_ylim(0, 1)
    ax.set_ylabel('IoU@0.5 recall on annotated frames')
    ax.set_title('Offline localization (UAV-labelled model)')
    fig.tight_layout()
    target = folder / 'localization_recall.png'
    fig.savefig(target, dpi=160)
    plt.close(fig)
    print('Saved:', target)
