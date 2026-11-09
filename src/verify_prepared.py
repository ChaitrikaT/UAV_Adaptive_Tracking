"""Render a few of the prepared detection labels for visual inspection."""
from pathlib import Path
import argparse
import cv2

ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--count', type=int, default=3)
    args = ap.parse_args()
    data = ROOT / 'datasets' / 'prepared'
    out = ROOT / 'results' / 'prepared_checks'
    out.mkdir(parents=True, exist_ok=True)
    total = 0
    for group in ('train', 'val', 'test'):
        candidates = sorted((data / 'images' / group).glob('*.jpg'))
        # Use samples from the start, middle and end where possible.
        positions = sorted(set([0, len(candidates)//2, len(candidates)-1])) if candidates else []
        for idx in positions[:args.count]:
            img_path = candidates[idx]
            label_path = data / 'labels' / group / (img_path.stem + '.txt')
            if not label_path.is_file():
                raise FileNotFoundError(label_path)
            frame = cv2.imread(str(img_path))
            if frame is None:
                raise RuntimeError(f'Could not read {img_path}')
            H, W = frame.shape[:2]
            for line in label_path.read_text(encoding='utf-8').splitlines():
                cls, xc, yc, bw, bh = map(float, line.split())
                if int(cls) != 0 or not all(0 <= v <= 1 for v in (xc, yc, bw, bh)):
                    raise ValueError(f'Bad YOLO label: {label_path}: {line}')
                x1, y1 = int((xc - bw/2)*W), int((yc - bh/2)*H)
                x2, y2 = int((xc + bw/2)*W), int((yc + bh/2)*H)
                cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)
            output = out / f'{group}_{img_path.name}'
            cv2.imwrite(str(output), frame)
            total += 1
            print('Saved:', output)
    print(f'\nVisual checks: {total} images. Open these and inspect boxes manually.')


if __name__ == '__main__':
    main()
