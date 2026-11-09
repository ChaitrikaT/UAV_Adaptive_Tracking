"""Prepare a small, offline Purdue image/label dataset without altering sources."""
from pathlib import Path
from collections import defaultdict
import argparse
import csv
import shutil
import cv2

ROOT = Path(__file__).resolve().parents[1]
SPLITS = {'train': range(1, 7), 'val': range(7, 9), 'test': range(9, 11)}


def read_boxes(path):
    frames = defaultdict(list)
    with path.open('r', encoding='utf-8-sig', errors='replace', newline='') as fh:
        for row in csv.reader(fh):
            if len(row) < 6:
                continue
            try:
                number = int(float(row[0]))
                x, y, w, h = map(float, row[2:6])
            except (ValueError, OverflowError):
                continue
            if number >= 1 and w > 0 and h > 0:
                frames[number].append((x, y, w, h))
    return frames


def convert_box(x, y, w, h, width, height):
    # Preserve only the portion that actually overlaps the image.
    x1, y1 = max(0.0, x), max(0.0, y)
    x2, y2 = min(float(width), x + w), min(float(height), y + h)
    if x2 <= x1 or y2 <= y1:
        return None
    return (0, (x1 + x2) / (2 * width), (y1 + y2) / (2 * height),
            (x2 - x1) / width, (y2 - y1) / height)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--stride', type=int, default=20, help='Use every Nth annotated frame')
    ap.add_argument('--max-per-clip', type=int, default=100)
    ap.add_argument('--reset', action='store_true', help='Clear ONLY derived images/labels')
    args = ap.parse_args()
    if args.stride < 1 or args.max_per_clip < 1:
        ap.error('--stride and --max-per-clip must be positive')

    out = ROOT / 'datasets' / 'prepared'
    if args.reset:
        for folder in ('images', 'labels'):
            shutil.rmtree(out / folder, ignore_errors=True)
    for group in ('train', 'val', 'test'):
        for kind in ('images', 'labels'):
            (out / kind / group).mkdir(parents=True, exist_ok=True)

    manifest = []
    for group, clips in SPLITS.items():
        for clip_no in clips:
            movie = ROOT / 'data' / 'videos' / f'Clip_{clip_no}.mov'
            txt = ROOT / 'data' / 'annotations' / f'Clip_{clip_no}_refined.txt'
            if not (movie.is_file() and txt.is_file()):
                raise FileNotFoundError(f'Clip {clip_no}: missing {movie} or {txt}')
            annotations = read_boxes(txt)
            if not annotations:
                raise ValueError(f'Clip {clip_no}: no valid annotation rows')
            cap = cv2.VideoCapture(str(movie))
            if not cap.isOpened():
                raise RuntimeError(f'Cannot open {movie}')
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            frame_ids = [f for f in sorted(annotations)
                         if f <= total and (f - 1) % args.stride == 0]
            frame_ids = frame_ids[:args.max_per_clip]
            saved = 0
            for frame_no in frame_ids:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no - 1)  # source is 1-indexed
                ok, image = cap.read()
                if not ok:
                    continue
                height, width = image.shape[:2]
                lines = []
                for x, y, w, h in annotations[frame_no]:
                    box = convert_box(x, y, w, h, width, height)
                    if box is not None:
                        lines.append(' '.join([str(box[0])] +
                                              [f'{v:.8f}' for v in box[1:]]))
                if not lines:
                    continue
                stem = f'Clip_{clip_no}_frame_{frame_no:06d}'
                image_path = out / 'images' / group / f'{stem}.jpg'
                label_path = out / 'labels' / group / f'{stem}.txt'
                if not cv2.imwrite(str(image_path), image):
                    raise RuntimeError(f'Cannot save {image_path}')
                label_path.write_text('\n'.join(lines) + '\n', encoding='utf-8')
                manifest.append((group, clip_no, frame_no, len(lines)))
                saved += 1
            cap.release()
            print(f'{group.upper()} Clip {clip_no}: {saved} image/label pairs')

    if not manifest:
        raise RuntimeError('No files prepared. Check input files and sampling settings.')
    with (out / 'prepared_manifest.csv').open('w', newline='', encoding='utf-8') as fh:
        writer = csv.writer(fh)
        writer.writerow(['split', 'clip', 'frame', 'visible_boxes'])
        writer.writerows(manifest)
    yaml_path = out / 'data.yaml'
    yaml_path.write_text(
        f"path: '{out.resolve().as_posix()}'\n"
        "train: images/train\nval: images/val\ntest: images/test\n"
        "names:\n  0: uav\n", encoding='utf-8')
    print(f'\nPrepared {len(manifest)} frame pairs; manifest: {out / "prepared_manifest.csv"}')
    print(f'Dataset configuration: {yaml_path}')
    print('NOTE: frames without GT rows were not treated as true negative examples.')


if __name__ == '__main__':
    main()
