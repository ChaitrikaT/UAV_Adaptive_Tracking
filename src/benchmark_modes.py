"""Offline FAST/ROBUST detector benchmark on sampled, prerecorded test images.

No camera, vehicle, tracking, or autonomous switching interfaces.
COCO model is for speed only. UAV localization metrics are enabled only when
--uav-model is supplied with weights whose class name is 'uav' or 'drone'.
"""
from pathlib import Path
import argparse
import csv
import time
from statistics import mean
import cv2
import torch
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
MODES = {'FAST': 640, 'ROBUST': 960}


def sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def load_ground_truth(path, width, height):
    boxes = []
    if not path.is_file():
        raise FileNotFoundError(path)
    for line in path.read_text(encoding='utf-8').splitlines():
        cls, xc, yc, bw, bh = map(float, line.split())
        if int(cls) != 0:
            continue
        boxes.append(((xc-bw/2)*width, (yc-bh/2)*height,
                      (xc+bw/2)*width, (yc+bh/2)*height))
    return boxes


def iou(a, b):
    x1, y1, x2, y2 = max(a[0],b[0]), max(a[1],b[1]), min(a[2],b[2]), min(a[3],b[3])
    area = max(0., x2-x1) * max(0., y2-y1)
    size_a = max(0.,a[2]-a[0])*max(0.,a[3]-a[1])
    size_b = max(0.,b[2]-b[0])*max(0.,b[3]-b[1])
    return area / (size_a + size_b - area) if size_a + size_b > area else 0.


def counts(gt, pred, threshold=0.5):
    # Greedy one-to-one matching; an illustrative IoU@0.5 metric, not mAP.
    choices = sorted([(iou(g,p), i,j) for i,g in enumerate(gt)
                      for j,p in enumerate(pred)], reverse=True)
    used_gt, used_pred = set(), set()
    for score,i,j in choices:
        if score < threshold:
            break
        if i not in used_gt and j not in used_pred:
            used_gt.add(i); used_pred.add(j)
    n = len(used_gt)
    return n, len(pred)-n, len(gt)-n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--weights', default='yolo11n.pt')
    ap.add_argument('--limit', type=int, default=30, help='Number of test images')
    ap.add_argument('--uav-model', action='store_true', help='Enable UAV quality metrics for a UAV-labelled model')
    args = ap.parse_args()
    if args.limit < 1:
        ap.error('--limit must be positive')
    images = sorted((ROOT/'datasets'/'prepared'/'images'/'val').glob('*.jpg'))[:args.limit]
    if not images:
        raise RuntimeError('No test images. Run prepare_dataset.py first.')
    model = YOLO(args.weights)
    device = 0 if torch.cuda.is_available() else 'cpu'
    names = model.names
    uav_indices = [k for k,v in names.items() if str(v).lower().strip() in ('uav','drone')]
    if args.uav_model and not uav_indices:
        raise ValueError('Cannot enable UAV metrics: model class names have no uav/drone label.')
    print('Device:', device, '| Model classes:', names)
    if not args.uav_model:
        print('COCO/unverified weights: TIMING ONLY. UAV quality is NOT measured.')
    out = ROOT/'results'/'offline_benchmark'
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    # Warm up both shapes; CUDA calls are asynchronous.
    warm_frame = cv2.imread(str(images[0]))
    if warm_frame is None:
        raise RuntimeError(f'Cannot read {images[0]}')
    for imgsz in MODES.values():
        for _ in range(3):
            model.predict(warm_frame, imgsz=imgsz, conf=0.15,
                          device=device, classes=uav_indices if args.uav_model else None,
                          verbose=False)
            sync()
    for index, img_path in enumerate(images):
        frame = cv2.imread(str(img_path))
        if frame is None:
            raise RuntimeError(f'Cannot read {img_path}')
        h, w = frame.shape[:2]
        gt = load_ground_truth(ROOT/'datasets'/'prepared'/'labels'/'val'/f'{img_path.stem}.txt', w, h)
        for mode, imgsz in MODES.items():
            sync()
            start = time.perf_counter()
            result = model.predict(frame, imgsz=imgsz, conf=0.15,
                                   device=device, classes=uav_indices if args.uav_model else None,
                                   verbose=False)[0]
            sync()
            elapsed_ms = (time.perf_counter()-start)*1000.
            preds = result.boxes.xyxy.cpu().numpy().tolist() if result.boxes is not None else []
            row = {'frame':img_path.name, 'mode':mode, 'imgsz':imgsz,
                   'latency_ms':round(elapsed_ms, 3), 'predictions':len(preds)}
            if args.uav_model:
                tp, fp, fn = counts(gt, preds)
                row.update({'gt_boxes':len(gt), 'TP':tp, 'FP':fp, 'FN':fn})
            rows.append(row)
            if index == 0:
                cv2.imwrite(str(out/f'{mode}_example.jpg'), result.plot())
        if (index + 1) % 10 == 0:
            print(f'Processed {index+1}/{len(images)} test frames')
    with (out/'per_frame.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader(); writer.writerows(rows)
    summary = []
    for mode in MODES:
        subset = [r for r in rows if r['mode'] == mode]
        avg = mean(r['latency_ms'] for r in subset)
        line = {'mode':mode, 'imgsz':MODES[mode], 'frames':len(subset),
                'mean_latency_ms':round(avg,3), 'fps_equivalent':round(1000./avg,2)}
        if args.uav_model:
            TP = sum(r['TP'] for r in subset)
            FP = sum(r['FP'] for r in subset)
            FN = sum(r['FN'] for r in subset)
            line.update({'TP':TP, 'FP':FP, 'FN':FN,
                         'precision_iou50':round(TP/(TP+FP),4) if TP+FP else 0.,
                         'recall_iou50':round(TP/(TP+FN),4) if TP+FN else 0.})
        summary.append(line)
    with (out/'summary.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=summary[0].keys())
        writer.writeheader(); writer.writerows(summary)
    for line in summary:
        print(line)
    print('Saved:', out/'summary.csv')
    print('Speed is approximate model-call throughput on sampled still images, not end-to-end video FPS.')


if __name__ == '__main__':
    main()
