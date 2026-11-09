"""Offline diagnostic only: compare pretrained-model boxes with labelled Purdue video.

Uses saved files, never accesses a camera, never controls or tracks a UAV.
"""
from pathlib import Path
from collections import defaultdict
import argparse
import csv
import cv2
import torch
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]


def load_reference(path):
    per_frame = defaultdict(list)
    with path.open('r', encoding='utf-8-sig', errors='replace', newline='') as stream:
        for row in csv.reader(stream):
            if len(row) < 6:
                continue
            try:
                frame_num, target_id = int(float(row[0])), int(float(row[1]))
                x, y, width, height = map(float, row[2:6])
            except (ValueError, OverflowError):
                continue
            if frame_num >= 1 and width > 0 and height > 0:
                per_frame[frame_num].append((target_id, x, y, width, height))
    return per_frame


def box_iou(a, b):
    x1 = max(a[0], b[0]); y1 = max(a[1], b[1])
    x2 = min(a[2], b[2]); y2 = min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = max(0, a[2] - a[0]) * max(0, a[3] - a[1])
    area_b = max(0, b[2] - b[0]) * max(0, b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def main():
    ap = argparse.ArgumentParser(description='Diagnostic video: reference vs independent detector predictions')
    ap.add_argument('--clip', type=int, default=1)
    ap.add_argument('--start', type=int, default=130)
    ap.add_argument('--end', type=int, default=180)
    ap.add_argument('--imgsz', type=int, default=960)
    ap.add_argument('--conf', type=float, default=0.10)
    args = ap.parse_args()
    if args.start < 1 or args.end < args.start:
        ap.error('Expected 1 <= start <= end')

    video_path = ROOT / 'data' / 'videos' / f'Clip_{args.clip}.mov'
    reference_path = ROOT / 'data' / 'annotations' / f'Clip_{args.clip}_refined.txt'
    weights = ROOT / 'results' / 'training' / 'uav_detector_v1' / 'weights' / 'best.pt'
    output_dir = ROOT / 'results' / 'videos'
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f'clip_{args.clip}_detector_review.mp4'
    csv_path = output_dir / f'clip_{args.clip}_detector_review.csv'
    for path in (video_path, reference_path, weights):
        if not path.exists():
            raise FileNotFoundError(f'Expected file not found: {path}')

    reference = load_reference(reference_path)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f'Could not open {video_path}')
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if frame_count and args.start > frame_count:
        cap.release()
        raise ValueError(f'Clip contains only {frame_count} frames')

    writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*'mp4v'), 10.0, (width, height))
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f'Could not write output video at {output_path}')

    model = YOLO(str(weights))
    device = 0 if torch.cuda.is_available() else 'cpu'
    rows = []
    cap.set(cv2.CAP_PROP_POS_FRAMES, args.start - 1)

    try:
        for frame_num in range(args.start, args.end + 1):
            ok, image = cap.read()
            if not ok:
                print(f'Stopped at frame {frame_num}: video could not be read')
                break

            result = model.predict(image, imgsz=args.imgsz, conf=args.conf, device=device, verbose=False)[0]
            pred_boxes = result.boxes.xyxy.cpu().numpy().tolist() if result.boxes is not None else []
            pred_scores = result.boxes.conf.cpu().numpy().tolist() if result.boxes is not None else []
            gt_boxes = []
            display = image.copy()

            for target_id, x, y, bw, bh in reference.get(frame_num, []):
                x1, y1 = max(0, x), max(0, y)
                x2, y2 = min(width, x + bw), min(height, y + bh)
                if x2 <= x1 or y2 <= y1:
                    continue
                gt_boxes.append((x1, y1, x2, y2))
                p1 = (int(round(x1)), int(round(y1)))
                p2 = (int(round(x2)), int(round(y2)))
                cv2.rectangle(display, p1, p2, (0, 220, 0), 3)
                cv2.circle(display, ((p1[0]+p2[0])//2, (p1[1]+p2[1])//2), 17, (0, 220, 0), 2)

            for pred, score in zip(pred_boxes, pred_scores):
                x1, y1, x2, y2 = (int(round(v)) for v in pred)
                cv2.rectangle(display, (x1, y1), (x2, y2), (220, 0, 220), 3)
                cv2.putText(display, f'pred {score:.2f}', (max(0, x1), max(20, y1-5)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 0, 220), 2)

            max_iou = max((box_iou(g, p) for g in gt_boxes for p in pred_boxes), default=0.0)
            banner = f'Frame {frame_num} | GREEN: Purdue GT | MAGENTA: Model prediction | IoU(max)={max_iou:.2f}'
            cv2.rectangle(display, (0, 0), (width, 43), (20, 20, 20), -1)
            cv2.putText(display, banner, (15, 29), cv2.FONT_HERSHEY_SIMPLEX, 0.60, (255, 255, 255), 2)

            # A reference-guided crop is for HUMAN INSPECTION only, not model input.
            if gt_boxes and width >= 390 and height >= 390:
                g = gt_boxes[0]
                cx, cy = int((g[0]+g[2])/2), int((g[1]+g[3])/2)
                pad = 44
                left, top = max(0, cx-pad), max(0, cy-pad)
                right, bottom = min(width, cx+pad), min(height, cy+pad)
                crop = display[top:bottom, left:right]
                if crop.size:
                    zoom = cv2.resize(crop, (300, 300), interpolation=cv2.INTER_NEAREST)
                    ox, oy = width-315, height-325
                    cv2.rectangle(display, (ox-5, oy-22), (ox+305, oy+305), (25, 25, 25), -1)
                    display[oy:oy+300, ox:ox+300] = zoom
                    cv2.putText(display, 'GT-guided zoom (review only)', (ox, oy-5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.47, (255,255,255), 1)

            writer.write(display)
            rows.append({'frame': frame_num, 'reference_boxes': len(gt_boxes),
                         'predicted_boxes': len(pred_boxes), 'max_iou': round(max_iou, 4)})
    finally:
        cap.release()
        writer.release()

    with csv_path.open('w', newline='', encoding='utf-8') as out:
        csv_writer = csv.DictWriter(out, fieldnames=['frame', 'reference_boxes', 'predicted_boxes', 'max_iou'])
        csv_writer.writeheader()
        csv_writer.writerows(rows)
    frames_with_pred = sum(r['predicted_boxes'] > 0 for r in rows)
    frames_with_overlap = sum(r['max_iou'] >= 0.5 for r in rows)
    print(f'Device: {device} | Frames reviewed: {len(rows)}')
    print(f'Frames with >=1 model prediction: {frames_with_pred}')
    print(f'Frames with max IoU >=0.5: {frames_with_overlap}')
    print(f'VIDEO: {output_path}')
    print(f'CSV:   {csv_path}')
    print('Diagnostic only: not tracker evaluation, not test-set performance.')

if __name__ == '__main__':
    main()
