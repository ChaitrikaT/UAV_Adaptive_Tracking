"""Audit historical predicted identity continuity against stored reference labels.

The CSV must contain saved predictions with columns:
clip,frame,track_id,x,y,w,h
One row per prediction. This does NOT perform detection or tracking.

Reports only directly interpretable identity-change counts after IoU matching,
not HOTA, IDF1, MOTA or a standard multi-object tracking benchmark.
"""
from collections import defaultdict
from pathlib import Path
import argparse
import csv
import math

ROOT = Path(__file__).resolve().parents[1]


def overlap(a, b):
    x1 = max(a[0], b[0]); y1 = max(a[1], b[1])
    x2 = min(a[0]+a[2], b[0]+b[2]); y2 = min(a[1]+a[3], b[1]+b[3])
    inter = max(0.0, x2-x1) * max(0.0, y2-y1)
    area_a = max(0.0, a[2])*max(0.0,a[3])
    area_b = max(0.0, b[2])*max(0.0,b[3])
    union = area_a + area_b - inter
    return inter/union if union>0 else 0.0


def read_records(path, prediction=False):
    data=defaultdict(list)
    with path.open(newline='', encoding='utf-8-sig') as f:
        if prediction:
            rdr=csv.DictReader(f)
            req={'clip','frame','track_id','x','y','w','h'}
            if not req.issubset(rdr.fieldnames or []):
                raise ValueError(f'Prediction CSV missing: {req-set(rdr.fieldnames or [])}')
            for r in rdr:
                clip=int(r['clip']); frame=int(r['frame']); tid=int(r['track_id'])
                bbox=tuple(float(r[k]) for k in ('x','y','w','h'))
                if frame<1 or bbox[2]<=0 or bbox[3]<=0 or not all(map(math.isfinite,bbox)):
                    raise ValueError(f'Invalid prediction row: {r}')
                data[(clip,frame)].append((tid,bbox))
        else:
            for row in csv.reader(f):
                if len(row)<6: continue
                try:
                    frame=int(float(row[0])); tid=int(float(row[1]))
                    bbox=tuple(float(v) for v in row[2:6])
                except ValueError: continue
                if frame>=1 and bbox[2]>0 and bbox[3]>0:
                    data[frame].append((tid,bbox))
    return data


def summarize(predictions, threshold):
    clips=sorted({c for c,_ in predictions})
    if not clips:raise ValueError('Prediction CSV has no rows')
    results=[]
    for clip in clips:
        gt_path=ROOT/'data'/'annotations'/f'Clip_{clip}_refined.txt'
        if not gt_path.exists():raise FileNotFoundError(gt_path)
        gt=read_records(gt_path)
        history=defaultdict(list)
        matched_rows=0
        for (c,frame),boxes in sorted(predictions.items()):
            if c!=clip:continue
            refs=gt.get(frame,[])
            candidates=sorted(((overlap(gbox,pbox),gi,pi)
                               for gi,(_,gbox) in enumerate(refs)
                               for pi,(_,pbox) in enumerate(boxes)),reverse=True)
            used_g=set();used_p=set()
            for iou,gi,pi in candidates:
                if iou<threshold:break
                if gi in used_g or pi in used_p:continue
                used_g.add(gi);used_p.add(pi)
                ref_id=refs[gi][0]; pred_id=boxes[pi][0]
                history[ref_id].append((frame,pred_id))
                matched_rows+=1
        changes=0
        for observations in history.values():
            observations.sort()
            for (oldframe,oldid),(newframe,newid) in zip(observations,observations[1:]):
                # An identity change is counted only between adjacent frames
                # with a matched observation (unmatched gaps excluded).
                if newframe==oldframe+1 and newid!=oldid:
                    changes+=1
        results.append({'clip':clip,'matched_reference_boxes':matched_rows,
                        'matched_reference_ids':len(history),
                        'adjacent_matched_id_changes':changes})
    return results


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--csv', required=True)
    ap.add_argument('--iou',type=float,default=0.5)
    args=ap.parse_args()
    if not 0<args.iou<=1:ap.error('--iou must be in (0,1]')
    predictions=read_records(Path(args.csv),prediction=True)
    results=summarize(predictions,args.iou)
    out=ROOT/'results'/'saved_track_id_review.csv'
    out.parent.mkdir(parents=True,exist_ok=True)
    with out.open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=results[0].keys())
        writer.writeheader(); writer.writerows(results)
    for r in results:print(r)
    print('Saved:',out)
    print('This is a LIMITED offline identity-change diagnostic, not IDF1/HOTA.')

if __name__=='__main__':main()
