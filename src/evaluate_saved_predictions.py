"""Offline localization assessment for EXISTING saved detector outputs.

Input CSV required columns:
image,mode,x1,y1,x2,y2,confidence
"image" is the filename of one prepared test image (not a path).
Scores only listed positive test frames; NOT mAP or full detector validation.
This script does not generate target detections or control tracking.
"""
from collections import defaultdict
from pathlib import Path
import argparse
import csv
import math

ROOT=Path(__file__).resolve().parents[1]

def box_iou(a,b):
    x1=max(a[0],b[0]);y1=max(a[1],b[1]);x2=min(a[2],b[2]);y2=min(a[3],b[3])
    inter=max(0,x2-x1)*max(0,y2-y1)
    area_a=max(0,a[2]-a[0])*max(0,a[3]-a[1])
    area_b=max(0,b[2]-b[0])*max(0,b[3]-b[1])
    u=area_a+area_b-inter
    return inter/u if u>0 else 0.

def score(gt,pred,threshold=0.5):
    # Greedy one-to-one matches, only on this frame.
    candidate=sorted(((box_iou(a,b),i,j) for i,a in enumerate(gt) for j,b in enumerate(pred)),reverse=True)
    used_g,used_p=set(),set()
    for iou,i,j in candidate:
        if iou<threshold:break
        if i not in used_g and j not in used_p:
            used_g.add(i);used_p.add(j)
    tp=len(used_g)
    return tp,len(pred)-tp,len(gt)-tp

def read_gt(label_path):
    result=[]
    for line in label_path.read_text(encoding="utf-8").splitlines():
        parts=line.split()
        if len(parts)!=5:raise ValueError(f"Bad label in {label_path}: {line}")
        cls,xc,yc,w,h=map(float,parts)
        if cls!=0:continue
        result.append((xc-w/2,yc-h/2,xc+w/2,yc+h/2))
    return result

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--csv", required=True,help="Precomputed detections CSV")
    args=ap.parse_args()
    src=Path(args.csv)
    if not src.exists():raise FileNotFoundError(src)
    # Precomputed boxes are expressed in original image pixels. Image dimensions
    # come from each prepared image, not guessed constants.
    try: import cv2
    except ImportError as e: raise SystemExit("Install opencv-python") from e
    pred=defaultdict(list)
    with src.open(newline="",encoding="utf-8-sig") as f:
        reader=csv.DictReader(f)
        required={"image","mode","x1","y1","x2","y2","confidence"}
        if not required.issubset(set(reader.fieldnames or [])):
            raise ValueError(f"Required CSV columns: {sorted(required)}")
        for row in reader:
            image=Path(row["image"]).name
            mode=row["mode"].strip()
            if not mode:raise ValueError("Empty mode name")
            vals=[float(row[k]) for k in ("x1","y1","x2","y2","confidence")]
            if not all(math.isfinite(x) for x in vals) or vals[2]<=vals[0] or vals[3]<=vals[1]:
                raise ValueError(f"Invalid prediction: {row}")
            pred[(mode,image)].append(tuple(vals[:4]))
    if not pred:
        raise ValueError("Prediction CSV is empty: no accuracy claims possible")
    dataset=ROOT/"datasets"/"prepared"
    images=sorted((dataset/"images"/"test").glob("*.jpg"))
    if not images:raise FileNotFoundError("Prepare test images first")
    modes=sorted({key[0] for key in pred})
    summaries=[]
    for mode in modes:
        tp=fp=fn=0
        for image_path in images:
            frame=cv2.imread(str(image_path))
            if frame is None:raise ValueError(f"Cannot open {image_path}")
            h,w=frame.shape[:2]
            gt=read_gt(dataset/"labels"/"test"/(image_path.stem+".txt"))
            scaled=[(a[0]*w,a[1]*h,a[2]*w,a[3]*h) for a in gt]
            t,p,n=score(scaled,pred.get((mode,image_path.name),[]))
            tp+=t;fp+=p;fn+=n
        summaries.append({"mode":mode,"test_images":len(images),"TP":tp,"FP":fp,"FN":fn,
                          "precision_iou50":round(tp/(tp+fp),4) if tp+fp else 0,
                          "recall_iou50":round(tp/(tp+fn),4) if tp+fn else 0})
    dest=ROOT/"results"/"saved_detection_evaluation.csv"
    dest.parent.mkdir(parents=True,exist_ok=True)
    with dest.open("w",newline="",encoding="utf-8") as f:
        writer=csv.DictWriter(f,fieldnames=summaries[0].keys())
        writer.writeheader();writer.writerows(summaries)
    for r in summaries:print(r)
    print("Saved:",dest)
    print("NOTE: IoU@0.5 counts on sampled positive test images only; NOT mAP/HOTA/IDF1.")

if __name__=="__main__":main()
