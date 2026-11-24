"""READ-ONLY diagnosis: why is detection recall ~0 on clips 9-10 (and how do clips 7-8 compare)? Changes nothing, retunes nothing.

  python check_test_clips.py --clips 7 8 9 10
Uses the saved caches (results/policy/cache_clip_<N>_default.pkl) and the recorded videos/annotations.
Writes results/final/check/alignment_stats.csv, alignment_summary.txt and two PNG montages per clip.
Tests three explanations: (a) detector finds nothing, (b) detections exist but are shifted/mis-sized (low IoU),
(c) annotation frame numbers are offset from the video (recall jumps at a non-zero shift).
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2
import numpy as np

import run_evaluation as RE
from uavtrack.tracker import iou_xyxy


def best_ious(refs, dets):
    return [max([iou_xyxy(g[1:], d[:4]) for d in dets], default=0.0) for g in refs]


def analyse(cache):
    fr = [r for r in cache["frames"] if r["ref"]]
    by = {r["frame"]: r for r in cache["frames"]}
    best, sizes, ndet, confs, near = [], [], [], [], []
    for r in fr:
        d = r["dets_robust"]
        best += best_ious(r["ref"], d)
        sizes += [float(np.sqrt((g[3] - g[1]) * (g[4] - g[2]))) for g in r["ref"]]
        ndet.append(len(d)); confs += [float(x) for x in d[:, 4]] if len(d) else []
        for g in r["ref"]:
            if len(d):
                cx, cy = (g[1] + g[3]) / 2, (g[2] + g[4]) / 2
                dist = min(np.hypot((x[0] + x[2]) / 2 - cx, (x[1] + x[3]) / 2 - cy) for x in d)
                near.append(dist / max(1.0, np.sqrt((g[3] - g[1]) * (g[4] - g[2]))))
    best = np.array(best)
    out = {"annotated_frames": len(fr), "ref_boxes": len(best), "median_ref_size_px": round(float(np.median(sizes)), 1), "p10_ref_size_px": round(float(np.percentile(sizes, 10)), 1),
           "mean_dets_per_frame": round(float(np.mean(ndet)), 3), "frames_with_any_det": round(float(np.mean(np.array(ndet) > 0)), 4),
           "median_det_conf": round(float(np.median(confs)), 3) if confs else None, "p90_det_conf": round(float(np.percentile(confs, 90)), 3) if confs else None,
           "refs_with_IoU>=0.5": round(float((best >= .5).mean()), 4), "refs_with_IoU>=0.3": round(float((best >= .3).mean()), 4), "refs_with_IoU>=0.1": round(float((best >= .1).mean()), 4),
           "median_nearest_det_distance_in_ref_sizes": round(float(np.median(near)), 2) if near else None}
    shifts = {}
    for k in range(-5, 6):                       # reference of frame f+k scored against detections of frame f
        hit = tot = 0
        for r in fr:
            o = by.get(r["frame"] + k)
            if o is not None and o["ref"]:
                b = best_ious(o["ref"], r["dets_robust"]); hit += sum(x >= .5 for x in b); tot += len(b)
        shifts[k] = round(hit / tot, 4) if tot else None
    out["recall_by_annotation_shift(frames)"] = shifts
    return out


def montage(clip, cache, tag, outdir):
    cap = cv2.VideoCapture(str(RE.find_video(clip)))
    fr = [r for r in cache["frames"] if r["ref"]]
    picks = [fr[int(i)] for i in np.linspace(0, len(fr) - 1, 6)]
    full, crops = [], []
    for r in picks:
        cap.set(cv2.CAP_PROP_POS_FRAMES, r["frame"] - 1); ok, img = cap.read()
        if not ok:
            continue
        for g in r["ref"]:
            cv2.rectangle(img, (int(g[1]) - 4, int(g[2]) - 4), (int(g[3]) + 4, int(g[4]) + 4), (60, 220, 60), 3)
        for d in r["dets_robust"]:
            cv2.rectangle(img, (int(d[0]), int(d[1])), (int(d[2]), int(d[3])), (220, 0, 220), 3)
            cv2.putText(img, f"{d[4]:.2f}", (int(d[0]), max(20, int(d[1]) - 6)), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (220, 0, 220), 2)
        cv2.putText(img, f"clip {clip} frame {r['frame']}  green=reference magenta=ROBUST detections", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 255, 255), 3)
        full.append(cv2.resize(img, (640, 360)))
        g = r["ref"][0]; cx, cy = int((g[1] + g[3]) / 2), int((g[2] + g[4]) / 2); half = 120
        x0, y0 = max(0, cx - half), max(0, cy - half)
        c = img[y0:y0 + 2 * half, x0:x0 + 2 * half]
        crops.append(cv2.resize(c, (320, 320), interpolation=cv2.INTER_NEAREST))
    cap.release()
    def grid(tiles, cols, tw, th):
        while len(tiles) % cols: tiles.append(np.zeros((th, tw, 3), np.uint8))
        return np.vstack([np.hstack(tiles[i:i + cols]) for i in range(0, len(tiles), cols)])
    if full:
        cv2.imwrite(str(outdir / f"clip{clip}_full_frames.png"), grid(full, 3, 640, 360))
        cv2.imwrite(str(outdir / f"clip{clip}_zoom_on_reference.png"), grid(crops, 3, 320, 320))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--clips", type=int, nargs="+", default=[7, 8, 9, 10]); ap.add_argument("--tag", default="default")
    a = ap.parse_args()
    outdir = RE.ROOT / "results" / "final" / "check"; outdir.mkdir(parents=True, exist_ok=True)
    rows, lines = [], []
    for c in a.clips:
        cache = RE.load_cache(c, a.tag); s = analyse(cache)
        shifts = s.pop("recall_by_annotation_shift(frames)")
        rows.append({"clip": c, **s, "recall_by_shift": str(shifts)})
        lines += [f"== clip {c}", *[f"  {k}: {v}" for k, v in s.items()], f"  recall at IoU>=0.5 by annotation shift k (ref of frame f+k vs detections of frame f): {shifts}", ""]
        montage(c, cache, a.tag, outdir)
    with (outdir / "alignment_stats.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    (outdir / "alignment_summary.txt").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines)); print(f"Saved to {outdir}")


if __name__ == "__main__":
    main()
