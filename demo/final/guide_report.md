# Mini-project: offline experimental status

## What works
- Dataset prepared and a lightweight one-class UAV detector fine-tuned on GPU.
- Recorded Clip 7 detector review: 43/120 frames match at IoU ≥ 0.5 (35.83%).
- Longest contiguous period without a match: frames 39–106 (68 frames).
- Global image features extracted; the observed averages did not clearly separate matched and unmatched frames.
- Fixed-mode and learned mode-policy offline replay evaluated on archived sampled frames.

## Offline policy comparison (Clip 8)

| Policy | TP | FP | FN | Precision | Recall | Model-call latency |
|---|---:|---:|---:|---:|---:|---:|
| ALWAYS_FAST | 4 | 58 | 172 | 6.45% | 2.27% | 29.305 ms |
| ALWAYS_ROBUST | 25 | 92 | 151 | 21.37% | 14.20% | 30.913 ms |
| LEARNED_OFFLINE_REPLAY | 12 | 60 | 164 | 16.67% | 6.82% | 29.619 ms |

**Conclusion:** The learned replay does not achieve the highest recall; do not claim it improves on always ROBUST.

## Important limitations
- The demo video shows detector predictions versus reference boxes, **not predicted tracking identities**.
- The selection experiment replays saved still-frame outcomes; it is **not adaptive video tracking**.
- Model-call latency excludes frame decoding, feature computation, switching, and tracker cost.
- Clip 7 and Clip 8 offer limited sequence-level evidence. The split and any other reserved test clips must remain documented.
- These results do not demonstrate reliable real-time operation.

## Guide demonstration (in order)
1. Open `results/videos/clip_7_detector_review.mp4` (green = reference, magenta = predicted).
2. Show `results/final/clip7_detection_timeline.png` to explain the long failure interval.
3. Show `results/final/policy_recall.png` and `results/final/policy_latency.png` for measured comparisons.
4. Explain honestly that independently predicted track IDs and integrated adaptive video evaluation are missing.

## Source paths
- Detector review: `results\videos\clip_7_detector_review.csv`
- Selector comparison: `results\selector\comparison.csv`
- Benchmark clip counts (rows across both modes): `{'Clip 7': 180, 'Clip 8': 180}`
