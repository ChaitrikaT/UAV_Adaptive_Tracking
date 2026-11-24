# Detection and condition analysis (tag `default`)

Real detector outputs on recorded clips; frames without reference boxes are not scored (project variant,
not the official MOTChallenge protocol). AP50 uses detections with conf >= the detector floor, so it is
not directly comparable with Ultralytics mAP50.

## Clip 7: 1800 annotated of 1800 frames

| mode | conf | TP | FP | FN | precision | recall | F1 | frame hit rate |
|---|---|---|---|---|---|---|---|---|
| FAST | conf>=0.10 | 435 | 879 | 1365 | 0.3311 | 0.2417 | 0.2794 | 0.2417 |
| FAST | conf>=0.25 | 289 | 334 | 1511 | 0.4639 | 0.1606 | 0.2385 | 0.1606 |
| FAST | conf>=0.50 | 5 | 11 | 1795 | 0.3125 | 0.0028 | 0.0055 | 0.0028 |
| ROBUST | conf>=0.10 | 663 | 1416 | 1137 | 0.3189 | 0.3683 | 0.3418 | 0.3683 |
| ROBUST | conf>=0.25 | 425 | 581 | 1375 | 0.4225 | 0.2361 | 0.3029 | 0.2361 |
| ROBUST | conf>=0.50 | 86 | 86 | 1714 | 0.5 | 0.0478 | 0.0872 | 0.0478 |

AP50: FAST 0.1195, ROBUST 0.1625 | mean detector ms: FAST 34.152, ROBUST 33.966

| descriptor | Spearman rho vs FAST hit (p) | rho vs ROBUST hit (p) |
|---|---|---|
| brightness | -0.086 (0.0002) | -0.201 (0.0) |
| contrast | -0.121 (0.0) | -0.05 (0.0325) |
| sharpness | -0.171 (0.0) | -0.203 (0.0) |
| frame_change | -0.126 (0.0) | -0.139 (0.0) |
| target_size_px | -0.097 (0.0) | -0.323 (0.0) |
| local_contrast | -0.074 (0.0018) | 0.049 (0.0383) |
| edge_distance_frac | -0.077 (0.0011) | 0.032 (0.1807) |
| target_speed_px_per_frame | -0.172 (0.0) | -0.195 (0.0) |

## Clip 8: 1800 annotated of 1800 frames

| mode | conf | TP | FP | FN | precision | recall | F1 | frame hit rate |
|---|---|---|---|---|---|---|---|---|
| FAST | conf>=0.10 | 99 | 1732 | 3436 | 0.0541 | 0.028 | 0.0369 | 0.055 |
| FAST | conf>=0.25 | 68 | 787 | 3467 | 0.0795 | 0.0192 | 0.031 | 0.0378 |
| FAST | conf>=0.50 | 5 | 45 | 3530 | 0.1 | 0.0014 | 0.0028 | 0.0028 |
| ROBUST | conf>=0.10 | 639 | 2710 | 2896 | 0.1908 | 0.1808 | 0.1856 | 0.3217 |
| ROBUST | conf>=0.25 | 449 | 1151 | 3086 | 0.2806 | 0.127 | 0.1749 | 0.2389 |
| ROBUST | conf>=0.50 | 80 | 179 | 3455 | 0.3089 | 0.0226 | 0.0422 | 0.0444 |

AP50: FAST 0.0027, ROBUST 0.0555 | mean detector ms: FAST 37.138, ROBUST 37.21

| descriptor | Spearman rho vs FAST hit (p) | rho vs ROBUST hit (p) |
|---|---|---|
| brightness | 0.042 (0.0767) | -0.14 (0.0) |
| contrast | -0.006 (0.8087) | 0.084 (0.0003) |
| sharpness | -0.044 (0.0627) | 0.002 (0.9185) |
| frame_change | -0.049 (0.0362) | -0.069 (0.0032) |
| target_size_px | 0.031 (0.1898) | -0.076 (0.0012) |
| local_contrast | -0.018 (0.4412) | 0.1 (0.0) |
| edge_distance_frac | -0.082 (0.0005) | 0.034 (0.1453) |
| target_speed_px_per_frame | -0.007 (0.7536) | 0.059 (0.0128) |

## Can conditions predict that FAST will miss? (AUC, 0.5 = chance; fit on one clip, test on the other)

| train -> test | scene only | online policy features | scene + target descriptors (analysis only) |
|---|---|---|---|
| 7 -> 8 | 0.402 | 0.449 | 0.501 |
| 8 -> 7 | 0.431 | 0.418 | 0.468 |
