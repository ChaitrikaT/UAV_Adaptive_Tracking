# Final evaluation: train clips [7] -> eval clips [8] (tag `default`)

Validation run (held-out validation clip; the detector itself used clips 7-8 for model selection).

Tracker settings: {'high_conf': 0.4, 'max_age': 60, 'emit_coasting': False, 'chosen_on': 'clip 7 (validation), best ROBUST IDF1', 'IDF1_on_clip7': 0.1213}. Tracking IDs come from the tracker; reference IDs are used only for scoring (annotated frames only, IoU>=0.5).
Cost = decode + global features + chosen-mode detector call + policy/tracker overhead (composed from measured timings).

## Eval clip 8

| policy | track recall | track precision | IDF1 | MOTA | ID switches | fragmentations | predicted IDs | ROBUST frac | total ms | fps |
|---|---|---|---|---|---|---|---|---|---|---|
| fast | 0.0107 | 0.0412 | 0.0166 | -0.2396 | 1 | 20 | 4 | 0.0 | 45.342 | 22.05 |
| robust | 0.0829 | 0.1531 | 0.0496 | -0.3819 | 22 | 83 | 7 | 1.0 | 45.523 | 21.97 |
| rule | 0.0829 | 0.1531 | 0.0496 | -0.3819 | 22 | 83 | 7 | 1.0 | 45.517 | 21.97 |
| difficulty | 0.0803 | 0.1485 | 0.0463 | -0.3867 | 22 | 83 | 7 | 0.9933 | 45.563 | 21.95 |
| learned | 0.026 | 0.0915 | 0.0273 | -0.2328 | 2 | 32 | 5 | 0.2283 | 45.697 | 21.88 |

## Difficulty estimator on the held-out clip(s) (AUC, 0.5 = chance; block-bootstrap 95% CI)

| clip | predicts 'FAST misses' | predicts 'ROBUST helps' |
|---|---|---|
| 8 | 0.4671 [0.369, 0.5543] | 0.418 [0.3345, 0.5074] |

Difficulty model: tau=0.057 (never switches: False), cost weight lam=0.1, standardised coefficients {'brightness': 1.6857, 'contrast': 1.2252, 'sharpness': 1.289, 'frame_change': 0.1111, 'has_track': 3.7891, 'frames_since_seen': 0.9409, 'target_size': -2.2496, 'target_speed': -0.0799, 'last_conf': -1.0027, 'prev_robust': -0.0161}

## Paired comparisons over all eval frames (delta = first - second; 95% block bootstrap)

| comparison | delta detection recall [point, lo, hi] | delta total ms/frame [point, lo, hi] |
|---|---|---|
| robust_vs_fast | [0.1528, 0.1164, 0.1916] | [0.181, -0.3138, 0.7811] |
| difficulty_vs_fast | [0.1508, 0.1152, 0.1891] | [0.2204, -0.2763, 0.8204] |
| difficulty_vs_robust | [-0.002, -0.0051, 0.0] | [0.0395, 0.0251, 0.054] |
| rule_vs_fast | [0.1528, 0.1164, 0.1916] | [0.1748, -0.3227, 0.7717] |
| learned_vs_fast | [0.0289, 0.0144, 0.0465] | [0.3549, 0.2304, 0.4873] |
| learned_vs_robust | [-0.1239, -0.1599, -0.0889] | [0.174, -0.3692, 0.6266] |

**Verdict, difficulty policy: INCONCLUSIVE** | learned tree: INCONCLUSIVE
Rule (uavtrack/protocol.py::verdict): SUPPORTED = better than FAST AND cheaper than ROBUST AND within 2 points recall of ROBUST; NOT SUPPORTED = not better than FAST, or clearly worse than ROBUST without being cheaper; else INCONCLUSIVE.
