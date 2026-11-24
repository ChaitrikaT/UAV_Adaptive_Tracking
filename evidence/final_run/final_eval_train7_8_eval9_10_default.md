# Final evaluation: train clips [7, 8] -> eval clips [9, 10] (tag `default`)

**RESERVED TEST SET RUN**

Tracker settings: {'high_conf': 0.4, 'max_age': 60, 'emit_coasting': False, 'chosen_on': 'clip 7 (validation), best ROBUST IDF1', 'IDF1_on_clip7': 0.1213}. Tracking IDs come from the tracker; reference IDs are used only for scoring (annotated frames only, IoU>=0.5).
Cost = decode + global features + chosen-mode detector call + policy/tracker overhead (composed from measured timings).

## Eval clip 9

| policy | track recall | track precision | IDF1 | MOTA | ID switches | fragmentations | predicted IDs | ROBUST frac | total ms | fps |
|---|---|---|---|---|---|---|---|---|---|---|
| fast | 0.0 | 0.0 | 0.0 | 0.0 | 0 | 0 | 0 | 0.0 | 37.267 | 26.83 |
| robust | 0.0017 | 0.1667 | 0.0033 | -0.0067 | 0 | 1 | 1 | 1.0 | 36.681 | 27.26 |
| rule | 0.0017 | 0.1667 | 0.0033 | -0.0067 | 0 | 1 | 1 | 1.0 | 36.681 | 27.26 |
| difficulty | 0.0 | 0.0 | 0.0 | 0.0 | 0 | 0 | 0 | 0.895 | 36.716 | 27.24 |
| learned | 0.0 | 0.0 | 0.0 | 0.0 | 0 | 0 | 0 | 0.3083 | 37.385 | 26.75 |

## Eval clip 10

| policy | track recall | track precision | IDF1 | MOTA | ID switches | fragmentations | predicted IDs | ROBUST frac | total ms | fps |
|---|---|---|---|---|---|---|---|---|---|---|
| fast | 0.0 | 0.0 | 0.0 | 0.0 | 0 | 0 | 0 | 0.0 | 37.739 | 26.5 |
| robust | 0.0 | 0.0 | 0.0 | 0.0 | 0 | 0 | 0 | 1.0 | 36.841 | 27.14 |
| rule | 0.0 | 0.0 | 0.0 | 0.0 | 0 | 0 | 0 | 1.0 | 36.84 | 27.14 |
| difficulty | 0.0 | 0.0 | 0.0 | 0.0 | 0 | 0 | 0 | 0.8333 | 36.974 | 27.05 |
| learned | 0.0 | 0.0 | 0.0 | 0.0 | 0 | 0 | 0 | 0.2389 | 37.817 | 26.44 |

## Difficulty estimator on the held-out clip(s) (AUC, 0.5 = chance; block-bootstrap 95% CI)

| clip | predicts 'FAST misses' | predicts 'ROBUST helps' |
|---|---|---|
| 9 | 0.5564 [0.4796, 0.6501] | 0.3999 [0.3116, 0.4934] |
| 10 | n/a | n/a |

Difficulty model: tau=0.199 (never switches: False), cost weight lam=0.1, standardised coefficients {'brightness': 0.161, 'contrast': -0.5049, 'sharpness': 1.2988, 'frame_change': 0.3975, 'has_track': 4.4008, 'frames_since_seen': 3.2475, 'target_size': -0.8195, 'target_speed': 0.0747, 'last_conf': -0.4863, 'prev_robust': -0.152}

## Paired comparisons over all eval frames (delta = first - second; 95% block bootstrap)

| comparison | delta detection recall [point, lo, hi] | delta total ms/frame [point, lo, hi] |
|---|---|---|
| robust_vs_fast | [-0.0003, -0.0081, 0.0072] | [-0.7423, -0.9555, -0.535] |
| difficulty_vs_fast | [-0.0022, -0.0097, 0.0045] | [-0.6579, -0.8649, -0.4587] |
| difficulty_vs_robust | [-0.0019, -0.0036, -0.0006] | [0.0845, 0.0072, 0.1618] |
| rule_vs_fast | [-0.0003, -0.0081, 0.0072] | [-0.7427, -0.9558, -0.536] |
| learned_vs_fast | [0.0, 0.0, 0.0] | [0.098, -0.0026, 0.1951] |
| learned_vs_robust | [0.0003, -0.0072, 0.0081] | [0.8404, 0.6446, 1.046] |

**Verdict, difficulty policy: NOT SUPPORTED** | learned tree: NOT SUPPORTED
Rule (uavtrack/protocol.py::verdict): SUPPORTED = better than FAST AND cheaper than ROBUST AND within 2 points recall of ROBUST; NOT SUPPORTED = not better than FAST, or clearly worse than ROBUST without being cheaper; else INCONCLUSIVE.
