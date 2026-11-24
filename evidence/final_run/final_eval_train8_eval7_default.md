# Final evaluation: train clips [8] -> eval clips [7] (tag `default`)

Validation run (held-out validation clip; the detector itself used clips 7-8 for model selection).

Tracker settings: {'high_conf': 0.4, 'max_age': 60, 'emit_coasting': False, 'chosen_on': 'clip 7 (validation), best ROBUST IDF1', 'IDF1_on_clip7': 0.1213}. Tracking IDs come from the tracker; reference IDs are used only for scoring (annotated frames only, IoU>=0.5).
Cost = decode + global features + chosen-mode detector call + policy/tracker overhead (composed from measured timings).

## Eval clip 7

| policy | track recall | track precision | IDF1 | MOTA | ID switches | fragmentations | predicted IDs | ROBUST frac | total ms | fps |
|---|---|---|---|---|---|---|---|---|---|---|
| fast | 0.1706 | 0.473 | 0.1307 | -0.0206 | 2 | 63 | 3 | 0.0 | 42.323 | 23.63 |
| robust | 0.28 | 0.4311 | 0.1213 | -0.1044 | 27 | 72 | 6 | 1.0 | 42.185 | 23.7 |
| rule | 0.28 | 0.4311 | 0.1213 | -0.1044 | 27 | 72 | 6 | 1.0 | 42.19 | 23.7 |
| difficulty | 0.2833 | 0.4408 | 0.1211 | -0.0906 | 26 | 76 | 6 | 0.9844 | 42.201 | 23.7 |
| learned | 0.2733 | 0.4681 | 0.1179 | -0.05 | 23 | 80 | 6 | 0.6772 | 42.527 | 23.51 |

## Difficulty estimator on the held-out clip(s) (AUC, 0.5 = chance; block-bootstrap 95% CI)

| clip | predicts 'FAST misses' | predicts 'ROBUST helps' |
|---|---|---|
| 7 | 0.438 [0.3351, 0.5559] | 0.4705 [0.3973, 0.5461] |

Difficulty model: tau=0.146 (never switches: False), cost weight lam=0.1, standardised coefficients {'brightness': -2.368, 'contrast': -3.8099, 'sharpness': 1.9999, 'frame_change': -0.0102, 'has_track': 0.2073, 'frames_since_seen': -0.0569, 'target_size': -0.1269, 'target_speed': -0.3262, 'last_conf': 0.0964, 'prev_robust': -0.3184}

## Paired comparisons over all eval frames (delta = first - second; 95% block bootstrap)

| comparison | delta detection recall [point, lo, hi] | delta total ms/frame [point, lo, hi] |
|---|---|---|
| robust_vs_fast | [0.1267, 0.0711, 0.1872] | [-0.1371, -0.3913, 0.1224] |
| difficulty_vs_fast | [0.1267, 0.0717, 0.1867] | [-0.1218, -0.3726, 0.1233] |
| difficulty_vs_robust | [0.0, -0.0022, 0.0022] | [0.0154, -0.0293, 0.0497] |
| rule_vs_fast | [0.1267, 0.0711, 0.1872] | [-0.1326, -0.3831, 0.1289] |
| learned_vs_fast | [0.1067, 0.06, 0.1606] | [0.2039, -0.0197, 0.4157] |
| learned_vs_robust | [-0.02, -0.0356, -0.005] | [0.3411, 0.1796, 0.4934] |

**Verdict, difficulty policy: INCONCLUSIVE** | learned tree: NOT SUPPORTED
Rule (uavtrack/protocol.py::verdict): SUPPORTED = better than FAST AND cheaper than ROBUST AND within 2 points recall of ROBUST; NOT SUPPORTED = not better than FAST, or clearly worse than ROBUST without being cheaper; else INCONCLUSIVE.
