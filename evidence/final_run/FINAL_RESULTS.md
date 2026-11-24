# FINAL RESULTS (generated from the files this machine produced)

Tag `default`. GPU: NVIDIA GeForce RTX 3050 6GB Laptop GPU; torch 2.10.0+cu128; ultralytics 8.4.197; weights sha256 c81e0d53e2155964...
Everything below is copied from JSON/CSV outputs; nothing is typed by hand. Offline research on recorded clips; clips 9-10 are touched only in the section marked TEST.

## Stage status

| stage | result | seconds |
|---|---|---|
| cache | OK | 310.5 |
| verify | OK | 23.1 |
| analyze | OK | 11.2 |
| tune | OK | 20.7 |
| eval_7_to_8 | OK | 25.7 |
| eval_8_to_7 | OK | 26.3 |
| demo | OK | 74.8 |
| cache_test | OK | 289.2 |
| eval_test | OK | 34.4 |
| demo_test | OK | 61.4 |

Replay equals true online run (verify-online, 150 frames): **True**

Tracker settings frozen on clip 7: {'high_conf': 0.4, 'max_age': 60, 'emit_coasting': False, 'chosen_on': 'clip 7 (validation), best ROBUST IDF1', 'IDF1_on_clip7': 0.1213}

## Held-out validation, train 7 -> eval 8

| policy | track recall | precision | IDF1 | MOTA | ID switches | fragmentations | predicted IDs | ROBUST frac | cost ms/frame | fps |
|---|---|---|---|---|---|---|---|---|---|---|
| fast (clip 8) | 0.0107 | 0.0412 | 0.0166 | -0.2396 | 1 | 20 | 4 | 0.0 | 45.342 | 22.05 |
| robust (clip 8) | 0.0829 | 0.1531 | 0.0496 | -0.3819 | 22 | 83 | 7 | 1.0 | 45.523 | 21.97 |
| rule (clip 8) | 0.0829 | 0.1531 | 0.0496 | -0.3819 | 22 | 83 | 7 | 1.0 | 45.517 | 21.97 |
| difficulty (clip 8) | 0.0803 | 0.1485 | 0.0463 | -0.3867 | 22 | 83 | 7 | 0.9933 | 45.563 | 21.95 |
| learned (clip 8) | 0.026 | 0.0915 | 0.0273 | -0.2328 | 2 | 32 | 5 | 0.2283 | 45.697 | 21.88 |

Difficulty-policy verdict: **INCONCLUSIVE**; learned-tree verdict: **INCONCLUSIVE**.
difficulty vs FAST (delta recall, cost): [0.1508, 0.1152, 0.1891] / [0.2204, -0.2763, 0.8204]; difficulty vs ROBUST: [-0.002, -0.0051, 0.0] / [0.0395, 0.0251, 0.054]; ROBUST vs FAST: [0.1528, 0.1164, 0.1916] / [0.181, -0.3138, 0.7811].
Difficulty AUC on held-out clip: clip 8: FAST-miss {'auc': 0.4671, 'ci95': [0.369, 0.5543], 'n': 1800, 'positives': 1701} , ROBUST-helps {'auc': 0.418, 'ci95': [0.3345, 0.5074], 'n': 1800, 'positives': 519}
Estimator coefficients: {'brightness': 1.6857, 'contrast': 1.2252, 'sharpness': 1.289, 'frame_change': 0.1111, 'has_track': 3.7891, 'frames_since_seen': 0.9409, 'target_size': -2.2496, 'target_speed': -0.0799, 'last_conf': -1.0027, 'prev_robust': -0.0161}

## Held-out validation, train 8 -> eval 7

| policy | track recall | precision | IDF1 | MOTA | ID switches | fragmentations | predicted IDs | ROBUST frac | cost ms/frame | fps |
|---|---|---|---|---|---|---|---|---|---|---|
| fast (clip 7) | 0.1706 | 0.473 | 0.1307 | -0.0206 | 2 | 63 | 3 | 0.0 | 42.323 | 23.63 |
| robust (clip 7) | 0.28 | 0.4311 | 0.1213 | -0.1044 | 27 | 72 | 6 | 1.0 | 42.185 | 23.7 |
| rule (clip 7) | 0.28 | 0.4311 | 0.1213 | -0.1044 | 27 | 72 | 6 | 1.0 | 42.19 | 23.7 |
| difficulty (clip 7) | 0.2833 | 0.4408 | 0.1211 | -0.0906 | 26 | 76 | 6 | 0.9844 | 42.201 | 23.7 |
| learned (clip 7) | 0.2733 | 0.4681 | 0.1179 | -0.05 | 23 | 80 | 6 | 0.6772 | 42.527 | 23.51 |

Difficulty-policy verdict: **INCONCLUSIVE**; learned-tree verdict: **NOT SUPPORTED**.
difficulty vs FAST (delta recall, cost): [0.1267, 0.0717, 0.1867] / [-0.1218, -0.3726, 0.1233]; difficulty vs ROBUST: [0.0, -0.0022, 0.0022] / [0.0154, -0.0293, 0.0497]; ROBUST vs FAST: [0.1267, 0.0711, 0.1872] / [-0.1371, -0.3913, 0.1224].
Difficulty AUC on held-out clip: clip 7: FAST-miss {'auc': 0.438, 'ci95': [0.3351, 0.5559], 'n': 1800, 'positives': 1365} , ROBUST-helps {'auc': 0.4705, 'ci95': [0.3973, 0.5461], 'n': 1800, 'positives': 311}
Estimator coefficients: {'brightness': -2.368, 'contrast': -3.8099, 'sharpness': 1.9999, 'frame_change': -0.0102, 'has_track': 0.2073, 'frames_since_seen': -0.0569, 'target_size': -0.1269, 'target_speed': -0.3262, 'last_conf': 0.0964, 'prev_robust': -0.3184}

## Integrated end-to-end run (true online, clip 8, policy difficulty, frames [1, 900])

Video: `results/final/demo/clip8_difficulty_integrated_demo.mp4`. IDF1 0.0829, MOTA -0.4945, ID switches 14, fragmentations 65, predicted IDs 3 (reference IDs in clip: 2), recall 0.1331, precision 0.1769.
Speed (decode + features + policy + detector + tracker; video writing excluded): 21.38 FPS, 46.779 ms/frame, ROBUST fraction 0.9922.

## TEST SET (clips 9-10, run once with frozen settings)

- clip 9 fast: recall 0.0, IDF1 0.0, MOTA 0.0, ID switches 0, cost 37.267 ms
- clip 9 robust: recall 0.0017, IDF1 0.0033, MOTA -0.0067, ID switches 0, cost 36.681 ms
- clip 9 rule: recall 0.0017, IDF1 0.0033, MOTA -0.0067, ID switches 0, cost 36.681 ms
- clip 9 difficulty: recall 0.0, IDF1 0.0, MOTA 0.0, ID switches 0, cost 36.716 ms
- clip 9 learned: recall 0.0, IDF1 0.0, MOTA 0.0, ID switches 0, cost 37.385 ms
- clip 10 fast: recall 0.0, IDF1 0.0, MOTA 0.0, ID switches 0, cost 37.739 ms
- clip 10 robust: recall 0.0, IDF1 0.0, MOTA 0.0, ID switches 0, cost 36.841 ms
- clip 10 rule: recall 0.0, IDF1 0.0, MOTA 0.0, ID switches 0, cost 36.84 ms
- clip 10 difficulty: recall 0.0, IDF1 0.0, MOTA 0.0, ID switches 0, cost 36.974 ms
- clip 10 learned: recall 0.0, IDF1 0.0, MOTA 0.0, ID switches 0, cost 37.817 ms
- verdict difficulty policy: NOT SUPPORTED

## Synopsis objective status (derived from the numbers above)

1. Detection + tracking pipeline with independent IDs: DEMONSTRATED on validation clips (quality = numbers above; low detector recall limits identity metrics).
2. Difficulty estimation from scale/motion/blur/illumination/confidence/track stability: estimator trained and evaluated on held-out clips; AUC [0.4671, 0.438]; NOT SUPPORTED (no interval above 0.5). Occlusion is not modelled.
3. Adaptive mode selection balancing quality and cost: verdicts ['INCONCLUSIVE', 'INCONCLUSIVE']. Only SUPPORTED means the adaptive policy beat FAST and was cheaper than ROBUST within 2 points recall; otherwise report as a negative/inconclusive result.

Expected outcome: working prototype with tracking + adaptive effort + accuracy/identity/speed/cost measures: DEMONSTRATED offline on recorded clips (see integrated run).

Limits that remain regardless of outcome: single dataset; detector trained 12 epochs; validation clips informed detector selection; frames without reference rows are not scored; costs are model-and-pipeline timings on one laptop GPU (no CPU/edge measurement unless --tag cpu was run); occlusion not modelled.