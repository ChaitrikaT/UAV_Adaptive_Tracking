# Corrections to the first add-on (v2)

Overwrite the files from the first ZIP with these. Nothing outside `uavtrack/`, the listed scripts, `tests/`, and `demo/` is touched.

## Confirmed bugs (each has a test that FAILED on the old code, except where noted)
| # | Where | Problem | Fix |
|---|---|---|---|
| 1 | `evaluate.py` fragmentations | A target picked up late (first frames missed) was counted as a fragmentation. Standard Frag = tracked, lost, then tracked again. | Count only resumptions after having been tracked. Test: `test_leading_miss_is_not_a_fragmentation` |
| 2 | `evaluate.py` IDF1 | Two boxes sharing one predicted ID in the same frame were counted twice in IDTP, so IDF1 could exceed 1. | Count each (reference ID, predicted ID) pair once per frame. Test: `test_duplicate_prediction_same_id_not_double_counted` |
| 3 | `collect.py` / training | `prev_robust` was constant (1) in training but varies online; tracker-state features came from ROBUST-only detections that the deployed policy does not reproduce. | Rewritten: both modes cached once; training rolls the policy itself over the cache with the same `build_features` code the online pipeline uses (DAgger-style, 2 iterations). Tests: `test_training_rollout_matches_online_pipeline`, `test_prev_robust_is_not_constant_in_training_rows` |
| 4 | labels | Frames where both modes found nothing were labelled FAST by default. | `label_frame`: see tie rules below. Test: `test_label_table`, `test_both_fail_frames_are_excluded_not_defaulted_to_fast` |

## Verified correct (no change needed)
MOTA = 1 - (FN+FP+IDSW)/GT, ID-switch counting (with CLEAR-MOT "keep previous match" rule), precision/recall, and IDF1 global assignment. IDF1 was checked against an exhaustive-assignment brute force on 150 random cases plus hand-computed cases for missed detections, identity change, duplicates, gaps, and crossing swaps. NOT cross-checked against the `motmetrics` library (no network here).

## Tie handling in training labels
- ROBUST hits more reference boxes -> ROBUST; FAST hits more -> FAST.
- Both hit the same number (> 0) -> FAST (same outcome, cheaper). `--tie-equal-hit robust` flips it.
- Both hit nothing -> EXCLUDED (no evidence which mode is better). `--tie-both-fail fast|robust` forces a label.
- Consequence: the tree never learns from the hardest frames. This is a limitation, not a fix.

## Leakage review
Features contain no reference annotations and no future frames (test: `test_features_do_not_depend_on_reference_annotations`). Contiguous-block CV is reported because neighbouring frames are near-duplicates. Training accuracy on the training clip is not reported as performance.

## New option found by the synthetic demo (not a bug fix)
`TrackerConfig.state_mode="worst"`: policy features from the hardest confirmed track. The default `"primary"` is unchanged. With `"primary"` and two targets, the rule policy did not react when the second target became invisible to FAST (synthetic result: robust fraction 3%, same recall as always-FAST).

## Suspected, NOT changed
- Clips 7/8 were used for detector validation/early stopping, so they are not a clean test of the detector.
- `src/modes.py` (original offline experiment) has the same default-to-FAST tie rule. Left untouched.
- Sharpness in the saved Clip 7 condition CSV shows regular spikes about every 10 frames; possible codec/keyframe artifact (unconfirmed).
- Cached replay equals online output only if the detector is deterministic per (frame, size); check on real data.

---
# v3: consolidation with UAV_Correctness_Patch.zip + real-data evaluation tools

## Regression found in v2 by the patch (CONFIRMED, reproduced here)
v2 `evaluate()` solved the per-frame assignment without gating by IoU >= thr, then discarded invalid pairs.
Patch case: ref [32,44] and [28,40], predictions [32,44] and [33,45] -> v2 gave TP=1, correct is TP=2.
Same flaw existed in the greedy `count_hits` used for training labels.

## What was merged
| Component | Source | Note |
|---|---|---|
| `evaluate.py` | patch (gated CLEAR-MOT, input validation, protocol label, IDTP/IDFP/IDFN) + v2 fragmentation fix | The patch still counted a late first pickup as a fragmentation; the v2 fix is re-applied |
| `labels.py` | patch + `both_fail` / `equal_hit` options from v2 | Defaults identical to the patch |
| `collect.py` | v2 cache + rollout + DAgger (kept) | `count_hits` now uses the patch's optimal gated matching; `collect_clip` kept as a thin wrapper over cache+rollout so the patch tests still apply; multi-clip training added |
| `train_online_policy.py` | v2 (kept) | cache file names carry a `--tag`; patch's `--driver/--aggregate` path is superseded by DAgger over the cache |
| tests | v2 + all 22 patch tests | duplicate same-ID prediction is now a ValueError (patch behaviour), v2 test updated |

## New (v3)
`uavtrack/protocol.py` and `run_evaluation.py` (`cache`, `analyze`, `tune`, `policy`, `verify-online`): detection P/R/F1/AP50 per mode, condition descriptors vs detection success and cross-clip AUC, tracker tuning on one validation clip, FAST/ROBUST/rule/learned comparison with tracker IDs, composed cost, block-bootstrap CIs and a pre-stated verdict rule, test-set guards, provenance (weights hash, video hash, versions). `run_tracking.py` gained `--state-mode`.

## Not verified by me
Everything in `run_evaluation.py` was exercised only with a synthetic fake detector. No result on real clips exists.
