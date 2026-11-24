# Completion kit: run the real-data research pipeline (overlay on v3)

**Status of this kit: written and compiled only. It has NOT been run, not even on synthetic data, and never on your clips.**
If a stage fails, `results\final\run_log.txt` shows the error; send it to me.

## What it adds (v3 files are not modified)
| File | Purpose | Synopsis item |
|---|---|---|
| `uavtrack/difficulty.py` | explicit difficulty score (probability FAST misses) from scale, motion, blur, illumination, detector confidence, track stability + the policy that switches FAST/ROBUST with it | Objective 2, 3 |
| `run_final_evaluation.py` | trains on one validation clip, evaluates FAST / ROBUST / rule / learned / DIFFICULTY on the other: tracker IDs, IDF1, MOTA, ID switches, cost, 95% CIs, verdict, figures | Objective 1, 2, 3 |
| `make_integrated_demo.py` | end-to-end run on a recorded clip: features -> difficulty -> mode -> YOLO -> tracker IDs -> annotated video + metrics | Expected outcome |
| `run_full_project.py` | runs everything in order and writes `results\final\FINAL_RESULTS.md` | all |

## Apply (development copy only; original folder untouched)
```powershell
cd "D:\UAV_Adaptive_Tracking - Copy"
Expand-Archive -Force "$HOME\Downloads\UAV_Completion_Kit.zip" .
python -c "import uavtrack.difficulty, run_final_evaluation, make_integrated_demo, run_full_project; print('imports ok')"
```
Expected: `imports ok`.

## Run
```powershell
python run_full_project.py --quick     # ~300 frames per clip, tag "quick": first pass, to catch problems early
python run_full_project.py             # full clips 7 and 8 (2 detector calls per frame for caching; the demo is 900 frames)
```
Expected files in `results\final\`: `FINAL_RESULTS.md`, `stage_status.json`, `run_log.txt`, `frozen_settings_<tag>.json`,
`final_eval_train7_eval8_<tag>.{json,md}`, `final_eval_train8_eval7_<tag>.{json,md}`, `difficulty_scores_*.csv`, `figures\*.png`,
`demo\clip8_difficulty_integrated_demo.mp4` (+ `_frames.csv`, `_tracks.csv`, `_id_summary.csv`, `_metrics.json`).
Open the video: `start results\final\demo\clip8_difficulty_integrated_demo.mp4`.

Validity checks before trusting anything:
1. `stage_status.json`: every stage `ok: true`.
2. `FINAL_RESULTS.md` line "Replay equals true online run": must be **True**.
3. `results\evaluation\ANALYSIS_<tag>.md`: clip-7 ROBUST frame-hit numbers should be in the same range as the saved review (43 of the first 120 frames matched at conf 0.10).
4. In the demo video, orange boxes carry predicted IDs; green boxes are the reference. The ID should stay the same across a gap unless it is a genuine switch.
If `quick` fails with "Only one difficulty class" the 300-frame slice is too short or too easy/hard; use the full run.

## Reserved test set (clips 9-10): ONLY after the full validation run, ONCE
```powershell
python run_full_project.py --final --confirm-test
```
Uses tracker settings frozen on clip 7 and a model trained on clips 7+8. Logged in `results\final\test_runs.log`. A second run voids the held-out claim.

## Return to me
```powershell
Compress-Archive -Force -DestinationPath D:\UAV_final_return.zip -Path results\final, results\evaluation\ANALYSIS_*.md, results\evaluation\environment_*.json, results\evaluation\verify_online_*.json
```
(Do not include the `.pkl` caches or the full video if it is large; the `_metrics.json`, CSVs and `FINAL_RESULTS.md` are what matter.)

## How results map to the synopsis (the report derives this automatically; it never claims more than the numbers show)
- Obj 1: predicted IDs from the tracker, IDF1/MOTA/ID switches/fragmentations vs reference, per policy and clip.
- Obj 2: difficulty score; held-out AUC (with CI) for "FAST misses" and "ROBUST helps"; standardised coefficients show which conditions drive it. Occlusion is not modelled.
- Obj 3: online FAST/ROBUST switching; paired-bootstrap comparison with a stated verdict rule; cost measured end to end.
- Expected outcome: integrated demo video + metrics (accuracy, identity consistency, speed, cost). Edge deployment stays future work. For a CPU cost figure run `python run_evaluation.py cache --clips 7 8 --end 300 --device cpu --tag cpu` then `python run_final_evaluation.py --train 7 --eval 8 --tag cpu`.

## Honest expectations
Detector recall is low (about 37% on clip 7 and 14% on clip 8 for ROBUST on sampled stills), so identity metrics will be limited and a negative or inconclusive adaptive-policy result is a real possibility. That is a reportable finding, not a failure of the kit.
