# v3: apply, verify, and run the real-data evaluation (Windows PowerShell)

Nothing below has been run on your recorded data by me. Every real-data result must come from your machine.

## A. Apply to the development copy (original folder untouched)
```powershell
cd "D:\UAV_Adaptive_Tracking - Copy"
$orig_before = (Get-ChildItem "D:\UAV_Adaptive_Tracking" -Recurse -File | Measure-Object LastWriteTime -Maximum).Maximum
$stamp = Get-Date -Format yyyyMMdd_HHmm; $bk = "D:\UAV_backup_$stamp"; New-Item -ItemType Directory $bk | Out-Null
foreach ($p in "uavtrack","tests","run_tracking.py","train_online_policy.py","run_video_demo.py","demo_archived_review.py","make_synthetic_demo.py","run_evaluation.py") { if (Test-Path $p) { Copy-Item $p $bk -Recurse -Force } }
Expand-Archive -Force "$HOME\Downloads\UAV_Tracking_Addon_v3.zip" .
python -m unittest discover -s tests 2>&1 | Select-Object -Last 4
$orig_after = (Get-ChildItem "D:\UAV_Adaptive_Tracking" -Recurse -File | Measure-Object LastWriteTime -Maximum).Maximum
"original folder unchanged: " + ($orig_before -eq $orig_after)
```
Expected: `Ran 83 tests ... OK (skipped=1)` (your 17 repo tests + 66 add-on tests; the skip is the optional motmetrics cross-check) and `original folder unchanged: True`.
Optional, 1 minute: `python -m pip install motmetrics` then re-run the tests; the oracle test then runs instead of skipping. I did not run it (no network); the patch author reports 0 disagreements after the fix.

## B. Real-data evaluation (needs best.pt, Clip_7/8 videos + annotations in `data\`)
Time: about 2 detector calls per frame. Add `--end 300` to every `cache` command for a quick first pass.
```powershell
# 1. Run the detector in FAST and ROBUST on every frame, once. Stores detections + timings + provenance.
python run_evaluation.py cache --clips 7 8
#    expect per clip: "N frames cached, M with reference boxes"; writes results\policy\cache_clip_*_default.pkl, results\evaluation\environment_default.json

# 2. Prove the cached replay equals a true online run on real frames (must say replay_equals_online: true)
python run_evaluation.py verify-online --clip 8 --end 150 --policy rule

# 3. Detection P/R/AP50, condition descriptors vs detection success, can conditions predict FAST misses?
python run_evaluation.py analyze --clips 7 8
#    writes results\evaluation\ANALYSIS_default.md

# 4. Choose tracker settings on clip 7 ONLY, then freeze them
python run_evaluation.py tune --clip 7
#    prints "Best ROBUST setting ...: --high-conf X --max-age Y [--emit-coasting]"; use those flags below

# 5. Train on one validation clip, evaluate on the other (both directions), same frozen flags
python run_evaluation.py policy --train 7 --eval 8 --high-conf 0.3 --max-age 30
python run_evaluation.py policy --train 8 --eval 7 --high-conf 0.3 --max-age 30
#    writes results\evaluation\policy_train7_eval8_default.md : tracking IDs/IDF1, cost, 95% CIs, verdict

# 6. True online end-to-end run (independent of the replay), with annotated video
python run_tracking.py --clip 8 --end 300 --policy all --save-video --high-conf 0.3 --max-age 30

# 7. Cost on CPU, where 960 vs 640 should differ much more than on your GPU (use a short slice)
python run_evaluation.py cache --clips 7 8 --end 300 --device cpu --tag cpu
python run_evaluation.py policy --train 7 --eval 8 --tag cpu --high-conf 0.3 --max-age 30
```
## C. Reserved test set: ONLY after steps 1-7, ONCE, with frozen settings
```powershell
python run_evaluation.py cache --clips 9 10 --allow-test
python run_evaluation.py policy --train 7 8 --eval 9 10 --final --high-conf 0.3 --max-age 30
```
`results\evaluation\test_runs.log` records every test run; a second entry means the held-out claim is void.

## D. Return these files (small; do NOT send the .pkl caches)
```powershell
Compress-Archive -Force -DestinationPath D:\UAV_results_return.zip -Path results\evaluation, results\policy\*_report.json, results\policy\*_tree.txt, results\tracking\*\comparison.csv
```
Send `D:\UAV_results_return.zip`. Also tell me: GPU model, number of frames in clips 7-10, and whether `verify-online` printed `replay_equals_online: true`.
