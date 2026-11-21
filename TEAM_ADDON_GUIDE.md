# Offline Research Add-on — 2026-11-11

This add-on is based on inspection of **your `project_code_review.zip`** and **your teammate's `teammate_code_review.zip`**. It deliberately adds reusable review code instead of replacing the training or video scripts that already run on your machine.

## What it adds

- `research_addon/benchmark.py`: strict paired FAST/ROBUST CSV validation; catches duplicate modes, mismatched ground-truth counts, missing pairs and nonsensical latencies.
- `research_addon/policy_audit.py`: recomputes always-FAST, always-ROBUST and saved-learned-policy outcomes from the same recorded evaluation frames; enforces split consistency, all decisions present, and no training/eval clip overlap.
- `research_addon/track_audit.py`: independent **retrospective evaluation only** of a separately saved predicted-track CSV against MOT-format reference TXT, including identity changes across missing observations. Does **not** generate IDs or run a tracker.
- `research_addon/report.py`: self-contained HTML, Markdown and JSON reports, with SHA256 input provenance.
- `research_addon/timeline.py`: joins your **existing** Clip 7 detector video diagnostic CSV with the condition-analysis CSV, showing unmatched intervals without new detector inference.
- `run_research_review.py`: single CLI entry point.
- `tests/test_research_addon.py`: regression tests.
- `examples/`: clearly invented toy CSVs, to test the program even without Purdue or any saved predictions.

## Install without overwriting your project

1. Back up / commit your current work if desired.
2. Extract this ZIP in the **project root** `D:\UAV_Adaptive_Tracking`. It creates the `research_addon/`, `tests/`, `examples/` folders and `run_research_review.py`.
3. Run these from PowerShell, one command per line:

```powershell
cd D:\UAV_Adaptive_Tracking
python -m unittest discover -s tests -p test_research_addon.py -v
python run_research_review.py --example
python run_research_review.py
Start-Process .\results\research_review\index.html
```

If your existing `results/videos/clip_7_detector_review.csv` and `results/conditions/clip_7_condition_analysis.csv` are present, it also creates `results/research_review/timeline_review.csv` and reports the longest unmatched interval.

The last command opens your actual saved-results report **if** your local `results/offline_benchmark/per_frame.csv` and `results/selector/mode_decisions.csv` exist.

The `--example` command writes to `results/research_review_example/` using **invented** data and should never be used as a real project result.

No new dependencies are required: Python standard library only. Inference needs your existing separate project environment; this add-on does not execute the detector.

## Optional independent saved track audit

Only once you possess true independent predicted track IDs from a separately verified benign research system:

```powershell
python run_research_review.py --tracks results/saved_predictions/clip7_tracks.csv --reference data/annotations/Clip_7_refined.txt --clip 7
```

Expected saved prediction format (`x,y,w,h` is pixel-space top-left with width and height):

```
clip,frame,track_id,x,y,w,h
7,1,42,10,20,12,10
```

The audit is not IDF1/HOTA or tracking implementation. Do not put Purdue reference IDs into the predicted CSV. The supplied evaluator performs one-to-one match for retrospective scoring only.

## What is unchanged / still missing

- Your existing trained `best.pt`, video and saved dashboard are left untouched.
- The selector is still an offline replay; not an online adaptive controller.
- The existing trained detector has weak localization results; this add-on does not claim it improves detection.
- No real air-to-air tracking with independently generated IDs or end-to-end tracking-quality benchmark is provided.
- Teammate Streamlit UI and tracker wrappers have **not** been copied; their internal compatibility and real UAV outcomes were unverified.
- The evaluation clip may have been seen during detector validation, so **do not call it a final untouched test result**.

### For the guide

Use `demo/videos/clip_7_detector_review_h264.mp4` for the existing recorded video and `results/research_review/research_report.md` for a reproducible analysis. The new report recalculates results directly from your own CSV and labels its limitations explicitly.
