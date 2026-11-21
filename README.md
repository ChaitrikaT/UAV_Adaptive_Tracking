# Condition-Aware Adaptive Perception — offline mini-project

**Status:** research prototype with a saved prerecorded-video detector demonstration and a sampled-frame processing-mode study. **This repository does not yet implement independent UAV track IDs or end-to-end adaptive video processing.**

## Quick start: teammate demonstration

The `demo/` folder contains a pre-rendered annotated video, comparison figures and dashboard. You do **not** need CUDA, the Purdue dataset or a model checkpoint to view it.

1. Clone this repository and check out `feature/data-validation` (or the branch containing this code).
2. Install Python 3.
3. From the project root, run:

   ```powershell
   python src\check_project.py
   python run_demo.py --results-dir demo
   ```

4. The browser opens a local page. Keep the terminal open during viewing; use Ctrl+C to stop.

`run_demo.py` serves local files using only Python's standard library; this is not a separate hosted website. If the video does not play, check that `demo/videos/clip_7_detector_review_h264.mp4` exists. You can also open the MP4 directly in a media player.

## What source scripts do

| Entry point | Role |
| --- | --- |
| `src/prepare_dataset.py` | Prepare sampled and labelled frames from the local Purdue recordings |
| `src/train_detector.py` | One-class detector training experiment; requires a working GPU PyTorch environment |
| `src/review_detector_video.py` | Archived-video detector-versus-reference visual diagnostic; does **not** assign track IDs |
| `src/benchmark_modes.py` | Sampled **validation**-image FAST/ROBUST evaluation; not video FPS |
| `src/condition_analysis.py` | Whole-frame appearance summaries on an existing recorded clip |
| `src/train_selector.py` | Learn a decision tree from saved FAST/ROBUST evaluation results |
| `src/adaptive.py` | Replay selected modes against archived still-image outcomes; no runtime controller |
| `src/evaluate_saved_track_ids.py` | Score IDs supplied independently in a saved CSV; does not generate tracks |
| `src/finalize_project.py` | Rebuild `results/final/` using local experiment outputs (ignored by Git) |
| `src/check_project.py` | Verify that the committed viewer works and report what's absent |

`src/audit_*`, `src/verify_*`, and `src/inspect_*` are optional one-time data checks. **Don't rerun them during a routine demo.**

## Reproducing experiments (different from viewing the demo)

The `.gitignore` intentionally excludes `data/`, `datasets/`, `results/`, trained `*.pt` weights and virtual environments. Those are required for a *new experiment* but **not** for viewing `demo/`.

- Install a PyTorch build appropriate to your computer, following the official PyTorch installation selector; the Windows machine used for this project previously had a working CUDA environment. Avoid replacing a working CUDA build just to view the demo.
- Run `python -m pip install -r requirements.txt` in an experiment virtual environment.
- The prepared dataset configuration currently assumes local Purdue files. The provisional train/validation/test division is by video: Clips 1–6 / 7–8 / 9–10. Those clips have **not** been demonstrated to represent independent flights.
- The classifier was trained using Clip 7 saved outcomes and evaluated on Clip 8. Clip 8 was unseen to *selector training*, but both Clips 7 and 8 were validation clips for the detector. The selector results must **not** be called a final untouched detector test.
- Fixed-mode means compare input resolutions 640 and 960. Mean latency is a synchronized model-call measurement, **not** full video throughput. Feature extraction, decode, postprocessing and switching costs are not included.

## Verified results from the committed demo

- Clip 7 recorded-video diagnostic: 120 consecutive frames; 59 frames with ≥1 detector box; 43 frames with a maximum IoU ≥0.5; longest unmatched run frames 39–106 (68 frames).
- Clip 8 sampled-frame replay, 90 images: always FAST recall 2.27%, always ROBUST recall 14.20%, learned offline replay recall 6.82%. The learned policy **did not** outperform ROBUST.
- The initial detector validation mAP50 was reported as about 12.3%; this checkpoint and its training logs are not contained in the repository archive.

## Remaining work against the original synopsis

1. **Independent video tracking IDs:** not implemented for Purdue footage. No saved Purdue reference IDs may be described as predicted IDs.
2. **Integrated video mode selection:** not implemented. A replay of saved image outcomes is not an active controller.
3. **Tracking quality and end-to-end runtime evaluation:** not available. In particular, the committed report does not establish IDF1/HOTA/MOTA, real-time FPS or reliable small-target localization.
4. **Generalization and publication:** only a limited development comparison is available, not a multi-sequence unseen-test result or a demonstrated improvement over fixed ROBUST.

These limitations should be stated explicitly to a guide or conference reviewer.

## Data credits

The prerecorded imagery is derived from Purdue's UAV-to-UAV Tracking Dataset. See `demo/PURDUE_DATA_LICENSE.txt` for the bundled notice and https://engineering.purdue.edu/~bouman/UAV_Dataset/ for its source. Verify the applicable publication/attribution conditions when redistributing derived clips.

## Team collaboration and handoff

Start at [`TEAM_HANDOFF.md`](TEAM_HANDOFF.md) for the current checkpoint, exact commands,
team task assignments, and honest status of remaining milestones. This research package
can be run without the private dataset or trained checkpoint.
