# TEAM HANDOFF — condition-aware adaptive perception

**Checkpoint:** 14 Nov 2026. Based on the `project_code_review.zip` snapshot from
GitHub branch `feature/data-validation`, commit `1be5a02`. The team has **not**
subsequently pushed the review or research add-ons at the time of this handoff.
Use `git log -1 --oneline` to verify which commit you have; don't assume `main`
contains the project work. New commits after this checkpoint are not included.

## What's genuinely implemented

- Purdue annotations prepared and validated on the originating computer.
- One-class YOLO11n detector fine-tuned and tested there; checkpoint is **not** in Git.
- Saved Clip 7 detection review video plus dashboard **are in Git**; no CUDA or data required.
- A paired FAST(640)/ROBUST(960) still-image benchmark, and an ML selector trained
  on Clip 7 preferences and evaluated offline on Clip 8.
- Initial whole-frame visual-condition analysis.
- This patch adds regression tests, corrected documentation, project preflight,
  offline CSV audits, independent *saved* track-ID scoring, and consolidated reports.

## Not yet demonstrated — do not claim these results

- No independently predicted IDs for Purdue UAVs were produced by the committed
  code. The separate synthetic demo is educational, not a real UAV tracker.
- No closed-loop integrated detector/tracker/selector processing and no real-time FPS.
- No final untouched detector test result or proven advantage of adaptive ML.
- No HOTA/IDF1/MOTA results established by the saved evaluation utilities.

## Start in a clean checkout

PowerShell in your **own** working directory:

```powershell
git clone --branch feature/data-validation --single-branch https://github.com/ChaitrikaT/UAV_Adaptive_Tracking.git UAV_Adaptive_Tracking
cd .\UAV_Adaptive_Tracking
python run_demo.py --results-dir demo
```

The ZIP install is described in the handoff ZIP's `START_HERE.md`. If working
from a manually copied directory instead of a clone, use that directory as the
`--repo` argument; **do not recopy data** just to use the offline viewer.

After installing the patch, run:

```powershell
python team_preflight.py
python src\check_project.py
python -m unittest discover -s tests -v
python run_research_review.py --example
python run_research_review.py
```

`--example` uses invented toy data and is a code smoke test only. A fresh
checkout contains no `results/offline_benchmark/per_frame.csv`, no
`results/selector/mode_decisions.csv`, and no `best.pt`. Therefore the real
report will *explicitly skip* absent research sections until someone supplies
saved results. This is expected; it is **not a successful research evaluation**.

## Sharing saved outputs safely

From the original research machine, **only if the team has permission**, share
these already-generated CSVs with the teammates who will analyze them:

- `results/offline_benchmark/per_frame.csv`
- `results/selector/mode_decisions.csv`
- `results/selector/experiment.json` (if present)
- `results/videos/clip_7_detector_review.csv` (if present)
- `results/conditions/clip_7_condition_analysis.csv` (if present)

Do not send the 1.9 GB dataset, private local files, personal credentials, or a
trained checkpoint just to make the **offline** report work. Keep the Purdue
redistribution/attribution requirements in mind for any imagery shared.
The research report checks input hashes so results can be traced to source files.

## Exact current evidence (from project notes and saved demo)

| Experiment | Observed result | Correct interpretation |
|---|---|---|
| Detector, initial validation | P 30.9%, R 24.1%, mAP50 12.3%, mAP50-95 2.4% | Detector quality limited; checkpoint and logs are local |
| Clip 7, 120 consecutive video frames | 59 frames with a detector prediction; 43 matched at IoU>=0.5; 68-frame unmatched interval | **Detection**, not independent tracking |
| Clip 8 replay, 90 sampled images | FAST R 2.27%; ROBUST R 14.20%; LEARNED R 6.82% | Learned selector does **not** outperform ROBUST |
| Initial video conditions | Global brightness/contrast/sharpness/frame-change averages similar across matches and failures | No robust condition predictor established |

Recall uses sampled, IoU-matched *detection* boxes. Model-call latency excludes
video decoding, feature computation, switching and tracking. Clip 8 is held out
from selector training but was part of detector validation. These are
**preliminary development** measurements, not final untouched-test claims.

## Working rules

1. Do not modify or commit data, results, video checkpoints, model weights, `.venv`.
2. Use a separate feature branch and PR for each non-overlapping task.
3. Run offline tests before sharing a PR; record exact command and result.
4. Do not put Purdue ground-truth IDs into a *predicted* track CSV.
5. Do not rebrand offline replay as active mode switching or demo-only data as experimental generalization.
6. For remaining end-to-end air-to-air tracking or control requirements, discuss
   safe project scope and independent validation requirements with the guide.

See `TEAM_TASKS.md` for the work split and completion checklist.
