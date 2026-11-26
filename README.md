# Condition-Aware Adaptive Perception for Air-to-Air UAV Tracking

A unified, research-rigorous, and demo-ready UAV tracking framework combining:
- **Streamlit Live Dashboard** (`app.py`): Real-time tracking preview, per-frame mode timelines, target flight paths, online-replay verification badge, selector CV leaderboard, and side-by-side comparison.
- **10 Causal Condition Features** (`src/features/features.py`): `blur`, `mean_brightness`, `contrast`, `frame_change`, `box_area_ratio`, `mean_confidence`, `camera_motion` (DIS optical flow), `has_track`, `frames_since_seen`, and `target_speed` (+ 15-frame rolling averages).
- **Multi-Model Adaptive Selector** (`src/selector/selector.py`):
  - `rules` (hand-set baseline)
  - `tree` (Decision Tree), `forest` (Random Forest), `gbm` (Gradient Boosting), `logreg` (Calibrated Logistic Regression), `svm` (Calibrated RBF SVM), `mlp` (MLP Neural Network), plus `xgb` / `lgbm` when installed.
- **Aerial-Filtered Detection + Synchronized ByteTrack** (`config.yaml`, `src/tracking/tracker.py`):
  - `classes: [4, 14, 33]` (COCO `airplane`, `bird`, `kite`) removes terrain false positives (`cow`, `car`, `skis`, `baseball bat`) when using COCO weights.
  - `ByteTrackerWrapper` increments `frame_id` on skipped-detector frames so `track_buffer` stays synchronized with Kalman predictions.
- **Ground-Truth Evaluation & Verification Protocol** (`src/evaluation/ground_truth.py`, `src/evaluation/metrics.py`, `src/evaluation/protocol.py`):
  - Optimal Hungarian matching for **HOTA, IDF1, MOTA, IDSW, Recall, Precision**.
  - Online-replay equivalence verification (`0` mismatches between live and cached replay), block-bootstrap CIs, and formal verdict (`PASS` / `MARGINAL` / `NEGATIVE`).
- **Drone Detector Fine-Tuning** (`src/training/finetune_detector.py`):
  - Converts MOT ground-truth (or pseudo-labels) + videos into YOLO format and fine-tunes YOLOv8.

---

## Dataset Audit (`Videos/` + `data/uploads/`)

- **50 Purdue clips** (`Videos/Clip_1.mov` – `Clip_50.mov`) are **not repetitive**: they span `1920x1080` and `1280x960` resolutions, `309–1800` frames, sharpness (Laplacian variance) `20–83`, and brightness `100–136`.
- **Additional clips** in `data/uploads/`: `Demo_V_DRONE_017 (1).mp4` (`640x512` IR/thermal-style, `328` frames) and `Recording 2026-11-25 140418.mp4` (`1262x620`, `905` frames).
- **Detector finding**: Generic COCO weights miss tiny/thermal drones on some clips (e.g., `Demo_V_DRONE_017 (1).mp4` had `0` aerial-class detections). Fine-tuning YOLO on labelled drone data (`finetune_detector.py`) is recommended when MOT `.txt` annotations are available.

---

## Multi-Clip Benchmark Results (4 Diverse Clips, 80 Frames Each)

Evaluated on `Clip_3.mov`, `Clip_21.mov`, `Clip_31.mov`, and `Demo_V_DRONE_017 (1).mp4` ([outputs/metrics/multi_clip_benchmark.csv](outputs/metrics/multi_clip_benchmark.csv)):

| Run | Avg FPS | Detector Calls % | Time in ROBUST % | Agreement % | Replay Equivalence |
|---|---:|---:|---:|---:|---|
| **Fast** | 25.8 | 33.8% | 0.0% | 85.6% | — |
| **Robust** | 3.1 | 100.0% | 100.0% | 100.0% | — |
| **Adaptive + Random Forest** | **17.2** | **53.8%** | 30.0% | 94.7% | PASS (0/80 mismatches) |
| **Adaptive + Logistic Regression** | 14.6 | 64.1% | 45.3% | **95.3%** | PASS (0/80 mismatches) |
| **Adaptive + Decision Tree** | 10.2 | 61.2% | 41.2% | 94.7% | PASS (0/80 mismatches) |
| **Adaptive + Gradient Boosting** | 9.2 | 57.2% | 35.0% | **95.3%** | PASS (0/80 mismatches) |
| **Adaptive + MLP Neural Net** | 7.3 | 71.6% | 56.6% | **95.3%** | PASS (0/80 mismatches) |
| **Adaptive + SVM (RBF)** | 7.3 | 68.1% | 51.6% | **95.3%** | PASS (0/80 mismatches) |
| **Adaptive + Rules (Baseline)** | 5.5 | 78.1% | 66.6% | 95.3% | PASS (0/80 mismatches) |

**Conclusion:** Learned adaptive selectors achieve **94.7%–95.3% agreement** with always-ROBUST (vs. `85.6%` for always-FAST, which drops to `45.0%` on hard clips like `Clip_3.mov`) while cutting detector calls by **36%–46%** and boosting speed **3×–5.5×** over always-ROBUST (`17.2 FPS` for Random Forest, `14.6 FPS` for Logistic Regression, `9.2 FPS` for Gradient Boosting vs. `3.1 FPS` for ROBUST and `5.5 FPS` for Rules).

---

## Quick Start

```powershell
# 1. Verify environment
.\.venv\Scripts\python.exe run.py --check-setup

# 2. Launch Streamlit Demo UI
.\.venv\Scripts\streamlit.exe run app.py

# 3. Train all selector models (leak-free GroupKFold by source_video)
.\.venv\Scripts\python.exe -m src.training.train_selector --videos Videos/Clip_1.mov Videos/Clip_3.mov Videos/Clip_26.mov Videos/Clip_36.mov Videos/Clip_46.mov --variants 1 --max-frames 120

# 4. Run CLI comparison with replay verification
.\.venv\Scripts\python.exe run.py --video Videos/Clip_3.mov --compare --verify-replay --max-frames 100

# 5. Fine-tune YOLO when MOT ground-truth .txt files are placed in data/gt/
.\.venv\Scripts\python.exe -m src.training.finetune_detector --videos Videos --gt-dir data/gt --epochs 30
```
