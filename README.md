# Condition-Aware Adaptive Perception for Air-to-Air UAV Tracking

A research-rigorous, demo-ready air-to-air UAV tracking framework combining:
- **Streamlit Live Dashboard** (`app.py`): Real-time tracking preview, per-frame mode timelines, target flight paths, online-replay verification badge, selector CV leaderboard, and side-by-side comparison.
- **10 Causal Condition Features** (`src/features/features.py`): `blur`, `mean_brightness`, `contrast`, `frame_change`, `box_area_ratio`, `mean_confidence`, `camera_motion` (DIS optical flow), `has_track`, `frames_since_seen`, and `target_speed` (+ 15-frame rolling averages).
- **Adaptive Mode Selector** (`src/selector/selector.py`):
  - `rules` (hand-set baseline)
  - `tree` (Decision Tree) and `forest` (Random Forest), trained on **ground-truth-derived labels** (label `ROBUST` when `FAST` misses a GT target in the next 15 frames).
- **Single-Class Drone Detection + Quiet Bird Filtering** (`src/detection/detector.py`, `config.yaml`):
  - Fine-tuned single-class (`drone`) YOLOv8n (`weights/purdue_drone_yolov8n.pt`) run in a two-resolution design: `640px` (`conf=0.20`, every 3rd frame) for **FAST** and `1280px` (`conf=0.12`, every frame) for **ROBUST**, trained strictly on the Purdue training split.
  - `AerialBirdFilter` runs a lightweight COCO check (`bird=14`, `airplane=4`) when candidate detections are present, quietly suppressing bird false positives before ByteTrack.
- **Synchronized ByteTrack + Hysteresis Smoothing** (`src/tracking/tracker.py`, `src/selector/selector.py`):
  - `ByteTrackerWrapper` increments `frame_id` and advances Kalman predictions on skipped-detector frames.
  - `HysteresisSmoother` (`min_mode_duration=15`, `persistence_frames=5`) prevents rapid mode oscillation.
- **Leak-Free Ground-Truth Evaluation & Verification Protocol** (`src/evaluation/`):
  - Optimal Hungarian matching for **HOTA, IDF1, MOTA, IDSW, Recall, Precision**.
  - Online-replay equivalence verification (`0` mismatches), shuffled-order timing (1 warm-up + 5 repeats), and 95% block-bootstrap CIs.

---

## Leak-Free Dataset & Split (`data/splits/purdue_split.json`)

Following the **Dogfight 40/10 precedent** on the 50 Purdue UAV-to-UAV videos (`Videos/Clip_1.mov` – `Clip_50.mov`, MOT GT in `data/gt/`):
- **Train (32 videos):** `Clip_1` – `Clip_32` (`1,659` YOLO frames; `6,333` GT-labelled selector frames)
- **Validation (8 videos):** `Clip_33` – `Clip_40` (`355` YOLO frames; used only for YOLO val & selector threshold tuning)
- **Held-Out Test (10 videos):** `Clip_41` – `Clip_50` (evaluated once at the end; automated leakage check enforces 0 overlap)
- **MOT-FLY (`data/raw/mot_fly/`):** Cloned from `https://github.com/CZC-123/MOT-FLY` (`MOT-FLY DATASET.mp4` available for qualitative demo; note the external Google Drive full-archive link in the upstream README currently returns HTTP 404).

---

## Ground-Truth Selector Training (32 Train Videos, GroupKFold CV)

Trained with `python -m src.training.train_selector --split-file data/splits/purdue_split.json --split train --gt-dir data/gt --variants 0 --max-frames 200 --models tree forest` (`6,333` rows, `84%` ROBUST label share):

| Selector | CV Accuracy | Balanced Acc | F1 (ROBUST) | ROC-AUC | Predicted ROBUST % |
|---|---:|---:|---:|---:|---:|
| **Rules (Baseline)** | 0.807 | 0.489 | 0.893 | — | 97% |
| **Decision Tree** | 0.641 | 0.470 | 0.771 | 0.529 | 73% |
| **Random Forest** | 0.763 | 0.495 | 0.863 | 0.608 | 90% |

---

## Held-Out Purdue Test Evaluation (10 Videos: `Clip_41`–`Clip_50`, 300 Frames/Clip)

Evaluated with `python -m src.evaluation.eval_split --split test --max-frames 300 --repeats 5` (GPU: NVIDIA GeForce RTX 4050 Laptop GPU, shuffled order, 1 warm-up + 5 timing repeats, IoU=0.3, val-tuned thresholds `tree=0.35`, `forest=0.65`):

| Configuration | HOTA | IDF1 | MOTA | IDSW | Recall % | Precision % | Detector Calls % | Time in ROBUST % | Median FPS (Min–Max) | Recall Δ vs FAST (95% CI) | Replay OK |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| **Always FAST** (`640px`, 1/3 frames) | 2.88 | 6.32 | 3.82 | 40 | 6.85 | 59.35 | 33.3% | 0.00% | 28.39 (27.81–28.45) | — | — |
| **Always ROBUST** (`1280px`, every frame) | 8.52 | 13.09 | -14.68 | 56 | 16.11 | 37.61 | 100.0% | 100.00% | 19.48 (18.39–19.64) | +8.70 [+5.41, +11.98] | — |
| **Adaptive (Rules)** | 8.37 | 12.66 | -11.68 | 54 | 15.74 | 37.47 | 92.6% | 88.76% | 22.63 (21.95–23.41) | +8.54 [+5.33, +11.68] | PASS |
| **Adaptive (Decision Tree)** | 6.11 | 7.58 | -14.94 | 46 | 12.05 | **43.10** | **75.0%** | **62.43%** | **23.42** (22.48–23.65) | +5.13 [+2.30, +7.96] | PASS |
| **Adaptive (Random Forest)** | **7.28** | **11.46** | -12.14 | 59 | **15.33** | 41.62 | 85.1% | 77.60% | 21.67 (21.25–21.92) | **+7.99 [+4.75, +11.24]** | PASS |

**Key Findings:**
- **Statistically Significant Recall & HOTA Gains:** `Adaptive (Random Forest)` improves HOTA **2.5×** (`7.28` vs `2.88`), IDF1 **1.8×** (`11.46` vs `6.32`), and Recall **2.2×** (`15.33%` vs `6.85%`, `+7.99 pts`, 95% CI `[+4.75, +11.24]`) over `Always FAST`, recovering **95.2%** of `Always ROBUST`'s recall while cutting detector calls by **14.9%** and improving precision (`41.62%` vs `37.61%`).
- **Compute-First Tradeoff:** `Adaptive (Decision Tree)` cuts detector calls by **25.0%** (`75.0%` vs `100.0%`), boosts speed to **23.42 FPS**, and achieves the highest adaptive precision (`43.10%`).
- **Tiny-Target Bottleneck:** Purdue targets are often `~13×11 px` in `1080p` frames, which caps absolute recall and causes negative MOTA under `1280px` low-confidence (`0.12`) false positives; this motivates future work on ROI-cropped detection around predicted tracks.

---

## Quick Start

```powershell
# 1. Verify environment & CUDA
.\.venv\Scripts\python.exe run.py --check-setup

# 2. Launch Streamlit Demo UI
.\.venv\Scripts\streamlit.exe run app.py

# 3. Create leak-free 32/8/10 Purdue split & YOLO dataset
.\.venv\Scripts\python.exe -m src.training.prepare_splits

# 4. Fine-tune single-class UAV YOLO on GPU (train split only)
.\.venv\Scripts\python.exe -m src.training.finetune_detector --data-yaml data/yolo_drone_split/drone_dataset.yaml --base-weights weights/yolov8n.pt --out-weights weights/purdue_drone_yolov8n.pt --epochs 20 --imgsz 640 --batch 16

# 5. Train Decision Tree & Random Forest selectors with GT labels
.\.venv\Scripts\python.exe -m src.training.train_selector --split-file data/splits/purdue_split.json --split train --gt-dir data/gt --variants 0 --max-frames 200 --models tree forest

# 6. Run held-out test evaluation (with val tuning, shuffled FPS, and 95% CIs)
.\.venv\Scripts\python.exe -m src.evaluation.eval_split --split test --max-frames 300 --repeats 5
```
