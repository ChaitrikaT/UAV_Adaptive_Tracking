# Condition-Aware Adaptive Perception for Air-to-Air UAV Tracking

A research-rigorous, demo-ready air-to-air UAV tracking framework combining:
- **Streamlit Live Dashboard** (`app.py`): Real-time tracking preview, per-frame mode timelines, target flight paths, online-replay verification badge, selector CV leaderboard, and side-by-side comparison.
- **10 Causal Condition Features** (`src/features/features.py`): `blur`, `mean_brightness`, `contrast`, `frame_change`, `box_area_ratio`, `mean_confidence`, `camera_motion` (DIS optical flow), `has_track`, `frames_since_seen`, and `target_speed` (+ 15-frame rolling averages).
- **Adaptive Mode Selector** (`src/selector/selector.py`):
  - `rules` (hand-set baseline)
  - `tree` (Decision Tree) and `forest` (Random Forest), trained on **ground-truth-derived labels** (label `ROBUST` when `FAST` misses a GT target in the next 15 frames).
- **Single-Class Drone Detection + Quiet Bird Filtering** (`src/detection/detector.py`, `config.yaml`):
  - Fine-tuned single-class (`drone`) YOLOv8n (`640px`, FAST) and YOLOv8s (`1280px`, ROBUST) trained strictly on the Purdue training split.
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

Evaluated with `python -m src.evaluation.eval_split --split test --max-frames 300 --repeats 5` (GPU: NVIDIA GeForce RTX 4050 Laptop GPU, shuffled order, 1 warm-up + 5 timing repeats, IoU=0.3):

| Configuration | HOTA | IDF1 | MOTA | IDSW | Recall % | Precision % | Detector Calls % | Time in ROBUST % | Median FPS (Min–Max) | Replay OK |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| **Always FAST** (`yolov8n`, 640px) | 2.88 | 6.32 | 3.82 | 40 | 6.85 | 59.35 | 33.30% | 0.00% | 29.26 (28.92–29.78) | — |
| **Always ROBUST** (`yolov8s`, 1280px) | 2.04 | 4.17 | 1.73 | 23 | 4.40 | 51.82 | 100.00% | 100.00% | 18.51 (18.35–18.64) | — |
| **Adaptive (Rules)** | 2.07 | 4.19 | 1.95 | 25 | 4.58 | 55.20 | 90.66% | 85.86% | 23.52 (22.98–23.73) | PASS |
| **Adaptive (Decision Tree)** | 2.36 | 4.88 | 2.01 | **23** | 4.93 | 58.36 | **72.26%** | **58.33%** | **23.50** (23.07–24.26) | PASS |
| **Adaptive (Random Forest)** | 2.03 | 4.41 | 1.96 | **23** | 4.66 | 53.15 | 82.63% | 73.90% | 20.00 (19.73–20.19) | PASS |

**Key Findings:**
- **ID Switches & Compute:** Both learned selectors (`Decision Tree` and `Random Forest`) cut ID switches from **40** (Always FAST) down to **23** (matching Always ROBUST) while reducing detector calls by **17.4%–27.7%** and boosting speed over Always ROBUST (`23.50 FPS` vs `18.51 FPS`).
- **Tiny-Target Bottleneck:** Purdue targets are often `~13×11 px` in `1080p` frames, keeping overall recall low (`4.4%–6.9%`) under default ByteTrack association thresholds; this motivates future work on ROI-cropped detection around predicted tracks.

---

## Quick Start

```powershell
# 1. Verify environment & CUDA
.\.venv\Scripts\python.exe run.py --check-setup

# 2. Launch Streamlit Demo UI
.\.venv\Scripts\streamlit.exe run app.py

# 3. Create leak-free 32/8/10 Purdue split & YOLO dataset
.\.venv\Scripts\python.exe -m src.training.prepare_splits

# 4. Fine-tune FAST (640px) & ROBUST (1280px) YOLO on GPU (train split only)
.\.venv\Scripts\python.exe -m src.training.finetune_detector --data-yaml data/yolo_drone_split/drone_dataset.yaml --base-weights weights/yolov8n.pt --out-weights weights/purdue_drone_yolov8n.pt --epochs 20 --imgsz 640 --batch 16
.\.venv\Scripts\python.exe -m src.training.finetune_detector --data-yaml data/yolo_drone_split/drone_dataset.yaml --base-weights weights/yolov8s.pt --out-weights weights/purdue_drone_yolov8s.pt --epochs 20 --imgsz 1280 --batch 4

# 5. Train Decision Tree & Random Forest selectors with GT labels
.\.venv\Scripts\python.exe -m src.training.train_selector --split-file data/splits/purdue_split.json --split train --gt-dir data/gt --variants 0 --max-frames 200 --models tree forest

# 6. Run held-out test evaluation (with val tuning, shuffled FPS, and 95% CIs)
.\.venv\Scripts\python.exe -m src.evaluation.eval_split --split test --max-frames 300 --repeats 5
```
