# Condition-Aware Adaptive Perception for Air-to-Air UAV Tracking

An end-to-end adaptive perception framework for air-to-air Unmanned Aerial Vehicle (UAV) tracking. On every video frame, the system extracts lightweight visual and temporal tracking cues prior to object detection, estimates frame difficulty using a calibrated logistic model, and dynamically switches between a fast low-resolution detector (`FAST`, 640 px) and a high-resolution detector (`ROBUST`, 960 px) while maintaining consistent target identities.

## Key Features

- **Online Condition Feature Extraction (`uavtrack/features.py`):** Computes 10 causal scene and track-state indicators (brightness, contrast, Laplacian sharpness, frame difference, prior target scale, velocity, detector confidence, and track age) with zero ground-truth leakage.
- **Calibrated Difficulty Estimator (`uavtrack/difficulty.py`):** Predicts the probability that `FAST` (640 px) detection will miss the target, driving a thresholded switching controller (`uavtrack/policy.py`).
- **Multi-Object UAV Tracker (`uavtrack/tracker.py`):** Associates YOLO11n detections across frames using IoU matching and configurable track persistence (`high_conf=0.4`, `max_age=60`).
- **Strict Reproducibility & Verification (`uavtrack/protocol.py`):** Verifies 100% equivalence between cached policy replay and true online frame-by-frame execution (0 mismatches across 150 frames) and enforces single-run held-out test evaluation.
- **Interactive Demo Dashboard (`demo_dashboard.html`):** Self-contained browser dashboard displaying annotated tracking videos, live difficulty timelines, benchmark tables, and diagnostic plots.

## Repository Structure

- `uavtrack/` — Core adaptive perception, difficulty, policy & tracking package
- `tests/` — Automated unit and protocol verification tests
- `evidence/final_run/` — Verified metrics, figures, alignment diagnostics & logs
- `docs/` — Execution guides and repository notes
- `run_evaluation.py` — Feature/detection caching, online verification & tuning
- `run_final_evaluation.py` — Held-out cross-validation & single-run test evaluation
- `make_integrated_demo.py` — End-to-end online video & log generator
- `render_demo_from_logs.py` — H.264 diagnostic video renderer from saved logs
- `check_test_clips.py` — Ground-truth alignment & temporal shift diagnostics
- `build_final_report.py` — Automated report generator (outputs `FINAL_REPORT.md`)
- `demo_dashboard.html` — Interactive results and video demo dashboard
- `FINAL_REPORT.md` — Complete experimental results and statistical analysis
- `requirements.txt` — Python dependencies

## Quick Start

1. Install dependencies: `pip install -r requirements.txt`
2. Run unit tests: `pytest tests/`
3. Open `demo_dashboard.html` in any browser or inspect `FINAL_REPORT.md` for full validation and test metrics.
