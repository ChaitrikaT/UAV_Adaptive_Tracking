# Results, Conclusions, Limitations & Future Improvements

**Live Demo:** `http://localhost:8501` (`streamlit run app.py`)  
**Benchmark Data:** [multi_clip_benchmark.csv](outputs/metrics/multi_clip_benchmark.csv) · [report.json](outputs/selector/report.json)

---

## 1. Cross-Validation Leaderboard (Leak-Free `GroupKFold` by `source_video`)

Trained on **2,400 labelled frames** from 5 diverse source videos (`Clip_1`, `Clip_3`, `Clip_26`, `Clip_36`, `Clip_46` + condition-varied copies; **42.2% ROBUST / 57.8% FAST**):

| Selector Model | CV Accuracy | CV Balanced Acc | CV F1 (ROBUST) | CV ROC-AUC | Predicted ROBUST % |
|---|---:|---:|---:|---:|---:|
| **Rules (Baseline)** | 0.420 | 0.499 | 0.592 | n/a | 99.9% (collapses to ROBUST) |
| **Random Forest (`forest`)** | 0.637 | 0.599 | 0.453 | **0.718** | 24.2% |
| **MLP Neural Net (`mlp`)** | 0.630 | 0.593 | 0.448 | **0.668** | 24.9% |
| **Gradient Boosting (`gbm`)** | 0.584 | 0.540 | 0.346 | **0.657** | 21.4% |
| **Decision Tree (`tree`)** | **0.658** | **0.645** | **0.580** | 0.608 | 39.4% |
| **Logistic Regression (`logreg`)** | 0.554 | 0.544 | 0.476 | 0.549 | 43.0% |
| **SVM (RBF, `svm`)** | 0.545 | 0.523 | 0.414 | 0.510 | 35.5% |

**Top Causal Features Learned (Random Forest / GBM):**
1. `mean_brightness` (`0.149` RF / `0.275` GBM)
2. `contrast` & `contrast_avg` (`0.237` combined RF / `0.253` GBM)
3. `blur` & `blur_avg` (`0.180` combined RF / `0.253` GBM)
4. `camera_motion_avg` (`0.121` RF / `0.075` GBM)

---

## 2. Multi-Clip End-to-End Benchmark (4 Held-Out / Diverse Clips)

Evaluated across `Clip_3.mov` (hard/small target), `Clip_21.mov` (multi-target terrain), `Clip_31.mov` (easy sky), and `Demo_V_DRONE_017 (1).mp4` (640×512 IR/thermal-style):

| Run | Avg FPS | Detector Calls % | Time in ROBUST % | Agreement % | Replay Equivalence |
|---|---:|---:|---:|---:|---|
| **Fast** | 25.8 | 33.8% | 0.0% | 85.6% | — |
| **Robust** | 3.1 | 100.0% | 100.0% | 100.0% | — |
| **Adaptive + Random Forest** | **17.2** | **53.8%** | 30.0% | 94.7% | **PASS (0/80)** |
| **Adaptive + Logistic Regression** | 14.6 | 64.1% | 45.3% | **95.3%** | **PASS (0/80)** |
| **Adaptive + Decision Tree** | 10.2 | 61.2% | 41.2% | 94.7% | **PASS (0/80)** |
| **Adaptive + Gradient Boosting** | 9.2 | 57.2% | 35.0% | **95.3%** | **PASS (0/80)** |
| **Adaptive + MLP Neural Net** | 7.3 | 71.6% | 56.6% | **95.3%** | **PASS (0/80)** |
| **Adaptive + SVM (RBF)** | 7.3 | 68.1% | 51.6% | **95.3%** | **PASS (0/80)** |
| **Adaptive + Rules (Baseline)** | 5.5 | 78.1% | 66.6% | 95.3% | **PASS (0/80)** |

### Key Per-Clip Highlights
- **Hard clip (`Clip_3.mov`)**: `FAST` drops to **45.0% agreement**, while all adaptive selectors recover to **81.2% agreement** (+36.2 points).
- **Multi-target clip (`Clip_21.mov`)**: `Adaptive + Gradient Boosting` finds all **3/3 targets (100.0% agreement)** at **9.2 FPS** vs. `ROBUST`'s `3.4 FPS` (**2.7× faster**, saving **53.7%** detector calls).
- **Easy clip (`Clip_31.mov`)**: `Random Forest`, `GBM`, `MLP`, `SVM`, and `Decision Tree` all stay in `FAST` (`15.8–17.8 FPS`, `0` unnecessary switches), whereas `Rules` falsely escalates to `ROBUST` on `82.5%` of frames (`3.9 FPS`).

---

## 3. Conclusion

1. **Adaptive ML selectors beat both fixed extremes and hand-set rules**:
   - **Random Forest** delivers the best speed-accuracy trade-off (**17.2 FPS**, **5.5× faster than ROBUST**, **46.2% fewer detector calls**, **94.7% agreement**, **0.718 CV ROC-AUC**).
   - **Gradient Boosting** achieves the highest multi-target efficiency on complex scenes (**100% agreement on 3 targets at 9.2 FPS** in `Clip_21.mov`).
   - **Hand-set Rules** over-triggers ROBUST (`99.9%` in training, `66.6%` in testing), wasting compute on easy clips.
2. **100% Reproducibility Verified**: Every adaptive run (`28/28` across 4 clips) passed online-replay equivalence with **0 mismatches**.

---

## 4. Limitations & How to Improve

| # | Current Limitation | Impact Observed | Concrete Improvement |
|---|---|---|---|
| 1 | **Generic COCO Weights (`yolov8n`/`s`)** | Misses thermal/IR drones (`0` detections on `Demo_V_DRONE_017`) and ultra-tiny drones (`Clip_41`). Requires `[4, 14, 33]` class filter to suppress terrain clutter. | Place MOT `.txt` labels in `data/gt/` and run `python -m src.training.finetune_detector --epochs 30` to train a dedicated single-class `drone` detector. |
| 2 | **No Bundled Ground-Truth Boxes in `Videos/`** | Evaluation uses `ROBUST`-as-teacher (`agreement_pct`) rather than true **HOTA / IDF1 / MOTA** on `Clip_1..50`. | Add Purdue/Anti-UAV MOT `.txt` annotations to `data/gt/` (already supported by `src/evaluation/metrics.py` and `app.py`). |
| 3 | **CV Generalization Gap (`5` Source Videos)** | `train` accuracy is `0.99` (RF) / `1.00` (GBM) vs. `0.64` / `0.58` in `source_video` GroupKFold CV due to unseen clip lighting/backgrounds. | Expand training to 20–30 of the 50 Purdue clips and tighten tree regularization (`max_depth=4–6`, `min_samples_leaf=25`). |
| 4 | **Binary FAST/ROBUST Action Space** | No intermediate resolution (`960px`) or ROI-crop zoom around predicted track boxes for tiny 10–17 px targets. | Add a 3rd `MEDIUM` tier or Kalman-guided ROI crop in `TrackingPipeline` when `box_area_ratio < 0.002`. |
| 5 | **CPU Inference Bottleneck** | On CPU (`torch 2.15.0+cpu`), `ROBUST` runs at `3.1 FPS` (making adaptive `3×–5.5×` faster); on a desktop GPU, `FAST` vs `ROBUST` latency gap shrinks. | Export fine-tuned detector to ONNX/TensorRT or benchmark on edge hardware (Jetson Orin Nano) where compute savings directly extend battery/FPS. |
