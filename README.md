# Condition-Aware Adaptive Perception for Air-to-Air UAV Tracking (BAI507P mini project)

Offline research prototype on prerecorded Purdue UAV-to-UAV clips. No flight control, interception or live following.

## What is here
- Detector (YOLO11n) + ByteTrack-style tracker with its own predicted IDs + identity metrics (IDF1, MOTA, ID switches).
- Difficulty estimator (scale, motion, blur, illumination, detector confidence, track stability) and FAST (640) / ROBUST (960) switching policies.
- Integrated end-to-end run, reproducible evaluation protocol with a reserved test set (clips 9-10, run once).

## Results in one paragraph (see `evidence/final_run/FINAL_REPORT.md` for the numbers and caveats)
The pipeline runs end to end. ROBUST detects more than FAST on validation clips, but identity quality is low (IDF1 <= 0.13), the difficulty estimator does not generalise to a held-out clip, the adaptive policy collapses to always-ROBUST and ROBUST costs about the same as FAST on the test GPU, and the detector fails on the reserved clips. These are reported as negative results.

## Reproduce
`python run_full_project.py` (validation) then, once, `python run_full_project.py --final --confirm-test`. Requires the Purdue clips, annotations and the trained checkpoint (not in the repo). Unit tests: `python -m unittest discover -s tests`.

## Layout
`uavtrack/` library | `run_*.py`, `make_*.py` entry points | `src/`, `research_addon/` original experiments | `tests/` | `demo/` | `evidence/` final outputs | `docs/` notes
