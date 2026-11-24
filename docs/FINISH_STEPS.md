# Finish steps (in order; each is one command)

1. Apply: copy this pack into `D:\UAV_Adaptive_Tracking - Copy` (overwrite `uavtrack\pipeline.py`; it contains the video-writer fix).
2. Videos (no detector run, ~1-3 min each):
   `python render_demo_from_logs.py --clip 8 --policy difficulty --model results\policy\difficulty_train7_default.joblib`
   `python render_demo_from_logs.py --clip 9 --policy difficulty --allow-test --model results\policy\difficulty_train7_8_default.joblib`
   Expected last line: `Rendered 900 frames -> ...mp4 (h264); file reports 900 frames, N MB`. If playback still fails, open with VLC.
3. Diagnose the test clips (read-only, changes no result): `python check_test_clips.py --clips 7 8 9 10`. Send me the folder `results\final\check`.
4. Send `results\evaluation\tune_clip7_default.csv` (to see whether the tuning grid was too narrow).
5. Regenerate the report: `python build_final_report.py` (writes `FINAL_REPORT.md`).
6. Clean the repo: see `CLEAN_REPO.md`.
