# Repo cleaning: exactly what is required

Rule: nothing is deleted. Unneeded files are moved to `D:\UAV_archive_<time>` and can be restored. Do this in `D:\UAV_Adaptive_Tracking - Copy` only.

## Do NOT move or rename (code locates files relative to the script)
`uavtrack/`, `run_evaluation.py`, `run_final_evaluation.py`, `run_full_project.py`, `run_tracking.py`, `make_integrated_demo.py`, `render_demo_from_logs.py`, `check_test_clips.py`, `train_online_policy.py`, `demo_archived_review.py`, `make_synthetic_demo.py`, `run_video_demo.py`, `run_demo.py`, `run_research_review.py`, `src/`, `research_addon/`, `tests/`, `demo/`, `examples/`, `requirements.txt`, `team_preflight.py`.
Moving the scripts into a subfolder would break `import run_evaluation` and the `ROOT = Path(__file__)` paths.

## What the script does (`cleanup_repo.ps1`, dry run first)
| Action | Items |
|---|---|
| archive | all `*_quick*` outputs in `results\` (first-pass run), `__pycache__` folders, `*.tmp.mp4`, the two 257-byte broken demo videos |
| `docs\` | `CHANGES.md`, `NEXT_STEPS.md`, `RUN_ME.md`, `FINISH_STEPS.md`, `CLEAN_REPO.md` |
| `evidence\final_run\` | small final artifacts to commit: `FINAL_REPORT.md`, `FINAL_RESULTS.md`, eval JSON/MD, figures, per-frame CSVs, metrics, ANALYSIS, check outputs (videos only with `-IncludeVideos`) |
| `.gitignore` | adds `*.pkl`, `*.joblib`, `*.tmp.mp4` (results/ and data/ are already ignored) |

## Steps
1. `powershell -ExecutionPolicy Bypass -File .\cleanup_repo.ps1` (dry run) and read the list. Nothing outside the lists above should appear.
2. `powershell -ExecutionPolicy Bypass -File .\cleanup_repo.ps1 -Apply`
3. `python -m unittest discover -s tests` (expect OK; skips are the optional motmetrics and demo-data tests).
4. `git status --short`, then commit on a branch, e.g. `git checkout -b final-submission; git add -A; git commit -m "Final research pipeline, results and report"`.
5. Large videos (the 4 `*_integrated_demo.mp4` and the review video): do not commit; put them in a release/Drive folder and link them from the README.
6. Replace `README.md` with `README_FINAL.md` (short, honest, links to `evidence\final_run\FINAL_REPORT.md`); keep the old README as `docs\README_original.md`.

## Not cleaned automatically (decide yourself)
- `src\biundary_visibility_check.py` (typo in file name) and other one-off `src\*` scripts: they produced the original results, so keep them. If you rename, update any README mention.
- `results\policy\cache_clip_*.pkl` (large): keep until the report is accepted (needed to re-run any evaluation without the GPU), they are git-ignored.
- Old sandbox/backup folders `D:\UAV_Tracking_Addon_Sandbox`, `D:\UAV_backup_*`: delete only after the commit.
