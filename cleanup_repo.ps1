# cleanup_repo.ps1 - DRY RUN by default. Nothing is deleted: files are MOVED to an archive folder outside the repo.
#   Dry run :  powershell -ExecutionPolicy Bypass -File .\cleanup_repo.ps1
#   Apply   :  powershell -ExecutionPolicy Bypass -File .\cleanup_repo.ps1 -Apply
# Run from the development copy only (D:\UAV_Adaptive_Tracking - Copy). Refuses to run in the protected original.
param([switch]$Apply, [switch]$IncludeVideos)
$ErrorActionPreference = "Stop"
$root = (Get-Location).Path.TrimEnd('\')
if ($root -ieq "D:\UAV_Adaptive_Tracking") { throw "This is the protected original folder. Run from the development copy." }
if (-not (Test-Path "$root\run_evaluation.py")) { throw "Run this from the project root (run_evaluation.py not found)." }
$stamp = Get-Date -Format yyyyMMdd_HHmm; $arch = "D:\UAV_archive_$stamp"
$mode = if ($Apply) { "APPLY" } else { "DRY RUN" }; Write-Host "== $mode  root=$root  archive=$arch"

# 1) things to ARCHIVE (moved, not deleted)
$move = @()
$move += Get-ChildItem -Recurse -File -Path "$root\results" -ErrorAction SilentlyContinue | Where-Object { $_.Name -match '_quick(\.|_)' }          # first-pass 'quick' outputs
$move += Get-ChildItem -Recurse -Directory -Filter "__pycache__" -ErrorAction SilentlyContinue | Where-Object { $_.FullName -notmatch '\\\.venv\\' }
$move += Get-ChildItem -Recurse -File -Include "*.tmp.mp4","*.render.tmp.mp4" -ErrorAction SilentlyContinue | Where-Object { $_.FullName -notmatch '\\\.venv\\' }
$move += Get-ChildItem "$root\results\final\demo" -Filter "*.mp4" -File -ErrorAction SilentlyContinue | Where-Object { $_.Length -lt 10KB }       # the empty (257-byte) videos
foreach ($f in ($move | Sort-Object FullName -Unique)) {
    $rel = $f.FullName.Substring($root.Length + 1); $size = if ($f.PSIsContainer) { "dir" } else { "{0:N0} B" -f $f.Length }
    Write-Host "  archive: $rel ($size)"
    if ($Apply) { $dest = Join-Path $arch $rel; New-Item -ItemType Directory -Force -Path (Split-Path $dest) | Out-Null; Move-Item -LiteralPath $f.FullName -Destination $dest }
}

# 2) docs into docs\ (safe: no code imports them)
foreach ($d in "CHANGES.md","NEXT_STEPS.md","RUN_ME.md","FINISH_STEPS.md","CLEAN_REPO.md") {
    if (Test-Path "$root\$d") { Write-Host "  to docs\: $d"; if ($Apply) { New-Item -ItemType Directory -Force -Path "$root\docs" | Out-Null; Move-Item "$root\$d" "$root\docs\$d" -Force } }
}

# 3) evidence\ : small, final, committable artifacts (caches and big videos stay out)
$ev = "$root\evidence\final_run"
$pat = @("FINAL_REPORT.md","results\final\FINAL_RESULTS.md","results\final\stage_status.json","results\final\test_runs.log","results\final\frozen_settings_default.json",
         "results\final\final_eval_*_default.json","results\final\final_eval_*_default.md","results\final\difficulty_scores_*_default.csv","results\final\figures\*_default_*.png",
         "results\final\demo\*_metrics.json","results\final\demo\*_frames.csv","results\final\demo\*_tracks.csv","results\final\demo\*_id_summary.csv","results\final\check\*",
         "results\evaluation\ANALYSIS_default.md","results\evaluation\environment_default.json","results\evaluation\verify_online_*_default.json","results\evaluation\tune_clip7_default.csv")
if ($IncludeVideos) { $pat += "results\final\demo\*_integrated_demo.mp4" }
foreach ($p in $pat) { foreach ($f in (Get-ChildItem -Path "$root\$p" -File -ErrorAction SilentlyContinue)) {
    Write-Host "  evidence: $($f.Name)"; if ($Apply) { New-Item -ItemType Directory -Force -Path $ev | Out-Null; Copy-Item $f.FullName $ev -Force } } }

# 4) .gitignore additions
$add = @("*.pkl","*.joblib","*.tmp.mp4","*.render.tmp.mp4","*.mp4","!demo/videos/*.mp4")
if ($Apply) { $gi = Get-Content "$root\.gitignore" -ErrorAction SilentlyContinue; foreach ($l in $add) { if ($gi -notcontains $l) { Add-Content "$root\.gitignore" $l } } }
else { Write-Host "  .gitignore would gain: $($add -join ', ')" }
Write-Host "== done ($mode). Next: git status --short  (review), then commit."
