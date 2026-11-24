"""One command that runs the whole offline research pipeline on YOUR recorded clips and writes FINAL_RESULTS.md.

  python run_full_project.py --quick        # first pass: 300 frames per clip (about 10 minutes)
  python run_full_project.py                # full clips 7 and 8
  python run_full_project.py --final --confirm-test   # reserved test clips 9/10, ONCE, after the above
Stages: cache -> verify-online -> analyze -> tune (freezes tracker settings on clip 7) -> final evaluation (7->8 and 8->7)
        -> integrated demo video on clip 8 -> FINAL_RESULTS.md.  A failed stage is logged; later independent stages still run.
Needs: data/videos/Clip_N.mov, data/annotations/Clip_N_refined.txt, results/training/uav_detector_v1/weights/best.pt
Offline research on prerecorded footage only: no live following, no control output.
"""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
FINAL = ROOT / "results" / "final"
PY = sys.executable
LOG = FINAL / "run_log.txt"


def log(msg: str):
    FINAL.mkdir(parents=True, exist_ok=True)
    line = f"{time.strftime('%H:%M:%S')} {msg}"
    print(line, flush=True)
    with LOG.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def stage(name: str, argv: list[str], status: dict) -> bool:
    log(f"=== {name}: {' '.join(argv)}")
    t0 = time.time()
    r = subprocess.run([PY, *argv], cwd=str(ROOT))
    ok = r.returncode == 0
    status[name] = {"ok": ok, "seconds": round(time.time() - t0, 1)}
    log(f"--- {name}: {'OK' if ok else 'FAILED (exit ' + str(r.returncode) + ')'} in {time.time() - t0:.0f}s")
    (FINAL / "stage_status.json").write_text(json.dumps(status, indent=2), encoding="utf-8")
    return ok


def freeze_settings(tag: str) -> dict | None:
    """Pick tracker settings on clip 7 ONLY (best ROBUST IDF1, ties -> fewer ID switches) and write them down."""
    p = ROOT / "results" / "evaluation" / f"tune_clip7_{tag}.csv"
    if not p.is_file():
        return None
    rows = [r for r in csv.DictReader(p.open(encoding="utf-8")) if r["mode"] == "ROBUST"]
    if not rows:
        return None
    best = sorted(rows, key=lambda r: (-float(r["IDF1"]), int(r["ID_switches"])))[0]
    s = {"high_conf": float(best["high_conf"]), "max_age": int(best["max_age"]), "emit_coasting": bool(int(best["emit_coasting"])),
         "chosen_on": "clip 7 (validation), best ROBUST IDF1", "IDF1_on_clip7": float(best["IDF1"])}
    (FINAL / f"frozen_settings_{tag}.json").write_text(json.dumps(s, indent=2), encoding="utf-8")
    log(f"frozen tracker settings: {s}")
    return s


def jload(p: Path):
    return json.loads(p.read_text(encoding="utf-8")) if p.is_file() else None


def build_report(tag: str, demo_policy: str, demo_clip: int, status: dict, final_mode: bool):
    ev = {k: jload(FINAL / f"final_eval_train{k}_{tag}.json") for k in ("7_eval8", "8_eval7")}
    ev = {"7->8": jload(FINAL / f"final_eval_train7_eval8_{tag}.json"), "8->7": jload(FINAL / f"final_eval_train8_eval7_{tag}.json")}
    test = jload(FINAL / f"final_eval_train7_8_eval9_10_{tag}.json")
    demo = jload(FINAL / "demo" / f"clip{demo_clip}_{demo_policy}_metrics.json")
    env = jload(ROOT / "results" / "evaluation" / f"environment_{tag}.json") or {}
    ver = jload(ROOT / "results" / "evaluation" / f"verify_online_clip8_rule_{tag}.json")
    fz = jload(FINAL / f"frozen_settings_{tag}.json")
    L = ["# FINAL RESULTS (generated from the files this machine produced)", "",
         f"Tag `{tag}`. GPU: {env.get('gpu')}; torch {env.get('torch')}; ultralytics {env.get('ultralytics')}; weights sha256 {str(env.get('weights_sha256'))[:16]}...",
         "Everything below is copied from JSON/CSV outputs; nothing is typed by hand. Offline research on recorded clips; clips 9-10 are touched only in the section marked TEST.", "",
         "## Stage status", "", "| stage | result | seconds |", "|---|---|---|"]
    L += [f"| {k} | {'OK' if v['ok'] else 'FAILED'} | {v['seconds']} |" for k, v in status.items()]
    L += ["", f"Replay equals true online run (verify-online, 150 frames): **{ver['replay_equals_online'] if ver else 'not run'}**", ""]
    if fz:
        L += [f"Tracker settings frozen on clip 7: {fz}", ""]
    for k, e in ev.items():
        if not e: continue
        L += [f"## Held-out validation, train {k.split('->')[0]} -> eval {k.split('->')[1]}", "", "| policy | track recall | precision | IDF1 | MOTA | ID switches | fragmentations | predicted IDs | ROBUST frac | cost ms/frame | fps |", "|---|---|---|---|---|---|---|---|---|---|---|"]
        for clip, per in e["per_clip"].items():
            for p, r in per.items():
                t = r["tracking"]
                L.append(f"| {p} (clip {clip}) | {t['recall']} | {t['precision']} | {t['IDF1']} | {t['MOTA']} | {t['ID_switches']} | {t['fragmentations']} | {t['distinct_pred_ids']} | {r['robust_fraction']} | {r['mean_total_ms']} | {r['composed_fps']} |")
        c = e["comparison"]
        L += ["", f"Difficulty-policy verdict: **{c['verdict_difficulty']}**; learned-tree verdict: **{c.get('verdict_learned', 'n/a')}**.",
              f"difficulty vs FAST (delta recall, cost): {c['difficulty_vs_fast']['d_recall']} / {c['difficulty_vs_fast']['d_cost_ms']}; difficulty vs ROBUST: {c['difficulty_vs_robust']['d_recall']} / {c['difficulty_vs_robust']['d_cost_ms']}; ROBUST vs FAST: {c['robust_vs_fast']['d_recall']} / {c['robust_vs_fast']['d_cost_ms']}.",
              "Difficulty AUC on held-out clip: " + "; ".join(f"clip {cl}: FAST-miss {a['predicts_FAST_miss']} , ROBUST-helps {a['predicts_ROBUST_helps']}" for cl, a in e["difficulty_auc_heldout"].items()),
              f"Estimator coefficients: {e['difficulty_training']['standardised_coefficients']}", ""]
    if demo:
        t, tm = demo["tracking_vs_reference"], demo["timing_end_to_end_incl_decode_excl_video_writing"]
        L += [f"## Integrated end-to-end run (true online, clip {demo['clip']}, policy {demo['policy']}, frames {demo['frames']})", "",
              f"Video: `results/final/demo/clip{demo['clip']}_{demo['policy']}_integrated_demo.mp4`. IDF1 {t['IDF1']}, MOTA {t['MOTA']}, ID switches {t['ID_switches']}, fragmentations {t['fragmentations']}, predicted IDs {t['distinct_pred_ids']} (reference IDs in clip: {demo['reference_ids_in_clip']}), recall {t['recall']}, precision {t['precision']}.",
              f"Speed (decode + features + policy + detector + tracker; video writing excluded): {tm.get('end_to_end_fps')} FPS, {tm.get('mean_total_ms_incl_decode')} ms/frame, ROBUST fraction {tm.get('robust_fraction')}.", ""]
    if test:
        L += ["## TEST SET (clips 9-10, run once with frozen settings)", ""]
        for clip, per in test["per_clip"].items():
            for p, r in per.items():
                t = r["tracking"]; L.append(f"- clip {clip} {p}: recall {t['recall']}, IDF1 {t['IDF1']}, MOTA {t['MOTA']}, ID switches {t['ID_switches']}, cost {r['mean_total_ms']} ms")
        L.append(f"- verdict difficulty policy: {test['comparison']['verdict_difficulty']}")
        L.append("")
    # objective status derived from the outputs, never hand-written
    L += ["## Synopsis objective status (derived from the numbers above)", ""]
    have_track = any(e for e in ev.values()) or demo
    L += [f"1. Detection + tracking pipeline with independent IDs: {'DEMONSTRATED on validation clips (quality = numbers above; low detector recall limits identity metrics)' if have_track else 'NOT DEMONSTRATED (stage failed or not run)'}."]
    aucs = [a[k] for e in ev.values() if e for a in e["difficulty_auc_heldout"].values() for k in ("predicts_FAST_miss",) if a[k]]
    if aucs:
        sup = [x for x in aucs if x["ci95"] and x["ci95"][0] > 0.5]
        L += [f"2. Difficulty estimation from scale/motion/blur/illumination/confidence/track stability: estimator trained and evaluated on held-out clips; AUC {[x['auc'] for x in aucs]}; "
              f"{'SUPPORTED (CI above 0.5 on ' + str(len(sup)) + ' of ' + str(len(aucs)) + ')' if sup else 'NOT SUPPORTED (no interval above 0.5)'}. Occlusion is not modelled."]
    else:
        L += ["2. Difficulty estimation: NOT DEMONSTRATED (stage failed or not run)."]
    vs = [e["comparison"]["verdict_difficulty"] for e in ev.values() if e]
    L += [f"3. Adaptive mode selection balancing quality and cost: verdicts {vs}. Only SUPPORTED means the adaptive policy beat FAST and was cheaper than ROBUST within 2 points recall; otherwise report as a negative/inconclusive result."]
    L += ["", "Expected outcome: working prototype with tracking + adaptive effort + accuracy/identity/speed/cost measures: "
          + ("DEMONSTRATED offline on recorded clips (see integrated run)." if demo else "NOT DEMONSTRATED (integrated run missing)."), "",
          "Limits that remain regardless of outcome: single dataset; detector trained 12 epochs; validation clips informed detector selection; frames without reference rows are not scored; costs are model-and-pipeline timings on one laptop GPU (no CPU/edge measurement unless --tag cpu was run); occlusion not modelled."]
    (FINAL / "FINAL_RESULTS.md").write_text("\n".join(L), encoding="utf-8")
    log(f"wrote {FINAL / 'FINAL_RESULTS.md'}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--quick", action="store_true", help="300 frames per clip, tag 'quick'")
    ap.add_argument("--tag", default=None); ap.add_argument("--demo-clip", type=int, default=8); ap.add_argument("--demo-frames", type=int, default=None)
    ap.add_argument("--demo-policy", choices=["difficulty", "rule", "fast", "robust"], default="difficulty")
    ap.add_argument("--skip", nargs="*", default=[], help="stage names to skip: cache verify analyze tune eval demo")
    ap.add_argument("--final", action="store_true"); ap.add_argument("--confirm-test", action="store_true")
    a = ap.parse_args()
    tag = a.tag or ("quick" if a.quick else "default")
    end = ["--end", "300"] if a.quick else []
    demo_end = str(a.demo_frames or (300 if a.quick else 900))
    status = jload(FINAL / "stage_status.json") or {}
    if a.final:
        if not a.confirm_test:
            raise SystemExit("--final runs the reserved test clips. Add --confirm-test to confirm settings are frozen and this is the single test run.")
        ok = stage("cache_test", ["run_evaluation.py", "cache", "--clips", "9", "10", "--allow-test", "--tag", tag, *end], status)
        if ok:
            stage("eval_test", ["run_final_evaluation.py", "--train", "7", "8", "--eval", "9", "10", "--final", "--tag", tag], status)
            stage("demo_test", ["make_integrated_demo.py", "--clip", "9", "--end", demo_end, "--policy", "difficulty", "--allow-test", "--tag", tag,
                                "--model", str(ROOT / "results" / "policy" / f"difficulty_train7_8_{tag}.joblib")], status)
        build_report(tag, a.demo_policy, a.demo_clip, status, True)
        return
    if "cache" not in a.skip:
        stage("cache", ["run_evaluation.py", "cache", "--clips", "7", "8", "--tag", tag, *end], status)
    if "verify" not in a.skip:
        stage("verify", ["run_evaluation.py", "verify-online", "--clip", "8", "--end", "150", "--policy", "rule", "--tag", tag], status)
    if "analyze" not in a.skip:
        stage("analyze", ["run_evaluation.py", "analyze", "--clips", "7", "8", "--tag", tag], status)
    if "tune" not in a.skip:
        if stage("tune", ["run_evaluation.py", "tune", "--clip", "7", "--tag", tag], status):
            freeze_settings(tag)
    if "eval" not in a.skip:
        stage("eval_7_to_8", ["run_final_evaluation.py", "--train", "7", "--eval", "8", "--tag", tag], status)
        stage("eval_8_to_7", ["run_final_evaluation.py", "--train", "8", "--eval", "7", "--tag", tag], status)
    if "demo" not in a.skip:
        model = ["--model", str(ROOT / "results" / "policy" / f"difficulty_train7_{tag}.joblib")] if a.demo_policy == "difficulty" else []
        stage("demo", ["make_integrated_demo.py", "--clip", str(a.demo_clip), "--end", demo_end, "--policy", a.demo_policy, "--tag", tag, *model], status)
    build_report(tag, a.demo_policy, a.demo_clip, status, False)


if __name__ == "__main__":
    main()
