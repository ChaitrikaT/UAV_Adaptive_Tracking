"""Offline research evaluation on YOUR recorded Purdue clips (needs best.pt + clips + annotations).

Stages (run in this order; every stage writes files under results/evaluation or results/policy):
  cache    run the detector in FAST and ROBUST mode on every frame, once, and store detections + timings
  analyze  detection precision/recall/AP50 per mode; condition descriptors vs detection success;
           can conditions predict that FAST will miss? (cross-clip AUC)
  policy   train the learned FAST/ROBUST tree on --train clips, then compare always-FAST, always-ROBUST,
           rule and learned on --eval clips (tracking IDs from the tracker, composed cost, 95% CIs, verdict)
  tune     choose tracker settings (high-conf, max-age, coasting) on ONE validation clip, from the cache
  verify-online  check on real frames that the cached replay equals a true online run (modes + tracks)

Split discipline: train/tune on clip 7 (or 8) only; evaluate on the other validation clip; clips 9/10 are the
reserved test set: `cache --allow-test`, then `policy --eval 9 10 --final` ONCE with frozen settings.
All numbers come from real files you provide; nothing here is simulated.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import platform
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
TEST_CLIPS = (9, 10)


# ---------------------------------------------------------------- paths / guards
def eval_dir() -> Path:
    d = ROOT / "results" / "evaluation"; d.mkdir(parents=True, exist_ok=True); return d


def policy_dir() -> Path:
    d = ROOT / "results" / "policy"; d.mkdir(parents=True, exist_ok=True); return d


def cache_path(clip: int, tag: str) -> Path:
    return policy_dir() / f"cache_clip_{clip}_{tag}.pkl"


def find_video(clip: int) -> Path:
    for ext in (".mov", ".mp4", ".avi", ".MOV", ".MP4"):
        p = ROOT / "data" / "videos" / f"Clip_{clip}{ext}"
        if p.is_file():
            return p
    raise SystemExit(f"Missing video for clip {clip}: data/videos/Clip_{clip}.mov")


def find_ref(clip: int) -> Path:
    p = ROOT / "data" / "annotations" / f"Clip_{clip}_refined.txt"
    if not p.is_file():
        raise SystemExit(f"Missing annotation: {p}")
    return p


def load_cache(clip: int, tag: str) -> dict:
    p = cache_path(clip, tag)
    if not p.is_file():
        raise SystemExit(f"No cache for clip {clip} (tag '{tag}'). Run: python run_evaluation.py cache --clips {clip}")
    return pickle.loads(p.read_bytes())


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def tracker_cfg(a):
    from uavtrack.tracker import TrackerConfig
    return TrackerConfig(high_conf=a.high_conf, max_age=a.max_age, emit_coasting=a.emit_coasting,
                         state_mode=a.state_mode)


def add_tracker_args(ap):
    ap.add_argument("--high-conf", type=float, default=0.30)
    ap.add_argument("--max-age", type=int, default=30)
    ap.add_argument("--emit-coasting", action="store_true")
    ap.add_argument("--state-mode", choices=["primary", "worst"], default="primary")


def make_detector(weights: Path, conf: float, device):
    from uavtrack.pipeline import YoloDetector
    return YoloDetector(weights, conf=conf, device=device)


def dump(path: Path, obj):
    path.write_text(json.dumps(obj, indent=2, default=float), encoding="utf-8")


# ---------------------------------------------------------------- cache
def cmd_cache(a):
    import cv2
    from uavtrack.collect import cache_clip
    if any(c in TEST_CLIPS for c in a.clips) and not a.allow_test:
        raise SystemExit("Clips 9/10 are the reserved test set: add --allow-test only for the final run.")
    if not a.weights.is_file():
        raise SystemExit(f"Missing weights: {a.weights}")
    device = None if a.device == "auto" else (a.device if a.device == "cpu" else int(a.device))
    det = make_detector(a.weights, a.conf, device)
    meta = {"weights": str(a.weights), "weights_sha256": sha256(a.weights), "conf_floor": a.conf, "device": str(det.device),
            "python": sys.version.split()[0], "platform": platform.platform(), "created": time.strftime("%Y-%m-%d %H:%M:%S"),
            "sizes": {"FAST": 640, "ROBUST": 960}}
    try:
        import torch, ultralytics
        meta.update(torch=torch.__version__, ultralytics=ultralytics.__version__,
                    gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
    except Exception:
        pass
    for clip in a.clips:
        video, ref = find_video(clip), find_ref(clip)
        cap = cv2.VideoCapture(str(video)); ok, first = cap.read(); cap.release()
        if not ok:
            raise SystemExit(f"Cannot read {video}")
        det.warmup(first)
        print(f"== caching clip {clip} ({video.name})", flush=True)
        cache = cache_clip(video, ref, det, a.start, a.end)
        cache["conf_floor"] = a.conf
        cache["meta"] = {**meta, "clip": clip, "video_sha256": sha256(video), "frames_cached": len(cache["frames"])}
        cache_path(clip, a.tag).write_bytes(pickle.dumps(cache))
        n_ref = sum(1 for r in cache["frames"] if r["ref"])
        print(f"   {len(cache['frames'])} frames cached, {n_ref} with reference boxes -> {cache_path(clip, a.tag)}")
    dump(eval_dir() / f"environment_{a.tag}.json", meta)


# ---------------------------------------------------------------- analyze
def cmd_analyze(a):
    from uavtrack.protocol import condition_report, condition_rows, detection_report, difficulty_auc
    import csv
    caches = {c: load_cache(c, a.tag) for c in a.clips}
    cfg = tracker_cfg(a)
    out = {"tag": a.tag, "clips": {}}
    md = [f"# Detection and condition analysis (tag `{a.tag}`)", "",
          "Real detector outputs on recorded clips; frames without reference boxes are not scored (project variant,",
          "not the official MOTChallenge protocol). AP50 uses detections with conf >= the detector floor, so it is",
          "not directly comparable with Ultralytics mAP50.", ""]
    for c, cache in caches.items():
        det = detection_report(cache)
        rows = condition_rows(cache)
        cond = condition_report(rows)
        out["clips"][str(c)] = {"detection": det, "condition": cond}
        if rows:
            with (eval_dir() / f"condition_rows_clip{c}_{a.tag}.csv").open("w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
        md += [f"## Clip {c}: {det['annotated_frames']} annotated of {det['total_frames']} frames", "",
               "| mode | conf | TP | FP | FN | precision | recall | F1 | frame hit rate |", "|---|---|---|---|---|---|---|---|---|"]
        for mode, per in det["modes"].items():
            for k, v in per.items():
                if k.startswith("conf"):
                    md.append(f"| {mode} | {k} | {v['TP']} | {v['FP']} | {v['FN']} | {v['precision']} | {v['recall']} | {v['F1']} | {v['frame_hit_rate']} |")
        md += ["", "AP50: " + ", ".join(f"{m} {p['AP50']}" for m, p in det["modes"].items()) +
               " | mean detector ms: " + ", ".join(f"{m} {p['mean_detect_ms']}" for m, p in det["modes"].items()), ""]
        md += ["| descriptor | Spearman rho vs FAST hit (p) | rho vs ROBUST hit (p) |", "|---|---|---|"]
        for d, v in cond["descriptors"].items():
            f_, r_ = v["spearman_fast"], v["spearman_robust"]
            md.append(f"| {d} | {f_['rho'] if f_ else 'n/a'} ({f_['p'] if f_ else ''}) | {r_['rho'] if r_ else 'n/a'} ({r_['p'] if r_ else ''}) |")
        md.append("")
    ids = list(caches)
    if len(ids) >= 2:
        out["difficulty_cross_clip_auc"] = {}
        md += ["## Can conditions predict that FAST will miss? (AUC, 0.5 = chance; fit on one clip, test on the other)", "",
               "| train -> test | scene only | online policy features | scene + target descriptors (analysis only) |", "|---|---|---|---|"]
        for tr, te in ((ids[0], ids[1]), (ids[1], ids[0])):
            r = difficulty_auc(caches[tr], caches[te], cfg)
            out["difficulty_cross_clip_auc"][f"{tr}->{te}"] = r
            md.append(f"| {tr} -> {te} | {r['scene']} | {r['online_policy_features']} | {r['target_plus_scene']} |")
        md.append("")
    dump(eval_dir() / f"analysis_{a.tag}.json", out)
    (eval_dir() / f"ANALYSIS_{a.tag}.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md))
    print(f"\nSaved: {eval_dir() / f'ANALYSIS_{a.tag}.md'} and analysis_{a.tag}.json")


# ---------------------------------------------------------------- policy
def render_policy_report(res: dict) -> str:
    L = [f"# FAST vs ROBUST vs LEARNED (tag `{res['tag']}`, train clips {res['train']}, eval clips {res['eval']})", "",
         res["split_note"], "",
         "Cost = decode + global features + chosen-mode detector call + policy/state/tracker overhead per frame, composed from",
         "timings measured in the cache stage (detector, decode) and in the replay (policy + tracker). Video writing is excluded.",
         "Tracking IDs are produced by the tracker; reference IDs are used only for scoring. Protocol: annotated frames only", ""]
    for clip, per in res["per_clip"].items():
        L += [f"## Eval clip {clip}", "",
              "| policy | det recall (chosen mode) | track recall | track precision | IDF1 | MOTA | ID sw | frag | pred IDs | ROBUST frac | detect ms | total ms | fps |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for name, r in per.items():
            t = r["tracking"]
            L.append(f"| {name} | {r['detection_recall_chosen_mode']} | {t['recall']} | {t['precision']} | {t['IDF1']} | {t['MOTA']} | "
                     f"{t['ID_switches']} | {t['fragmentations']} | {t['distinct_pred_ids']} | {r['robust_fraction']} | "
                     f"{r['mean_detect_ms']} | {r['mean_total_ms']} | {r['composed_fps']} |")
        L.append("")
    cmp_ = res.get("comparison")
    if cmp_:
        L += ["## Paired block-bootstrap comparison (all eval frames pooled; delta = first minus second)", "",
              "| comparison | delta detection recall [point, 95% CI] | delta total ms/frame [point, 95% CI] |", "|---|---|---|"]
        for k in ("learned_vs_fast", "learned_vs_robust", "rule_vs_fast", "robust_vs_fast"):
            c = cmp_[k]; L.append(f"| {k} | {c['d_recall']} | {c['d_cost_ms']} |")
        L += ["", f"**Verdict for the learned policy: {cmp_['verdict_learned']}** (rule in the code: `protocol.verdict`, margin {cmp_['margin']} recall).", ""]
    L += res.get("notes", [])
    return "\n".join(L)


def cmd_policy(a):
    import joblib
    from uavtrack.collect import dagger_train
    from uavtrack.policy import FixedPolicy, LearnedPolicy, RulePolicy
    from uavtrack.protocol import compare_learned, replay_eval, strip_arrays
    from uavtrack.features import FEATURE_NAMES
    if any(c in TEST_CLIPS for c in a.train):
        raise SystemExit("Never train on clips 9/10.")
    final = any(c in TEST_CLIPS for c in a.eval)
    if final and not a.final:
        raise SystemExit("Evaluating on clips 9/10 needs --final (reserved test set; run once with frozen settings).")
    overlap = set(a.train) & set(a.eval)
    if overlap and not a.allow_train_eval:
        raise SystemExit(f"Clips {sorted(overlap)} are in both --train and --eval: that is not a held-out evaluation.")
    cfg = tracker_cfg(a)
    train_caches = [load_cache(c, a.tag) for c in a.train]
    eval_caches = {c: load_cache(c, a.tag) for c in a.eval}
    notes = []
    learned = None
    try:
        clf, trep = dagger_train(train_caches, cfg, a.iterations, a.max_depth, a.tie_both_fail, a.tie_equal_hit)
        bundle = {"model": clf, "feature_names": FEATURE_NAMES, "detector_conf": train_caches[0].get("conf_floor"),
                  "high_conf": a.high_conf, "max_age": a.max_age, "train_clips": a.train, "tie_rules": trep["tie_rules"]}
        name = "_".join(map(str, a.train))
        joblib.dump(bundle, policy_dir() / f"policy_train{name}_{a.tag}.joblib")
        joblib.dump(bundle, policy_dir() / "online_policy.joblib")  # default file used by run_tracking.py
        from sklearn.tree import export_text
        (policy_dir() / f"policy_train{name}_{a.tag}_tree.txt").write_text(export_text(clf, feature_names=FEATURE_NAMES), encoding="utf-8")
        dump(policy_dir() / f"policy_train{name}_{a.tag}_report.json", trep)
        learned = LearnedPolicy(bundle)
        notes += ["## Training of the learned tree", "", "```", json.dumps(trep, indent=2), "```", ""]
    except ValueError as exc:
        notes += [f"## Learned policy NOT trained: {exc}", "",
                  "Only one usable label class remained, so a learned selector would be constant. This is a finding, not a bug.", ""]
    policies = {"fast": FixedPolicy("FAST"), "robust": FixedPolicy("ROBUST"), "rule": RulePolicy()}
    if learned is not None:
        policies["learned"] = learned
    per_clip, results = {}, {k: [] for k in policies}
    for clip, cache in eval_caches.items():
        per_clip[str(clip)] = {}
        for pname, pol in policies.items():
            r = replay_eval(cache, pol, cfg)
            results[pname].append(r)
            per_clip[str(clip)][pname] = strip_arrays(r)
    res = {"tag": a.tag, "train": a.train, "eval": a.eval, "per_clip": per_clip,
           "tracker_config": vars(cfg) if hasattr(cfg, "__dict__") else str(cfg),
           "split_note": ("FINAL TEST RUN on reserved clips; settings must have been frozen on validation clips." if final else
                          "Validation run: train and eval clips are different; the detector itself saw clips 1-6 in training and used clips 7-8 for early stopping."),
           "notes": notes}
    if learned is not None:
        res["comparison"] = compare_learned(results, a.block, a.n_boot, a.margin)
        fast_ms = sum(r["mean_total_ms"] for r in results["fast"]) / len(results["fast"])
        rob_ms = sum(r["mean_total_ms"] for r in results["robust"]) / len(results["robust"])
        gap = (rob_ms - fast_ms) / fast_ms * 100 if fast_ms else 0.0
        res["notes"] += [f"Cost gap: ROBUST costs {gap:.1f}% more than FAST per frame end-to-end on this setup "
                         f"(FAST {fast_ms:.1f} ms, ROBUST {rob_ms:.1f} ms). A selector cannot save more compute than this gap; "
                         "if it is small, recall/IDF1 gains are the only meaningful benefit. Re-run `cache --device cpu --tag cpu` "
                         "to measure a setting where the gap is larger.", ""]
    base = f"policy_train{'_'.join(map(str, a.train))}_eval{'_'.join(map(str, a.eval))}_{a.tag}"
    dump(eval_dir() / f"{base}.json", res)
    text = render_policy_report(res)
    (eval_dir() / f"{base}.md").write_text(text, encoding="utf-8")
    print(text)
    print(f"\nSaved: {eval_dir() / (base + '.md')}")
    if final:
        with (eval_dir() / "test_runs.log").open("a", encoding="utf-8") as f:
            f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} FINAL {base} args={vars(a)}\n")
        n = sum(1 for _ in (eval_dir() / "test_runs.log").open(encoding="utf-8"))
        if n > 1:
            print(f"WARNING: this is test run #{n} (see test_runs.log). Repeated test runs invalidate the held-out claim.")


# ---------------------------------------------------------------- verify-online
def cmd_verify(a):
    from uavtrack.collect import rollout
    from uavtrack.pipeline import run_video
    from uavtrack.policy import make_policy
    cache = load_cache(a.clip, a.tag)
    n = min(a.end, len(cache["frames"]))
    sub = {**cache, "frames": cache["frames"][:n]}
    cfg = tracker_cfg(a)
    policy = make_policy(a.policy, policy_dir() / "online_policy.joblib")
    device = None if a.device == "auto" else (a.device if a.device == "cpu" else int(a.device))
    det = make_detector(a.weights, cache.get("conf_floor") or 0.10, device)
    import cv2
    video = find_video(a.clip)
    cap = cv2.VideoCapture(str(video)); ok, first = cap.read(); cap.release()
    det.warmup(first)
    tracks: list = []
    replay = rollout(sub, policy, cfg, tracks)
    online, otracks = run_video(video, det, make_policy(a.policy, policy_dir() / "online_policy.joblib"), 1, n, cfg, log_every=0)
    mode_diff = sum(1 for r, o in zip(replay, online) if r["mode"] != o["mode"])
    ndet_diff = sum(1 for r, o in zip(sub["frames"], online)
                    if len(r["dets_robust" if o["mode"] == "ROBUST" else "dets_fast"]) != o["n_det"])
    key = lambda t: (t["frame"], t["track_id"])
    ra, oa = {key(t): t for t in tracks}, {key(t): t for t in otracks}
    track_diff = len(set(ra) ^ set(oa)) + sum(1 for k in set(ra) & set(oa)
                                              if max(abs(ra[k][c] - oa[k][c]) for c in ("x1", "y1", "x2", "y2")) > 1.0)
    out = {"clip": a.clip, "policy": a.policy, "frames": n, "mode_mismatches": mode_diff,
           "detection_count_mismatches": ndet_diff, "track_row_mismatches": track_diff,
           "replay_equals_online": mode_diff == 0 and ndet_diff == 0 and track_diff == 0}
    dump(eval_dir() / f"verify_online_clip{a.clip}_{a.policy}_{a.tag}.json", out)
    print(json.dumps(out, indent=2))


# ---------------------------------------------------------------- tune (tracker settings, validation clip only)
def cmd_tune(a):
    import csv
    import itertools
    from uavtrack.policy import FixedPolicy
    from uavtrack.protocol import replay_eval, strip_arrays
    from uavtrack.tracker import TrackerConfig
    if a.clip in TEST_CLIPS:
        raise SystemExit("Never tune on clips 9/10.")
    cache = load_cache(a.clip, a.tag)
    rows = []
    for hc, ma, em in itertools.product(a.high_conf_grid, a.max_age_grid, (False, True)):
        cfg = TrackerConfig(high_conf=hc, max_age=ma, emit_coasting=em)
        for mode in ("FAST", "ROBUST"):
            t = replay_eval(cache, FixedPolicy(mode), cfg)["tracking"]
            rows.append({"mode": mode, "high_conf": hc, "max_age": ma, "emit_coasting": int(em), "IDF1": t["IDF1"],
                         "MOTA": t["MOTA"], "recall": t["recall"], "precision": t["precision"],
                         "ID_switches": t["ID_switches"], "fragmentations": t["fragmentations"],
                         "distinct_pred_ids": t["distinct_pred_ids"]})
    rows.sort(key=lambda r: (-r["IDF1"], r["ID_switches"]))
    with (eval_dir() / f"tune_clip{a.clip}_{a.tag}.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    best = [r for r in rows if r["mode"] == "ROBUST"][0]
    print("Top 5 by IDF1 (all modes):")
    for r in rows[:5]:
        print("  ", r)
    print(f"\nBest ROBUST setting on clip {a.clip}: --high-conf {best['high_conf']} --max-age {best['max_age']}"
          + (" --emit-coasting" if best["emit_coasting"] else ""))
    print("Freeze this setting, then use it for every later command (including clip 8 and the final test).")


def build_parser():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("cache"); c.set_defaults(fn=cmd_cache)
    c.add_argument("--clips", type=int, nargs="+", required=True)
    c.add_argument("--start", type=int, default=1); c.add_argument("--end", type=int, default=-1)
    c.add_argument("--conf", type=float, default=0.10)
    c.add_argument("--weights", type=Path, default=ROOT / "results/training/uav_detector_v1/weights/best.pt")
    c.add_argument("--device", default="auto", help="auto | cpu | GPU index such as 0")
    c.add_argument("--tag", default="default"); c.add_argument("--allow-test", action="store_true")
    z = sub.add_parser("analyze"); z.set_defaults(fn=cmd_analyze)
    z.add_argument("--clips", type=int, nargs="+", required=True); z.add_argument("--tag", default="default")
    add_tracker_args(z)
    p = sub.add_parser("policy"); p.set_defaults(fn=cmd_policy)
    p.add_argument("--train", type=int, nargs="+", required=True); p.add_argument("--eval", type=int, nargs="+", required=True)
    p.add_argument("--tag", default="default"); p.add_argument("--final", action="store_true")
    p.add_argument("--allow-train-eval", action="store_true")
    p.add_argument("--iterations", type=int, default=2); p.add_argument("--max-depth", type=int, default=3)
    p.add_argument("--tie-both-fail", choices=["exclude", "fast", "robust"], default="exclude")
    p.add_argument("--tie-equal-hit", choices=["fast", "robust"], default="fast")
    p.add_argument("--block", type=int, default=30); p.add_argument("--n-boot", type=int, default=2000)
    p.add_argument("--margin", type=float, default=0.02)
    add_tracker_args(p)
    t = sub.add_parser("tune"); t.set_defaults(fn=cmd_tune)
    t.add_argument("--clip", type=int, required=True); t.add_argument("--tag", default="default")
    t.add_argument("--high-conf-grid", type=float, nargs="+", default=[0.2, 0.3, 0.4])
    t.add_argument("--max-age-grid", type=int, nargs="+", default=[10, 30, 60])
    v = sub.add_parser("verify-online"); v.set_defaults(fn=cmd_verify)
    v.add_argument("--clip", type=int, required=True); v.add_argument("--end", type=int, default=150)
    v.add_argument("--policy", choices=["fast", "robust", "rule", "learned"], default="rule")
    v.add_argument("--weights", type=Path, default=ROOT / "results/training/uav_detector_v1/weights/best.pt")
    v.add_argument("--device", default="auto"); v.add_argument("--tag", default="default")
    add_tracker_args(v)
    return ap


def main(argv=None):
    a = build_parser().parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    main()
