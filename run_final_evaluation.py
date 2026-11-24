"""Train the difficulty estimator + learned tree on one validation clip; compare FAST / ROBUST / rule / learned /
DIFFICULTY policies (tracker IDs, IDF1, MOTA, ID switches, cost) on the other clip. Real cached detections only.

  python run_final_evaluation.py --train 7 --eval 8
  python run_final_evaluation.py --train 8 --eval 7
  python run_final_evaluation.py --train 7 8 --eval 9 10 --final     (reserved test set, run ONCE)
Reads results/policy/cache_clip_*_<tag>.pkl written by `python run_evaluation.py cache`.
Settings frozen on clip 7 are read from results/final/frozen_settings_<tag>.json if present.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

import run_evaluation as RE
from uavtrack.collect import dagger_train, rollout
from uavtrack.difficulty import DifficultyPolicy, train_difficulty, _xy
from uavtrack.policy import FixedPolicy, LearnedPolicy, RulePolicy
from uavtrack.protocol import paired_block_bootstrap, replay_eval, strip_arrays, verdict
from uavtrack.tracker import TrackerConfig

ROOT = Path(__file__).resolve().parent


def final_dir() -> Path:
    d = RE.ROOT / "results" / "final"; (d / "figures").mkdir(parents=True, exist_ok=True); return d


def load_settings(tag: str, args) -> dict:
    p = final_dir() / f"frozen_settings_{tag}.json"
    s = {"high_conf": 0.30, "max_age": 30, "emit_coasting": False}
    if p.is_file():
        s.update(json.loads(p.read_text(encoding="utf-8")))
    for k in ("high_conf", "max_age"):
        if getattr(args, k) is not None:
            s[k] = getattr(args, k)
    if args.emit_coasting:
        s["emit_coasting"] = True
    return s


def auc_with_ci(scores, labels, block=30, n=1000, seed=7):
    from sklearn.metrics import roc_auc_score
    scores, labels = np.asarray(scores), np.asarray(labels)
    if len(set(labels)) < 2:
        return None
    auc = float(roc_auc_score(labels, scores))
    rng, starts, vals = np.random.default_rng(seed), np.arange(0, len(scores), block), []
    for _ in range(n):
        pick = rng.integers(0, len(starts), len(starts))
        idx = np.concatenate([np.arange(starts[i], min(starts[i] + block, len(scores))) for i in pick])
        if len(set(labels[idx])) == 2:
            vals.append(roc_auc_score(labels[idx], scores[idx]))
    return {"auc": round(auc, 4), "ci95": [round(float(np.percentile(vals, 2.5)), 4), round(float(np.percentile(vals, 97.5)), 4)] if vals else None,
            "n": int(len(scores)), "positives": int(labels.sum())}


def block_ci(arr: dict, block=30, n=1000, seed=3):
    """95% block-bootstrap CI of recall (sum tp / sum ref) and mean cost for one policy over all eval clips."""
    blocks = []
    for c in arr:
        for s in range(0, len(c["ref"]), block):
            sl = slice(s, s + block)
            blocks.append([c["tp"][sl].sum(), c["ref"][sl].sum(), c["cost"][sl].sum(), len(c["ref"][sl])])
    B = np.array(blocks, float); idx = np.random.default_rng(seed).integers(0, len(B), (n, len(B))); T = B[idx].sum(axis=1)
    rec, cost = T[:, 0] / np.maximum(T[:, 1], 1), T[:, 2] / np.maximum(T[:, 3], 1)
    S = B.sum(axis=0)
    return ((S[0] / max(S[1], 1), *np.percentile(rec, [2.5, 97.5])), (S[2] / max(S[3], 1), *np.percentile(cost, [2.5, 97.5])))


def run_direction(train, evals, args):
    st = load_settings(args.tag, args)
    cfg = TrackerConfig(high_conf=st["high_conf"], max_age=st["max_age"], emit_coasting=st["emit_coasting"])
    train_caches = [RE.load_cache(c, args.tag) for c in train]
    eval_caches = {c: RE.load_cache(c, args.tag) for c in evals}
    name = "_".join(map(str, train)); ename = "_".join(map(str, evals))
    notes = []
    est, tau, drep = train_difficulty(train_caches, cfg, args.iterations, args.lam, args.hold)
    import joblib
    pol_dir = RE.policy_dir()
    joblib.dump({"estimator": est, "tau": tau, "hold": args.hold, "lam": args.lam, "tracker": st, "train_clips": train},
                pol_dir / f"difficulty_train{name}_{args.tag}.joblib")
    policies = {"fast": FixedPolicy("FAST"), "robust": FixedPolicy("ROBUST"), "rule": RulePolicy(),
                "difficulty": DifficultyPolicy(est, tau, args.hold)}
    try:
        clf, trep = dagger_train(train_caches, cfg, args.iterations, 3, "exclude", "fast")
        policies["learned"] = LearnedPolicy(clf)
    except ValueError as exc:
        notes.append(f"Learned tree not trained: {exc}")
    per_clip, results = {}, {k: [] for k in policies}
    for c, cache in eval_caches.items():
        per_clip[str(c)] = {}
        for pname, pol in policies.items():
            r = replay_eval(cache, pol, cfg)
            results[pname].append(r); per_clip[str(c)][pname] = strip_arrays(r)
    def pair(a, b):
        return paired_block_bootstrap([{"tp_a": x["_arrays"]["tp"], "tp_b": y["_arrays"]["tp"], "ref": x["_arrays"]["ref"],
                                        "cost_a": x["_arrays"]["cost"], "cost_b": y["_arrays"]["cost"]} for x, y in zip(results[a], results[b])])
    comp = {"robust_vs_fast": pair("robust", "fast"), "difficulty_vs_fast": pair("difficulty", "fast"),
            "difficulty_vs_robust": pair("difficulty", "robust"), "rule_vs_fast": pair("rule", "fast")}
    comp["verdict_difficulty"] = verdict(comp["difficulty_vs_fast"], comp["difficulty_vs_robust"], args.margin)
    if "learned" in policies:
        comp["learned_vs_fast"] = pair("learned", "fast"); comp["learned_vs_robust"] = pair("learned", "robust")
        comp["verdict_learned"] = verdict(comp["learned_vs_fast"], comp["learned_vs_robust"], args.margin)
    pooled = {p: block_ci([r["_arrays"] for r in results[p]]) for p in policies}
    # does the difficulty score predict FAST misses / ROBUST-helps frames on the held-out clip(s)?
    auc, score_rows = {}, []
    for c, cache in eval_caches.items():
        rows = [r for r in rollout(cache, DifficultyPolicy(est, tau, args.hold), cfg) if r["labelled"]]
        X, hard = _xy(rows); s = est.score_matrix(X)
        helps = np.array([int((r["robust_tp"] or 0) > (r["fast_tp"] or 0)) for r in rows])
        auc[str(c)] = {"predicts_FAST_miss": auc_with_ci(s, hard), "predicts_ROBUST_helps": auc_with_ci(s, helps)}
        score_rows += [{"clip": c, "frame": r["frame"], "difficulty": round(float(si), 4), "fast_tp": r["fast_tp"], "robust_tp": r["robust_tp"]} for r, si in zip(rows, s)]
    res = {"tag": args.tag, "train": train, "eval": evals, "tracker_settings": st, "difficulty_training": drep, "learned_training": trep if "learned" in policies else None,
           "per_clip": per_clip, "comparison": comp, "pooled": {p: {"recall": list(map(float, v[0])), "mean_total_ms": list(map(float, v[1]))} for p, v in pooled.items()},
           "difficulty_auc_heldout": auc, "notes": notes, "created": time.strftime("%Y-%m-%d %H:%M:%S"),
           "final_test": any(c in RE.TEST_CLIPS for c in evals)}
    base = f"final_eval_train{name}_eval{ename}_{args.tag}"
    fd = final_dir()
    (fd / f"{base}.json").write_text(json.dumps(res, indent=2, default=float), encoding="utf-8")
    import csv
    with (fd / f"difficulty_scores_train{name}_eval{ename}_{args.tag}.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["clip", "frame", "difficulty", "fast_tp", "robust_tp"]); w.writeheader(); w.writerows(score_rows)
    (fd / f"{base}.md").write_text(render_md(res), encoding="utf-8")
    make_figures(res, results, score_rows, fd / "figures", base)
    print(render_md(res))
    return res


def render_md(res) -> str:
    L = [f"# Final evaluation: train clips {res['train']} -> eval clips {res['eval']} (tag `{res['tag']}`)", "",
         ("**RESERVED TEST SET RUN**" if res["final_test"] else "Validation run (held-out validation clip; the detector itself used clips 7-8 for model selection)."), "",
         f"Tracker settings: {res['tracker_settings']}. Tracking IDs come from the tracker; reference IDs are used only for scoring (annotated frames only, IoU>=0.5).",
         "Cost = decode + global features + chosen-mode detector call + policy/tracker overhead (composed from measured timings).", ""]
    for clip, per in res["per_clip"].items():
        L += [f"## Eval clip {clip}", "", "| policy | track recall | track precision | IDF1 | MOTA | ID switches | fragmentations | predicted IDs | ROBUST frac | total ms | fps |", "|---|---|---|---|---|---|---|---|---|---|---|"]
        for p, r in per.items():
            t = r["tracking"]
            L.append(f"| {p} | {t['recall']} | {t['precision']} | {t['IDF1']} | {t['MOTA']} | {t['ID_switches']} | {t['fragmentations']} | {t['distinct_pred_ids']} | {r['robust_fraction']} | {r['mean_total_ms']} | {r['composed_fps']} |")
        L.append("")
    L += ["## Difficulty estimator on the held-out clip(s) (AUC, 0.5 = chance; block-bootstrap 95% CI)", "", "| clip | predicts 'FAST misses' | predicts 'ROBUST helps' |", "|---|---|---|"]
    for c, a in res["difficulty_auc_heldout"].items():
        f = lambda x: "n/a" if x is None else f"{x['auc']} {x['ci95']}"
        L.append(f"| {c} | {f(a['predicts_FAST_miss'])} | {f(a['predicts_ROBUST_helps'])} |")
    d = res["difficulty_training"]
    L += ["", f"Difficulty model: tau={d['tau']:.3f} (never switches: {d['never_switches']}), cost weight lam={d['lam']}, standardised coefficients {d['standardised_coefficients']}", "",
          "## Paired comparisons over all eval frames (delta = first - second; 95% block bootstrap)", "", "| comparison | delta detection recall [point, lo, hi] | delta total ms/frame [point, lo, hi] |", "|---|---|---|"]
    for k, v in res["comparison"].items():
        if isinstance(v, dict):
            L.append(f"| {k} | {v['d_recall']} | {v['d_cost_ms']} |")
    L += ["", f"**Verdict, difficulty policy: {res['comparison']['verdict_difficulty']}**" + (f" | learned tree: {res['comparison'].get('verdict_learned')}" if 'verdict_learned' in res['comparison'] else ""),
          "Rule (uavtrack/protocol.py::verdict): SUPPORTED = better than FAST AND cheaper than ROBUST AND within 2 points recall of ROBUST; NOT SUPPORTED = not better than FAST, or clearly worse than ROBUST without being cheaper; else INCONCLUSIVE.", ""]
    L += res["notes"]
    return "\n".join(L)


def make_figures(res, results, score_rows, figdir: Path, base: str):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    cols = {"fast": "#4c78a8", "robust": "#e45756", "rule": "#f58518", "learned": "#54a24b", "difficulty": "#7b3294"}
    fig, ax = plt.subplots(figsize=(7.5, 5))
    for p, v in res["pooled"].items():
        (r, rl, rh), (c, cl, ch) = v["recall"], v["mean_total_ms"]
        ax.errorbar(c, r * 100, xerr=[[c - cl], [ch - c]], yerr=[[(r - rl) * 100], [(rh - r) * 100]], fmt="o", color=cols.get(p), capsize=3, label=p, ms=8)
    ax.set_xlabel("mean per-frame cost (ms, composed; see report)"); ax.set_ylabel("detection recall of chosen mode (%)")
    ax.set_title(f"Quality vs cost, train {res['train']} -> eval {res['eval']}"); ax.grid(alpha=.3); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(figdir / f"{base}_quality_cost.png", dpi=150); plt.close(fig)
    fig, ax = plt.subplots(figsize=(8, 4.5)); names = list(res["per_clip"][str(res["eval"][0])].keys()); w = .8 / len(res["eval"])
    for i, c in enumerate(res["eval"]):
        ax.bar(np.arange(len(names)) + i * w, [res["per_clip"][str(c)][p]["tracking"]["IDF1"] for p in names], w, label=f"clip {c}")
    ax.set_xticks(np.arange(len(names)) + w * (len(res["eval"]) - 1) / 2); ax.set_xticklabels(names); ax.set_ylabel("IDF1"); ax.set_title("Identity consistency (IDF1) by policy"); ax.legend(); ax.grid(axis="y", alpha=.3)
    fig.tight_layout(); fig.savefig(figdir / f"{base}_idf1.png", dpi=150); plt.close(fig)
    for c in res["eval"]:
        rows = [r for r in score_rows if r["clip"] == c]
        if not rows: continue
        fig, ax = plt.subplots(figsize=(10, 3.8)); fr = [r["frame"] for r in rows]
        ax.plot(fr, [r["difficulty"] for r in rows], color="#7b3294", lw=1, label="difficulty score")
        ax.scatter([r["frame"] for r in rows if r["fast_tp"] == 0], [-0.05] * sum(r["fast_tp"] == 0 for r in rows), s=6, color="#999", label="FAST missed")
        ax.scatter([r["frame"] for r in rows if (r["robust_tp"] or 0) > (r["fast_tp"] or 0)], [-0.12] * sum((r["robust_tp"] or 0) > (r["fast_tp"] or 0) for r in rows), s=6, color="#e45756", label="ROBUST helped")
        ax.axhline(res["difficulty_training"]["tau"], ls=":", color="k", lw=1); ax.set_ylim(-.2, 1.05); ax.set_xlabel("frame"); ax.set_ylabel("score")
        ax.set_title(f"Clip {c}: estimated difficulty vs outcomes (threshold dotted)"); ax.legend(fontsize=7, ncol=3); ax.grid(alpha=.3)
        fig.tight_layout(); fig.savefig(figdir / f"{base}_difficulty_clip{c}.png", dpi=150); plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--train", type=int, nargs="+", required=True); ap.add_argument("--eval", type=int, nargs="+", required=True)
    ap.add_argument("--tag", default="default"); ap.add_argument("--final", action="store_true")
    ap.add_argument("--lam", type=float, default=0.1, help="extra target hits one ROBUST call must buy (cost weight)")
    ap.add_argument("--hold", type=int, default=3); ap.add_argument("--iterations", type=int, default=2)
    ap.add_argument("--margin", type=float, default=0.02)
    ap.add_argument("--high-conf", type=float, default=None); ap.add_argument("--max-age", type=int, default=None); ap.add_argument("--emit-coasting", action="store_true")
    a = ap.parse_args()
    if any(c in RE.TEST_CLIPS for c in a.train):
        raise SystemExit("Never train on clips 9/10.")
    if any(c in RE.TEST_CLIPS for c in a.eval) and not a.final:
        raise SystemExit("Clips 9/10 are the reserved test set: add --final (run once, frozen settings).")
    if set(a.train) & set(a.eval):
        raise SystemExit("Train and eval clips overlap: not a held-out evaluation.")
    res = run_direction(a.train, a.eval, a)
    if res["final_test"]:
        with (final_dir() / "test_runs.log").open("a", encoding="utf-8") as f:
            f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} FINAL train={a.train} eval={a.eval} tag={a.tag}\n")


if __name__ == "__main__":
    main()
