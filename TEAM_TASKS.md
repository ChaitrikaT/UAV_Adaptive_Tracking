# TEAM TASKBOARD — suggested division for three active developers

This is a **prioritized backlog**, not a claim that the original system is complete.
Use the existing `feature/data-validation` branch as the shared base. Each person
creates an independent feature branch and PR; merge only after review.

## Immediate acceptance checks — 1 shared baseline

- [ ] A: `python team_preflight.py` explains what is present/absent.
- [ ] B: `python src/check_project.py` prints `SAVED DEMO READY`.
- [ ] C: `python -m unittest discover -s tests -v` passes.
- [ ] C: `python run_research_review.py --example` produces a report *labelled toy*.
- [ ] A: With genuine local CSVs, `python run_research_review.py` produces a
      report containing the 3 saved policies; or explicitly notes missing inputs.
- [ ] Everyone: inspect `git status` before `git add`/commit and keep the
      experiment data, model weights and installer backups out of the PR.

## Person A — research evidence and selector audit

Own: `research_addon/benchmark.py`, `research_addon/policy_audit.py`,
`tests/test_research_addon.py`.

- [ ] Verify same frame and GT count for FAST and ROBUST; use original saved CSV.
- [ ] Check all 90 Clip 8 decisions match exactly one saved outcome.
- [ ] Compare learned policy with BOTH fixed baselines (recall, precision,
      F1, model-call latency), not classifier agreement alone.
- [ ] Report selector underperformance accurately; document Clip 7/8
      detector-validation overlap and unit of timing.
- [ ] For later academic improvements, design a **held-out offline evaluation
      protocol** that doesn't reuse prior validation clips.

Deliverable: auditable `research_report.md`, written limitations, tests.

## Person B — demo, reproducibility, reporting

Own: `run_demo.py`, `src/check_project.py`, docs, and report presentation.

- [ ] Make the committed saved-results demo understandable on a CPU-only laptop.
- [ ] Confirm the dashboard video plays (H.264 included in branch).
- [ ] Put the tested commands and expected outputs in the README.
- [ ] Ensure no fabricated tracker/accuracy/real-time metrics appear in reports.
- [ ] Collect short screenshots or a video from the *existing saved demo* for
      the presentation, with Purdue attribution rules observed.

Deliverable: viewer smoke-test record, guide-facing status presentation.

## Person C — testing, integration of offline evidence, code quality

Own: `research_addon/timeline.py`, `research_addon/track_audit.py`,
`run_research_review.py`, `team_preflight.py`.

- [ ] Verify the Clip 7 detector CSV joins correctly with condition CSV by
      frame index, with no shifted samples or label leakage.
- [ ] Add edge-case tests for missing CSVs, mismatched frame IDs, duplicate
      saved track IDs, and empty predicted outputs.
- [ ] Where independently generated non-operational track records exist,
      check continuity diagnostics for both adjacent frames and observation gaps.
- [ ] Save input hashes and an unambiguous report of available/absent evidence.

Deliverable: one-command retrospective report with passing tests and no data leaks.

## Completion / guide-facing definition of done

**Practical mini-project demonstration ready when:**
- [ ] Fresh-clone demo and 17 regression tests work (3 from reviewed fix + 14 from research add-on).
- [ ] Actual offline report is reproducible from archived CSVs.
- [ ] Claims match the evidence and all negative outcomes are shown.
- [ ] Repository contains accurate setup instructions, report, demonstration,
      and honest known-limitations section.

**Original synopsis not fully satisfied until independently demonstrated:**
- [ ] UAV identity tracking with genuine independent predicted IDs;
- [ ] Actual per-frame adaptive processing, not archived decision replay;
- [ ] End-to-end tracking/latency evaluation on genuinely held-out sequences;
- [ ] Evidence of a useful quality-versus-computation trade-off.

The last items are outside the included package and cannot be certified by its
unit tests; agree revised project scope with the guide if these are mandatory
for the immediate review. Do not mark them complete based on a synthetic demo.
