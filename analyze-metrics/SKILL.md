---
name: analyze-metrics
description: >-
  Compress TensorBoard/CSV training metrics so agents avoid dumping long logs
  into context. Use for run logs, metrics.csv, tfevents, compare runs.
  Teaches a screen→detail method; outputs numbers only — interpret yourself.
  Project-specific logging cadence belongs in the project, not this skill.
---

# Analyze Metrics

Goal: shrink log context. Not: declare converge / diverge / overfit.

## Method

1. **Never** `Read` / `cat` full CSV or tfevents.
2. **Screen** — one compact digest of the run (or both runs if comparing).
3. **Decide what matters** from names + user goal (primary losses, val, rollout, …).
4. **Detail** only those series (`--metrics a,b`). Skip if screen already enough.
5. **Interpret** last/best/early/late/jump/nan counts yourself. The JSON is evidence, not a verdict.
6. Project rules (which metric is king, LR phases, ckpt checks) stay in the project.

```bash
python scripts/summarize.py <path>                          # screen
python scripts/summarize.py <path> --mode detail --metrics a,b
python scripts/summarize.py <a> --compare <b>
```

## What the JSON is for

- `top` / `metrics`: compact numbers to reason over
- `other`: names you can pull into a later detail pass
- `next`: suggested detail targets (still your call)
- schedules (`lr`, `Lambda_*`, `PhysRamp`, …) stay out of Top-K on purpose
- steps are sorted ascending when the file is newest-first

TB needs: `pip install -r requirements.txt` (CSV is stdlib-only).
