---
name: analyze-metrics
description: >-
  Compress TensorBoard/CSV training metrics so agents avoid dumping long logs
  into context. Use for run logs, metrics.csv, tfevents, compare runs.
  Screen→detail method; JSON is numeric evidence plus invalidation warnings.
  Project cadence via optional .metrics.toml.
---

# Analyze Metrics

Goal: shrink log context. Not: declare converge / diverge / overfit.

## Method

1. **Never** `Read` / `cat` full CSV or tfevents.
2. **Screen** — one compact digest (or both runs if comparing).
3. **Decide** which series matter from names + user goal.
4. **Detail** only those (`--metrics a,b`). Skip if screen is enough.
5. **Interpret** yourself from numbers; read `warnings` before citing scalars.

```bash
python scripts/summarize.py <path>
python scripts/summarize.py <path> --mode detail --metrics a,b,aux_lr
python scripts/summarize.py <a> --compare <b>
python scripts/summarize.py <path> --config path/to/.metrics.toml
```

`--compare` returns deltas only (no second full digest).

## Warnings

If `warnings` is non-empty on a metric, **do not treat** its top-level `last` / `best` / `early` / `late` as a single population. For `bimodal`, use each group's own early/late/best/last_improvement.

Kinds: `bimodal`, `sparse_logging` (period ≥ 2), `late_start`, `still_improving`, `plateaued_since`, `misaligned` (compare).

- `sparse_logging`: regular finite cadence with period ≥ 2 (not every-step).
- `late_start`: leading NaNs before first finite step (inactive term), not a logging period.
- `still_improving.final_step`: run timeline end, not last finite of that series.

## Optional `.metrics.toml`

Walks up from the log path; `--config` overrides.

```toml
king = "Val"              # always in screen Top-K
val_every = 10            # sparse val cadence hint
phase_from = "AvgLoss_Data"  # phase labels; attach bimodal only if series separates
```

`phase_from` keeps `gap_ratio` from the source split; metrics that do not separate under those labels get no phase bimodal. Detail mode still loads `phase_from` / `king` when filtering.

Project rules (ckpt checks, test scripts) stay in the project.

TB: `pip install -r requirements.txt` (CSV is stdlib-only).
