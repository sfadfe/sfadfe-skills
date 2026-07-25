---
name: analyze-metrics
description: >-
  Two-pass compression and analysis of TensorBoard or CSV metrics without
  loading long training dumps into context. Use when inspecting training,
  RL, PINNs, or ML run logs, TensorBoard runs, metrics.csv, convergence,
  divergence, plateaus, NaNs, or comparing runs.
---

# Analyze Metrics

## Hard rules

1. **Never** `Read` / `cat` full TensorBoard event files or large metrics CSVs.
2. Always use `scripts/summarize.py`. Prefer **screen → detail**, not one giant dump.
3. Infer domain from metric names + user goal. Do not assume PINNs/RL/etc.
4. Match answer shape to purpose (`health`, `converge`, `compare`, `debug`). If unclear, ask one short question or use `--purpose auto`.

## Two-pass workflow

### Pass 1 — screen (default, tiny)

```bash
python scripts/summarize.py <path> --mode screen --purpose <auto|health|converge|compare|debug>
```

Use the JSON for:
- run `status`: `ok` / `warn` / `bad`
- `verdicts` (heuristic)
- `top_metrics` only (anomalies + primary objectives)
- `next.suggested_metrics` for follow-up

Do **not** request samples yet.

### Pass 2 — detail (only if needed)

```bash
python scripts/summarize.py <path> --mode detail --purpose <purpose> --metrics name1,name2
```

- Restrict with `--metrics` from pass-1 suggestions or the user's focus.
- Reason from importance-sampled points + flags.
- Avoid `--mode detail` on all metrics unless the run is tiny.

### Compare

```bash
python scripts/summarize.py <pathA> --compare <pathB> --mode screen --purpose compare
```

Then detail only metrics that disagree or look unhealthy.

## Purpose → focus

| Purpose | Focus |
|---------|--------|
| health | NaN/Inf, explosion, spikes, empty |
| converge | best/last, plateau, stall, regret from best |
| compare | shared deltas, winner_by_last |
| debug | anomaly steps + detail samples around jumps |
| auto | mixed screen; refine after first read |

Answer pattern: **verdict → evidence (from JSON) → next action**.

## Compression notes (what the script already does)

- Scores metrics (NaN/Inf, explosion, spike, plateau, stall, regression); screen keeps Top-K.
- Down-ranks aux series (`lr`, `grad_norm`, …) unless catastrophic.
- Detail samples are **importance-sampled** (endpoints, extrema, largest jumps), not only uniform.
- Numbers are rounded; screen omits `sampled` entirely.

## Deps

CSV: stdlib only. TensorBoard: `pip install -r requirements.txt`.
