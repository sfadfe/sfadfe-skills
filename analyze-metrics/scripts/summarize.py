#!/usr/bin/env python3
"""Compress TensorBoard / CSV metrics into compact JSON for agents.

Two-pass design:
  screen  — tiny digest: status, Top-K anomalies, heuristic verdicts (default)
  detail  — focused series with importance-sampled points
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any, Literal

Purpose = Literal["auto", "health", "converge", "compare", "debug"]
Mode = Literal["screen", "detail"]

POSITIVE_NAME_PARTS = (
    "reward",
    "return",
    "accuracy",
    "acc",
    "success",
    "score",
    "psnr",
    "ssim",
    "iou",
    "f1",
)
AUX_NAME_PARTS = ("lr", "learning_rate", "wd", "weight_decay", "grad_norm", "clip", "eps")


def is_finite(x: float) -> bool:
    return math.isfinite(x)


def parse_float(value: str) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def r4(x: float) -> float:
    """Round for compact JSON."""
    if not is_finite(x):
        return x
    ax = abs(x)
    if ax >= 100 or ax == 0:
        return round(x, 4)
    if ax >= 1:
        return round(x, 6)
    return round(x, 8)


def guess_higher_is_better(name: str) -> bool:
    n = name.lower()
    return any(p in n for p in POSITIVE_NAME_PARTS)


def is_aux_metric(name: str) -> bool:
    n = name.lower()
    return any(p in n for p in AUX_NAME_PARTS)


def mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def median(xs: list[float]) -> float:
    if not xs:
        return 0.0
    s = sorted(xs)
    m = len(s) // 2
    return s[m] if len(s) % 2 else 0.5 * (s[m - 1] + s[m])


def sample_importance(
    steps: list[float],
    values: list[float],
    max_points: int,
    extra_idxs: list[int] | None = None,
) -> list[dict[str, float]]:
    """Keep first/last/extrema + high-delta points + optional anomaly idxs."""
    n = len(values)
    if max_points <= 0 or n == 0:
        return []
    if n <= max_points:
        return [{"step": r4(s), "value": r4(v)} for s, v in zip(steps, values)]

    idxs: set[int] = {0, n - 1}
    # Global extrema
    idxs.add(min(range(n), key=lambda i: values[i]))
    idxs.add(max(range(n), key=lambda i: values[i]))
    if extra_idxs:
        for i in extra_idxs:
            if 0 <= i < n:
                idxs.add(i)
                if i - 1 >= 0:
                    idxs.add(i - 1)
                if i + 1 < n:
                    idxs.add(i + 1)

    # Rank consecutive absolute deltas; keep largest jumps.
    deltas = [(abs(values[i] - values[i - 1]), i) for i in range(1, n)]
    deltas.sort(reverse=True)
    for _, i in deltas[: max(4, max_points // 3)]:
        idxs.add(i)

    # Fill remaining with uniform coverage.
    if len(idxs) < max_points:
        for k in range(max_points):
            idxs.add(round(k * (n - 1) / (max_points - 1)))
            if len(idxs) >= max_points:
                break

    chosen = sorted(idxs)[:max_points] if len(idxs) > max_points else sorted(idxs)
    # If trimmed, force endpoints back in.
    if 0 not in chosen:
        chosen[0] = 0
    if n - 1 not in chosen:
        chosen[-1] = n - 1
    chosen = sorted(set(chosen))
    return [{"step": r4(steps[i]), "value": r4(values[i])} for i in chosen]


def analyze_series(
    name: str,
    steps: list[float],
    values: list[float],
    *,
    max_points: int,
    include_samples: bool,
) -> dict[str, Any]:
    higher = guess_higher_is_better(name)
    aux = is_aux_metric(name)
    n = len(values)
    nan_count = sum(1 for v in values if isinstance(v, float) and math.isnan(v))
    inf_count = sum(1 for v in values if isinstance(v, float) and math.isinf(v))
    finite = [(s, v) for s, v in zip(steps, values) if is_finite(v)]

    base: dict[str, Any] = {
        "name": name,
        "count": n,
        "nan_count": nan_count,
        "inf_count": inf_count,
        "finite_count": len(finite),
        "higher_is_better": higher,
        "aux": aux,
        "flags": [],
        "score": 0.0,
    }
    if not finite:
        base["flags"] = ["empty"]
        base["score"] = 100.0
        base["status"] = "bad"
        return base

    fsteps = [s for s, _ in finite]
    fvals = [v for _, v in finite]
    fn = len(fvals)

    best_idx = max(range(fn), key=lambda i: fvals[i]) if higher else min(
        range(fn), key=lambda i: fvals[i]
    )
    last_v, best_v = fvals[-1], fvals[best_idx]
    vmin, vmax = min(fvals), max(fvals)
    span = vmax - vmin
    scale = abs(median(fvals)) + abs(mean(fvals)) * 0.1 + 1e-12

    # Relative jumps
    max_jump = 0.0
    max_jump_idx = 0
    max_rel_jump = 0.0
    max_rel_idx = 0
    for i in range(1, fn):
        jump = abs(fvals[i] - fvals[i - 1])
        rel = jump / (abs(fvals[i - 1]) + scale * 0.01)
        if jump > max_jump:
            max_jump, max_jump_idx = jump, i
        if rel > max_rel_jump:
            max_rel_jump, max_rel_idx = rel, i

    # Recent plateau / dead flatline
    window = max(5, fn // 5)
    head, mid, tail = fvals[:window], fvals[fn // 2 - window // 2 : fn // 2 + window // 2 + 1], fvals[-window:]
    tail_span = max(tail) - min(tail) if tail else 0.0
    plateau = fn >= 5 and tail_span <= max(1e-8, 0.01 * scale, 0.02 * span if span > 0 else 0.0)
    dead_flat = fn >= 5 and span <= max(1e-10, 1e-6 * scale)

    # Explosion / collapse: compare late peak (not only median) to early median.
    early = fvals[: max(3, fn // 10)]
    late = fvals[-max(3, fn // 10) :]
    early_m = median(early)
    late_m = median(late)
    late_extreme = max(late) if not higher else min(late)
    exploded = (not higher) and (
        late_extreme > max(8 * (abs(early_m) + 1e-12), early_m + 4 * scale)
        or last_v > max(8 * (abs(early_m) + 1e-12), early_m + 4 * scale)
    )
    collapsed = higher and (
        late_extreme < 0.15 * (abs(early_m) + 1e-12)
        or last_v < 0.15 * (abs(early_m) + 1e-12)
    )

    # Improvement stall: best is far from end while recent is flat
    progress = 0.0
    if span > 0:
        if higher:
            progress = (last_v - fvals[0]) / span
        else:
            progress = (fvals[0] - last_v) / span
    best_near_end = best_idx >= int(0.85 * (fn - 1))
    stalled = plateau and not best_near_end and fn >= 20

    # Regression from best
    if higher:
        regret = (best_v - last_v) / (abs(best_v) + scale * 0.01)
    else:
        regret = (last_v - best_v) / (abs(best_v) + scale * 0.01)

    # Trailing non-finite: last raw values are nan/inf
    trailing_nonfinite = False
    for v in reversed(values):
        if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
            trailing_nonfinite = True
            break
        if is_finite(v):
            break

    flags: list[str] = []
    score = 0.0
    if nan_count:
        flags.append("nan")
        score += 40 + min(40.0, float(nan_count))
    if inf_count:
        flags.append("inf")
        score += 50 + min(40.0, float(inf_count))
    if trailing_nonfinite:
        flags.append("ends_nonfinite")
        score += 15
    if exploded:
        flags.append("explosion")
        score += 45
    if collapsed:
        flags.append("collapse")
        score += 40
    if dead_flat and not aux:
        flags.append("dead_flat")
        score += 25
    if max_rel_jump >= 3.5:
        flags.append("spike")
        score += min(35.0, 10.0 + max_rel_jump)
    elif max_rel_jump >= 1.5:
        flags.append("jump")
        score += min(20.0, 5.0 + max_rel_jump)
    if stalled and not aux:
        flags.append("stalled")
        score += 15
    elif plateau and not aux and fn >= 10:
        flags.append("plateau")
        score += 8
    if regret >= 0.25 and not aux and fn >= 5:
        flags.append("regressed_from_best")
        score += min(20.0, regret * 20)

    if aux:
        score *= 0.35  # down-rank lr etc. unless catastrophic
        if not (nan_count or inf_count or exploded):
            flags = [f for f in flags if f in {"nan", "inf", "explosion", "spike"}]

    if score >= 40 or "nan" in flags or "inf" in flags or "explosion" in flags:
        status = "bad"
    elif score >= 12:
        status = "warn"
    else:
        status = "ok"

    anomaly_idxs = [max_jump_idx, max_rel_idx, best_idx]
    out: dict[str, Any] = {
        **base,
        "status": status,
        "flags": flags,
        "score": r4(score),
        "min": r4(vmin),
        "max": r4(vmax),
        "last": {"step": r4(fsteps[-1]), "value": r4(last_v)},
        "best": {"step": r4(fsteps[best_idx]), "value": r4(best_v)},
        "progress_0_to_1": r4(max(0.0, min(1.0, progress))),
        "regret_from_best": r4(max(0.0, regret)),
        "plateau_recent": plateau,
        "max_abs_jump": {"step": r4(fsteps[max_jump_idx]), "value": r4(max_jump)},
        "max_rel_jump": {"step": r4(fsteps[max_rel_idx]), "value": r4(max_rel_jump)},
    }
    # Compact early/late anchors (always cheap)
    out["early_median"] = r4(early_m)
    out["late_median"] = r4(late_m)
    if include_samples:
        out["sampled"] = sample_importance(
            fsteps, fvals, max_points, extra_idxs=anomaly_idxs
        )
    return out


def slim_series(series: dict[str, Any], purpose: Purpose, mode: Mode) -> dict[str, Any]:
    """Drop fields not needed for the current purpose/mode."""
    common = ("name", "status", "flags", "score", "count", "last", "best")
    if mode == "screen":
        keys = common + ("nan_count", "inf_count", "regret_from_best", "progress_0_to_1", "aux")
        if purpose in ("health", "auto", "debug"):
            keys += ("max_rel_jump", "early_median", "late_median")
        if purpose in ("converge", "auto"):
            keys += ("plateau_recent",)
        return {k: series[k] for k in keys if k in series}

    # detail
    keys = list(series.keys())
    if purpose == "health":
        drop = {"progress_0_to_1"}
        keys = [k for k in keys if k not in drop]
    elif purpose == "converge":
        drop = {"max_abs_jump"}
        keys = [k for k in keys if k not in drop]
    return {k: series[k] for k in keys}


def pack_run(
    path: str,
    source: str,
    metrics: dict[str, dict[str, Any]],
    *,
    mode: Mode,
    purpose: Purpose,
    top_k: int,
    step_column: str | None = None,
) -> dict[str, Any]:
    ranked = sorted(
        metrics.values(),
        key=lambda m: (m.get("score", 0.0), 0 if m.get("status") == "bad" else 1),
        reverse=True,
    )
    bad = [m["name"] for m in ranked if m.get("status") == "bad"]
    warn = [m["name"] for m in ranked if m.get("status") == "warn"]
    ok = [m["name"] for m in ranked if m.get("status") == "ok"]

    if bad:
        run_status = "bad"
    elif warn:
        run_status = "warn"
    elif not ranked:
        run_status = "empty"
    else:
        run_status = "ok"

    verdicts: list[str] = []
    if not ranked:
        verdicts.append("no_numeric_metrics")
    if any("nan" in m.get("flags", []) for m in ranked):
        verdicts.append("has_nan")
    if any("inf" in m.get("flags", []) for m in ranked):
        verdicts.append("has_inf")
    if any("explosion" in m.get("flags", []) for m in ranked):
        verdicts.append("possible_divergence")
    if any("ends_nonfinite" in m.get("flags", []) for m in ranked):
        verdicts.append("ends_with_nan_or_inf")
    if any("stalled" in m.get("flags", []) for m in ranked):
        verdicts.append("improvement_stalled")
    if any("regressed_from_best" in m.get("flags", []) for m in ranked):
        verdicts.append("regressed_from_best")
    if any("collapse" in m.get("flags", []) for m in ranked):
        verdicts.append("reward_or_score_collapsed")
    primary = [
        m
        for m in ranked
        if not m.get("aux") and any(
            x in m["name"].lower()
            for x in ("loss", "reward", "return", "residual", "error", "objective")
        )
    ]
    if purpose in ("converge", "auto") and primary:
        p = primary[0]
        if p.get("plateau_recent") and p.get("progress_0_to_1", 0) > 0.5:
            verdicts.append(f"primary_plateau:{p['name']}")
        if p.get("progress_0_to_1", 0) > 0.8 and p.get("status") == "ok":
            verdicts.append(f"primary_looking_healthy:{p['name']}")

    if mode == "screen":
        focus = ranked[:top_k]
        # Always include a primary objective metric if missing from Top-K.
        for p in primary[:2]:
            if p not in focus:
                focus.append(p)
        metrics_out = {
            m["name"]: slim_series(m, purpose, mode) for m in focus
        }
        return {
            "path": path,
            "source": source,
            "mode": mode,
            "purpose": purpose,
            "status": run_status,
            "verdicts": verdicts,
            "metric_counts": {
                "total": len(ranked),
                "bad": len(bad),
                "warn": len(warn),
                "ok": len(ok),
            },
            "top_metrics": metrics_out,
            "other_metric_names": [m["name"] for m in ranked if m["name"] not in metrics_out],
            "next": {
                "action": "detail",
                "hint": "Re-run with --mode detail --metrics <names> for sampled points",
                "suggested_metrics": list(metrics_out.keys())[:top_k],
            },
            **({"step_column": step_column} if step_column else {}),
        }

    # detail: include requested/all metrics with samples already attached
    metrics_out = {m["name"]: slim_series(m, purpose, mode) for m in ranked}
    return {
        "path": path,
        "source": source,
        "mode": mode,
        "purpose": purpose,
        "status": run_status,
        "verdicts": verdicts,
        "metric_counts": {
            "total": len(ranked),
            "bad": len(bad),
            "warn": len(warn),
            "ok": len(ok),
        },
        "metrics": metrics_out,
        **({"step_column": step_column} if step_column else {}),
    }


def find_step_value_columns(headers: list[str]) -> tuple[str | None, list[str]]:
    lower = {h: h.lower().strip() for h in headers}
    step_col = None
    for h, lh in lower.items():
        if lh in {
            "step",
            "steps",
            "global_step",
            "iteration",
            "iter",
            "epoch",
            "episodes",
            "episode",
        }:
            step_col = h
            break
    return step_col, [h for h in headers if h != step_col]


def load_csv_series(
    path: Path, metrics_filter: set[str] | None
) -> tuple[str | None, dict[str, dict[str, list[float]]]]:
    with path.open(newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            return None, {}
        headers = list(reader.fieldnames)
        step_col, metric_cols = find_step_value_columns(headers)
        if metrics_filter:
            want = {m.lower() for m in metrics_filter}
            metric_cols = [c for c in metric_cols if c in metrics_filter or c.lower() in want]

        series: dict[str, dict[str, list[float]]] = {
            c: {"steps": [], "values": []} for c in metric_cols
        }
        row_i = 0
        for row in reader:
            step = float(row_i)
            if step_col and row.get(step_col) not in (None, ""):
                parsed = parse_float(row[step_col])
                if parsed is not None:
                    step = parsed
            for c in metric_cols:
                raw = row.get(c)
                if raw is None or raw == "":
                    continue
                val = parse_float(raw)
                if val is None:
                    continue
                series[c]["steps"].append(step)
                series[c]["values"].append(val)
            row_i += 1
    return step_col, series


def build_metrics(
    series: dict[str, dict[str, list[float]]],
    *,
    max_points: int,
    include_samples: bool,
) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for name, data in series.items():
        if not data["values"]:
            continue
        out[name] = analyze_series(
            name,
            data["steps"],
            data["values"],
            max_points=max_points,
            include_samples=include_samples,
        )
    return out


def summarize_csv(
    path: Path,
    metrics_filter: set[str] | None,
    *,
    mode: Mode,
    purpose: Purpose,
    max_points: int,
    top_k: int,
) -> dict[str, Any]:
    step_col, series = load_csv_series(path, metrics_filter)
    if step_col is None and not series:
        return {"path": str(path), "error": "empty_or_missing_header"}
    metrics = build_metrics(
        series, max_points=max_points, include_samples=(mode == "detail")
    )
    return pack_run(
        str(path),
        "csv",
        metrics,
        mode=mode,
        purpose=purpose,
        top_k=top_k,
        step_column=step_col,
    )


def find_tfevent_files(root: Path) -> list[Path]:
    return sorted(root.rglob("events.out.tfevents.*"))


def summarize_tensorboard(
    path: Path,
    metrics_filter: set[str] | None,
    *,
    mode: Mode,
    purpose: Purpose,
    max_points: int,
    top_k: int,
) -> dict[str, Any]:
    try:
        from tensorboard.backend.event_processing.event_accumulator import (
            EventAccumulator,
        )
    except ImportError:
        return {
            "path": str(path),
            "source": "tensorboard",
            "error": "tensorboard_not_installed",
            "hint": "pip install -r analyze-metrics/requirements.txt",
        }

    if path.is_file():
        run_dirs = [path.parent]
    else:
        run_dirs = sorted({p.parent for p in find_tfevent_files(path)})
    if not run_dirs:
        return {"path": str(path), "source": "tensorboard", "error": "no_tfevents_found"}

    runs = []
    for run_dir in run_dirs:
        ea = EventAccumulator(str(run_dir), size_guidance={"scalars": 0})
        ea.Reload()
        tags = ea.Tags().get("scalars", [])
        if metrics_filter:
            want = {m.lower() for m in metrics_filter}
            tags = [t for t in tags if t in metrics_filter or t.lower() in want]
        series: dict[str, dict[str, list[float]]] = {}
        for tag in tags:
            events = ea.Scalars(tag)
            series[tag] = {
                "steps": [float(e.step) for e in events],
                "values": [float(e.value) for e in events],
            }
        metrics = build_metrics(
            series, max_points=max_points, include_samples=(mode == "detail")
        )
        runs.append(
            pack_run(
                str(run_dir),
                "tensorboard",
                metrics,
                mode=mode,
                purpose=purpose,
                top_k=top_k,
            )
        )

    if len(runs) == 1:
        return runs[0]
    # Multi-run screen: compact each
    overall = "ok"
    if any(r.get("status") == "bad" for r in runs):
        overall = "bad"
    elif any(r.get("status") == "warn" for r in runs):
        overall = "warn"
    return {
        "path": str(path),
        "source": "tensorboard",
        "mode": mode,
        "purpose": purpose,
        "status": overall,
        "runs": runs,
    }


def detect_and_summarize(
    path: Path,
    metrics_filter: set[str] | None,
    *,
    mode: Mode,
    purpose: Purpose,
    max_points: int,
    top_k: int,
) -> dict[str, Any]:
    kwargs = dict(
        mode=mode, purpose=purpose, max_points=max_points, top_k=top_k
    )
    if path.is_file() and path.suffix.lower() == ".csv":
        return summarize_csv(path, metrics_filter, **kwargs)
    if path.is_file() and "tfevents" in path.name:
        return summarize_tensorboard(path, metrics_filter, **kwargs)
    if path.is_dir():
        csvs = sorted(path.glob("*.csv"))
        tfevents = find_tfevent_files(path)
        if tfevents and not csvs:
            return summarize_tensorboard(path, metrics_filter, **kwargs)
        if csvs and not tfevents:
            if len(csvs) == 1:
                return summarize_csv(csvs[0], metrics_filter, **kwargs)
            runs = [
                summarize_csv(c, metrics_filter, **kwargs) for c in csvs
            ]
            overall = "ok"
            if any(r.get("status") == "bad" for r in runs):
                overall = "bad"
            elif any(r.get("status") == "warn" for r in runs):
                overall = "warn"
            return {
                "path": str(path),
                "source": "csv",
                "mode": mode,
                "purpose": purpose,
                "status": overall,
                "runs": runs,
            }
        if tfevents and csvs:
            return {
                "path": str(path),
                "note": "both_csv_and_tensorboard_present",
                "tensorboard": summarize_tensorboard(path, metrics_filter, **kwargs),
                "csv_runs": [summarize_csv(c, metrics_filter, **kwargs) for c in csvs],
            }
        return {"path": str(path), "error": "no_csv_or_tfevents_found"}
    return {"path": str(path), "error": "unsupported_path"}


def primary_metrics(summary: dict[str, Any]) -> dict[str, Any]:
    if "top_metrics" in summary:
        return summary["top_metrics"]
    if "metrics" in summary:
        return summary["metrics"]
    return {}


def compare_runs(a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any]:
    ma, mb = primary_metrics(a), primary_metrics(b)
    # If screen used top_metrics, also try full names via other_metric_names — compare shared present ones.
    shared = sorted(set(ma) & set(mb))
    winners: dict[str, str] = {}
    deltas = {}
    for name in shared:
        sa, sb = ma[name], mb[name]
        if "last" not in sa or "last" not in sb:
            continue
        higher = sa.get("higher_is_better") or guess_higher_is_better(name)
        last_delta = sb["last"]["value"] - sa["last"]["value"]
        best_delta = sb.get("best", {}).get("value", 0) - sa.get("best", {}).get("value", 0)
        if higher:
            winner = "b" if last_delta > 0 else "a" if last_delta < 0 else "tie"
        else:
            winner = "b" if last_delta < 0 else "a" if last_delta > 0 else "tie"
        winners[name] = winner
        deltas[name] = {
            "last_delta": r4(last_delta),
            "best_delta": r4(best_delta),
            "winner_by_last": winner,
            "a_last": sa["last"],
            "b_last": sb["last"],
            "a_best": sa.get("best"),
            "b_best": sb.get("best"),
            "a_status": sa.get("status"),
            "b_status": sb.get("status"),
        }
    score_a = sum(1 for w in winners.values() if w == "a")
    score_b = sum(1 for w in winners.values() if w == "b")
    overall = "a" if score_a > score_b else "b" if score_b > score_a else "tie"
    return {
        "shared_metrics": shared,
        "only_a": sorted(set(ma) - set(mb)),
        "only_b": sorted(set(mb) - set(ma)),
        "deltas": deltas,
        "win_counts": {"a": score_a, "b": score_b},
        "overall_winner_by_last": overall,
        "verdicts": [
            f"overall_winner_by_last:{overall}",
            f"a_status:{a.get('status')}",
            f"b_status:{b.get('status')}",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Screen/detail summarizer for TensorBoard or CSV metrics."
    )
    parser.add_argument("path", type=Path, help="CSV file, tfevents file, or run directory")
    parser.add_argument(
        "--mode",
        choices=("screen", "detail"),
        default="screen",
        help="screen=compact digest (default); detail=sampled series",
    )
    parser.add_argument(
        "--purpose",
        choices=("auto", "health", "converge", "compare", "debug"),
        default="auto",
        help="Shapes flags emphasis and output fields",
    )
    parser.add_argument(
        "--metrics",
        type=str,
        default="",
        help="Comma-separated metric/tag names (detail pass)",
    )
    parser.add_argument(
        "--max-points",
        type=int,
        default=24,
        help="Max importance-sampled points per series in detail mode (default 24)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=8,
        help="Top anomalous/primary metrics to include in screen mode (default 8)",
    )
    parser.add_argument(
        "--compare",
        type=Path,
        default=None,
        help="Optional second path for A/B comparison",
    )
    args = parser.parse_args()

    if not args.path.exists():
        json.dump({"error": "path_not_found", "path": str(args.path)}, sys.stdout, indent=2)
        print()
        return 1

    purpose: Purpose = args.purpose  # type: ignore[assignment]
    mode: Mode = args.mode  # type: ignore[assignment]
    if args.compare is not None and purpose == "auto":
        purpose = "compare"

    metrics_filter = {m.strip() for m in args.metrics.split(",") if m.strip()} or None
    # detail without filter still allowed, but keep samples; screen ignores samples.
    if mode == "detail" and metrics_filter is None and args.max_points > 16:
        # Soft cap when dumping all metrics in detail
        pass

    summary = detect_and_summarize(
        args.path,
        metrics_filter,
        mode=mode,
        purpose=purpose,
        max_points=args.max_points,
        top_k=args.top_k,
    )

    result: dict[str, Any] = {
        "mode": mode,
        "purpose": purpose,
        "summary": summary,
    }
    if args.compare is not None:
        if not args.compare.exists():
            result["compare_error"] = {
                "error": "path_not_found",
                "path": str(args.compare),
            }
        else:
            other = detect_and_summarize(
                args.compare,
                metrics_filter,
                mode=mode,
                purpose=purpose,
                max_points=args.max_points,
                top_k=args.top_k,
            )
            result["compare_summary"] = other
            result["comparison"] = compare_runs(summary, other)

    json.dump(result, sys.stdout, indent=2, allow_nan=False)
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
