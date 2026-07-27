#!/usr/bin/env python3
"""Compact TB/CSV metric digests for agents. Numbers + invalidation warnings only."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any

try:
    import tomllib
except ImportError:  # pragma: no cover
    import tomli as tomllib  # type: ignore

STEP_NAMES = {
    "step", "steps", "global_step", "iteration", "iter", "epoch", "episodes", "episode",
}
POS = ("reward", "return", "accuracy", "acc", "success", "score", "psnr", "ssim", "iou", "f1")
AUX = (
    "lr", "learning_rate", "wd", "weight_decay", "grad_norm", "clip", "lambda",
    "physramp", "phys_ramp", "schedule", "aux_",
)
PRIMARY = ("loss", "reward", "return", "residual", "error", "objective", "val", "metric_")

BIMODAL_K = 8.0
STILL_IMPROVING_FRAC = 0.1
STILL_IMPROVING_ABS = 10
SPARSE_GAP_REL_TOL = 0.2
SPARSE_MIN_FINITE = 3


def finite(x: float) -> bool:
    return math.isfinite(x)


def fnum(s: str) -> float | None:
    try:
        return float(s)
    except (TypeError, ValueError):
        return None


def r(x: float) -> float:
    if not finite(x):
        return x
    a = abs(x)
    return round(x, 4 if a >= 100 or a == 0 else 6 if a >= 1 else 8)


def higher_better(name: str) -> bool:
    n = name.lower()
    return any(p in n for p in POS)


def is_aux(name: str) -> bool:
    n = name.lower().replace("-", "_")
    if n.startswith("aux_"):
        return True
    if n in {"lr", "eps", "beta", "ramp", "physramp"}:
        return True
    if any(p in n for p in AUX) or n.startswith("lambda") or "_lambda" in n:
        return True
    return "ramp" in n and ("phys" in n or n.endswith("ramp"))


def is_primary(name: str) -> bool:
    n = name.lower()
    return any(x in n for x in PRIMARY)


def med(xs: list[float]) -> float:
    if not xs:
        return 0.0
    s = sorted(xs)
    m = len(s) // 2
    return s[m] if len(s) % 2 else 0.5 * (s[m - 1] + s[m])


def std(xs: list[float]) -> float:
    if len(xs) < 2:
        return 0.0
    m = sum(xs) / len(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / len(xs))


def find_metrics_config(data_path: Path, override: Path | None) -> tuple[str | None, dict[str, Any]]:
    if override is not None:
        if not override.is_file():
            return None, {}
        with override.open("rb") as f:
            return str(override), tomllib.load(f)
    cur = data_path if data_path.is_dir() else data_path.parent
    for _ in range(30):
        cand = cur / ".metrics.toml"
        if cand.is_file():
            with cand.open("rb") as f:
                return str(cand), tomllib.load(f)
        if cur.parent == cur:
            break
        cur = cur.parent
    return None, {}


def bimodal_split(values: list[float]) -> tuple[float, float] | None:
    fv = sorted(v for v in values if finite(v))
    if len(fv) < 8:
        return None
    gaps = [fv[i + 1] - fv[i] for i in range(len(fv) - 1)]
    if not gaps:
        return None
    med_gap = med(gaps)
    if med_gap <= 0:
        med_gap = max(gaps) or 1e-12
    max_i = max(range(len(gaps)), key=lambda i: gaps[i])
    if gaps[max_i] <= BIMODAL_K * med_gap:
        return None
    threshold = (fv[max_i] + fv[max_i + 1]) / 2
    g0 = [v for v in fv if v < threshold]
    g1 = [v for v in fv if v >= threshold]
    if len(g0) < 3 or len(g1) < 3:
        return None
    m0, m1 = sum(g0) / len(g0), sum(g1) / len(g1)
    s0, s1 = std(g0), std(g1)
    sep = abs(m1 - m0)
    if sep < 3 * max(s0, s1, 1e-9):
        return None
    return threshold, gaps[max_i] / med_gap


def group_digest(ss: list[float], vv: list[float], hi: bool, group_id: int) -> dict[str, Any]:
    n = len(vv)
    bi = max(range(n), key=lambda i: vv[i]) if hi else min(range(n), key=lambda i: vv[i])
    early, late = vv[: max(3, n // 10)], vv[-max(3, n // 10) :]
    li = last_improvement_index(vv, hi)
    return {
        "group": group_id,
        "n": n,
        "mean": r(sum(vv) / n),
        "last_step": r(ss[-1]),
        "early": r(med(early)),
        "late": r(med(late)),
        "best": {"step": r(ss[bi]), "value": r(vv[bi])},
        "last": {"step": r(ss[-1]), "value": r(vv[-1])},
        "last_improvement_step": r(ss[li]),
    }


def group_stats(steps: list[float], values: list[float], threshold: float, hi: bool) -> list[dict[str, Any]]:
    g0s, g0v, g1s, g1v = [], [], [], []
    for s, v in zip(steps, values):
        if not finite(v):
            continue
        if v < threshold:
            g0s.append(s)
            g0v.append(v)
        else:
            g1s.append(s)
            g1v.append(v)
    out = []
    for n, ss, vv in ((0, g0s, g0v), (1, g1s, g1v)):
        if not vv:
            continue
        out.append(group_digest(ss, vv, hi, n))
    return out


def groups_separated(by_phase: dict[int, list[tuple[float, float]]]) -> bool:
    phases = [p for p in sorted(by_phase) if p >= 0 and len(by_phase[p]) >= 3]
    if len(phases) < 2:
        return False
    stats = []
    for ph in phases[:2]:
        vv = [v for _, v in by_phase[ph]]
        stats.append((sum(vv) / len(vv), std(vv)))
    (m0, s0), (m1, s1) = stats[0], stats[1]
    return abs(m1 - m0) >= 3 * max(s0, s1, 1e-9)


def warn_bimodal(steps: list[float], values: list[float], hi: bool) -> dict[str, Any] | None:
    split = bimodal_split(values)
    if split is None:
        return None
    threshold, gap_ratio = split
    groups = group_stats(steps, values, threshold, hi)
    if len(groups) < 2:
        return None
    return {
        "kind": "bimodal",
        "gap_ratio": r(gap_ratio),
        "groups": groups,
        "note": "last/best/early/late computed over mixed population",
    }


def warn_phase_bimodal(
    steps: list[float],
    values: list[float],
    phase_label: list[int],
    hi: bool,
    gap_ratio: float,
) -> dict[str, Any] | None:
    """Attach phase_from split only when this series is actually separated under those labels."""
    by_phase: dict[int, list[tuple[float, float]]] = {}
    for s, v, ph in zip(steps, values, phase_label):
        if not finite(v) or ph < 0:
            continue
        by_phase.setdefault(ph, []).append((s, v))
    if not groups_separated(by_phase):
        return None
    groups = [group_digest([s for s, _ in by_phase[ph]], [v for _, v in by_phase[ph]], hi, ph) for ph in sorted(by_phase)]
    return {
        "kind": "bimodal",
        "gap_ratio": r(gap_ratio),
        "groups": groups,
        "note": "phase_from split; scalars mix populations",
    }


def warn_late_start(steps: list[float], values: list[float]) -> dict[str, Any] | None:
    first_i = next((i for i, v in enumerate(values) if finite(v)), None)
    if first_i is None or first_i == 0:
        return None
    if not all(not finite(values[i]) for i in range(first_i)):
        return None
    return {
        "kind": "late_start",
        "first_finite_step": r(steps[first_i]),
        "leading_nan": first_i,
        "note": "leading nan before first finite; inactive term, not sparse cadence",
    }


def warn_sparse(steps: list[float], values: list[float], val_every: int | None) -> dict[str, Any] | None:
    nan_n = sum(1 for v in values if isinstance(v, float) and math.isnan(v))
    if nan_n == 0:
        return None
    finite_steps = [s for s, v in zip(steps, values) if finite(v)]
    if len(finite_steps) >= SPARSE_MIN_FINITE:
        gaps = [finite_steps[i + 1] - finite_steps[i] for i in range(len(finite_steps) - 1)]
        if gaps:
            mg = med([abs(g) for g in gaps])
            if mg > 0 and all(abs(abs(g) - mg) <= SPARSE_GAP_REL_TOL * mg for g in gaps):
                period = int(round(mg)) if mg >= 1 else r(mg)
                # period 1 = logged every step → not sparse (usually leading/trailing nan)
                if isinstance(period, (int, float)) and float(period) < 2:
                    return None
                if val_every is not None and abs(period - val_every) <= max(1, val_every * 0.1):
                    period = val_every
                return {
                    "kind": "sparse_logging",
                    "period": period,
                    "note": "regular cadence; nan count is not numeric failure",
                }
    if val_every and nan_n > 0 and val_every >= 2:
        ratio = nan_n / max(len(values), 1)
        if ratio > 0.5:
            return {
                "kind": "sparse_logging",
                "period": val_every,
                "note": "regular cadence; nan count is not numeric failure",
            }
    return None


def last_improvement_index(vals: list[float], hi: bool) -> int:
    best_i = 0
    best = vals[0]
    for i in range(1, len(vals)):
        if hi:
            if vals[i] > best:
                best, best_i = vals[i], i
        else:
            if vals[i] < best:
                best, best_i = vals[i], i
    return best_i


def warn_improvement(steps: list[float], values: list[float], hi: bool) -> list[dict[str, Any]]:
    pairs = [(s, v) for s, v in zip(steps, values) if finite(v)]
    if len(pairs) < 3 or not steps:
        return []
    fs, fv = [p[0] for p in pairs], [p[1] for p in pairs]
    run_final = steps[-1]  # run timeline end, not last finite of this series
    li = last_improvement_index(fv, hi)
    span = run_final - fs[0]
    gap = run_final - fs[li]
    out: list[dict[str, Any]] = []
    near = gap <= STILL_IMPROVING_ABS or (span > 0 and gap / span <= STILL_IMPROVING_FRAC)
    if li > 0 and near:
        out.append(
            {
                "kind": "still_improving",
                "last_improvement_step": r(fs[li]),
                "final_step": r(run_final),
                "note": "last improvement near final step; run may be truncated",
            }
        )
    elif li < len(fs) - 1 and not (li > 0 and near):
        out.append(
            {
                "kind": "plateaued_since",
                "since_step": r(fs[li]),
                "final_step": r(run_final),
            }
        )
    return out


def collect_warnings(
    name: str,
    steps: list[float],
    values: list[float],
    hi: bool,
    cfg: dict[str, Any],
    phase_label: list[int] | None,
    phase_gap_ratio: float | None,
) -> list[dict[str, Any]]:
    val_every = cfg.get("val_every")
    if isinstance(val_every, float):
        val_every = int(val_every)
    warnings: list[dict[str, Any]] = []

    wb = None
    if phase_label is not None and phase_gap_ratio is not None and len(phase_label) == len(values):
        wb = warn_phase_bimodal(steps, values, phase_label, hi, phase_gap_ratio)
    if wb is None:
        wb = warn_bimodal(steps, values, hi)
    if wb:
        warnings.append(wb)

    wl = warn_late_start(steps, values)
    if wl:
        warnings.append(wl)

    ws = warn_sparse(steps, values, val_every if isinstance(val_every, int) else None)
    if ws:
        warnings.append(ws)

    if not is_aux(name):
        warnings.extend(warn_improvement(steps, values, hi))

    return warnings


def phase_labels_from_column(steps: list[float], values: list[float]) -> tuple[list[int], float] | None:
    split = bimodal_split(values)
    if split is None:
        return None
    threshold, gap_ratio = split
    labels = [0 if (finite(v) and v < threshold) else (1 if finite(v) else -1) for v in values]
    return labels, gap_ratio


def sample(steps: list[float], vals: list[float], n: int, extra: list[int]) -> list[dict]:
    m = len(vals)
    if n <= 0 or m == 0:
        return []
    if m <= n:
        return [{"step": r(s), "value": r(v)} for s, v in zip(steps, vals)]
    idxs = {0, m - 1, min(range(m), key=lambda i: vals[i]), max(range(m), key=lambda i: vals[i])}
    for i in extra:
        if 0 <= i < m:
            idxs.update({i, max(0, i - 1), min(m - 1, i + 1)})
    deltas = sorted(((abs(vals[i] - vals[i - 1]), i) for i in range(1, m)), reverse=True)
    for _, i in deltas[: max(3, n // 3)]:
        idxs.add(i)
    k = 0
    while len(idxs) < n and k < n:
        idxs.add(round(k * (m - 1) / max(n - 1, 1)))
        k += 1
    chosen = sorted(idxs)[:n]
    if 0 not in chosen:
        chosen[0] = 0
    if m - 1 not in chosen:
        chosen[-1] = m - 1
    return [{"step": r(steps[i]), "value": r(vals[i])} for i in sorted(set(chosen))]


def analyze(
    name: str,
    steps: list[float],
    values: list[float],
    max_pts: int,
    samples: bool,
    cfg: dict[str, Any],
    phase_label: list[int] | None,
    phase_gap_ratio: float | None,
) -> dict[str, Any]:
    hi, aux = higher_better(name), is_aux(name)
    nan_n = sum(1 for v in values if isinstance(v, float) and math.isnan(v))
    inf_n = sum(1 for v in values if isinstance(v, float) and math.isinf(v))
    pairs = [(s, v) for s, v in zip(steps, values) if finite(v)]
    warnings = collect_warnings(name, steps, values, hi, cfg, phase_label, phase_gap_ratio)
    out: dict[str, Any] = {
        "name": name,
        "count": len(values),
        "finite": len(pairs),
        "nan": nan_n,
        "inf": inf_n,
        "aux": aux,
    }
    if warnings:
        out["warnings"] = warnings
    if not pairs:
        return out

    fs, fv = [p[0] for p in pairs], [p[1] for p in pairs]
    n = len(fv)
    bi = max(range(n), key=lambda i: fv[i]) if hi else min(range(n), key=lambda i: fv[i])
    last, best = fv[-1], fv[bi]
    vmin, vmax = min(fv), max(fv)
    span = vmax - vmin
    scale = abs(med(fv)) + 1e-12

    max_rel = 0.0
    j_i = r_i = 0
    for i in range(1, n):
        jump = abs(fv[i] - fv[i - 1])
        rel = jump / (abs(fv[i - 1]) + scale * 0.01)
        if rel > max_rel:
            max_rel, r_i = rel, i
            j_i = i

    early, late = fv[: max(3, n // 10)], fv[-max(3, n // 10) :]
    out["_var"] = span / scale

    out.update(
        min=r(vmin),
        max=r(vmax),
        last={"step": r(fs[-1]), "value": r(last)},
        best={"step": r(fs[bi]), "value": r(best), "higher_better": hi},
        early=r(med(early)),
        late=r(med(late)),
        jump={"step": r(fs[j_i]), "rel": r(max_rel)},
    )
    if samples:
        out["sampled"] = sample(fs, fv, max_pts, [j_i, r_i, bi])
    return out


def slim(m: dict[str, Any], mode: str) -> dict[str, Any]:
    keys = [
        "name", "count", "finite", "nan", "inf", "aux",
        "min", "max", "last", "best", "early", "late", "jump", "warnings",
    ]
    if mode == "detail":
        keys.append("sampled")
    return {k: m[k] for k in keys if k in m}


def pack(
    path: str,
    source: str,
    metrics: dict[str, dict],
    *,
    mode: str,
    top_k: int,
    step_col: str | None = None,
    step_order: str | None = None,
    config_path: str | None = None,
    cfg: dict[str, Any] | None = None,
    filt_explicit: set[str] | None = None,
) -> dict[str, Any]:
    cfg = cfg or {}
    king = cfg.get("king")
    objs = [m for m in metrics.values() if not m.get("aux")]
    auxs = [m for m in metrics.values() if m.get("aux")]
    ranked = sorted(
        objs,
        key=lambda m: (0 if is_primary(m["name"]) else 1, -m.get("_var", 0.0), m["name"]),
    )

    if mode == "screen":
        focus = ranked[:top_k]
        for p in (m for m in ranked if is_primary(m["name"])):
            if p not in focus and len(focus) < top_k + 2:
                focus.append(p)
        if king and king in metrics:
            km = metrics[king]
            if km not in focus:
                if len(focus) >= top_k:
                    focus[-1] = km
                else:
                    focus.append(km)
        top = {m["name"]: slim(m, mode) for m in focus}
        other = [m["name"] for m in ranked if m["name"] not in top] + [m["name"] for m in auxs]
        out: dict[str, Any] = {
            "path": path,
            "source": source,
            "mode": mode,
            "n": {"total": len(metrics), "obj": len(objs), "aux": len(auxs)},
            "top": top,
            "other": other[:40],
            "next": list(top.keys())[:top_k],
        }
    else:
        if filt_explicit:
            chosen = [metrics[n] for n in sorted(filt_explicit) if n in metrics]
        else:
            chosen = ranked + auxs
        out = {
            "path": path,
            "source": source,
            "mode": mode,
            "n": {"total": len(metrics), "obj": len(objs), "aux": len(auxs)},
            "metrics": {m["name"]: slim(m, mode) for m in chosen if m["name"] in metrics},
        }
    if step_col:
        out["step"] = step_col
    if step_order and step_order != "already_asc":
        out["order"] = step_order
    if config_path:
        out["config"] = config_path
    return out


def load_csv(
    path: Path,
    filt: set[str] | None,
    always_include: set[str] | None = None,
) -> tuple[str | None, dict[str, dict[str, list[float]]]]:
    with path.open(newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            return None, {}
        headers = list(reader.fieldnames)
        step = next((h for h in headers if h.lower().strip() in STEP_NAMES), None)
        cols = [h for h in headers if h != step]
        if filt:
            want = {m.lower() for m in filt}
            extra = always_include or set()
            cols = [c for c in cols if c in filt or c.lower() in want or c in extra]
        series = {c: {"steps": [], "values": []} for c in cols}
        for i, row in enumerate(reader):
            st = float(i)
            if step and row.get(step) not in (None, ""):
                p = fnum(row[step])
                if p is not None:
                    st = p
            for c in cols:
                raw = row.get(c)
                if raw in (None, ""):
                    v = float("nan")
                else:
                    v = fnum(raw)
                    if v is None:
                        v = float("nan")
                series[c]["steps"].append(st)
                series[c]["values"].append(v)
    return step, series


def sort_series(series: dict[str, dict[str, list[float]]]) -> tuple[dict, str]:
    out = {}
    rev = mix = False
    for name, data in series.items():
        steps, vals = data["steps"], data["values"]
        if len(steps) < 2:
            out[name] = data
            continue
        if all(steps[i] >= steps[i + 1] for i in range(len(steps) - 1)) and steps[0] > steps[-1]:
            rev = True
        elif any(steps[i] > steps[i + 1] for i in range(len(steps) - 1)):
            mix = True
        combined = sorted(zip(steps, vals))
        out[name] = {"steps": [s for s, _ in combined], "values": [v for _, v in combined]}
    order = "newest_first" if rev else ("mixed" if mix else "already_asc")
    return out, order


def cfg_always_include(cfg: dict[str, Any]) -> set[str]:
    out: set[str] = set()
    for key in ("phase_from", "king"):
        v = cfg.get(key)
        if isinstance(v, str) and v:
            out.add(v)
    return out


def build(
    series: dict,
    max_pts: int,
    samples: bool,
    cfg: dict[str, Any],
) -> tuple[dict, str]:
    series, order = sort_series(series)
    phase_from = cfg.get("phase_from")
    phase_label: list[int] | None = None
    phase_gap_ratio: float | None = None
    if phase_from and phase_from in series:
        pl = phase_labels_from_column(series[phase_from]["steps"], series[phase_from]["values"])
        if pl is not None:
            phase_label, phase_gap_ratio = pl

    metrics = {
        n: analyze(n, d["steps"], d["values"], max_pts, samples, cfg, phase_label, phase_gap_ratio)
        for n, d in series.items()
        if d["values"]
    }
    return metrics, order


def summarize_csv(
    path: Path,
    filt: set[str] | None,
    mode: str,
    max_pts: int,
    top_k: int,
    cfg: dict[str, Any],
    config_path: str | None,
    filt_explicit: set[str] | None,
) -> dict:
    step, series = load_csv(path, filt, always_include=cfg_always_include(cfg))
    if step is None and not series:
        return {"path": str(path), "error": "empty_or_missing_header"}
    metrics, order = build(series, max_pts, mode == "detail", cfg)
    return pack(
        str(path),
        "csv",
        metrics,
        mode=mode,
        top_k=top_k,
        step_col=step,
        step_order=order,
        config_path=config_path,
        cfg=cfg,
        filt_explicit=filt_explicit,
    )


def tfevents(root: Path) -> list[Path]:
    return sorted(root.rglob("events.out.tfevents.*"))


def summarize_tb(
    path: Path,
    filt: set[str] | None,
    mode: str,
    max_pts: int,
    top_k: int,
    cfg: dict[str, Any],
    config_path: str | None,
    filt_explicit: set[str] | None,
) -> dict:
    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    except ImportError:
        return {"path": str(path), "error": "tensorboard_not_installed", "hint": "pip install -r requirements.txt"}

    run_dirs = [path.parent] if path.is_file() else sorted({p.parent for p in tfevents(path)})
    if not run_dirs:
        return {"path": str(path), "error": "no_tfevents_found"}

    runs = []
    for rd in run_dirs:
        ea = EventAccumulator(str(rd), size_guidance={"scalars": 0})
        ea.Reload()
        tags = ea.Tags().get("scalars", [])
        if filt:
            want = {m.lower() for m in filt}
            extra = cfg_always_include(cfg)
            tags = [t for t in tags if t in filt or t.lower() in want or t in extra]
        series = {
            t: {"steps": [float(e.step) for e in ea.Scalars(t)], "values": [float(e.value) for e in ea.Scalars(t)]}
            for t in tags
        }
        metrics, order = build(series, max_pts, mode == "detail", cfg)
        runs.append(
            pack(
                str(rd),
                "tensorboard",
                metrics,
                mode=mode,
                top_k=top_k,
                step_order=order,
                config_path=config_path,
                cfg=cfg,
                filt_explicit=filt_explicit,
            )
        )
    if len(runs) == 1:
        return runs[0]
    return {"path": str(path), "source": "tensorboard", "runs": runs}


def detect(
    path: Path,
    filt: set[str] | None,
    mode: str,
    max_pts: int,
    top_k: int,
    cfg: dict[str, Any],
    config_path: str | None,
    filt_explicit: set[str] | None,
) -> dict:
    kw = dict(
        mode=mode,
        max_pts=max_pts,
        top_k=top_k,
        cfg=cfg,
        config_path=config_path,
        filt_explicit=filt_explicit,
    )
    if path.is_file() and path.suffix.lower() == ".csv":
        return summarize_csv(path, filt, **kw)
    if path.is_file() and "tfevents" in path.name:
        return summarize_tb(path, filt, **kw)
    if not path.is_dir():
        return {"path": str(path), "error": "unsupported_path"}
    csvs, ev = sorted(path.glob("*.csv")), tfevents(path)
    if ev and not csvs:
        return summarize_tb(path, filt, **kw)
    if csvs and not ev:
        if len(csvs) == 1:
            return summarize_csv(csvs[0], filt, **kw)
        return {"path": str(path), "source": "csv", "runs": [summarize_csv(c, filt, **kw) for c in csvs]}
    if ev and csvs:
        return {
            "path": str(path),
            "note": "both_csv_and_tb",
            "tb": summarize_tb(path, filt, **kw),
            "csv": [summarize_csv(c, filt, **kw) for c in csvs],
        }
    return {"path": str(path), "error": "no_csv_or_tfevents_found"}


def metrics_map(s: dict) -> dict:
    return s.get("top") or s.get("metrics") or {}


def final_step(summary: dict) -> float | None:
    mm = metrics_map(summary)
    steps = []
    for m in mm.values():
        if "last" in m:
            steps.append(m["last"]["step"])
    return max(steps) if steps else None


def has_bimodal(m: dict) -> bool:
    return any(w.get("kind") == "bimodal" for w in m.get("warnings", []))


def compare(a: dict, b: dict, *, path_b: str | None = None) -> dict:
    ma, mb = metrics_map(a), metrics_map(b)
    shared = sorted(set(ma) & set(mb))
    fa, fb = final_step(a), final_step(b)
    misaligned = fa is not None and fb is not None and abs(fa - fb) > 1e-9
    cmp_warnings: list[dict[str, Any]] = []
    if misaligned:
        cmp_warnings.append(
            {
                "kind": "misaligned",
                "final_step_a": r(fa),
                "final_step_b": r(fb),
                "note": "d_last suppressed; endpoints differ",
            }
        )

    deltas = {}
    for name in shared:
        sa, sb = ma[name], mb[name]
        if "last" not in sa or "last" not in sb:
            continue
        entry: dict[str, Any] = {
            "d_best": r(sb.get("best", {}).get("value", 0) - sa.get("best", {}).get("value", 0)),
            "a": sa["last"],
            "b": sb["last"],
        }
        suppress = misaligned or has_bimodal(sa) or has_bimodal(sb)
        if not suppress:
            entry["d_last"] = r(sb["last"]["value"] - sa["last"]["value"])
        else:
            entry["d_last"] = None
            entry["warnings"] = [{"kind": "misaligned", "note": "do not cite d_last"}]
        deltas[name] = entry

    out: dict[str, Any] = {
        "shared": shared,
        "only_a": sorted(set(ma) - set(mb)),
        "only_b": sorted(set(mb) - set(ma)),
        "deltas": deltas,
    }
    if path_b:
        out["path_b"] = path_b
    if cmp_warnings:
        out["warnings"] = cmp_warnings
    return out


def run(path: Path, *, compare_path: Path | None = None, **kwargs) -> dict[str, Any]:
    config_path, cfg = find_metrics_config(path, kwargs.pop("config_override", None))
    filt_explicit = kwargs.pop("filt_explicit", None)
    filt = kwargs.pop("filt", None)
    summary = detect(path, filt, config_path=config_path, cfg=cfg, filt_explicit=filt_explicit, **kwargs)
    result: dict[str, Any] = {"mode": kwargs.get("mode", "screen"), "summary": summary}
    if compare_path:
        if not compare_path.exists():
            result["compare"] = {"error": "path_not_found", "path": str(compare_path)}
        else:
            # B side for deltas only — do not ship a second full screen digest
            other = detect(
                compare_path, filt, config_path=config_path, cfg=cfg, filt_explicit=filt_explicit, **kwargs
            )
            result["compare"] = compare(summary, other, path_b=str(compare_path))
    return result


def main() -> int:
    p = argparse.ArgumentParser(description="Compact TB/CSV metric digest (numbers only)")
    p.add_argument("path", type=Path)
    p.add_argument("--mode", choices=("screen", "detail"), default="screen")
    p.add_argument("--metrics", default="")
    p.add_argument("--max-points", type=int, default=12)
    p.add_argument("--top-k", type=int, default=5)
    p.add_argument("--compare", type=Path, default=None)
    p.add_argument("--config", type=Path, default=None, help="Override .metrics.toml")
    p.add_argument("--pretty", action="store_true", help="Indent JSON (default compact)")
    args = p.parse_args()

    if not args.path.exists():
        json.dump({"error": "path_not_found", "path": str(args.path)}, sys.stdout)
        print()
        return 1

    filt_list = [m.strip() for m in args.metrics.split(",") if m.strip()]
    filt = set(filt_list) or None
    filt_explicit = set(filt_list) if filt_list and args.mode == "detail" else None

    result = run(
        args.path,
        compare_path=args.compare,
        mode=args.mode,
        max_pts=args.max_points,
        top_k=args.top_k,
        filt=filt,
        filt_explicit=filt_explicit,
        config_override=args.config,
    )

    dump_kw: dict[str, Any] = {"allow_nan": False, "separators": (",", ":")}
    if args.pretty:
        dump_kw = {"allow_nan": False, "indent": 2}
    json.dump(result, sys.stdout, **dump_kw)
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
