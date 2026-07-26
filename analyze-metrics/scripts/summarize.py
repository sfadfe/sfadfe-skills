"""

Compact TB/CSV metric digests for agents. 

"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any

STEP_NAMES = {
    "step", "steps", "global_step", "iteration", "iter", "epoch", "episodes", "episode",
}
POS = ("reward", "return", "accuracy", "acc", "success", "score", "psnr", "ssim", "iou", "f1")
AUX = (
    "lr", "learning_rate", "wd", "weight_decay", "grad_norm", "clip", "lambda",
    "physramp", "phys_ramp", "schedule",
)
PRIMARY = ("loss", "reward", "return", "residual", "error", "objective", "val")


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


def analyze(name: str, steps: list[float], values: list[float], max_pts: int, samples: bool) -> dict[str, Any]:
    hi, aux = higher_better(name), is_aux(name)
    nan_n = sum(1 for v in values if isinstance(v, float) and math.isnan(v))
    inf_n = sum(1 for v in values if isinstance(v, float) and math.isinf(v))
    pairs = [(s, v) for s, v in zip(steps, values) if finite(v)]
    out: dict[str, Any] = {
        "name": name, "count": len(values), "finite": len(pairs),
        "nan": nan_n, "inf": inf_n, "aux": aux,
    }
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
    # variation used only to pick screen Top-K (compression), not as a verdict
    out["_var"] = span / scale

    out.update(
        min=r(vmin), max=r(vmax),
        last={"step": r(fs[-1]), "value": r(last)},
        best={"step": r(fs[bi]), "value": r(best), "higher_better": hi},
        early=r(med(early)), late=r(med(late)),
        jump={"step": r(fs[j_i]), "rel": r(max_rel)},
    )
    if samples:
        out["sampled"] = sample(fs, fv, max_pts, [j_i, r_i, bi])
    return out


def slim(m: dict[str, Any], mode: str) -> dict[str, Any]:
    keys = [
        "name", "count", "finite", "nan", "inf", "aux",
        "min", "max", "last", "best", "early", "late", "jump",
    ]
    if mode == "detail":
        keys.append("sampled")
    return {k: m[k] for k in keys if k in m}


def pack(
    path: str, source: str, metrics: dict[str, dict], *,
    mode: str, top_k: int,
    step_col: str | None = None, step_order: str | None = None,
) -> dict[str, Any]:
    objs = [m for m in metrics.values() if not m.get("aux")]
    auxs = [m for m in metrics.values() if m.get("aux")]
    # Top-K: primary names first, then highest variation (more signal per token)
    ranked = sorted(
        objs,
        key=lambda m: (0 if is_primary(m["name"]) else 1, -m.get("_var", 0.0), m["name"]),
    )

    if mode == "screen":
        focus = ranked[:top_k]
        for p in (m for m in ranked if is_primary(m["name"])):
            if p not in focus and len(focus) < top_k + 2:
                focus.append(p)
        top = {m["name"]: slim(m, mode) for m in focus}
        other = [m["name"] for m in ranked if m["name"] not in top] + [m["name"] for m in auxs]
        out: dict[str, Any] = {
            "path": path, "source": source, "mode": mode,
            "n": {"total": len(metrics), "obj": len(objs), "aux": len(auxs)},
            "top": top,
            "other": other[:40],
            "next": list(top.keys())[:top_k],
        }
    else:
        out = {
            "path": path, "source": source, "mode": mode,
            "n": {"total": len(metrics), "obj": len(objs), "aux": len(auxs)},
            "metrics": {m["name"]: slim(m, mode) for m in ranked},
        }
    if step_col:
        out["step"] = step_col
    if step_order and step_order != "already_asc":
        out["order"] = step_order
    return out


def load_csv(path: Path, filt: set[str] | None) -> tuple[str | None, dict[str, dict[str, list[float]]]]:
    with path.open(newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            return None, {}
        headers = list(reader.fieldnames)
        step = next((h for h in headers if h.lower().strip() in STEP_NAMES), None)
        cols = [h for h in headers if h != step]
        if filt:
            want = {m.lower() for m in filt}
            cols = [c for c in cols if c in filt or c.lower() in want]
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
                    continue
                v = fnum(raw)
                if v is None:
                    continue
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
        dedup = {s: v for s, v in sorted(zip(steps, vals))}
        ss = sorted(dedup)
        out[name] = {"steps": ss, "values": [dedup[s] for s in ss]}
    order = "newest_first" if rev else ("mixed" if mix else "already_asc")
    return out, order


def build(series: dict, max_pts: int, samples: bool) -> tuple[dict, str]:
    series, order = sort_series(series)
    metrics = {
        n: analyze(n, d["steps"], d["values"], max_pts, samples)
        for n, d in series.items() if d["values"]
    }
    return metrics, order


def summarize_csv(path: Path, filt: set[str] | None, mode: str, max_pts: int, top_k: int) -> dict:
    step, series = load_csv(path, filt)
    if step is None and not series:
        return {"path": str(path), "error": "empty_or_missing_header"}
    metrics, order = build(series, max_pts, mode == "detail")
    return pack(str(path), "csv", metrics, mode=mode, top_k=top_k, step_col=step, step_order=order)


def tfevents(root: Path) -> list[Path]:
    return sorted(root.rglob("events.out.tfevents.*"))


def summarize_tb(path: Path, filt: set[str] | None, mode: str, max_pts: int, top_k: int) -> dict:
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
            tags = [t for t in tags if t in filt or t.lower() in want]
        series = {
            t: {"steps": [float(e.step) for e in ea.Scalars(t)],
                "values": [float(e.value) for e in ea.Scalars(t)]}
            for t in tags
        }
        metrics, order = build(series, max_pts, mode == "detail")
        runs.append(pack(str(rd), "tensorboard", metrics, mode=mode, top_k=top_k, step_order=order))
    if len(runs) == 1:
        return runs[0]
    return {"path": str(path), "source": "tensorboard", "runs": runs}


def detect(path: Path, filt: set[str] | None, mode: str, max_pts: int, top_k: int) -> dict:
    kw = dict(mode=mode, max_pts=max_pts, top_k=top_k)
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
        return {"path": str(path), "note": "both_csv_and_tb",
                "tb": summarize_tb(path, filt, **kw),
                "csv": [summarize_csv(c, filt, **kw) for c in csvs]}
    return {"path": str(path), "error": "no_csv_or_tfevents_found"}


def metrics_map(s: dict) -> dict:
    return s.get("top") or s.get("metrics") or {}


def compare(a: dict, b: dict) -> dict:
    ma, mb = metrics_map(a), metrics_map(b)
    shared = sorted(set(ma) & set(mb))
    deltas = {}
    for name in shared:
        sa, sb = ma[name], mb[name]
        if "last" not in sa or "last" not in sb:
            continue
        deltas[name] = {
            "d_last": r(sb["last"]["value"] - sa["last"]["value"]),
            "d_best": r(
                sb.get("best", {}).get("value", 0) - sa.get("best", {}).get("value", 0)
            ),
            "a": sa["last"], "b": sb["last"],
        }
    return {"shared": shared, "only_a": sorted(set(ma) - set(mb)), "only_b": sorted(set(mb) - set(ma)), "deltas": deltas}


def main() -> int:
    p = argparse.ArgumentParser(description="Compact TB/CSV metric digest (numbers only)")
    p.add_argument("path", type=Path)
    p.add_argument("--mode", choices=("screen", "detail"), default="screen")
    p.add_argument("--metrics", default="")
    p.add_argument("--max-points", type=int, default=12)
    p.add_argument("--top-k", type=int, default=5)
    p.add_argument("--compare", type=Path, default=None)
    p.add_argument("--pretty", action="store_true", help="Indent JSON (default compact)")
    args = p.parse_args()

    if not args.path.exists():
        json.dump({"error": "path_not_found", "path": str(args.path)}, sys.stdout)
        print()
        return 1

    filt = {m.strip() for m in args.metrics.split(",") if m.strip()} or None
    summary = detect(args.path, filt, args.mode, args.max_points, args.top_k)
    result: dict[str, Any] = {"mode": args.mode, "summary": summary}
    if args.compare:
        if not args.compare.exists():
            result["compare_error"] = {"error": "path_not_found", "path": str(args.compare)}
        else:
            other = detect(args.compare, filt, args.mode, args.max_points, args.top_k)
            result["compare"] = other
            result["deltas"] = compare(summary, other)

    dump_kw: dict[str, Any] = {"allow_nan": False, "separators": (",", ":")}
    if args.pretty:
        dump_kw = {"allow_nan": False, "indent": 2}
    json.dump(result, sys.stdout, **dump_kw)
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
