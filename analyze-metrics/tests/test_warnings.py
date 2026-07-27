#!/usr/bin/env python3
"""Regression: warnings must catch known misleading summaries."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FIX = ROOT / "tests" / "fixtures"
SCRIPT = ROOT / "scripts" / "summarize.py"


def run(*args: str) -> dict:
    out = subprocess.check_output([sys.executable, str(SCRIPT), *args], text=True)
    return json.loads(out)


def kinds(summary: dict, name: str) -> list[str]:
    block = summary.get("top") or summary.get("metrics") or {}
    m = block.get(name)
    if not m:
        return []
    return [w["kind"] for w in m.get("warnings", [])]


def warning(summary: dict, name: str, kind: str) -> dict:
    block = summary.get("top") or summary.get("metrics") or {}
    return next(w for w in block[name]["warnings"] if w["kind"] == kind)


def any_kind(summary: dict, kind: str) -> bool:
    block = summary.get("top") or summary.get("metrics") or {}
    for m in block.values():
        for w in m.get("warnings", []):
            if w.get("kind") == kind:
                return True
    return False


class TestWarningsPositive(unittest.TestCase):
    def test_bimodal_loss_a(self):
        s = run(str(FIX / "alternating_two_phase.csv"))["summary"]
        self.assertIn("bimodal", kinds(s, "loss_a"))
        m = s["top"]["loss_a"]
        groups = next(w for w in m["warnings"] if w["kind"] == "bimodal")["groups"]
        means = sorted(g["mean"] for g in groups)
        self.assertAlmostEqual(means[0], 0.023, places=2)
        self.assertAlmostEqual(means[1], 2.237, places=2)

    def test_bimodal_group_trends(self):
        s = run(str(FIX / "alternating_two_phase.csv"))["summary"]
        groups = warning(s, "loss_a", "bimodal")["groups"]
        high = max(groups, key=lambda g: g["mean"])
        for key in ("early", "late", "best", "last", "last_improvement_step"):
            self.assertIn(key, high)
        # flip group: early ~2.8 → late ~2.0 (monotonic decline within phase)
        self.assertGreater(high["early"], high["late"])

    def test_sparse_logging(self):
        s = run(str(FIX / "alternating_two_phase.csv"))["summary"]
        self.assertIn("sparse_logging", kinds(s, "metric_sparse"))
        w = warning(s, "metric_sparse", "sparse_logging")
        self.assertEqual(w["period"], 10)

    def test_late_start_not_sparse_period_1(self):
        s = run(
            str(FIX / "alternating_two_phase.csv"),
            "--mode",
            "detail",
            "--metrics",
            "loss_roll,loss_c,aux_ramp",
        )["summary"]
        for name in ("loss_roll", "loss_c", "aux_ramp"):
            self.assertNotIn("sparse_logging", kinds(s, name), name)
            self.assertIn("late_start", kinds(s, name), name)
        self.assertEqual(warning(s, "loss_roll", "late_start")["first_finite_step"], 10.0)
        self.assertEqual(warning(s, "loss_c", "late_start")["first_finite_step"], 2.0)

    def test_misaligned_compare(self):
        c = run(
            str(FIX / "alternating_two_phase.csv"),
            "--compare",
            str(FIX / "diverging_tail.csv"),
        )["compare"]
        self.assertNotIn("summary", c)
        self.assertTrue(any(w.get("kind") == "misaligned" for w in c.get("warnings", [])))
        d0 = c["deltas"]["loss_a"]
        self.assertIsNone(d0["d_last"])

    def test_still_improving_uses_run_final_step(self):
        s = run(
            str(FIX / "alternating_two_phase.csv"),
            "--mode",
            "detail",
            "--metrics",
            "metric_sparse,best_val",
        )["summary"]
        w = warning(s, "metric_sparse", "still_improving")
        self.assertEqual(w["last_improvement_step"], 130.0)
        self.assertEqual(w["final_step"], 139.0)  # run end, not last finite of series
        w2 = warning(s, "best_val", "still_improving")
        self.assertEqual(w2["final_step"], 139.0)


class TestWarningsNegative(unittest.TestCase):
    def test_unimodal_no_bimodal(self):
        s = run(str(FIX / "unimodal_clean.csv"))["summary"]
        self.assertFalse(any_kind(s, "bimodal"))

    def test_three_phase_no_bimodal(self):
        s = run(str(FIX / "three_phase.csv"))["summary"]
        self.assertFalse(any_kind(s, "bimodal"))

    def test_genuine_nan_not_sparse(self):
        s = run(str(FIX / "genuinely_nan.csv"))["summary"]
        self.assertNotIn("sparse_logging", kinds(s, "metric_sparse"))

    def test_plateaued_not_still_improving(self):
        s = run(str(FIX / "plateaued.csv"))["summary"]
        self.assertFalse(any_kind(s, "still_improving"))


class TestPhaseFrom(unittest.TestCase):
    def _cfg(self) -> str:
        d = Path(tempfile.mkdtemp())
        p = d / ".metrics.toml"
        p.write_text('phase_from = "loss_a"\nking = "best_val"\nval_every = 10\n')
        return str(p)

    def test_phase_from_keeps_gap_ratio_and_skips_unseparated(self):
        cfg = self._cfg()
        s = run(str(FIX / "alternating_two_phase.csv"), "--config", cfg)["summary"]
        bm = warning(s, "loss_a", "bimodal")
        self.assertIn("gap_ratio", bm)
        self.assertGreater(bm["gap_ratio"], 100)
        # best_val is structurally phase-independent → no phase_from false bimodal
        self.assertNotIn("bimodal", kinds(s, "best_val"))

    def test_screen_detail_bimodal_consistent(self):
        cfg = self._cfg()
        path = str(FIX / "alternating_two_phase.csv")
        screen = run(path, "--config", cfg)["summary"]
        detail = run(path, "--config", cfg, "--mode", "detail", "--metrics", "best_val,loss_a")["summary"]
        self.assertEqual(
            "bimodal" in kinds(screen, "best_val"),
            "bimodal" in kinds(detail, "best_val"),
        )
        self.assertEqual(
            "bimodal" in kinds(screen, "loss_a"),
            "bimodal" in kinds(detail, "loss_a"),
        )
        self.assertIn("gap_ratio", warning(detail, "loss_a", "bimodal"))


class TestDetailAux(unittest.TestCase):
    def test_explicit_aux_in_detail(self):
        s = run(
            str(FIX / "alternating_two_phase.csv"),
            "--mode",
            "detail",
            "--metrics",
            "loss_a,aux_lr",
        )["summary"]
        self.assertIn("aux_lr", s["metrics"])
        self.assertIn("loss_a", s["metrics"])
        self.assertEqual(set(s["metrics"]), {"loss_a", "aux_lr"})


if __name__ == "__main__":
    unittest.main()
