#!/usr/bin/env python3
"""End-to-end tests for the sparsity study CLI (T6) — the single seam."""
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
PYTHON = str(ROOT / ".venv/bin/python")


def _run_study(tmp_path, extra: list[str] | None = None):
    cmd = [PYTHON, str(ROOT / "scripts/sparsity_study.py"),
           "--transceivers", "2", "--chamber", "10cm", "--n-samples", "1",
           "--seed", "42", "--n-freq", "16", "--grid", "12",
           "--out", str(tmp_path)] + (extra or [])
    res = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    assert res.returncode == 0, res.stderr
    return tmp_path


def test_verdict_files_written(tmp_path):
    out = _run_study(tmp_path)
    md = out / "sparsity_verdict.md"
    js = out / "sparsity_verdict.json"
    assert md.exists() and js.exists()
    text = md.read_text()
    assert "# MicroTomo — Transceiver Sparsity Verdict" in text
    assert "| #Tx | Mode |" in text
    data = json.loads(js.read_text())
    assert len(data["rows"]) == 1
    row = data["rows"][0]
    assert row["n_transceivers"] == 2 and row["mode"] == "static"
    assert row["verdict"] in {"green", "yellow", "red"}
    assert row["chamfer_median_cm"] > 0.0
    assert row["cost_score"] == 2.0          # 2 channels, static
    assert row["complexity_score"] == 2.0    # 2 channels, static


def test_scan_mode_included(tmp_path):
    out = _run_study(tmp_path, ["--scan", "--n-stations", "4"])
    data = json.loads((out / "sparsity_verdict.json").read_text())
    assert [r["mode"] for r in data["rows"]] == ["static", "scan"]
    scan = data["rows"][1]
    assert scan["cost_score"] > 2.0          # motor surcharge
    assert scan["n_stations"] == 4


def test_deterministic_across_runs(tmp_path):
    out1 = _run_study(tmp_path)
    out2 = _run_study(tmp_path)
    d1 = json.loads((out1 / "sparsity_verdict.json").read_text())
    d2 = json.loads((out2 / "sparsity_verdict.json").read_text())
    assert d1["rows"] == d2["rows"]