#!/usr/bin/env python3
"""Tests for gui/state.py — the pure simulation-state core of the app."""
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from gui.state import SimState, default_state  # noqa: E402


def test_default_state_geometry():
    s = default_state()
    scatter = s.scatter()
    assert scatter.shape[1] == 6
    assert len(scatter) == s.n_scatter_per_object * len(s.objects)
    assert scatter[:, :3].min() >= 0.0 and scatter[:, :3].max() <= s.chamber_m


def test_default_ring_xy_plane():
    s = SimState(chamber_m=0.10, n_tx=6)
    ring = s.default_ring()
    assert len(ring) == 6
    assert np.allclose(ring[:, 2], s.chamber_m / 2)      # z = mid-height
    d = np.linalg.norm(ring - s.chamber_m / 2, axis=1)   # xy radius
    assert np.allclose(d, 0.4 * s.chamber_m, atol=1e-9)


def test_custom_transceiver_table():
    s = SimState(chamber_m=0.10, n_tx=3)
    custom = np.array([[0.01, 0.02, 0.03], [0.04, 0.05, 0.06], [0.07, 0.08, 0.09]])
    s.tx_positions = custom
    assert np.allclose(s.physical_positions(), custom)


def test_scan_virtual_positions_count_and_honour_edits():
    s = SimState(chamber_m=0.10, n_tx=2, mode="scan", n_stations=8)
    virt = s.virtual_positions()
    assert len(virt) == s.n_tx * s.n_stations
    # custom table: station 0 of the scan must be exactly the edited positions
    s.tx_positions = np.array([[0.01, 0.02, 0.05], [0.02, 0.01, 0.05]])
    virt2 = s.virtual_positions()
    assert np.allclose(virt2[: s.n_tx], s.tx_positions)


def test_run_pipeline_static_smoke():
    s = SimState(chamber_m=0.10, n_tx=2, mode="static", n_freq=16, grid=12,
                 seed=1, objects=[default_state().objects[0]])
    recon, metrics, wall = s.run_pipeline()
    assert recon.shape[1] == 3 and len(recon) > 0
    assert metrics["mode"] == "static" and metrics["n_tx"] == 2
    assert metrics["chamfer_mean_cm"] >= 0.0
    assert metrics["wall_time_s"] >= 0.0
    # determinism
    recon2, metrics2, _ = s.run_pipeline()
    assert np.array_equal(recon, recon2)
    assert metrics == metrics2


def test_run_pipeline_scan_smoke():
    s = SimState(chamber_m=0.10, n_tx=1, mode="scan", n_stations=3, n_freq=16,
                 grid=12, seed=2, objects=[default_state().objects[0]])
    recon, metrics, _ = s.run_pipeline()
    assert metrics["mode"] == "scan" and metrics["n_stations"] == 3
    assert len(recon) > 0


def test_state_roundtrip():
    s = default_state()
    s.tx_positions = s.default_ring()
    d = s.to_dict()
    t = SimState.from_dict(d)
    assert t.mode == s.mode and t.n_tx == s.n_tx and t.grid == s.grid
    assert len(t.objects) == len(s.objects)
    assert t.objects[0].center == s.objects[0].center
    assert np.allclose(t.tx_positions, s.tx_positions)