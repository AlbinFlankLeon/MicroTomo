#!/usr/bin/env python3
"""Tests for the reflection-mode forward model (sparsity study T3)."""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from sim.forward_surface import SurfaceScatterModel, write_scene_h5  # noqa: E402
from sim.transceivers import StaticLayout, ScanLayout  # noqa: E402

C = 299_792_458.0
FREQS = np.linspace(57e9, 64e9, 16)


def _model(scatter, wall="absorbing", seed=0, snr_db=40.0):
    return SurfaceScatterModel(
        chamber_m=0.10, freqs_hz=FREQS, scatter=scatter,
        gamma=None, wall_mode=wall, snr_db=snr_db, seed=seed)


def test_two_way_delay_single_scatterer():
    """Phase rotation between frequencies equals -2pi df * two-way delay."""
    scatter = np.array([[0.05, 0.05, 0.05, 1.0, 0.0, 0.0]], dtype=float)
    pos = StaticLayout(2, 0.10).positions
    m = _model(scatter)
    x, _, y = m.phase_history(pos, write_noise=False)
    # monostatic channel tx=0 -> rx=0: planes/axes avoid bistatic degeneracy
    f0, f1 = FREQS[0], FREQS[7]
    ch = 0 * len(pos) + 0
    y0, y1 = y[0, ch], y[7, ch]
    assert abs(y0) > 1e-9 and abs(y1) > 1e-9
    d_two_way = 2.0 * np.linalg.norm(scatter[0, :3] - pos[0])
    tau = d_two_way / C
    expected = -2.0 * np.pi * (f1 - f0) * tau
    got = np.angle(y1 / y0)
    assert got == pytest.approx(np.angle(np.exp(1j * expected)), abs=0.2), (got, expected)


def test_visibility_culling():
    """Scatterer facing AWAY from the Tx contributes almost nothing."""
    # normal -x, antenna at +x looking back: face-away
    scatter = np.array([[0.05, 0.05, 0.05, -1.0, 0.0, 0.0]], dtype=float)
    txpos = np.array([[0.09, 0.05, 0.05]])            # Tx at +x near wall
    rxpos = np.array([[0.05, 0.09, 0.05]])
    pos = np.concatenate([txpos, rxpos])
    m = _model(scatter)
    _, _, y = m.phase_history(pos, write_noise=False)
    assert np.max(np.abs(y)) < 1e-10
    # and with the normal flipped it must contribute
    scatter2 = np.array([[0.05, 0.05, 0.05, 1.0, 0.0, 0.0]], dtype=float)
    m2 = _model(scatter2)
    _, _, y2 = m2.phase_history(pos, write_noise=False)
    assert np.max(np.abs(y2)) > 1e-6


def test_metal_wall_changes_background():
    scatter = np.array([[0.05, 0.05, 0.05, 0.0, 0.0, 1.0]], dtype=float)
    pos = StaticLayout(2, 0.10).positions
    ma = _model(scatter, wall="absorbing")
    mm = _model(scatter, wall="metal")
    _, xh_a, _ = ma.phase_history(pos, write_noise=False)
    _, xh_m, _ = mm.phase_history(pos, write_noise=False)
    assert np.max(np.abs(xh_a - xh_m)) > 1e-6


def test_deterministic_with_noise():
    scatter = np.array([[0.05, 0.05, 0.05, 0.0, 0.0, 1.0]], dtype=float)
    pos = StaticLayout(2, 0.10).positions
    a = _model(scatter, seed=3).phase_history(pos, write_noise=True)
    b = _model(scatter, seed=3).phase_history(pos, write_noise=True)
    assert np.array_equal(a[0], b[0]) and np.array_equal(a[2], b[2])


def test_scan_layout_virtual_positions():
    sc = ScanLayout(2, 0.10, n_stations=24)
    assert sc.positions.shape == (48, 3)
    assert np.allclose(sc.positions[:2], StaticLayout(2, 0.10).positions)


def test_hdf5_roundtrip(tmp_path):
    scatter = np.array([[0.05, 0.05, 0.05, 0.0, 0.0, 1.0]], dtype=float)
    pos = StaticLayout(2, 0.10).positions
    m = _model(scatter)
    x, xh, y = m.phase_history(pos, write_noise=False)
    out = tmp_path / "s.h5"
    write_scene_h5(out, x, xh, y,
                   np.full((8, 8, 8), 1.0, dtype=np.float32),
                   np.full((8, 8, 8), 1.0, dtype=np.float32),
                   {"k": "v"})
    import h5py
    with h5py.File(out, "r") as f:
        assert set(f["samples/0"].keys()) == {"x", "x_hat", "y", "geometry", "eps", "meta"}
        assert f["samples/0"]["x"].dtype == np.complex64
        assert f["samples/0"]["x"].shape == (len(FREQS), 4)
        assert json.loads(f["samples/0"]["meta"][()]) == {"k": "v"}
        assert np.max(np.abs(f["samples/0"]["x"][:] - f["samples/0"]["x_hat"][:]
                             - f["samples/0"]["y"][:])) < 1e-3


def test_torch_loader_reads_scene(tmp_path):
    scatter = np.array([[0.05, 0.05, 0.05, 0.0, 0.0, 1.0]], dtype=float)
    pos = StaticLayout(2, 0.10).positions
    m = _model(scatter)
    x, xh, y = m.phase_history(pos, write_noise=False)
    out = tmp_path / "t.h5"
    write_scene_h5(out, x, xh, y,
                   np.full((8, 8, 8), 1.0, dtype=np.float32),
                   np.full((8, 8, 8), 1.0, dtype=np.float32), {"k": "v"})
    from datasets.torch_loader import MicroTomoDataset

    ds = MicroTomoDataset(str(out), normalize=False)
    assert len(ds) == 1
    item = ds[0]
    assert item["x"].shape == (2, len(FREQS), 4)
    assert item["eps"].shape == (1, 8, 8, 8)