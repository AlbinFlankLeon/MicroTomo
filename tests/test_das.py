#!/usr/bin/env python3
"""Tests for DAS surface reconstruction (sparsity study T4)."""
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from recon.das import das_beamform  # noqa: E402
from sim.forward_surface import SurfaceScatterModel  # noqa: E402
from sim.transceivers import StaticLayout  # noqa: E402

FREQS = np.linspace(57e9, 64e9, 48)


def _forward(scatter, positions, seed=0):
    m = SurfaceScatterModel(chamber_m=0.10, freqs_hz=FREQS, scatter=scatter,
                            gamma=None, wall_mode="absorbing", snr_db=35.0, seed=seed)
    x, xh, y = m.phase_history(positions, write_noise=False)
    return y


def test_single_scatterer_localization():
    """DAS peak lands near a known single scatterer (ring-plane case)."""
    truth = np.array([0.062, 0.052, 0.050])
    scatter = np.array([[*truth, 1.0, 0.0, 0.0]])
    pos = StaticLayout(4, 0.10).positions
    y = _forward(scatter, pos)
    pts, image, thr = das_beamform(y, pos, FREQS, chamber_m=0.10, grid=20,
                                   threshold_quantile=0.95, top_m=50)
    assert len(pts) > 0
    peak = _peak_voxel(image, pts, 0.10, 20)
    dist = np.linalg.norm(peak - truth)
    assert dist < 2.5e-2, f"peak {peak} vs truth {truth} dist={dist*100:.1f}cm"


def _peak_voxel(image, pts, chamber_m, grid):
    axis = (np.arange(grid) + 0.5) * (chamber_m / grid)
    idx = int(np.argmax(np.abs(image)))
    i, j, k = np.unravel_index(idx, image.shape)
    return np.array([axis[i], axis[j], axis[k]])


def test_peak_matches_max_energy_voxel():
    """The returned cloud contains the maximum-energy voxel."""
    truth = np.array([0.055, 0.05, 0.05])
    scatter = np.array([[*truth, 0.0, 1.0, 0.0]])
    pos = StaticLayout(4, 0.10).positions
    y = _forward(scatter, pos)
    pts, image, _ = das_beamform(y, pos, FREQS, chamber_m=0.10, grid=20,
                                 threshold_quantile=0.9, top_m=1000)
    peak = _peak_voxel(image, pts, 0.10, 20)
    dist = min(np.linalg.norm(peak - p) for p in pts)
    assert dist == pytest.approx(0.0, abs=1e-12)   # peak voxel is in the cloud


def test_empty_chamber_returns_empty():
    y = np.zeros((len(FREQS), 16), dtype=complex)
    pos = StaticLayout(4, 0.10).positions
    pts, image, thr = das_beamform(y, pos, FREQS, chamber_m=0.10, grid=12)
    assert pts.shape == (0, 3)
    assert thr == 0.0


def test_deterministic_given_same_data():
    truth = np.array([0.055, 0.05, 0.05])
    scatter = np.array([[*truth, 0.0, 1.0, 0.0]])
    pos = StaticLayout(4, 0.10).positions
    y = _forward(scatter, pos)
    a, _, _ = das_beamform(y, pos, FREQS, chamber_m=0.10, grid=12, threshold_quantile=0.9)
    b, _, _ = das_beamform(y, pos, FREQS, chamber_m=0.10, grid=12, threshold_quantile=0.9)
    assert np.array_equal(a, b)


def test_scan_layout_reconstructs_off_plane_point():
    """Scan layout (turntable) localizes a point OUTSIDE the ring plane."""
    truth = np.array([0.060, 0.055, 0.058])
    scatter = np.array([[*truth, 1.0, 0.0, 0.0]])
    from sim.transceivers import ScanLayout
    pos = ScanLayout(2, 0.10, n_stations=16).positions
    y = _forward(scatter, pos)
    pts, image, _ = das_beamform(y, pos, FREQS, chamber_m=0.10, grid=18,
                                 threshold_quantile=0.96, top_m=200)
    peak = _peak_voxel(image, pts, 0.10, 18)
    dist = np.linalg.norm(peak - truth)
    assert dist < 2.0e-2, f"scan peak {peak} vs truth {truth} dist={dist*100:.1f}cm"