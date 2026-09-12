#!/usr/bin/env python3
"""Tests for transceiver fillers (corners / random) in gui.state."""
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from gui.state import corners_tx, random_tx  # noqa: E402


def test_corners_four_cardinal():
    c = corners_tx(4, chamber_m=0.10)
    assert c.shape == (4, 3)
    # cardinal points sit ON the ring: |dist from centre| ≈ 0.4*chamber
    r = 0.4 * 0.10
    dist = np.linalg.norm(c[:, :2] - 0.05, axis=1)
    assert np.allclose(dist, r)
    assert np.allclose(c[:, 2], 0.05)


def test_corners_n5_reaches_ring_edge():
    c = corners_tx(5, chamber_m=0.10)
    assert c.shape == (5, 3)
    assert np.any(c[:, 0] > 0.08)            # corner antennas reach outward
    # no duplicated positions
    assert len({tuple(np.round(p, 6)) for p in c}) == 5
    assert np.all((c >= 0) & (c <= 0.10))


def test_corners_deterministic():
    a = corners_tx(8, 0.10)
    b = corners_tx(8, 0.10)
    assert np.array_equal(a, b)


def test_random_disc_bounded_and_seeded():
    r = random_tx(12, chamber_m=0.10, seed=7)
    a = random_tx(12, chamber_m=0.10, seed=7)
    b = random_tx(12, chamber_m=0.10, seed=8)
    assert r.shape == (12, 3)
    assert np.array_equal(r, a)              # deterministic under seed
    assert not np.array_equal(r, b)          # different seed -> different
    # all within 90% of the ring radius (kept off the wall)
    max_r = 0.9 * (0.4 * 0.10)
    dist = np.linalg.norm(r[:, :2] - 0.05, axis=1)
    assert np.all(dist <= max_r + 1e-9)
    assert np.allclose(r[:, 2], 0.05)


def test_random_unseeded_still_valid():
    r = random_tx(6, chamber_m=0.50)
    assert r.shape == (6, 3)
    assert np.all((r >= 0) & (r <= 0.50))
    assert np.all(np.isfinite(r))