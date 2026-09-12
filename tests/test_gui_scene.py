#!/usr/bin/env python3
"""Tests for the GUI scene view (sparsity study T8)."""
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from gui.main import _draw_scene, _load_scene  # noqa: E402

CHAMBER_M = 0.10


def test_load_scene_from_seed():
    scatter, cm, src = _load_scene("7", CHAMBER_M)
    assert scatter.shape[1] == 6
    assert scatter[:, :3].min() >= 0.0 and scatter[:, :3].max() <= CHAMBER_M
    assert src["seed"] == 7 and len(scatter) > 0


def test_load_scene_from_npy(tmp_path):
    arr = np.random.default_rng(0).uniform(0.02, 0.08, size=(50, 3))
    p = tmp_path / "pts.npy"
    np.save(p, arr)
    scatter, cm, src = _load_scene(str(p), CHAMBER_M)
    assert scatter.shape == (50, 6)          # 3-col input gets +z normals
    assert np.allclose(scatter[:, :3], arr)
    assert src == {"source": str(p)}


def test_draw_scene_offscreen(tmp_path):
    """Scene renders headless: seeded scatter + transceivers + recon cloud."""
    import pyvista as pv

    scatter, cm, _ = _load_scene("11", CHAMBER_M)
    txs = np.array([[0.0, -0.04, 0.0], [0.04, 0.0, 0.0]])
    recon = np.array([[0.005, 0.01, 0.0], [-0.01, 0.0, 0.02]])

    plotter = pv.Plotter(off_screen=True, window_size=(640, 480))
    _draw_scene(plotter, scatter, cm, transceivers=txs, recon=recon)
    names = {a.name for a in plotter.renderer.actors.values() if hasattr(a, "name")}
    assert {"chamber", "scene_scatter", "transceivers", "recon_cloud"} <= names
    out = tmp_path / "scene.png"
    plotter.screenshot(out)
    plotter.close()
    assert out.exists() and out.stat().st_size > 1000