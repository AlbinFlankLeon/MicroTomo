#!/usr/bin/env python3
"""Tests for the simulation viewer (desktop app + demo package)."""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from gui.main import View, _draw_scene, _load_scene, _resolve_demo  # noqa: E402

CHAMBER_M = 0.10


def test_load_scene_from_seed():
    scatter, src = _load_scene("7", CHAMBER_M)
    assert scatter.shape[1] == 6
    assert scatter[:, :3].min() >= 0.0 and scatter[:, :3].max() <= CHAMBER_M
    assert src["seed"] == 7 and len(scatter) > 0


def test_load_scene_from_npy(tmp_path):
    arr = np.random.default_rng(0).uniform(0.02, 0.08, size=(50, 3))
    p = tmp_path / "pts.npy"
    np.save(p, arr)
    scatter, src = _load_scene(str(p), CHAMBER_M)
    assert scatter.shape == (50, 6)          # 3-col input gets +z normals
    assert np.allclose(scatter[:, :3], arr)
    assert src == {"source": str(p)}


def test_draw_scene_offscreen(tmp_path):
    """Scene renders headless: scatter + transceivers + recon cloud, centered box."""
    import pyvista as pv

    scatter, _ = _load_scene("11", CHAMBER_M)
    txs = np.array([[0.0, 0.06, 0.0], [0.04, 0.0, 0.0]]) + CHAMBER_M / 2
    recon = np.array([[0.005, 0.01, 0.0], [-0.01, 0.0, 0.02]]) + CHAMBER_M / 2
    view = View(scatter=scatter, chamber_m=CHAMBER_M, recon=recon, transceivers=txs)

    plotter = pv.Plotter(off_screen=True, window_size=(640, 480))
    _draw_scene(plotter, view)
    names = {a.name for a in plotter.renderer.actors.values() if hasattr(a, "name")}
    assert {"chamber", "scene_scatter", "transceivers", "recon_cloud"} <= names
    out = tmp_path / "scene.png"
    plotter.screenshot(out)
    plotter.close()
    assert out.exists() and out.stat().st_size > 1000


def test_chamber_centering():
    """Scatter drawn at world [0,chamber] must land inside the centred box."""
    scatter = np.array([[0.0, 0.0, 0.0, 0, 0, 1.0], [CHAMBER_M, CHAMBER_M, CHAMBER_M, 0, 0, 1.0]])
    view = View(scatter=scatter, chamber_m=CHAMBER_M)
    import pyvista as pv

    plotter = pv.Plotter(off_screen=True, window_size=(320, 240))
    _draw_scene(plotter, view)
    actor = plotter.renderer.actors["scene_scatter"]
    bounds = actor.GetBounds()
    plotter.close()
    assert -0.05001 <= bounds[0] and bounds[1] <= 0.05001   # centred, not 0..0.10


def test_demo_package_load(monkeypatch):
    demo = ROOT / "datasets" / "demo"
    if not (demo / "manifest.json").exists():
        pytest.skip("demo package not built yet")
    monkeypatch.setattr("gui.main.DEMO_DIR", demo)
    v = _resolve_demo(None)
    assert v.info[0].startswith("Completed simulation")
    assert any("median chamfer" in line for line in v.info)
    assert v.recon is not None and len(v.recon) > 0