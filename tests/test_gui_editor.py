#!/usr/bin/env python3
"""Tests for the interactive editor (panel + main window), offscreen."""
import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("PYVISTA_OFF_SCREEN", "true")


@pytest.fixture(scope="module")
def qapp():
    from PyQt6.QtWidgets import QApplication

    app = QApplication.instance() or QApplication([])
    yield app


@pytest.fixture()
def panel(qapp):
    from gui.panel import SimPanel

    return SimPanel()


def test_panel_state_roundtrip(panel):
    from gui.state import default_state

    st = default_state()
    panel.set_state(st)
    got = panel.get_state()
    assert len(got.objects) == len(st.objects)
    assert got.objects[0].kind == st.objects[0].kind
    assert got.mode == st.mode and got.n_tx == st.n_tx
    assert got.grid == st.grid


def test_panel_edit_object_then_read(panel):
    got = panel.get_state()
    o = got.objects[0]
    o.center = [0.03, 0.03, 0.05]
    o.dim = [0.011, 0.012, 0.013]
    panel.set_state(got)
    # user-facing units are mm now: 33 mm == 0.033 m
    panel.cx.setValue(33.0)
    panel.d2.setValue(17.0)
    panel._emit_changed()
    read = panel.get_state().objects[0]
    assert read.center[0] == pytest.approx(0.033)
    assert read.dim[1] if len(read.dim) > 1 else read.dim[0] == pytest.approx(0.017)


def test_panel_add_remove_object(panel):
    n = panel.obj_list.count()
    panel._add_object()
    assert panel.obj_list.count() == n + 1
    panel._remove_object()
    assert panel.obj_list.count() == n


def test_panel_transceiver_presets(panel):
    import numpy as np

    panel.n_tx.setValue(5)
    panel._fill("ring")
    st = panel.get_state()
    assert st.n_tx == 5 and st.tx_positions.shape == (5, 3)
    panel._fill("corners")
    st = panel.get_state()
    assert st.tx_positions.shape == (5, 3)
    assert np.any(st.tx_positions[:, 0] > 0.08)


@pytest.fixture()
def editor(qapp):
    """A built MainWindow, or skip if no GL context is available."""
    from gui.main_window import MainWindow

    mw = MainWindow()
    win = mw.build()
    if not mw.plotter.renderer.actors:
        win.close()
        pytest.skip("QOpenGLWidget is not supported on this platform")
    return mw, win


def test_editor_build_and_run(editor):
    mw, _win = editor
    assert mw.win.windowTitle().startswith("MicroTomo — simulation editor (")
    # shrink the run to keep the test fast
    mw.panel.grid.setValue(8)
    mw.panel.n_freq.setValue(8)
    mw.panel.n_tx.setValue(2)
    mw.panel.set_results("")
    mw.run_simulation()
    text = mw.panel.results.text()
    assert "median" in text and "cm" in text
    # view toggle: true-only hides the recon actor
    mw.rb_recon.setChecked(True)
    mw.apply_view_mode()
    actors = {a.name: a for a in mw.plotter.renderer.actors.values() if hasattr(a, "name")}
    assert actors["scene_scatter"].visibility is False
    mw.rb_both.setChecked(True)
    mw.apply_view_mode()
    actors = {a.name: a for a in mw.plotter.renderer.actors.values() if hasattr(a, "name")}
    assert actors["scene_scatter"].visibility is True