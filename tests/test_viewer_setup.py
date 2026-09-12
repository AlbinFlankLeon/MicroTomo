#!/usr/bin/env python3
"""Tests for the viewer configuration/diagnostics helpers in gui.main_window.

These must run WITHOUT a GL context (dev machine is headless) — hence stub
plotter objects capturing calls.
"""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))


class StubInteractor:
    def __init__(self):
        self.calls = []

    def setFocus(self):
        self.calls.append("focus")


class StubPlotter:
    def __init__(self, has_renderer=True):
        self.calls = []
        self.interactor = StubInteractor()

        class RenWin:
            def GetOffScreenRendering(self):
                return 1

            def ReportCapabilities(self):
                return "openGL vendor string: StubCorp renderer: StubGL"

        class Ren:
            def __init__(self):
                self.actors = {}

        self.ren_win = RenWin()
        self.renderer = Ren() if has_renderer else object()

    def enable_trackball_style(self):
        self.calls.append("trackball")

    def set_background(self, *a):
        self.calls.append("background")


def test_configure_viewer_enables_interaction():
    from gui.main_window import configure_viewer

    stub = StubPlotter()
    configure_viewer(stub)
    assert "trackball" in stub.calls
    assert "focus" in stub.interactor.calls


def test_configure_viewer_sets_background():
    from gui.main_window import configure_viewer

    stub = StubPlotter()
    configure_viewer(stub)
    assert "background" in stub.calls


def test_diagnostics_reports_gl_and_actors():
    from gui.main_window import viewer_diagnostics

    stub = StubPlotter()
    out = viewer_diagnostics(stub)
    assert out["error"] is None
    assert out["gl"] and "StubGL" in out["gl"]
    assert "actors" in out and "off_screen" in out


def test_diagnostics_never_raises_on_degenerate_plotter():
    from gui.main_window import viewer_diagnostics

    stub = StubPlotter(has_renderer=False)
    out = viewer_diagnostics(stub)
    assert "error" in out              # returns a report instead of raising
    assert "actors" in out