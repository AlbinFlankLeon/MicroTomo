#!/usr/bin/env python3
"""MicroTomo GUI — launcher entry point.

PyQt6 + PyVista/pyvistaqt desktop window with a 3D viewport.
T1: skeleton only — renders a placeholder chamber + transceiver markers;
scene loading lands in T8.

Usage:
    ./launch_gui.sh                 # open the window
    ./launch_gui.sh --smoke         # offscreen render + exit (CI check)
"""
from __future__ import annotations

import argparse
import os
import sys
import tempfile


def _parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="MicroTomo GUI")
    p.add_argument(
        "--smoke",
        action="store_true",
        help="offscreen render, print OK, exit 0 (no window)",
    )
    return p.parse_args(argv)


def _build_window(app) -> "QMainWindow":
    """Create the main window with the 3D viewport."""
    from PyQt6.QtWidgets import (
        QLabel,
        QMainWindow,
        QVBoxLayout,
        QWidget,
    )
    from pyvistaqt import QtInteractor

    win = QMainWindow()
    win.setWindowTitle("MicroTomo — 60 GHz transceiver sparsity study")

    central = QWidget()
    layout = QVBoxLayout(central)
    header = QLabel("MicroTomo — simulation & sparsity study (v0.1, T1 skeleton)")
    layout.addWidget(header)

    # 3D viewport (PyVista embedded in Qt)
    plotter = QtInteractor(central)
    layout.addWidget(plotter.interactor, stretch=1)
    win.setCentralWidget(central)

    _draw_placeholder_scene(plotter)
    win.statusBar().showMessage("Placeholder scene — no data loaded yet (T8 adds scene view)")
    return win


def _draw_placeholder_scene(plotter: "QtInteractor") -> None:
    """Placeholder: 10 cm PoC chamber wireframe + 8 transceiver markers."""
    import pyvista as pv

    # Chamber box (10 cm PoC cube), edges white wireframe
    box = pv.Box(bounds=(-0.05, 0.05, -0.05, 0.05, -0.05, 0.05))
    plotter.add_mesh(box, color="white", style="wireframe", line_width=2.0, name="chamber")

    # Transceiver markers: 8 static positions (cube corners, 0.08 m apart)
    txs = pv.PolyData(
        [
            [-0.04, -0.04, -0.04],
            [0.04, -0.04, -0.04],
            [-0.04, 0.04, -0.04],
            [0.04, 0.04, -0.04],
            [-0.04, -0.04, 0.04],
            [0.04, -0.04, 0.04],
            [-0.04, 0.04, 0.04],
            [0.04, 0.04, 0.04],
        ]
    )
    plotter.add_mesh(
        txs, color="red", point_size=12.0, render_points_as_spheres=True, name="transceivers"
    )

    plotter.add_axes()
    plotter.view_isometric()


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv if argv is not None else sys.argv[1:])
    if args.smoke:
        # Headless verify of the scene-rendering pipeline: pure PyVista
        # offscreen render (Qt's QOpenGLWidget cannot render without a
        # display/GL context, so the Qt window itself is only exercised
        # interactively via ./launch_gui.sh).
        import pyvista as pv

        plotter = pv.Plotter(off_screen=True, window_size=(1280, 800))
        _draw_placeholder_scene(plotter)
        out = os.path.join(tempfile.gettempdir(), "microtomo_smoke.png")
        plotter.screenshot(out)
        print(f"GUI OK — smoke render saved to {out}")
        return 0

    from PyQt6.QtWidgets import QApplication

    app = QApplication(sys.argv)
    win = _build_window(app)
    win.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())