#!/usr/bin/env python3
"""MicroTomo GUI — launcher entry point.

PyQt6 + PyVista/pyvistaqt desktop window with a 3D viewport.
T8: scene view — renders a generated veggie scene (surface scatterers colored
by normal, chamber wireframe) plus optional transceiver ring and DAS recon
cloud.

Usage:
    ./launch_gui.sh                         # placeholder scene
    ./launch_gui.sh --scene 42 --chamber 10cm --transceivers 8
    ./launch_gui.sh --scene /tmp/das.npy --recon /tmp/das_recon.npy
    ./launch_gui.sh --smoke                 # offscreen render + exit (CI check)
"""
from __future__ import annotations

import argparse
import os
import sys
import tempfile
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="MicroTomo GUI")
    p.add_argument(
        "--scene",
        type=str,
        default=None,
        help="integer seed (generate veggie scene) or path to a .npy point cloud (6-col pts+normals)",
    )
    p.add_argument("--chamber", choices=["10cm", "50cm"], default="10cm")
    p.add_argument("--transceivers", type=int, default=None, help="draw N transceiver markers")
    p.add_argument("--recon", type=str, default=None, help="path to a DAS recon point cloud .npy")
    p.add_argument(
        "--smoke",
        action="store_true",
        help="offscreen render, print OK, exit 0 (no window)",
    )
    return p.parse_args(argv)


def _load_scene(scene_ref: str, chamber_m: float):
    """Return (scatter (N,6), chamber_m) from a seed or a .npy path."""
    if scene_ref.endswith((".npy", ".npz")):
        data = np.load(scene_ref)
        if isinstance(data, np.ndarray):
            arr = data
        else:
            arr = data[list(data.keys())[0]]
        arr = np.asarray(arr, dtype=float)
        if arr.shape[1] == 3:
            normals = np.zeros_like(arr)
            normals[:, 2] = 1.0
            arr = np.hstack([arr, normals])
        return arr, chamber_m, {"source": scene_ref}

    from phantoms.veggie import generate_scene

    seed = int(scene_ref)
    scene = generate_scene(seed=seed, chamber="10cm" if chamber_m == 0.10 else "50cm",
                           n_shapes=2, pedestal=True)
    rng = np.random.default_rng(seed)
    scatter = scene.surface_points(3000, rng)
    return scatter, chamber_m, {"seed": seed, "n_shapes": len(scene.shapes)}


def _draw_scene(
    plotter,
    scatter: np.ndarray,
    chamber_m: float,
    transceivers: np.ndarray | None = None,
    recon: np.ndarray | None = None,
    name: str = "scene",
) -> None:
    """Render a veggie scene into a PyVista-style plotter (shared by Qt + offscreen)."""
    import pyvista as pv

    half = chamber_m / 2.0
    box = pv.Box(bounds=(-half, half, -half, half, -half, half))
    plotter.add_mesh(box, color="white", style="wireframe", line_width=2.0, name="chamber")

    # surface scatterers, coloured by the outward normal's z-component
    pts = pv.PolyData(np.asarray(scatter[:, :3], dtype=float))
    normals = np.asarray(scatter[:, 3:], dtype=float)
    pts.point_data["normal_z"] = normals[:, 2]
    plotter.add_mesh(
        pts,
        scalars="normal_z",
        cmap="coolwarm",
        point_size=6.0,
        render_points_as_spheres=True,
        clim=(-1.0, 1.0),
        name=f"{name}_scatter",
    )

    if transceivers is not None and len(transceivers):
        plotter.add_mesh(
            pv.PolyData(np.asarray(transceivers, dtype=float)),
            color="red",
            point_size=14.0,
            render_points_as_spheres=True,
            name="transceivers",
        )
    if recon is not None and len(recon):
        plotter.add_mesh(
            pv.PolyData(np.asarray(recon, dtype=float)),
            color="lime",
            point_size=5.0,
            render_points_as_spheres=True,
            name="recon_cloud",
        )

    plotter.add_axes()
    plotter.view_isometric()


def _build_window(app, args: argparse.Namespace) -> "QMainWindow":
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
    header = QLabel("MicroTomo — simulation & sparsity study (v0.2)")
    layout.addWidget(header)

    plotter = QtInteractor(central)
    layout.addWidget(plotter.interactor, stretch=1)
    win.setCentralWidget(central)

    status = _paint_scene(plotter, args)
    win.statusBar().showMessage(status)
    return win


def _paint_scene(plotter, args: argparse.Namespace) -> str:
    """Draw whichever scene the CLI asked for; return a status message."""
    import numpy as np

    chamber_m = 0.10 if args.chamber == "10cm" else 0.50

    if args.scene:
        scatter, chamber_m, src = _load_scene(args.scene, chamber_m)
        txs = None
        if args.transceivers:
            from sim.transceivers import StaticLayout

            txs = StaticLayout(args.transceivers, chamber_m).positions - chamber_m / 2.0
        recon = None
        if args.recon:
            recon = np.asarray(np.load(args.recon), dtype=float) - chamber_m / 2.0
        _draw_scene(plotter, scatter, chamber_m, transceivers=txs, recon=recon,
                    name="scene")
        return (f"scene {src} — {len(scatter)} surface scatterers, "
                f"{args.transceivers or 0} transceivers, "
                f"{len(recon) if recon is not None else 0} recon pts")
    _draw_placeholder_scene(plotter)
    return "Placeholder scene — pass --scene <seed|.npy> to view a generated scene"


def _draw_placeholder_scene(plotter: "QtInteractor") -> None:
    """Placeholder: 10 cm PoC chamber wireframe + 8 transceiver markers."""
    import pyvista as pv

    box = pv.Box(bounds=(-0.05, 0.05, -0.05, 0.05, -0.05, 0.05))
    plotter.add_mesh(box, color="white", style="wireframe", line_width=2.0, name="chamber")

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
        status = _paint_scene(plotter, args)
        out = os.path.join(tempfile.gettempdir(), "microtomo_smoke.png")
        plotter.screenshot(out)
        print(f"GUI OK — smoke render saved to {out}")
        print(f"  {status}")
        return 0

    from PyQt6.QtWidgets import QApplication

    app = QApplication(sys.argv)
    win = _build_window(app, args)
    win.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())