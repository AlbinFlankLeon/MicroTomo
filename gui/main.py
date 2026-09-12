#!/usr/bin/env python3
"""MicroTomo — Completed Simulation Viewer (desktop app).

PyQt6 + PyVista 3D viewport, side panel with the study metrics. On launch
(default) it loads the committed demo package from datasets/demo/ and shows
the true surface + the DAS reconstruction + the metric numbers. A .desktop
entry is installed by scripts/install_app.sh so it starts from the system app
menu (CachyOS/any XDG launcher).

Usage:
    ./launch_gui.sh                     # open viewer on the demo package
    ./launch_gui.sh --demo              # same
    ./launch_gui.sh --scene 42 --recon datasets/demo/recon.npy
    ./launch_gui.sh --smoke             # headless render check (also makes the icon)
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DEMO_DIR = PROJECT_ROOT / "datasets" / "demo"


@dataclass
class View:
    scatter: np.ndarray                # (N,6) world pts + normals, [0, chamber]
    chamber_m: float
    recon: np.ndarray | None = None
    transceivers: np.ndarray | None = None
    info: list[str] = field(default_factory=list)


def _parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="MicroTomo — simulation viewer")
    p.add_argument("--demo", action="store_true",
                   help="load the committed demo package (default if no --scene)")
    p.add_argument("--scene", type=str, default=None,
                   help="integer seed (generate veggie scene) or .npy point cloud")
    p.add_argument("--chamber", choices=["10cm", "50cm"], default="10cm")
    p.add_argument("--transceivers", type=int, default=None)
    p.add_argument("--recon", type=str, default=None, help="DAS recon cloud .npy")
    p.add_argument("--verdict", type=str, default=None,
                   help="path to reports/sparsity_verdict.json for the panel (default: repo one)")
    p.add_argument("--smoke", action="store_true",
                   help="offscreen render, print OK, exit 0 (no window)")
    return p.parse_args(argv)


# ---------------------------------------------------------------------------
# view resolution (pure; no Qt)
# ---------------------------------------------------------------------------
def _load_scene(scene_ref: str, chamber_m: float) -> tuple[np.ndarray, dict]:
    """Scatter (N,6) from a seed or a .npy path; info dict."""
    if scene_ref.endswith((".npy", ".npz")):
        data = np.load(scene_ref)
        arr = data if isinstance(data, np.ndarray) else data[list(data.keys())[0]]
        arr = np.asarray(arr, dtype=float)
        if arr.shape[1] == 3:
            normals = np.zeros_like(arr)
            normals[:, 2] = 1.0
            arr = np.hstack([arr, normals])
        return arr, {"source": str(scene_ref)}

    from phantoms.veggie import generate_scene

    seed = int(scene_ref)
    scene = generate_scene(seed=seed, chamber="10cm" if chamber_m == 0.10 else "50cm",
                           n_shapes=2, pedestal=True)
    rng = np.random.default_rng(seed)
    return scene.surface_points(3000, rng), {"seed": seed, "n_shapes": len(scene.shapes)}


def _resolve(args: argparse.Namespace) -> View:
    if args.demo or not args.scene:
        return _resolve_demo(args)
    return _resolve_custom(args)


def _resolve_demo(args: argparse.Namespace | None = None) -> View:
    from sim.transceivers import StaticLayout

    manifest_path = DEMO_DIR / "manifest.json"
    if not manifest_path.exists():
        raise SystemExit(
            f"demo package not found at {DEMO_DIR} — run: "
            f"python scripts/make_demo.py")
    manifest = json.loads(manifest_path.read_text())
    chamber_m = manifest["chamber_m"]
    scatter = np.load(DEMO_DIR / manifest["files"]["scene"])
    recon = np.load(DEMO_DIR / manifest["files"]["recon"])
    metrics = json.loads((DEMO_DIR / manifest["files"]["metrics"]).read_text())

    lines = [
        "Completed simulation (demo package)",
        f"scene seed {manifest['seed']} · {manifest['mode']} "
        f"{manifest['n_transceivers']}×{manifest['n_stations']} stations",
        f"chamber {chamber_m*1000:.0f} mm · true surface {len(scatter)} pts",
        f"reconstruction {len(recon)} pts (DAS grid-24, "
        f"threshold {metrics['das_threshold']:.3g})",
        "",
        "— metrics —",
        f"median chamfer   {metrics['chamfer_median_cm']:.2f} cm",
        f"mean chamfer     {metrics['chamfer_mean_cm']:.2f} cm",
        f"max chamfer      {metrics['chamfer_max_cm']:.2f} cm",
        f"% of surface ≤1cm {metrics['pct_within_1cm']:.1f}%",
        f"runtime          {metrics['wall_time_s']:.1f} s",
    ]
    lines += _verdict_summary(None if args is None else args.verdict)
    return View(scatter=scatter, chamber_m=chamber_m, recon=recon,
                transceivers=StaticLayout(2, chamber_m).positions, info=lines)


def _resolve_custom(args: argparse.Namespace) -> View:
    chamber_m = 0.10 if args.chamber == "10cm" else 0.50
    scatter, src = _load_scene(args.scene, chamber_m)
    txs = None
    if args.transceivers:
        from sim.transceivers import StaticLayout

        txs = StaticLayout(args.transceivers, chamber_m).positions
    recon = None
    if args.recon:
        recon = np.asarray(np.load(args.recon), dtype=float)
    return View(scatter=scatter, chamber_m=chamber_m, recon=recon,
                transceivers=txs, info=[f"custom source {src}"])


def _verdict_summary(path: str | None) -> list[str]:
    """Pull the 2-trx verdict rows + trust flag from the study report (best-effort)."""
    p = Path(path) if path else PROJECT_ROOT / "reports" / "sparsity_verdict.json"
    if not p.exists():
        return []
    try:
        data = json.loads(p.read_text())
    except Exception:
        return []
    out = ["", "— study verdict —"]
    for r in data["rows"][:2]:
        out.append(f"  {r['n_transceivers']}×{r['mode']}: "
                   f"{r['chamfer_median_cm']:.2f} cm → {r['verdict']}")
    flag = (PROJECT_ROOT / "reports" / "analytical_vs_gprmax.md")
    if flag.exists():
        out += ["", "⚠ trust flag: analytical numbers pending a 60 GHz",
                "  cross-validation (T7) — see reports/"]
    return out


# ---------------------------------------------------------------------------
# render (shared by Qt + offscreen)
# ---------------------------------------------------------------------------
def _draw_scene(
    plotter,
    view: View,
    name: str = "scene",
) -> None:
    """Render a View into a PyVista-style plotter, chamber-centred coords."""
    import pyvista as pv

    half = view.chamber_m / 2.0
    box = pv.Box(bounds=(-half, half, -half, half, -half, half))
    plotter.add_mesh(box, color="white", style="wireframe", line_width=2.0, name="chamber")

    pts = pv.PolyData(np.asarray(view.scatter[:, :3], float) - half)
    pts.point_data["normal_z"] = np.asarray(view.scatter[:, 3:], float)[:, 2]
    plotter.add_mesh(
        pts, scalars="normal_z", cmap="coolwarm", point_size=6.0,
        render_points_as_spheres=True, clim=(-1.0, 1.0), name=f"{name}_scatter")

    if view.transceivers is not None and len(view.transceivers):
        plotter.add_mesh(pv.PolyData(np.asarray(view.transceivers, float) - half),
                         color="red", point_size=14.0,
                         render_points_as_spheres=True, name="transceivers")
    if view.recon is not None and len(view.recon):
        plotter.add_mesh(pv.PolyData(np.asarray(view.recon, float) - half),
                         color="lime", point_size=5.0,
                         render_points_as_spheres=True, name="recon_cloud")

    plotter.add_axes()
    plotter.view_isometric()


# ---------------------------------------------------------------------------
# Qt window
# ---------------------------------------------------------------------------
def _build_window(app, args: argparse.Namespace):
    from PyQt6.QtWidgets import (
        QHBoxLayout,
        QLabel,
        QMainWindow,
        QPushButton,
        QTextEdit,
        QVBoxLayout,
        QWidget,
    )
    from pyvistaqt import QtInteractor

    view = _resolve(args)

    win = QMainWindow()
    win.setWindowTitle("MicroTomo — Completed Simulation Viewer")
    win.resize(1280, 800)

    central = QWidget()
    outer = QHBoxLayout(central)
    left = QVBoxLayout()
    header = QLabel(f"MicroTomo — {Path(view.info[0]).name if view.info else ''} "
                    "(simulation viewer)")
    left.addWidget(header)

    plotter = QtInteractor(central)
    left.addWidget(plotter.interactor, stretch=1)
    left.addWidget(QLabel("red: transceivers · lime: reconstruction · coloured: true surface"))

    right = QVBoxLayout()
    panel = QTextEdit()
    panel.setReadOnly(True)
    panel.setMinimumWidth(360)
    panel.setText("\n".join(view.info))
    btn_demo = QPushButton("Demo scene")
    btn_random = QPushButton("New random scene")
    right.addWidget(panel)
    right.addWidget(btn_demo)
    right.addWidget(btn_random)

    outer.addLayout(left, stretch=1)
    outer.addLayout(right)
    win.setCentralWidget(central)

    _draw_scene(plotter, view)
    win.statusBar().showMessage(view.info[0])

    def repaint(v: View) -> None:
        plotter.clear()
        _draw_scene(plotter, v)
        panel.setText("\n".join(v.info))
        win.statusBar().showMessage(v.info[0])

    btn_demo.clicked.connect(lambda: repaint(_resolve_demo(args)))
    btn_random.clicked.connect(lambda: repaint(_random_view()))
    win._repaint = repaint  # keep a reference so closures survive GC
    return win


def _random_view() -> View:
    import random

    seed = random.randint(0, 9999)
    args = argparse.Namespace(demo=False, scene=str(seed), chamber="10cm",
                              transceivers=None, recon=None)
    v = _resolve_custom(args)
    v.info = [f"random veggie scene (seed {seed})", "regenerate with: "
              f"./launch_gui.sh --scene {seed}"]
    return v


# ---------------------------------------------------------------------------
# entry
# ---------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv if argv is not None else sys.argv[1:])

    if args.smoke:
        import pyvista as pv

        view = _resolve(args)
        plotter = pv.Plotter(off_screen=True, window_size=(1280, 800))
        _draw_scene(plotter, view)
        out = os.path.join(tempfile.gettempdir(), "microtomo_smoke.png")
        plotter.screenshot(out)
        plotter.close()
        print(f"GUI OK — smoke render saved to {out}")
        print(f"  {view.info[:6]}")
        return 0

    from PyQt6.QtWidgets import QApplication

    app = QApplication(sys.argv)
    win = _build_window(app, args)
    win.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())