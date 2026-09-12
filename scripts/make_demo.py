#!/usr/bin/env python3
"""MicroTomo demo artifacts for the desktop viewer app.

Generates one deterministic "completed simulation" package and writes it to
datasets/demo/ so the GUI app has something to visualise without re-running:
    manifest.json   paths + metrics + layout info
    scene.npy       (N,6) surface scatterers (pts + normals)  [= truth]
    recon.npy       (M,3) DAS reconstruction cloud (67-64 sweep qtl 0.96)
    metrics.json    chamfer / %within-1cm / detection
    icon.png        small offscreen render for the OS icon (optional)

Fully seeded & headless. Re-runnable (--force overwrites).

Usage:
    python scripts/make_demo.py [--seed 42] [--force]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from phantoms.veggie import generate_scene  # noqa: E402
from sim.forward_surface import SurfaceScatterModel  # noqa: E402
from sim.transceivers import ScanLayout  # noqa: E402
from recon.das import das_beamform  # noqa: E402
from metrics.contour_metrics import chamfer, frac_within_cm, detect_feature  # noqa: E402

CHAMBER_M = 0.10
F_MIN, F_MAX = 57.0, 64.0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="build the GUI demo package")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--force", action="store_true")
    p.add_argument("--out", type=Path, default=PROJECT_ROOT / "datasets" / "demo")
    args = p.parse_args(argv)

    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    manifest = {"version": "viewer-v1", "seed": args.seed, "chamber_m": CHAMBER_M,
                "mode": "scan", "n_transceivers": 2, "n_stations": 24}

    t0 = time.time()

    # ---- scene + true surface scatterers -----------------------------------
    scene = generate_scene(seed=args.seed, chamber="10cm", n_shapes=3, pedestal=True)
    rng = np.random.default_rng(args.seed)
    scatter = scene.surface_points(6000, rng)
    truth = scatter[:, :3]

    # ---- forward model (camera-ready scan config) ---------------------------
    layout = ScanLayout(2, CHAMBER_M, n_stations=24)
    positions = layout.positions
    freqs = np.linspace(F_MIN, F_MAX, 64) * 1e9
    model = SurfaceScatterModel(chamber_m=CHAMBER_M, freqs_hz=freqs, scatter=scatter,
                                gamma=None, wall_mode="absorbing", snr_db=30.0,
                                seed=args.seed)
    _, _, y = model.phase_history(positions, write_noise=True)

    # ---- DAS reconstruction --------------------------------------------------
    pts, image, thr = das_beamform(y, positions, freqs, CHAMBER_M,
                                   grid=24, threshold_quantile=0.96, top_m=1500)
    recon = pts[:, :3]

    # ---- metrics -------------------------------------------------------------
    ch = chamfer(recon, truth)
    coverage = frac_within_cm(recon, truth, threshold_cm=1.0)
    feat = detect_feature(recon, np.asarray(scene.shapes[0].center), 0.025)
    metrics = {
        "n_scatter": int(len(scatter)),
        "n_recon": int(len(recon)),
        "chamfer_mean_cm": round(ch["mean"] * 100, 3),
        "chamfer_median_cm": round(ch["median"] * 100, 3),
        "chamfer_max_cm": round(ch["max"] * 100, 3),
        "pct_within_1cm": round(coverage * 100, 2),
        "detect_feature_0": feat,
        "das_threshold": float(thr),
        "wall_time_s": round(time.time() - t0, 1),
    }

    np.save(out / "scene.npy", scatter.astype(np.float32))
    np.save(out / "recon.npy", recon.astype(np.float32))
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    manifest["files"] = {
        "scene": "scene.npy", "recon": "recon.npy", "metrics": "metrics.json",
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))

    # ---- optional OS icon render (offscreen pyvista) -------------------------
    try:
        _render_icon(out, scatter, recon)
        manifest["files"]["icon"] = "icon.png"
        (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    except Exception as e:  # icon is best-effort
        print(f"  (icon render skipped: {e})")

    print(f"demo -> {out}")
    print(f"  scatter={len(scatter)} recon={len(recon)} "
          f"chamfer_median={metrics['chamfer_median_cm']} cm "
          f"pct<=1cm={metrics['pct_within_1cm']}%  wall={metrics['wall_time_s']}s")
    return 0


def _render_icon(out: Path, scatter: np.ndarray, recon: np.ndarray) -> None:
    import pyvista as pv

    from gui.main import View, _draw_scene

    view = View(scatter=scatter[:, :6], chamber_m=CHAMBER_M, recon=recon,
                transceivers=None, info=[])
    plotter = pv.Plotter(off_screen=True, window_size=(384, 288), shape=(1, 1))
    plotter.set_background("black")
    _draw_scene(plotter, view)
    plotter.screenshot(out / "icon.png")
    plotter.close()


if __name__ == "__main__":
    sys.exit(main())