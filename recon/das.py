#!/usr/bin/env python3
"""MicroTomo delay-and-sum (DAS) surface-contour reconstruction (T4).

Back-projects the multistatic SFCW phase history onto a chamber voxel grid:

    I(q) = | sum_{channels, freqs} y[f, ch] * exp(+j 2 pi f tau_ch(q)) |

with tau_ch(q) = two-way time of flight from Tx to voxel q to Rx. Peaks in
|I| mark surface scatterers; a threshold + top-M selection turns the image
into a surface contour point cloud (no ML).

Input: an HDF5 scene written by sim/forward_surface.py (`y`, `meta`).
Output: point cloud (N,3), volumetric image (G,G,G), threshold.

Usage:
    python recon/das.py --scene datasets/h5/demo_reflect.h5 --grid 32 \
        --top 500 --out /tmp/das_cloud.npy --image /tmp/das_image.npy
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

C = 299_792_458.0
PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def das_beamform(
    y: np.ndarray,
    positions: np.ndarray,
    freqs_hz: np.ndarray,
    chamber_m: float,
    grid: int = 32,
    threshold_quantile: float = 0.98,
    top_m: int | None = None,
    rng_seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Coherent back-projection -> surface point cloud, image, threshold.

    y: (F, T*T) complex — channel index = tx*T + rx.
    Returns (points (N,3) in meters, image (G,G,G), threshold).
    """
    T = len(positions)
    F = len(freqs_hz)
    if y.shape != (F, T * T):
        raise ValueError(f"y shape {y.shape} incompatible with T={T}, F={F}")

    axis = (np.arange(grid) + 0.5) * (chamber_m / grid)
    X, Yv, Z = np.meshgrid(axis, axis, axis, indexing="ij")
    q = np.stack([X.ravel(), Yv.ravel(), Z.ravel()], axis=1)          # (G^3, 3)

    image = np.zeros(grid**3, dtype=complex)
    w = 2.0 * np.pi * freqs_hz                                        # (F,)
    for i in range(T):
        d_tx = np.linalg.norm(q - positions[i], axis=1)               # (G^3,)
        for j in range(T):
            ch = i * T + j
            d_rx = np.linalg.norm(q - positions[j], axis=1)           # (G^3,)
            tau = (d_tx + d_rx) / C                                   # (G^3,)
            # S_ch(q) = sum_f y[f, ch] exp(+j w_f tau(q))
            image += np.exp(1j * (w[:, None] * tau[None, :])).T @ y[:, ch]

    image = image.reshape((grid, grid, grid))
    mag = np.abs(image)
    if float(np.max(mag)) < 1e-30:                      # empty scene -> no surface
        return np.empty((0, 3)), image, 0.0
    thr = float(np.quantile(mag, threshold_quantile))
    # Local maxima in the magnitude volume (surface peaks)
    # A voxel is a local maximum if it's >= all 26 neighbors AND above threshold
    pad = np.pad(mag, 1, mode="constant", constant_values=-1.0)
    # Build 26-neighborhood max for each voxel
    nb_max = np.full_like(mag, -1.0)
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            for dz in (-1, 0, 1):
                if dx == 0 and dy == 0 and dz == 0:
                    continue
                nb = pad[1+dx:1+dx+grid, 1+dy:1+dy+grid, 1+dz:1+dz+grid]
                np.maximum(nb_max, nb, out=nb_max)
    is_peak = (mag >= nb_max) & (mag >= thr)
    idx = np.flatnonzero(is_peak)
    if len(idx) == 0:
        # fallback: boundary of mask
        mask = mag >= thr
        pad_m = np.pad(mask, 1, mode="constant", constant_values=False)
        boundary = (
            (mask & ~pad_m[2:, 1:-1, 1:-1]) |
            (mask & ~pad_m[:-2, 1:-1, 1:-1]) |
            (mask & ~pad_m[1:-1, 2:, 1:-1]) |
            (mask & ~pad_m[1:-1, :-2, 1:-1]) |
            (mask & ~pad_m[1:-1, 1:-1, 2:]) |
            (mask & ~pad_m[1:-1, 1:-1, :-2])
        )
        idx = np.flatnonzero(boundary)
    if top_m is not None and len(idx) > top_m:
        order = np.argsort(mag.ravel()[idx])[::-1][:top_m]
        idx = idx[order]
    return q[idx], image, thr


def _load_scene(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, dict]:
    import json

    import h5py

    with h5py.File(path, "r") as f:
        grp = f["samples/0"]
        y = grp["y"][:]
        meta = json.loads(grp["meta"][()])
        geometry = grp["geometry"][:]
    from sim.transceivers import build_layout

    if "mode" not in meta:
        raise ValueError("scene meta has no layout info (not a forward_surface scene?)")
    layout = build_layout(meta["mode"], meta["n_transceivers"], meta["chamber_m"],
                          n_stations=meta.get("n_stations", 24))
    positions = layout.positions
    freqs = np.asarray(meta["freqs_hz"])
    return y, positions, freqs, meta["chamber_m"], meta


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="MicroTomo DAS surface reconstruction")
    p.add_argument("--scene", required=True, type=Path, help="HDF5 scene from forward_surface.py")
    p.add_argument("--grid", type=int, default=32)
    p.add_argument("--top", type=int, default=None, help="max points in the cloud")
    p.add_argument("--quantile", type=float, default=0.98)
    p.add_argument("--out", type=str, default="/tmp/das_cloud.npy")
    p.add_argument("--image", type=str, default=None, help="save volumetric image .npy")
    args = p.parse_args(argv)

    y, positions, freqs, chamber_m, meta = _load_scene(args.scene)
    pts, image, thr = das_beamform(y, positions, freqs, chamber_m,
                                   grid=args.grid, threshold_quantile=args.quantile,
                                   top_m=args.top)
    np.save(args.out, pts)
    print(f"channels={y.shape[1]} freqs={y.shape[0]} grid={args.grid}")
    print(f"surface points: {len(pts)}  threshold={thr:.4g}")
    print(f"cloud  -> {args.out}")
    if args.image:
        np.save(args.image, image)
        print(f"image  -> {args.image}")
    return 0


if __name__ == "__main__":
    sys.exit(main())