#!/usr/bin/env python3
"""MicroTomo reflection-mode analytical forward model (sparsity study T3).

Surface-scatterer SFCW model for surface-contour imaging, 60 GHz band:
- scatterers sit on the phantom *surface* with outward normals (from the veggie
  generator); a scatterer is visible to Tx/Rx when its normal faces them
  (backside culling) and weighted by a specular-ish cos-factor of the
  bisector angle plus a 1/(d_tx*d_rx) spreading factor.
- two-way time-of-flight phases at each stepped frequency (57-64 GHz default);
- background x_hat = direct Tx->Rx coupling (+ nearest-wall single-bounce echo
  in the metal-wall stress mode); absorbing walls have no echo;
- complex gaussian noise at a configurable SNR (vs scattered RMS).

Outputs follow the torch_loader layout:
    /samples/NN/{x, x_hat, y} (F, T*T) complex64, x - x_hat == y
    /samples/NN/{geometry, eps} (H,W,D) float32, /samples/NN/meta json str

Usage:
    python sim/forward_surface.py --mode static --transceivers 4 \
        --chamber 10cm --seed 42 --out datasets/h5/demo_reflect.h5
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

C = 299_792_458.0
PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

F_MIN_GHZ = 57.0
F_MAX_GHZ = 64.0
DIRECT_AMP = 4.0          # amplitude of the Tx->Rx direct coupling
SELF_AMP = 1.5            # monostatic self-reflection (Tx == Rx)
WALL_AMP = 0.4            # per-bounce amplitude of the metal-wall echo
WALL_REFL = 0.9           # |reflection coefficient| at metal faces


@dataclass
class SurfaceScatterModel:
    chamber_m: float
    freqs_hz: np.ndarray
    scatter: np.ndarray                 # (P, 6) world points + unit normals
    gamma: np.ndarray | None = None     # (P,) contrast weights (default 0.9 water |PEC->1)
    wall_mode: str = "absorbing"        # "absorbing" | "metal"
    snr_db: float = 30.0
    amp_scale: float = 1.0
    seed: int = 0

    def __post_init__(self) -> None:
        if self.gamma is None:
            self.gamma = np.full(len(self.scatter), 0.9)
        self._rng = np.random.default_rng(self.seed)

    # ---- helpers -----------------------------------------------------------
    def _contrast(self, pec: bool) -> float:
        return 1.0 if pec else 0.9

    # ---- main entry --------------------------------------------------------
    def phase_history(self, positions: np.ndarray, write_noise: bool = True):
        """Return (x, x_hat, y) complex64 (F, T*T), ch = tx*T + rx."""
        T = len(positions)
        F = len(self.freqs_hz)
        pts = self.scatter[:, :3]
        norms = self.scatter[:, 3:]
        P = len(pts)

        fvec = self.freqs_hz
        x_hat = np.zeros((F, T * T), dtype=complex)
        y = np.zeros((F, T * T), dtype=complex)

        # background (per Tx->Rx pair, no scatterers)
        for i in range(T):
            d_tx = np.linalg.norm(pts - positions[i], axis=1)          # (P,)

            # ---- scenic loop over receivers: scattered field ----
            d_rx = np.linalg.norm(pts[:, None, :] - positions[None, :, :], axis=2)  # (P,T)
            tau = (d_tx[:, None] + d_rx) / C                                        # (P,T)
            spread = 1.0 / np.maximum(d_tx[:, None] * d_rx, 1e-12)
            # unit directions AWAY from each scatterer point toward Tx/Rx
            u_tx = (positions[i] - pts) / np.maximum(d_tx[:, None], 1e-12)          # (P,3)
            u_rx = (positions[None, :, :] - pts[:, None, :]) / \
                np.maximum(d_rx[:, :, None], 1e-12)                                 # (P,T,3)
            # illumination: the surface front (normal) must face the Tx
            illum = (norms * u_tx).sum(axis=1) > 0.0                                # (P,)
            # specular-ish return: align normal with the Tx->Rx bisector
            bis = u_tx[:, None, :] + u_rx                                           # (P,T,3)
            bis = bis / np.maximum(np.linalg.norm(bis, axis=2, keepdims=True), 1e-12)
            spec = (norms[:, None, :] * bis).sum(axis=2)                            # (P,T)
            spec = np.clip(spec, 0.0, 1.0)
            # diffuse floor 0.3 + specular 0.7 peak (rough produce surfaces)
            vis = np.where(illum[:, None], 0.3 + 0.7 * spec, 0.0)                  # (P,T)
            base = self.gamma[:, None] * spread * vis                                # (P,T)

            # direct coupling + wall echo background, one channel f vector later
            ch = i * T + np.arange(T)
            d_direct = np.linalg.norm(positions[i][None, :] - positions, axis=1)     # (T,)

            for jf in range(F):
                w = 2.0 * np.pi * fvec[jf]
                y[jf, ch] = (base * np.exp(-1j * w * tau)).sum(axis=0)
                # background
                coup = DIRECT_AMP * np.exp(-1j * w * d_direct / C)
                coup = np.where(np.arange(T) == i, SELF_AMP, coup)   # monostatic self echo
                if self.wall_mode == "metal":
                    coup = coup + self._wall_term(positions[i], positions, w)
                x_hat[jf, ch] = coup

        # normalize scattered field, add noise
        rms = float(np.sqrt(np.mean(np.abs(y) ** 2))) or 1.0
        y *= self.amp_scale / rms
        sigma = self.amp_scale * 10.0 ** (-self.snr_db / 20.0)
        if write_noise:
            noise = sigma / math.sqrt(2.0) * (
                self._rng.standard_normal((F, T * T)) + 1j * self._rng.standard_normal((F, T * T)))
            x = x_hat + y + noise
        else:
            x = x_hat + y
        return (x.astype(np.complex64), x_hat.astype(np.complex64), y.astype(np.complex64))

    # ---- metal-wall single-bounce echo via nearest-face image ---------------
    def _wall_term(self, tx: np.ndarray, rx_all: np.ndarray, w: float) -> np.ndarray:
        """Amplitude+phase of one bounce off the nearest chamber face (image Rx)."""
        L = self.chamber_m
        imgs = np.empty_like(rx_all)
        for j, rx in enumerate(rx_all):
            # candidate image across each of the 6 faces
            cand = []
            for d, face in ((0, 0.0), (0, L), (1, 0.0), (1, L), (2, 0.0), (2, L)):
                img = rx.copy()
                img[d] = 2.0 * face - rx[d]
                cand.append(img)
            cand = np.asarray(cand)
            d_img = np.linalg.norm(cand - tx, axis=1)
            best = int(np.argmin(d_img))
            imgs[j] = cand[best]
        d_img = np.linalg.norm(imgs - tx, axis=1)
        return WALL_AMP * WALL_REFL * np.exp(-1j * w * d_img / C) / np.maximum(d_img, 1e-12)


def write_scene_h5(
    path: Path,
    x: np.ndarray, x_hat: np.ndarray, y: np.ndarray,
    geometry: np.ndarray, eps: np.ndarray, meta: dict,
) -> None:
    """Write the torch_loader HDF5 layout."""
    import h5py

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as f:
        grp = f.create_group("samples/0")
        grp.create_dataset("x", data=x)
        grp.create_dataset("x_hat", data=x_hat)
        grp.create_dataset("y", data=y)
        grp.create_dataset("geometry", data=geometry)
        grp.create_dataset("eps", data=eps)
        grp.create_dataset("meta", data=json.dumps(meta))


def _load_scene(seed: int, chamber_m: float, n_shapes: int, pedestal: bool):
    from phantoms.veggie import generate_scene

    chamber_key = "10cm" if chamber_m == 0.10 else "50cm"
    scene = generate_scene(seed=seed, chamber=chamber_key, n_shapes=n_shapes,
                           pedestal=pedestal)
    rng = np.random.default_rng(seed + 100)
    scatter = scene.surface_points(1500, rng)
    # contract weights: water-like shapes get |Gamma| ~0.9, PEC -> 1.0
    gamma = np.asarray([1.0] * len(scatter), dtype=float)

    # per-scatterer contrast from the nearest shape (approx: uniform 0.9)
    meta_shape = []
    for s in scene.shapes:
        meta_shape.append({"kind": s.kind, "center": s.center.tolist(),
                           "eps": s.eps, "pec": s.pec})
    geometry, eps_vol = scene.voxelize(grid=48)
    return scene, scatter, gamma, geometry, eps_vol, meta_shape


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="MicroTomo reflection-mode forward model")
    p.add_argument("--mode", choices=["static", "scan"], default="static")
    p.add_argument("--transceivers", type=int, default=4)
    p.add_argument("--chamber", choices=["10cm", "50cm"], default="10cm")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n-shapes", type=int, default=1)
    p.add_argument("--wall", choices=["absorbing", "metal"], default="absorbing")
    p.add_argument("--n-stations", type=int, default=24)
    p.add_argument("--snr-db", type=float, default=30.0)
    p.add_argument("--n-freq", type=int, default=64)
    p.add_argument("--no-noise", action="store_true")
    p.add_argument("--out", type=str, default="datasets/h5/demo_reflect.h5")
    args = p.parse_args(argv)

    chamber_m = 0.10 if args.chamber == "10cm" else 0.50
    scene, scatter, gamma, geometry, eps_vol, shapes = _load_scene(
        args.seed, chamber_m, args.n_shapes, pedestal=False)

    from sim.transceivers import build_layout

    layout = build_layout(args.mode, args.transceivers, chamber_m,
                          n_stations=args.n_stations)
    positions = layout.positions
    freqs = np.linspace(F_MIN_GHZ, F_MAX_GHZ, args.n_freq) * 1e9

    model = SurfaceScatterModel(
        chamber_m=chamber_m, freqs_hz=freqs, scatter=scatter, gamma=gamma,
        wall_mode=args.wall, snr_db=args.snr_db, seed=args.seed)
    t0 = time.time()
    x, x_hat, y = model.phase_history(positions, write_noise=not args.no_noise)
    dt = time.time() - t0

    meta = {
        "version": "reflect-v1", "chamber_m": chamber_m, "mode": args.mode,
        "n_transceivers": args.transceivers, "n_stations": args.n_stations if args.mode == "scan" else 1,
        "forward": "surface-scatterer SFCW", "wall": args.wall,
        "snr_db": args.snr_db, "freqs_hz": freqs.tolist(), "seed": args.seed,
        "n_tot_positions": len(positions), "n_scatter": len(scatter),
        "shapes": shapes, "generated_by": "sim/forward_surface.py",
    }
    out = Path(args.out)
    write_scene_h5(out, x, x_hat, y, geometry, eps_vol, meta)
    err = float(np.max(np.abs(x - x_hat - y)))
    print(f"scene: mode={args.mode} n_trx={args.transceivers} "
          f"virtual positions={len(positions)} chamber={args.chamber} wall={args.wall}")
    print(f"  arrays x/x_hat/y: {x.shape} {x.dtype}; max|x-x_hat-y|={err:.2e}")
    print(f"  scatter={len(scatter)} pts, wall time {dt:.2f}s -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())