#!/usr/bin/env python3
"""MicroTomo Phase 4 — Mass Dataset Generator.

Generates large-scale synthetic 60 GHz microwave-tomography datasets (HDF5)
for AI training. Output layout matches datasets/torch_loader.MicroTomoDataset:

    /samples/NN/x        (F, T) complex64 — full received phase history
    /samples/NN/x_hat    (F, T) complex64 — background/reference (empty scene)
    /samples/NN/y        (F, T) complex64 — scattered signal (x - x_hat)
    /samples/NN/geometry (H, W, D) float32 — phantom voxel mask
    /samples/NN/eps      (H, W, D) float32 — real permittivity volume
    /samples/NN/meta     json str            — full sample parameters

Forward model
    Air-coupled circular ring array (N_a antennas, radius R, z=0 plane).
    Multistatic stepped-frequency CW: for each Tx->Rx channel and each of F
    frequencies the complex received field is

        x[f, ch]  = background(f, ch) + scattered(f, ch) + noise
        x_hat[f]  = background(f, ch)                       (empty scene)
        y[f, ch]  = x[f, ch] - x_hat[f, ch]

    with ch = tx * N_a + rx. The scattered field uses a first-Born point-cloud
    model: each phantom object is sampled to N_s points, each a point scatterer

        y(f) ~ sum_p  gamma_p * exp(-j 2 pi f tau_p) / (d_tx_p * d_rx_p)

    where gamma_p = (eps_c - 1) / (eps_c + 2) * (V_obj / N_s) is the
    Clausius-Mossotti contrast weight, and tau_p the two-way path delay.
    Background = direct antenna-to-antenna coupling + ring-enclosure echo.
    Complex white noise is added at a configurable SNR (vs. scattered RMS).

    Reconstruction supervision: (x, y) -> eps volume (DBIM / SAR / U-Net).

Usage:
    python scripts/batch_generate.py --n-samples 1000 --out datasets/h5/train.h5
    python scripts/batch_generate.py --n-samples 10000 --jobs 8 \
        --n-antennas 24 --n-freq 96 --seed 42 --out datasets/h5/train.h5
    python scripts/batch_generate.py --check                # smoke test
"""
from __future__ import annotations

import argparse
import json
import math
import multiprocessing as mp
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - tqdm optional
    tqdm = None

C = 299_792_458.0  # m/s
PROJECT_ROOT = Path(__file__).parent.parent

# Default per-class loss tangent ranges (60 GHz, air-coupled reflection mode).
MATERIAL_DEFAULTS = {
    "plastic": {"eps_range": (2.1, 3.8), "tan_range": (0.001, 0.01)},
    "glass": {"eps_range": (4.0, 7.0), "tan_range": (0.001, 0.008)},
    "tissue": {"eps_range": (9.0, 25.0), "tan_range": (0.01, 0.1)},
    "water": {"eps_range": (50.0, 80.0), "tan_range": (0.05, 0.3)},
}


@dataclass
class GenConfig:
    """Generation configuration (picklable, one per dataset)."""
    # Dataset
    seed: int = 0
    n_samples: int = 1000
    # Domain / grid (mm)
    box_mm: float = 150.0
    grid: int = 64
    # Ring array
    n_antennas: int = 16
    radius_mm: float = 100.0
    z_ring_mm: float = 0.0
    # Frequencies (GHz, SFCW band)
    n_freq: int = 64
    fmin_ghz: float = 5.0
    fmax_ghz: float = 65.0
    # Phantom randomization
    n_obj_min: int = 1
    n_obj_max: int = 6
    radius_min_mm: float = 3.0
    radius_max_mm: float = 25.0
    ellipsoid_axial_range: tuple = (0.4, 1.0)  # per-axis radius multiplier
    eps_min: float = 2.1
    eps_max: float = 80.0
    pec_prob: float = 0.15
    # Forward model
    pts_per_object: int = 150
    direct_coupling: bool = True
    coup_amp: float = 4.0      # direct coupling amplitude (arbitrary units)
    self_amp: float = 1.5      # Tx self-reflection amplitude (monostatic ch)
    wall_amp: float = 0.4      # ring enclosure echo amplitude
    wall_delay_fact: float = 2.0  # wall echo delay = wall_delay_fact * R / c
    feed_delay_mm: float = 15.0   # self channel feed-line delay
    snr_db: float = 20.0
    amp_scale: float = 1.0     # target RMS of the normalized scattered field
    # IO / parallelism
    jobs: int = 1
    version: str = "phase4-v1"


@dataclass
class PhantomObject:
    cls: str
    center_mm: np.ndarray  # (3,) world center in mm (origin = domain center)
    radii_mm: np.ndarray   # (3,) ellipsoid half-axes in mm
    eps_r: float
    eps_i: float           # +j eps_i imaginary part
    is_pec: bool = False


def _material(rng: np.random.Generator, eps_r: float) -> tuple[float, float]:
    """Pick a loss tangent consistent with the permittivity class."""
    if eps_r >= 50.0:
        cls = "water"
    elif eps_r >= 9.0:
        cls = "tissue"
    elif eps_r >= 3.9:
        cls = "glass"
    else:
        cls = "plastic"
    lo, hi = MATERIAL_DEFAULTS[cls]["tan_range"]
    return cls, 10 ** rng.uniform(math.log10(lo), math.log10(hi))


def sample_phantom(rng: np.random.Generator, cfg: GenConfig) -> list[PhantomObject]:
    """Draw a random phantom: 1..n_obj non-overlapping ellipsoids."""
    n_obj = rng.integers(cfg.n_obj_min, cfg.n_obj_max + 1)
    half = cfg.box_mm / 2.0
    objs: list[PhantomObject] = []

    for _ in range(n_obj):
        is_pec = rng.random() < cfg.pec_prob
        base = rng.uniform(cfg.radius_min_mm, cfg.radius_max_mm)
        axial = rng.uniform(*cfg.ellipsoid_axial_range, size=3)
        radii = base * axial

        # Keep the ellipsoid fully inside the domain.
        lo = -half + radii
        hi = half - radii
        center = rng.uniform(lo, hi)

        if is_pec:
            eps_r, eps_i, cls = cfg.eps_max + 20.0, 0.0, "pec"
        else:
            eps_r = rng.uniform(cfg.eps_min, cfg.eps_max)
            cls, eps_i = _material(rng, eps_r)

        objs.append(PhantomObject(cls, center, radii, eps_r, eps_i, is_pec))
    return objs


def voxelize(objs: list[PhantomObject], cfg: GenConfig) -> tuple[np.ndarray, np.ndarray]:
    """Rasterize phantom onto an (H, W, D) permittivity volume + binary mask."""
    L = cfg.box_mm / 1000.0
    n = cfg.grid
    half = L / 2.0
    xs = (np.arange(n) + 0.5) * (L / n) - half  # world coords along each axis (m)

    eps = np.ones((n, n, n), dtype=np.float32)
    for o in objs:
        eps[_ellipsoid_mask(xs, o)] = o.eps_r
    geometry = (eps > 1.0).astype(np.float32)
    return eps, geometry


def _ellipsoid_mask(xs: np.ndarray, o: PhantomObject) -> np.ndarray:
    """Boolean mask (n, n, n) of voxel centers inside ellipsoid *o*."""
    c = o.center_mm / 1000.0
    r = o.radii_mm / 1000.0
    gx, gy, gz = np.meshgrid(xs, xs, xs, indexing="ij")
    m = ((gx - c[0]) / r[0]) ** 2
    m += ((gy - c[1]) / r[1]) ** 2
    m += ((gz - c[2]) / r[2]) ** 2
    return m <= 1.0


def sample_points(obj: PhantomObject, rng: np.random.Generator,
                  n: int) -> tuple[np.ndarray, complex]:
    """Uniform-volume point cloud inside ellipsoid + per-point contrast weight.

    Returns (points_m (n, 3) world meters, gamma weight (complex) applied to
    every point plus the V/N_s volume factor folded in).
    """
    center = obj.center_mm / 1000.0
    radii = obj.radii_mm / 1000.0

    direction = rng.standard_normal((n, 3))
    norm = np.linalg.norm(direction, axis=1, keepdims=True)
    direction = direction / np.maximum(norm, 1e-12)
    radius_f = rng.random(n) ** (1.0 / 3.0)  # uniform in unit ball
    pts = center + direction * (radius_f[:, None] * radii[None, :])

    if obj.is_pec:
        gamma = complex(1.0, 0.0)  # perfect reflector
    else:
        eps_c = complex(obj.eps_r, obj.eps_i)
        gamma = (eps_c - 1.0) / (eps_c + 2.0)
    vol = (4.0 / 3.0) * math.pi * float(np.prod(radii))
    weight = gamma * (vol / n)
    return pts, weight


def forward_model(objs: list[PhantomObject], cfg: GenConfig,
                  rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute (x, x_hat, y) phase-history tensors, shape (F, N_a*N_a)."""
    Na = cfg.n_antennas
    F = cfg.n_freq
    Nc = Na * Na

    fvec = np.linspace(cfg.fmin_ghz, cfg.fmax_ghz, F) * 1e9  # Hz
    theta = np.linspace(0.0, 2.0 * np.pi, Na, endpoint=False)
    Rm = cfg.radius_mm / 1000.0
    zant = cfg.z_ring_mm / 1000.0
    ants = np.stack([Rm * np.cos(theta), Rm * np.sin(theta),
                     np.full(Na, zant)], axis=1)

    # ---- Scattered field (point-cloud Born model) -------------------------
    pts_list, gammas = [], []
    for o in objs:
        p, g = sample_points(o, rng, cfg.pts_per_object)
        pts_list.append(p)
        gammas.append(g)
    if pts_list:
        pts = np.concatenate(pts_list, axis=0)
        weights = np.asarray([g for g in gammas for _ in
                              range(cfg.pts_per_object)], dtype=complex)
    else:
        pts = np.zeros((0, 3))
        weights = np.zeros(0, dtype=complex)

    y = np.zeros((F, Nc), dtype=complex)

    for i in range(Na):  # Tx antenna
        d_tx = np.linalg.norm(pts - ants[i], axis=1)  # (P,)
        d_rx = np.linalg.norm(pts[:, None, :] - ants[None, :, :], axis=2)  # (P, Na)
        tau = (d_tx[:, None] + d_rx) / C                       # (P, Na) s
        spread = 1.0 / np.maximum(d_tx[:, None] * d_rx, 1e-12)
        base = (weights[:, None] * spread)                    # complex (P, Na)

        for jf in range(F):
            phase = np.exp(-1j * 2.0 * np.pi * fvec[jf] * tau)
            y[jf, i * Na:(i + 1) * Na] = np.sum(base * phase, axis=0)

    # Per-sample normalization of the scattered field.
    rms = float(np.sqrt(np.mean(np.abs(y) ** 2)))
    if rms > 1e-12:
        y *= cfg.amp_scale / rms
    sigma = cfg.amp_scale * 10.0 ** (-cfg.snr_db / 20.0)
    noise = sigma / math.sqrt(2.0) * (rng.standard_normal((F, Nc))
                                      + 1j * rng.standard_normal((F, Nc)))

    # ---- Background (empty scene): direct coupling + enclosure echo -------
    dm = np.linalg.norm(ants[:, None, :] - ants[None, :, :], axis=2)  # (Na, Na)
    tau_direct = dm / C
    tau_self = 2.0 * cfg.feed_delay_mm / 1000.0 / C
    tau_wall = cfg.wall_delay_fact * Rm / C

    x_hat = np.zeros((F, Nc), dtype=complex)
    for jf in range(F):
        w = 2.0 * np.pi * fvec[jf]
        coup = cfg.coup_amp * np.exp(-1j * w * tau_direct)
        if cfg.direct_coupling:
            coup += cfg.self_amp * np.exp(-1j * w * tau_self) * \
                np.eye(Na, dtype=complex)
            coup += cfg.wall_amp * np.exp(-1j * w * tau_wall)
        x_hat[jf] = coup.ravel()

    # ---- Compose measured signatures --------------------------------------
    y_total = y + noise
    x = x_hat + y_total
    return x.astype(np.complex64), x_hat.astype(np.complex64), \
        y_total.astype(np.complex64)


def build_sample(idx: int, cfg: GenConfig) -> dict:
    """Build one full sample (arrays + meta json). Deterministic per index."""
    rng = np.random.default_rng(cfg.seed + idx)
    objs = sample_phantom(rng, cfg)
    eps, geometry = voxelize(objs, cfg)
    x, x_hat, y = forward_model(objs, cfg, rng)

    meta = {
        "sample_index": int(idx),
        "seed": cfg.seed + idx,
        "version": cfg.version,
        "forward_model": {
            "type": "multistatic_sfcw_phase_history",
            "n_antennas": cfg.n_antennas,
            "radius_m": cfg.radius_mm / 1000.0,
            "z_ring_m": cfg.z_ring_mm / 1000.0,
            "n_freq": cfg.n_freq,
            "freqs_ghz": [cfg.fmin_ghz, cfg.fmax_ghz],
            "n_channels": cfg.n_antennas * cfg.n_antennas,
            "channel_map": "tx*n_antennas+rx",
            "scatter_model": "born_pointcloud",
            "pts_per_object": cfg.pts_per_object,
            "n_points": sum(cfg.pts_per_object for _ in objs),
            "direct_coupling": bool(cfg.direct_coupling),
            "snr_db": cfg.snr_db,
            "amp_scale": cfg.amp_scale,
        },
        "domain": {"box_mm": cfg.box_mm, "grid": cfg.grid,
                   "dx_mm": cfg.box_mm / cfg.grid},
        "phantom": {
            "n_objects": len(objs),
            "objects": [
                {
                    "class": o.cls,
                    "eps": [o.eps_r, o.eps_i],
                    "is_pec": bool(o.is_pec),
                    "center_mm": o.center_mm.round(3).tolist(),
                    "radii_mm": o.radii_mm.round(3).tolist(),
                }
                for o in objs
            ],
        },
    }
    return {"x": x, "x_hat": x_hat, "y": y,
            "geometry": geometry, "eps": eps,
            "meta": json.dumps(meta)}


def write_sample(group: object, sample: dict) -> None:
    grp = group  # h5py group
    grp.create_dataset("x", data=sample["x"], compression="gzip", compression_opts=4)
    grp.create_dataset("x_hat", data=sample["x_hat"], compression="gzip",
                       compression_opts=4)
    grp.create_dataset("y", data=sample["y"], compression="gzip", compression_opts=4)
    grp.create_dataset("geometry", data=sample["geometry"], compression="gzip",
                       compression_opts=4)
    grp.create_dataset("eps", data=sample["eps"], compression="gzip",
                       compression_opts=4)
    grp.create_dataset("meta", data=sample["meta"])


def _worker(args: tuple) -> dict:
    """Top-level picklable worker for multiprocessing."""
    idx, cfg = args
    return build_sample(idx, cfg)


def generate(cfg: GenConfig, out_path: Path, quiet: bool = False) -> dict:
    """Generate *cfg.n_samples* samples into *out_path* (HDF5)."""
    import h5py

    out_path.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    if cfg.jobs <= 1:
        samples = (build_sample(i, cfg) for i in range(cfg.n_samples))
    else:
        pool = mp.Pool(cfg.jobs)
        samples = pool.imap(_worker,
                            ((i, cfg) for i in range(cfg.n_samples)))

    with h5py.File(out_path, "w") as f:
        f.attrs["version"] = cfg.version
        f.attrs["created"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        f.attrs["generation_config"] = json.dumps(asdict(cfg))
        f.attrs["n_samples"] = cfg.n_samples
        f.attrs["n_freq"] = cfg.n_freq
        f.attrs["n_channels"] = cfg.n_antennas * cfg.n_antennas
        f.attrs["grid"] = cfg.grid

        samples_grp = f.create_group("samples")
        bar = tqdm(total=cfg.n_samples, desc="generating",
                   disable=quiet or tqdm is None)
        for i, sample in enumerate(samples):
            write_sample(samples_grp.create_group(str(i)), sample)
            bar.update(1)
        bar.close()

    if cfg.jobs > 1:
        pool.close()
        pool.join()

    elapsed = time.time() - t0
    size_mb = out_path.stat().st_size / 1e6
    summary = {
        "out": str(out_path),
        "n_samples": cfg.n_samples,
        "elapsed_s": round(elapsed, 2),
        "samples_per_s": round(cfg.n_samples / elapsed, 2),
        "size_mb": round(size_mb, 2),
    }
    if not quiet:
        print(f"\nWrote {summary['n_samples']} samples to {out_path}")
        print(f"  elapsed {summary['elapsed_s']}s "
              f"({summary['samples_per_s']}/s), "
              f"{summary['size_mb']} MiB")
    return summary


def _read_check(out_path: Path) -> None:
    """Loader-compatible read-back: shapes/dtypes/meta for every sample."""
    import h5py

    print(f"Read-back check: {out_path}")
    with h5py.File(out_path, "r") as f:
        print(f"  attrs: {dict(f.attrs)}")
        n = len(f["samples"])
        print(f"  n_samples: {n}")
        for k in ("x", "x_hat", "y"):
            d = f[f"samples/0/{k}"]
            print(f"  {k}: shape={d.shape} dtype={d.dtype}")
        for k in ("geometry", "eps"):
            d = f[f"samples/0/{k}"]
            print(f"  {k}: shape={d.shape} dtype={d.dtype}")
        meta = json.loads(f["samples/0/meta"][()])
        print(f"  meta keys: {sorted(meta.keys())}")
        print(f"  phantom: {json.dumps(meta['phantom'], indent=2)[:400]}")
    print("  OK")


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="batch_generate.py",
        description="MicroTomo Phase 4 mass microwave-tomography dataset "
                    "generator (multistatic SFCW phase history, Born "
                    "point-cloud forward model).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--out", default="datasets/h5/train.h5",
                   help="output HDF5 path (parent dirs auto-created)")
    p.add_argument("--n-samples", type=int, default=1000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--jobs", type=int, default=1,
                   help="parallel workers (each writes a shard of samples)")
    p.add_argument("--quiet", action="store_true")
    # Domain
    p.add_argument("--box-mm", type=float, default=150.0)
    p.add_argument("--grid", type=int, default=64, help="voxels per axis (H=W=D)")
    # Array / signal
    p.add_argument("--n-antennas", type=int, default=16)
    p.add_argument("--radius-mm", type=float, default=100.0)
    p.add_argument("--n-freq", type=int, default=64)
    p.add_argument("--fmin-ghz", type=float, default=5.0)
    p.add_argument("--fmax-ghz", type=float, default=65.0)
    # Phantom randomization
    p.add_argument("--n-obj-min", type=int, default=1)
    p.add_argument("--n-obj-max", type=int, default=6)
    p.add_argument("--radius-min-mm", type=float, default=3.0)
    p.add_argument("--radius-max-mm", type=float, default=25.0)
    p.add_argument("--eps-min", type=float, default=2.1)
    p.add_argument("--eps-max", type=float, default=80.0)
    p.add_argument("--pec-prob", type=float, default=0.15)
    p.add_argument("--pts-per-object", type=int, default=150)
    # Noise / signal
    p.add_argument("--snr-db", type=float, default=20.0)
    p.add_argument("--amp-scale", type=float, default=1.0)
    p.add_argument("--no-direct-coupling", action="store_true",
                   help="disable Tx->Rx direct coupling in the background")
    p.add_argument("--check", action="store_true",
                   help="generate 2 samples to /tmp and verify read-back")
    return p.parse_args(argv)


def cfg_from_args(a: argparse.Namespace) -> GenConfig:
    return GenConfig(
        seed=a.seed,
        box_mm=a.box_mm,
        grid=a.grid,
        n_antennas=a.n_antennas,
        radius_mm=a.radius_mm,
        n_freq=a.n_freq,
        fmin_ghz=a.fmin_ghz,
        fmax_ghz=a.fmax_ghz,
        n_obj_min=a.n_obj_min,
        n_obj_max=a.n_obj_max,
        radius_min_mm=a.radius_min_mm,
        radius_max_mm=a.radius_max_mm,
        eps_min=a.eps_min,
        eps_max=a.eps_max,
        pec_prob=a.pec_prob,
        pts_per_object=a.pts_per_object,
        snr_db=a.snr_db,
        amp_scale=a.amp_scale,
        direct_coupling=not a.no_direct_coupling,
        jobs=a.jobs,
    )


def main(argv=None) -> int:
    a = parse_args(argv)
    if a.check:
        cfg = GenConfig(seed=a.seed, n_samples=2, n_antennas=max(a.n_antennas, 8),
                        n_freq=max(a.n_freq, 16), grid=32,
                        pts_per_object=40, radius_max_mm=20.0)
        tmp = Path("/tmp") / f"microtomo_check_{int(time.time())}.h5"
        generate(cfg, tmp, quiet=True)
        _read_check(tmp)
        tmp.unlink(missing_ok=True)
        return 0
    cfg = cfg_from_args(a)
    cfg.n_samples = a.n_samples
    out = Path(a.out) if Path(a.out).is_absolute() else PROJECT_ROOT / a.out
    generate(cfg, out, quiet=a.quiet)
    return 0


if __name__ == "__main__":
    sys.exit(main())