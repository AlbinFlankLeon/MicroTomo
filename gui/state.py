#!/usr/bin/env python3
"""Simulation state for the MicroTomo app (pure logic, no Qt).

Holds an explicit, editable description of a simulation run — object specs,
transceiver layout, reconstruction params — and can execute it (forward
model + DAS + metrics). Separated from the GUI so it is fully testable
headless.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field

import numpy as np

# dim fields per object kind (labels)
DIM_LABELS = {
    "ellipsoid": ("semi a", "semi b", "semi c"),
    "cylinder": ("radius", "half len"),
    "pepper": ("body a", "body b", "body c"),
}


@dataclass
class ObjectSpec:
    kind: str = "ellipsoid"
    center: list = field(default_factory=lambda: [0.05, 0.05, 0.05])
    dim: list = field(default_factory=lambda: [0.020, 0.016, 0.025])
    tilt_deg: list = field(default_factory=lambda: [0.0, 0.0, 0.0])
    eps: float = 4.5
    label: str = "object"


@dataclass
class SimState:
    chamber_m: float = 0.10
    objects: list[ObjectSpec] = field(default_factory=lambda: [ObjectSpec()])
    mode: str = "static"                 # static | scan
    n_tx: int = 4
    tx_positions: np.ndarray | None = None     # (T,3) meters; None -> fill ring
    n_stations: int = 24                       # scan mode
    n_freq: int = 48
    grid: int = 20
    seed: int = 42
    snr_db: float = 30.0
    n_scatter_per_object: int = 1500
    wall: str = "absorbing"

    # ---- helpers ---------------------------------------------------------
    def shape_kinds(self) -> list[str]:
        return [o.kind for o in self.objects]

    def to_dict(self) -> dict:
        return {
            "chamber_m": self.chamber_m,
            "mode": self.mode,
            "n_tx": self.n_tx,
            "tx_positions": None if self.tx_positions is None
            else self.tx_positions.tolist(),
            "n_stations": self.n_stations,
            "n_freq": self.n_freq,
            "grid": self.grid,
            "seed": self.seed,
            "snr_db": self.snr_db,
            "objects": [o.__dict__ for o in self.objects],
        }

    @classmethod
    def from_dict(cls, d: dict) -> "SimState":
        s = cls(
            chamber_m=float(d.get("chamber_m", 0.10)),
            mode=d.get("mode", "static"),
            n_tx=int(d.get("n_tx", 4)),
            tx_positions=(None if d.get("tx_positions") is None
                          else np.asarray(d["tx_positions"], float)),
            n_stations=int(d.get("n_stations", 24)),
            n_freq=int(d.get("n_freq", 48)),
            grid=int(d.get("grid", 20)),
            seed=int(d.get("seed", 42)),
            snr_db=float(d.get("snr_db", 30.0)),
            objects=[ObjectSpec(**o) for o in d.get("objects", [ObjectSpec()])],
        )
        return s

    # ---- scene / scatter ---------------------------------------------------
    def build_shapes(self):
        """Fresh VeggieShape instances from the specs (no stale cache)."""
        from phantoms.veggie import (CucumberVeggie, EllipsoidVeggie, PepperVeggie)

        shapes = []
        for o in self.objects:
            if o.kind == "ellipsoid":
                shapes.append(EllipsoidVeggie(o.center, tuple(o.tilt_deg), o.eps,
                                              semi=o.dim[:3]))
            elif o.kind == "cylinder":
                shapes.append(CucumberVeggie(o.center, tuple(o.tilt_deg), o.eps,
                                             radius=o.dim[0], half_len=o.dim[1]))
            elif o.kind == "pepper":
                shapes.append(PepperVeggie(o.center, tuple(o.tilt_deg), o.eps,
                                           body=o.dim[:3]))
            else:
                raise ValueError(f"unknown kind {o.kind}")
        return shapes

    def scatter(self, seed: int | None = None) -> np.ndarray:
        """(N,6) surface scatterers xyz+normal for all objects."""
        from phantoms.veggie import VeggieScene

        rng = np.random.default_rng(seed if seed is not None else self.seed)
        scene = VeggieScene(chamber_m=self.chamber_m, shapes=self.build_shapes())
        return scene.surface_points(self.n_scatter_per_object, rng)

    def mesh(self) -> tuple[list[np.ndarray], list[np.ndarray]]:
        from phantoms.veggie import VeggieScene

        scene = VeggieScene(chamber_m=self.chamber_m, shapes=self.build_shapes())
        return scene.mesh()

    # ---- transceiver geometry ----------------------------------------------
    def default_ring(self) -> np.ndarray:
        """n_tx equally-spaced positions on the chamber ring plane (xy, z=mid)."""
        r = 0.4 * self.chamber_m
        z = self.chamber_m / 2.0
        theta = np.linspace(0, 2 * np.pi, self.n_tx, endpoint=False)
        return np.stack([self.chamber_m / 2 + r * np.cos(theta),
                         self.chamber_m / 2 + r * np.sin(theta),
                         np.full(self.n_tx, z)], axis=1)

    def physical_positions(self) -> np.ndarray:
        """(T,3) editable physical transceivers (override-able table)."""
        if self.tx_positions is not None and len(self.tx_positions) == self.n_tx:
            return np.asarray(self.tx_positions, float)
        return self.default_ring()


# ---- transceiver fillers (module-level, deterministic, testable) ----------

    def virtual_positions(self) -> np.ndarray:
        """All MIMO positions used by the forward model (scan -> T*n_stations).

        Mirrors sim.transceivers.ScanLayout's turntable transform but rotates
        the editable physical table, so manual transceiver edits are honoured.
        """
        phys = self.physical_positions()
        if self.mode != "scan" or self.n_stations <= 0:
            return phys
        ant = phys - self.chamber_m / 2.0
        r = np.hypot(ant[:, 0], ant[:, 1])
        theta = np.arctan2(ant[:, 1], ant[:, 0])
        phis = np.linspace(0, 2 * np.pi, self.n_stations, endpoint=False)
        out = []
        for phi in phis:
            t = theta + phi
            out.append(np.stack([self.chamber_m / 2 + r * np.cos(t),
                                 self.chamber_m / 2 + r * np.sin(t),
                                 ant[:, 2] + self.chamber_m / 2], axis=1))
        return np.concatenate(out, axis=0)

    # ---- pipeline -----------------------------------------------------------
    def run_pipeline(self, noise: bool = True) -> tuple[np.ndarray, dict, float]:
        """Forward SFCW + DAS recon vs the true scatterers.

        Returns (recon_xyz (M,3), metrics dict, wall seconds).
        """
        from sim.forward_surface import SurfaceScatterModel
        from recon.das import das_beamform
        from metrics.contour_metrics import chamfer, frac_within_cm

        t0 = time.time()
        scatter = self.scatter()
        truth = scatter[:, :3]
        positions = self.virtual_positions()
        freqs = np.linspace(57e9, 64e9, self.n_freq)

        model = SurfaceScatterModel(chamber_m=self.chamber_m, freqs_hz=freqs,
                                    scatter=scatter, gamma=None,
                                    wall_mode=self.wall, snr_db=self.snr_db,
                                    seed=self.seed)
        _, _, y = model.phase_history(positions, write_noise=noise)

        pts, _, thr = das_beamform(y, positions, freqs, self.chamber_m,
                                   grid=self.grid, threshold_quantile=0.995,
                                   top_m=1500)
        recon = pts[:, :3]
        ch = chamfer(recon, truth)
        metrics = {
            "n_tx": self.n_tx,
            "mode": self.mode,
            "n_stations": self.n_stations,
            "n_recon": int(len(recon)),
            "n_scatter": int(len(scatter)),
            "chamfer_mean_cm": round(ch["mean"] * 100, 3),
            "chamfer_median_cm": round(ch["median"] * 100, 3),
            "chamfer_max_cm": round(ch["max"] * 100, 3),
            "pct_within_1cm": round(frac_within_cm(recon, truth, 1.0) * 100, 2),
            "das_threshold": float(thr),
            "wall_time_s": round(time.time() - t0, 1),
        }
        return recon, metrics, time.time() - t0


def corners_tx(n: int, chamber_m: float) -> np.ndarray:
    """n antennas at the ring's cardinal 'corners' (and diagonals if n > 4).

    First 4 sit at 0°/90°/180°/270° on the chamber ring — the classic corner
    placement for a cylindrical scan. Extra antennas (n > 4) fill diagonal
    slots on the same ring so positions never degenerate or duplicate.
    Dimensions: (n, 3) metres, z at mid-height.
    """
    r = 0.4 * chamber_m
    z = chamber_m / 2.0
    n_cardinal = min(n, 4)
    angles = np.arange(n_cardinal) * (np.pi / 2.0)
    if n > 4:
        diag = np.pi / 4.0 + np.arange(n - 4) * (2.0 * np.pi / (n - 4))
        angles = np.concatenate([angles, diag])
    return np.stack([chamber_m / 2 + r * np.cos(angles),
                     chamber_m / 2 + r * np.sin(angles),
                     np.full(n, z)], axis=1)


def random_tx(n: int, chamber_m: float, seed: int | None = None) -> np.ndarray:
    """n antennas at uniformly random positions on a disc 90% of the ring.

    Kept off the wall so readings stay clean; z at mid-height.
    Deterministic for a given seed, so runs stay reproducible if desired.
    """
    rng = np.random.default_rng(seed)
    r = 0.9 * (0.4 * chamber_m)
    rr = r * np.sqrt(rng.uniform(0.0, 1.0, n))
    theta = rng.uniform(0.0, 2.0 * np.pi, n)
    z = chamber_m / 2.0
    return np.stack([chamber_m / 2 + rr * np.cos(theta),
                     chamber_m / 2 + rr * np.sin(theta),
                     np.full(n, z)], axis=1)

def default_state() -> SimState:
    s = SimState()
    s.objects = [
        ObjectSpec("ellipsoid", [0.055, 0.06, 0.05], [0.020, 0.016, 0.025],
                   [0, 0, 0], 4.5, "egg"),
        ObjectSpec("cylinder", [0.038, 0.045, 0.055], [0.012, 0.028], [0, 0, 0],
                   5.0, "cucumber"),
    ]
    return s