#!/usr/bin/env python3
"""MicroTomo transceiver layouts (sparsity study).

Two mounting modes, exactly as the study needs:

- StaticLayout: N transceivers fixed at sparse positions in the chamber
  (mid-height ring on the chamber equator, radius 0.4 x chamber half-width).
- ScanLayout: the same N transceivers on a rotating platform (turntable) —
  M angular stations around the chamber z-axis produce M x N virtual
  look-angles (synthetic aperture).

Layout objects expose ``positions`` -> (T, 3) in meters, where T is the
total number of virtual transceiver positions (M*n for scans), and ``n``,
``meta()`` for the HDF5 metadata.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class StaticLayout:
    n: int
    chamber_m: float
    radius_frac: float = 0.4
    z_frac: float = 0.5

    @property
    def positions(self) -> np.ndarray:
        theta = np.linspace(0.0, 2.0 * np.pi, self.n, endpoint=False)
        r = self.radius_frac * self.chamber_m
        z = self.z_frac * self.chamber_m
        return np.stack(
            [self.chamber_m / 2 + r * np.cos(theta),
             self.chamber_m / 2 + r * np.sin(theta),
             np.full(self.n, z)], axis=1)

    def meta(self) -> dict:
        return {"mode": "static", "n": self.n, "chamber_m": self.chamber_m,
                "radius_frac": self.radius_frac, "z_frac": self.z_frac}


@dataclass(frozen=True)
class ScanLayout:
    n: int
    chamber_m: float
    n_stations: int = 24
    radius_frac: float = 0.4
    z_frac: float = 0.5

    @property
    def positions(self) -> np.ndarray:
        """(M*n, 3): antenna j at station m sits at angle theta_j + 2 pi m/M."""
        base = StaticLayout(self.n, self.chamber_m, self.radius_frac, self.z_frac)
        ant = base.positions - self.chamber_m / 2.0            # relative to center
        phis = np.linspace(0.0, 2.0 * np.pi, self.n_stations, endpoint=False)
        out = []
        # turntable rotation around the chamber z-axis
        r = np.hypot(ant[:, 0], ant[:, 1])
        theta = np.arctan2(ant[:, 1], ant[:, 0])
        for phi in phis:
            t = theta + phi
            x = self.chamber_m / 2 + r * np.cos(t)
            y = self.chamber_m / 2 + r * np.sin(t)
            out.append(np.stack([x, y, ant[:, 2] + self.chamber_m / 2], axis=1))
        return np.concatenate(out, axis=0)

    def meta(self) -> dict:
        return {"mode": "scan", "n": self.n, "n_stations": self.n_stations,
                "chamber_m": self.chamber_m, "radius_frac": self.radius_frac,
                "z_frac": self.z_frac}


def build_layout(mode: str, n: int, chamber_m: float, n_stations: int = 24):
    if mode == "static":
        return StaticLayout(n, chamber_m)
    if mode == "scan":
        return ScanLayout(n, chamber_m, n_stations=n_stations)
    raise ValueError(f"unknown mode {mode!r}")


def _self_test() -> None:
    s = StaticLayout(4, 0.10)
    assert s.positions.shape == (4, 3)
    assert s.positions.min() >= 0.0 and s.positions.max() <= 0.10
    sc = ScanLayout(2, 0.10, n_stations=24)
    assert sc.positions.shape == (48, 3)
    # static subset appears as the first M rotations
    assert np.allclose(sc.positions[:2], StaticLayout(2, 0.10).positions)
    print("transceivers self-test OK")


if __name__ == "__main__":
    _self_test()