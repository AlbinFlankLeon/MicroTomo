#!/usr/bin/env python3
"""MicroTomo contour metrics (sparsity study T5).

Turns a reconstructed surface point cloud and the true surface point cloud
into the study's verdict numbers:

- chamfer(): two-sided mean/median/max nearest-neighbour distance (meters)
- frac_within_cm(): fraction of the true surface covered within a distance
  threshold (the ~1 cm accuracy claim)
- detect_feature(): does a known-size probe (sphere of given radius)
  reconstruct as a cluster at the right location?

All inputs are (N, 3) float arrays in meters. Empty clouds are handled
gracefully (return inf / 0.0, never crash).
"""
from __future__ import annotations

import numpy as np

try:
    from scipy.spatial import cKDTree
except ImportError:  # pragma: no cover
    cKDTree = None


def _nn_distances(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Nearest distance from each point of `a` to the set `b`."""
    if len(a) == 0 or len(b) == 0:
        return np.array([np.inf])
    if cKDTree is None:  # pragma: no cover - slow fallback
        return np.min(np.sqrt(((a[:, None, :] - b[None, :, :]) ** 2).sum(-1)), axis=1)
    return cKDTree(b).query(a, k=1)[0]


def chamfer(recon: np.ndarray, truth: np.ndarray) -> dict:
    """Two-sided Chamfer distances, meters. Keys: mean, median, max."""
    recon, truth = np.asarray(recon, float), np.asarray(truth, float)
    if len(recon) == 0 or len(truth) == 0:
        return {"mean": np.inf, "median": np.inf, "max": np.inf}
    d = np.concatenate([_nn_distances(recon, truth), _nn_distances(truth, recon)])
    return {
        "mean": float(np.mean(d)),
        "median": float(np.median(d)),
        "max": float(np.max(d)),
    }


def frac_within_cm(recon: np.ndarray, truth: np.ndarray, threshold_cm: float = 1.0) -> float:
    """Fraction of the TRUE surface within `threshold_cm` of the recon cloud."""
    recon, truth = np.asarray(recon, float), np.asarray(truth, float)
    if len(recon) == 0 or len(truth) == 0:
        return 0.0
    d = _nn_distances(truth, recon)
    return float(np.mean(d <= threshold_cm / 100.0))


def detect_feature(
    recon: np.ndarray,
    probe_center: np.ndarray,
    probe_radius_m: float,
    tol_cm: float = 1.5,
    min_frac: float = 0.1,
) -> dict:
    """Does the recon cloud contain the known probe sphere near its center?

    Feature = fraction of recon points within (probe_radius + tol) of the
    probe center. Returns {detected, frac, n_points}.
    """
    recon = np.asarray(recon, float)
    if len(recon) == 0:
        return {"detected": False, "frac": 0.0, "n_points": 0}
    d = np.linalg.norm(recon - np.asarray(probe_center, float), axis=1)
    frac = float(np.mean(d <= probe_radius_m + tol_cm / 100.0))
    return {"detected": frac >= min_frac, "frac": frac, "n_points": int(len(recon))}


if __name__ == "__main__":  # pragma: no cover - sanity demo
    truth = np.array([[0.0, 0.0, 0.0], [0.01, 0.0, 0.0], [0.0, 0.01, 0.0]])
    print("self-check chamfer(truth, truth):", chamfer(truth, truth))
    print("self-check frac_within 0.5cm     :", frac_within_cm(truth + 0.005, truth, 0.5))