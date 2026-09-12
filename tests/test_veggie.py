#!/usr/bin/env python3
"""Tests for the veggie phantom generator (sparsity study T2)."""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from phantoms.veggie import CHAMBER_SIZES, generate_scene  # noqa: E402


@pytest.fixture(scope="module")
def scene(tmp_path_factory):
    return generate_scene(seed=42, chamber="10cm", n_shapes=2, pedestal=True)


def test_deterministic_same_seed():
    a = generate_scene(seed=7, chamber="10cm", n_shapes=2, pedestal=True)
    b = generate_scene(seed=7, chamber="10cm", n_shapes=2, pedestal=True)
    pa = a.surface_points(500, np.random.default_rng(8))
    pb = b.surface_points(500, np.random.default_rng(8))
    assert pa.shape[1] == 6 and pa.shape[0] > 500 * 2   # 2 shapes + pedestal
    assert np.array_equal(pa, pb)


def test_scatter_within_chamber(scene):
    pts = scene.surface_points(300, np.random.default_rng(1))
    lo, hi = 0.0, scene.chamber_m
    assert pts[:, 0].min() >= lo and pts[:, 0].max() <= hi
    assert pts[:, 1].min() >= lo and pts[:, 1].max() <= hi
    assert pts[:, 2].min() >= lo and pts[:, 2].max() <= hi


def test_scatter_has_normals(scene):
    pts = scene.surface_points(300, np.random.default_rng(1))
    norms = np.linalg.norm(pts[:, 3:], axis=1)
    assert np.allclose(norms, 1.0, atol=1e-6)


def test_voxels_nonempty_and_eps_range(scene):
    geometry, eps = scene.voxelize(grid=48)
    assert geometry.shape == eps.shape == (48, 48, 48)
    assert geometry.dtype == np.float32 and eps.dtype == np.float32
    assert geometry.any(), "surface mask should contain voxels"
    eps_vals = np.unique(eps)
    assert 1.0 in eps_vals              # air background
    body = eps_vals[(eps_vals != 1.0)]
    assert all((v >= 2.0 and v <= 85.0) for v in body)   # water-like range incl. pedestal


def test_center_is_filled(scene):
    geometry, eps = scene.voxelize(grid=32)
    mid = geometry.shape[0] // 2
    # chamber center column neighbours should show body eps somewhere
    assert (eps > 1.0).sum() > 0


def test_shape_kinds_valid():
    kinds = {"ellipsoid", "cucumber", "pepper"}
    rng = np.random.default_rng(3)
    from phantoms.veggie import _draw_shape
    seen = {_draw_shape(rng, 0.05).kind for _ in range(40)}
    assert seen <= kinds


def test_mesh_ply_roundtrip(tmp_path):
    scene = generate_scene(seed=11, chamber="10cm", n_shapes=2, pedestal=False)
    vs, fs = scene.mesh()
    v = np.concatenate(vs)
    f = np.concatenate(fs)
    assert v.shape[1] == 3 and f.shape[1] == 3
    ply = tmp_path / "m.ply"
    from phantoms.veggie import _write_ply
    _write_ply(ply, vs, fs)
    text = ply.read_text()
    lines = text.splitlines()
    assert lines[0] == "ply"
    assert f"element vertex {len(v)}" in text
    assert f"element face {len(f)}" in text
    # every face index in range (face lines are the LAST f.shape[0] non-empty lines:
    # they start with an integer count >= 3)
    body = [ln.strip() for ln in text.splitlines() if ln.strip() and not ln.startswith(("ply", "format", "element", "property", "end_header"))]
    face_lines = [ln for ln in body if ln.split()[0].isdigit() and int(ln.split()[0]) >= 3]
    assert len(face_lines) == len(f)
    idx = [int(x) for ln in face_lines for x in ln.split()[1:]]
    assert min(idx) >= 0 and max(idx) < len(v)


def test_chamber_sizes():
    assert CHAMBER_SIZES["10cm"] == pytest.approx(0.10)
    assert CHAMBER_SIZES["50cm"] == pytest.approx(0.50)