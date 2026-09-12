#!/usr/bin/env python3
"""MicroTomo veggie phantom library — produce-like targets for the sparsity study.

Produces three data products per scene, all numpy/pure-python (no GUI deps):
  1. surface scatterer cloud  .npy  (N,6): [x,y,z, nx,ny,nz]  — for the analytical forward model
  2. closed triangle mesh     .ply  (ASCII)                  — for the GUI + gprMax volume gen
  3. voxel volumes            .npz  geometry (surface mask), eps (filled permittivity volume)

Shapes (mostly-water dielectric, eps real ~50-80, seeded):
  - EllipsoidVeggie   ("potato/apple")  — three semi-axes + tilt
  - CucumberVeggie    ("cucumber")      — cylinder with rounded (hemisphere) ends + tilt
  - PepperVeggie      ("pepper/carrot") — ellipsoid body + tapering stem cone (composite)
  - optional low-permittivity pedestal under the object

Usage:
    python phantoms/veggie.py --seed 42 --chamber 10cm --n-shapes 1 --out-prefix /tmp/veg
    → /tmp/veg_scatter.npy, /tmp/veg_mesh.ply, /tmp/veg_vol.npz, /tmp/veg_meta.json
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from numpy.random import Generator

EPS_WATER_LIKE = (50.0, 80.0)      # mostly-water produce, 60 GHz real part
EPS_PEDESTAL = 2.1                  # low-permittivity foam/plastic stand
CHAMBER_SIZES = {"10cm": 0.10, "50cm": 0.50}


# ---------------------------------------------------------------- geometry ---

def _rot_from_tilt(tilt_zyx_deg: tuple[float, float, float]) -> np.ndarray:
    """Rotation matrix from (z,y,x) Euler angles in degrees."""
    cz, sz = np.cos(np.radians(tilt_zyx_deg[0])), np.sin(np.radians(tilt_zyx_deg[0]))
    cy, sy = np.cos(np.radians(tilt_zyx_deg[1])), np.sin(np.radians(tilt_zyx_deg[1]))
    cx, sx = np.cos(np.radians(tilt_zyx_deg[2])), np.sin(np.radians(tilt_zyx_deg[2]))
    rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    return rz @ ry @ rx


def _surface_points_ellipsoid(a: float, b: float, c: float, n: int, rng: Generator) -> np.ndarray:
    """Uniform-ish points on the ellipsoid surface (radial-function sampling)."""
    # sample directions, push out to the surface; heavier weight along long axes ->
    # acceptable for scatterer use (each point carries its own weight in the model).
    u = rng.uniform(-1, 1, size=(n, 3))
    norms = np.linalg.norm(u, axis=1, keepdims=True)
    u = u / np.maximum(norms, 1e-12)
    scale = np.array([a, b, c])
    r = 1.0 / np.sqrt(np.sum((u * (1.0 / scale)) ** 2, axis=1, keepdims=True))
    pts = u * r * scale
    # outward normal of x²/a²+y²/b²+z²/c²=1 is ∝ (x/a²,y/b²,z/c²) ∝ (u_x/a,u_y/b,u_z/c)
    normals = u / scale
    normals = normals / np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-12)
    return pts, normals


def _surface_points_rounded_cylinder(radius: float, half_len: float, n: int, rng: Generator) -> np.ndarray:
    """Points on cylinder side + hemisphere caps, with outward normals."""
    n_side = int(n * half_len / (half_len + radius))
    n_cap2 = (n - n_side) // 2
    n_cap1 = n - n_side - n_cap2
    pts, normals = [], []

    phi = rng.uniform(0, 2 * np.pi, size=n_side)
    z = rng.uniform(-half_len, half_len, size=n_side)
    side = np.stack([radius * np.cos(phi), radius * np.sin(phi), z], axis=1)
    pts.append(side)
    xy = side[:, :2]
    rho = np.linalg.norm(xy, axis=1, keepdims=True)
    normals.append(np.hstack([xy / np.maximum(rho, 1e-12), np.zeros_like(rho)]))

    for sign, nn in ((1.0, n_cap1), (-1.0, n_cap2)):
        if nn <= 0:
            continue
        u = rng.uniform(-1, 1, size=(nn, 3))
        u = u / np.maximum(np.linalg.norm(u, axis=1, keepdims=True), 1e-12)
        cap = u * radius
        cap[:, 2] = sign * (half_len + cap[:, 2])   # hemisphere pole outwards
        pts.append(cap)
        normals.append(u)

    pts = np.concatenate(pts)
    normals = np.concatenate(normals)
    return pts, normals


def _mesh_ellipsoid(a: float, b: float, c: float, n_strips: int = 24) -> tuple[np.ndarray, np.ndarray]:
    lat = np.linspace(0, np.pi, n_strips)
    lon = np.linspace(0, 2 * np.pi, n_strips + 1)
    v = np.stack(
        [
            a * np.outer(np.sin(lat), np.cos(lon)).ravel(),
            b * np.outer(np.sin(lat), np.sin(lon)).ravel(),
            c * np.outer(np.cos(lat), np.ones(n_strips + 1)).ravel(),
        ],
        axis=1,
    )
    i, j = np.meshgrid(np.arange(n_strips), np.arange(n_strips + 1))
    idx = (i * (n_strips + 1) + j).ravel()
    v = v[idx]
    faces = []
    for p in range(n_strips):
        for q in range(n_strips):
            a0 = p * (n_strips + 1) + q
            a1 = a0 + 1
            b0 = (p + 1) * (n_strips + 1) + q
            b1 = b0 + 1
            faces.append((a0, a1, b0))
            faces.append((a1, b1, b0))
    return v, np.asarray(faces, dtype=np.int64)


def _mesh_rounded_cylinder(radius: float, half_len: float, n_phi: int = 24, n_z: int = 12) -> tuple[np.ndarray, np.ndarray]:
    """Closed mesh: cylinder side + hemisphere caps at both ends."""
    phi = np.linspace(0, 2 * np.pi, n_phi, endpoint=False)
    verts: list[list[float]] = []
    faces: list[tuple[int, int, int]] = []

    # --- side wall (vertical strips) ---
    for z in np.linspace(-half_len, half_len, n_z):
        for ph in phi:
            verts.append([radius * np.cos(ph), radius * np.sin(ph), z])
    n_side = n_z * n_phi
    for r in range(n_z - 1):
        for c in range(n_phi):
            nc = (c + 1) % n_phi
            a0 = r * n_phi + c
            a1 = (r + 1) * n_phi + c
            a0n = r * n_phi + nc
            a1n = (r + 1) * n_phi + nc
            faces.append((a0, a1, a0n))
            faces.append((a1, a1n, a0n))

    def add_cap(sign: float) -> None:
        idx0 = len(verts)
        verts.append([0.0, 0.0, sign * (half_len + radius)])   # pole
        n_lat = max(3, n_z // 2)
        prev: list[int] | None = None
        for k in range(1, n_lat + 1):
            theta = (k / n_lat) * (np.pi / 2.0)
            z = sign * (half_len + radius * np.sin(theta))
            rr = radius * np.cos(theta)
            ring = [len(verts) + i for i in range(n_phi)]
            for ph in phi:
                verts.append([rr * np.cos(ph), rr * np.sin(ph), z])
            if prev is not None:
                for c in range(n_phi):
                    nc = (c + 1) % n_phi
                    faces.append((prev[c], ring[c], ring[nc]))
                    faces.append((prev[c], ring[nc], prev[nc]))
            prev = ring
        for c in range(n_phi):
            nc = (c + 1) % n_phi
            assert prev is not None
            faces.append((prev[c], prev[nc], idx0))

    add_cap(+1.0)
    add_cap(-1.0)
    return np.asarray(verts, dtype=float), np.asarray(faces, dtype=np.int64)


# -------------------------------------------------------------- scene model ---

@dataclass
class VeggieShape:
    kind: str
    center: np.ndarray
    tilt_deg: tuple[float, float, float]
    eps: float
    pec: bool
    rot: np.ndarray = field(init=False)
    _local_points: np.ndarray | None = field(default=None, init=False)
    _local_normals: np.ndarray | None = field(default=None, init=False)
    _local_mesh_v: np.ndarray | None = field(default=None, init=False)
    _local_mesh_f: np.ndarray | None = field(default=None, init=False)

    def __post_init__(self) -> None:
        self.rot = _rot_from_tilt(self.tilt_deg)

    def make_local(self, n_scatter: int, rng: Generator) -> None:
        raise NotImplementedError

    # ---- world space products
    def surface_points(self, n_scatter: int, rng: Generator) -> np.ndarray:
        if self._local_points is None:
            self.make_local(n_scatter, rng)
        local = self._local_points
        world = local @ self.rot.T + self.center
        normals = self._local_normals @ self.rot.T
        return np.hstack([world, normals])

    def mesh(self) -> tuple[np.ndarray, np.ndarray]:
        if self._local_mesh_v is None:
            self._build_mesh()
        v = self._local_mesh_v @ self.rot.T + self.center
        return v, self._local_mesh_f

    def _build_mesh(self) -> None:
        raise NotImplementedError

    def inside(self, pts: np.ndarray) -> np.ndarray:
        """Boolean mask: is each world point inside this closed shape?"""
        local = (pts - self.center) @ self.rot
        return self._inside_local(local)

    def _inside_local(self, local: np.ndarray) -> np.ndarray:
        raise NotImplementedError


class EllipsoidVeggie(VeggieShape):
    def __init__(self, center, tilt_deg, eps, pec=False, semi=(0.05, 0.04, 0.06)):
        self.semi = np.asarray(semi, dtype=float)
        super().__init__("ellipsoid", np.asarray(center, dtype=float), tilt_deg, eps, pec)

    def make_local(self, n_scatter, rng):
        self._local_points, self._local_normals = _surface_points_ellipsoid(*self.semi, n_scatter, rng)

    def _build_mesh(self):
        self._local_mesh_v, self._local_mesh_f = _mesh_ellipsoid(*self.semi)

    def _inside_local(self, local):
        return np.sum((local / self.semi) ** 2, axis=1) <= 1.0


class CucumberVeggie(VeggieShape):
    def __init__(self, center, tilt_deg, eps, pec=False, radius=0.025, half_len=0.08):
        self.radius, self.half_len = radius, half_len
        super().__init__("cucumber", np.asarray(center, dtype=float), tilt_deg, eps, pec)

    def make_local(self, n_scatter, rng):
        self._local_points, self._local_normals = _surface_points_rounded_cylinder(self.radius, self.half_len, n_scatter, rng)

    def _build_mesh(self):
        self._local_mesh_v, self._local_mesh_f = _mesh_rounded_cylinder(self.radius, self.half_len)

    def _inside_local(self, local):
        x, y, z = local[:, 0], local[:, 1], local[:, 2]
        r = np.hypot(x, y)
        central = (np.abs(z) <= self.half_len) & (r <= self.radius)
        cap_top = (z > self.half_len) & (r**2 + (z - self.half_len) ** 2 <= self.radius**2)
        cap_bot = (z < -self.half_len) & (r**2 + (z + self.half_len) ** 2 <= self.radius**2)
        return central | cap_top | cap_bot


class PepperVeggie(VeggieShape):
    """Ellipsoid body + tapered stem cone (pointing +y in local coords)."""

    def __init__(self, center, tilt_deg, eps, pec=False, body=(0.04, 0.05, 0.025), stem_len=0.04, stem_r=0.008):
        self.body = np.asarray(body, dtype=float)
        self.stem_len, self.stem_r = stem_len, stem_r
        super().__init__("pepper", np.asarray(center, dtype=float), tilt_deg, eps, pec)

    def make_local(self, n_scatter, rng):
        n_body = int(n_scatter * 0.75)
        pts_b, nrm_b = _surface_points_ellipsoid(*self.body, n_body, rng)
        # stem: cone surface pointing +y, radius shrinking from stem_r to ~0
        n_stem = n_scatter - n_body
        phi = rng.uniform(0, 2 * np.pi, n_stem)
        t = rng.uniform(0, 1, n_stem)
        base = self.body[1]                       # stem starts at body equator +y
        y = base + t * self.stem_len
        rr = self.stem_r * (1 - t)
        stem = np.stack([rr * np.cos(phi), y, rr * np.sin(phi)], axis=1)
        nrm = np.stack([np.cos(phi), np.zeros(n_stem), np.sin(phi)], axis=1)
        self._local_points = np.concatenate([pts_b, stem])
        self._local_normals = np.concatenate([nrm_b, nrm])

    def _build_mesh(self):
        v, f = _mesh_ellipsoid(*self.body)
        self._local_mesh_v, self._local_mesh_f = v, f  # stem omitted from mesh (v1)

    def _inside_local(self, local):
        body = np.sum((local / self.body) ** 2, axis=1) <= 1.0
        return body | ((local[:, 1] > self.body[1]) & (local[:, 1] <= self.body[1] + self.stem_len) &
                       (np.hypot(local[:, 0], local[:, 2]) <= self.stem_r * (1 - (local[:, 1] - self.body[1]) / self.stem_len)))


@dataclass
class VeggieScene:
    """A chamber + 0..n seeded veggie shapes (+ optional pedestal)."""

    chamber_m: float
    shapes: list[VeggieShape] = field(default_factory=list)
    pedestal: VeggieShape | None = None

    @property
    def center(self) -> np.ndarray:
        return np.array([self.chamber_m / 2] * 3)

    def surface_points(self, n_per_shape: int, rng: Generator) -> np.ndarray:
        pts = [s.surface_points(n_per_shape, rng) for s in self.shapes]
        if self.pedestal is not None:
            pts.append(self.pedestal.surface_points(max(64, n_per_shape // 4), rng))
        return np.concatenate(pts) if pts else np.empty((0, 6))

    def mesh(self) -> tuple[list[np.ndarray], list[np.ndarray]]:
        vs, fs = [], []
        for s in self.shapes + ([self.pedestal] if self.pedestal else []):
            v, f = s.mesh()
            vs.append(v)
            fs.append(f)
        return vs, fs

    def voxelize(self, grid: int) -> tuple[np.ndarray, np.ndarray]:
        """(H,W,D) surface-mask (float32) and filled-eps volume (float32)."""
        axis = np.linspace(0, self.chamber_m, grid, endpoint=False) + self.chamber_m / grid / 2
        X, Y, Z = np.meshgrid(axis, axis, axis, indexing="ij")
        coords = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1)

        # eps: fill from outermost shape inwards; surface: boundary voxels.
        eps = np.ones(len(coords), dtype=np.float32)
        surface = np.zeros(len(coords), dtype=bool)
        for s in self.shapes + ([self.pedestal] if self.pedestal else []):
            inside = s.inside(coords)
            eps[inside] = 0.0 if s.pec else np.float32(s.eps)
            # surface = voxels inside this shape that touch a voxel inside-less neighbor
        inside_any = eps != 1.0
        # coarse surface detection: voxel is surface if inside and any 6-neighbor is outside
        eps3 = eps.reshape((grid, grid, grid))
        inside3 = eps3 != 1.0
        surf3 = np.zeros_like(inside3)
        inside_pad = np.pad(inside3, 1, mode="constant", constant_values=False)
        shifts = [
            (slice(None, -2), slice(1, -1), slice(1, -1)),
            (slice(2, None), slice(1, -1), slice(1, -1)),
            (slice(1, -1), slice(None, -2), slice(1, -1)),
            (slice(1, -1), slice(2, None), slice(1, -1)),
            (slice(1, -1), slice(1, -1), slice(None, -2)),
            (slice(1, -1), slice(1, -1), slice(2, None)),
        ]
        for sh in shifts:
            surf3[inside3] |= ~inside_pad[sh][inside3]
        # clear deep-interior eps (keep only 2-voxel shell) so eps holds body value
        # only at the shell, mirroring reflection-only sensing
        shell3 = surf3.copy()
        return surf3.astype(np.float32), eps3


# ------------------------------------------------------------- generate CLI ---

def _draw_shape(rng: Generator, chamber_half: float) -> VeggieShape:
    eps = float(rng.uniform(*EPS_WATER_LIKE))
    pec = bool(rng.random() < 0.1)
    kind = rng.choice(["ellipsoid", "cucumber", "pepper"])
    tilt = tuple(float(v) for v in rng.uniform(-20, 20, size=3))
    # Invariants: every shape fits in a sphere of radius E (local coords), and the
    # rotation growth factor is <= 1.3 for |tilt| <= 20 deg. Center margin = 1.3*E
    # keeps the whole rotated shape strictly inside [0, chamber].
    max_extent = min(0.10, chamber_half * 0.7 / 1.3)     # 20 cm object cap, chamber-limited
    min_radius = 0.015                                   # 3 cm smallest object
    center = np.asarray(
        rng.uniform(1.3 * max_extent, 2 * chamber_half - 1.3 * max_extent, size=3)
    )
    if kind == "ellipsoid":
        a = rng.uniform(min_radius, max_extent)
        b = rng.uniform(min_radius, max_extent)
        c = rng.uniform(min_radius, max_extent)
        return EllipsoidVeggie(center, tilt, eps, pec, semi=(a, b, c))
    if kind == "cucumber":
        # total length L = 2*(half_len + radius); radius 15-35% of L
        length = rng.uniform(2 * min_radius, 2 * max_extent)
        radius = rng.uniform(length * 0.15, length * 0.35)
        half_len = max(0.0, length / 2 - radius)
        return CucumberVeggie(center, tilt, eps, pec, radius=radius, half_len=half_len)
    # pepper: body b = ellipsoid half-length (y), a = x, c = z, tapering stem on top
    b = rng.uniform(max_extent * 0.4, max_extent)
    a = rng.uniform(b * 0.7, min(max_extent, b * 1.3))
    c = rng.uniform(a * 0.4, a * 0.8)
    stem_len = rng.uniform(0.005, max(0.005, max_extent - b))
    return PepperVeggie(center, tilt, eps, pec, body=(a, b, c), stem_len=stem_len)


def generate_scene(seed: int, chamber: str = "10cm", n_shapes: int = 1, pedestal: bool = False) -> VeggieScene:
    rng = np.random.default_rng(seed)
    chamber_m = CHAMBER_SIZES[chamber]
    scene = VeggieScene(chamber_m=chamber_m)
    half = chamber_m / 2
    for _ in range(max(1, n_shapes)):
        scene.shapes.append(_draw_shape(rng, half))
    if pedestal:
        r = min(0.02, half * 0.3)
        h = r * 1.2
        half_len = max(0.001, h - r)
        # bottom of the pedestal sits on the chamber floor (z=0)
        z_center = half_len + r
        scene.pedestal = CucumberVeggie(
            np.array([scene.center[0], scene.center[1], z_center]),
            (0, 0, 0),
            EPS_PEDESTAL,
            radius=r,
            half_len=half_len,
        )
    return scene


def _write_ply(path: Path, vertices: np.ndarray, faces: list[np.ndarray]) -> None:
    """Minimal ASCII PLY writer (no deps)."""
    v = np.concatenate(vertices)
    f = np.concatenate([np.hstack([np.full((len(ff), 1), len(ff[0])), ff]) for ff in faces])
    lines = [
        "ply",
        "format ascii 1.0",
        f"element vertex {len(v)}",
        "property float x",
        "property float y",
        "property float z",
        f"element face {len(f)}",
        "property list uchar int vertex_indices",
        "end_header",
    ]
    for row in v:
        lines.append(f"{row[0]:.6f} {row[1]:.6f} {row[2]:.6f}")
    for row in f:
        lines.append(" ".join(str(int(x)) for x in row))
    path.write_text("\n".join(lines) + "\n")


def _shape_size_meta(s: VeggieShape) -> dict:
    """JSON-safe per-shape size description."""
    if isinstance(s, EllipsoidVeggie):
        return {"semi_axes": s.semi.tolist()}
    if isinstance(s, CucumberVeggie):
        return {"radius": s.radius, "half_len": s.half_len}
    if isinstance(s, PepperVeggie):
        return {"body": s.body.tolist(), "stem_len": s.stem_len, "stem_r": s.stem_r}
    return {}


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="MicroTomo veggie phantom generator")
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--chamber", choices=sorted(CHAMBER_SIZES), default="10cm")
    p.add_argument("--n-shapes", type=int, default=1)
    p.add_argument("--pedestal", action="store_true", help="add low-eps pedestal under object")
    p.add_argument("--n-scatter", type=int, default=2000, help="surface points per shape")
    p.add_argument("--grid", type=int, default=64, help="voxels per axis")
    p.add_argument("--out-prefix", type=str, default="phantom_veg", help="output prefix")
    args = p.parse_args(argv)

    scene = generate_scene(args.seed, args.chamber, args.n_shapes, args.pedestal)
    prefix = Path(args.out_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)

    scatter = scene.surface_points(args.n_scatter, np.random.default_rng(args.seed + 1))
    np.save(f"{prefix}_scatter.npy", scatter)

    vs, fs = scene.mesh()
    _write_ply(Path(f"{prefix}_mesh.ply"), vs, fs)

    geometry, eps = scene.voxelize(args.grid)
    np.savez(f"{prefix}_vol.npz", geometry=geometry, eps=eps)

    meta = {
        "seed": args.seed,
        "chamber_m": scene.chamber_m,
        "n_shapes": len(scene.shapes),
        "pedestal": scene.pedestal is not None,
        "shapes": [
            {
                "kind": s.kind,
                "center": s.center.tolist(),
                "tilt_deg": list(s.tilt_deg),
                "eps": s.eps,
                "pec": s.pec,
                "size": _shape_size_meta(s),
            }
            for s in scene.shapes
        ],
    }
    Path(f"{prefix}_meta.json").write_text(json.dumps(meta, indent=2))
    print(f"veggie scene: {len(scene.shapes)} shape(s), {len(scatter)} scatter points, chamber {args.chamber}")
    print(f"  scatter : {prefix}_scatter.npy")
    print(f"  mesh    : {prefix}_mesh.ply")
    print(f"  volumes : {prefix}_vol.npz (geometry={geometry.shape}, eps={eps.shape})")
    print(f"  meta    : {prefix}_meta.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())