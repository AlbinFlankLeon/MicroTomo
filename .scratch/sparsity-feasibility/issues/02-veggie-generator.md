# T2 — Veggie phantom / structure generator

- **Blocks:** T3, T7, T8 (needs shapes to sense)
- **Depends on:** T1 (none strictly)

## Goal
Extend `phantoms/generator.py` into the "tool to generate structures to attempt to detect": a seeded vegetable-like target library + a generic structure builder, emitting (a) an analytical surface-scatterer set, (b) a mesh/point cloud for the GUI, (c) voxel `geometry`/`eps` volumes for the HDF5 layout.

## Deliverables
- `veggie_library` builders: ellipsoid ("potato/apple"), cylinder-with-rounded-ends ("cucumber"), composite (ellipsoid + stem = "pepper/carrot"), optional low-permittivity pedestal.
- Parameters: sizes 3–20 cm, eccentricity, tilt angles, material = mostly-water dielectric (eps real ~50–80), PEC optional.
- CLI: `python phantoms/generator.py --type veggie_library --seed N --output-prefix X` → scatterer set + `.ply`/`.stl` mesh + `(H,W,D)` voxel volumes (respect chamber size arg).
- Deterministic per seed.

## Verification
- Generator determinism test: same seed → identical outputs.
- Shape bounds test: ellipsoid stays within requested chamber box.
- Visualizable mesh: opens in the T8 GUI / pyvista read.