# T4 — DAS contour reconstruction

- **Blocks:** T5, T6, T8 (needs a reconstructed contour to measure/view)
- **Depends on:** T3

## Goal
`recon/das.py` — delay-and-sum beamformer: back-project the multistatic phase history onto a voxel/point grid and extract the **(partial) surface contour** as a point cloud. No ML, per spec.

## Deliverables
- `recon/das.py`: takes HDF5 sample (or in-memory `(F,T)` multistatic data + geometry meta), outputs:
  - back-projection energy volume `(H,W,D)` (thresholded),
  - surface point cloud = energy peaks / isosurface points.
- Config: grid resolution, threshold, coherence vs incoherent sum over transceivers and frequencies.

## Verification
- Unit test: single small PEC sphere at known position → DAS peak within 1 voxel of truth.
- Returns a point cloud the GUI (T8) can plot over the true phantom (T2).