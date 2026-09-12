# T5 — Contour metrics + unit tests

- **Blocks:** T6 (verdict correctness), T8
- **Depends on:** T4

## Goal
`metrics/contour_metrics.py`: numbers that decide the verdict — contour error in cm, measured between true surface and reconstructed point cloud.

## Deliverables
- `metrics/contour_metrics.py`:
  - Chamfer distance (mean / median / max cm). NOTE: one-way mean distance from ground-truth surface to reconstruction, and `% of true surface within 1 cm`.
  - Feature detection check: known-size test objects (1 cm sphere, 1 cm notch) — reconstruction peak within 1.5 cm of true position.
- Unit tests in `tests/`: Chamfer against hand-made truth; the 1 cm sphere detection; determinism.

## Verification
- `python -m pytest tests/ -v` green (subset for metrics).
- Metrics are invariant to reconstruction point-density (normalizing for fair comparison).