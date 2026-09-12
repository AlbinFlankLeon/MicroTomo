# T3 — Reflection-mode analytical forward model

- **Blocks:** T4, T5, T6, T7, T8 (produces the phase-history data everything downstream reads)
- **Depends on:** T2

## Goal
Adapt the `batch_generate.py` Born point-cloud pattern to **reflection-only surface imaging**: scatterers sit on the target *surface* (visible from the Tx), SFCW sweep 57–64 GHz, two-way time-of-flight, spreading, specular-ish backscatter amplitude, direct-coupling + wall background (absorbing baseline, `--metal-walls` stress addition), additive noise. Supports chamber sizes 10 cm and 50 cm and both **static** and **scanned** transceiver layouts.

## Deliverables
- `sim/forward_surface.py` (or extended `batch_generate.py`) with:
  - `SurfaceScatterModel(scene, transceiver_layout, freqs, snr)` → `(F, T)` complex phase history per channel.
  - Generic scatterer model with surface visibility culling (backside occluded).
  - Static layout: N fixed positions on chamber faces. Scan layout: N transceivers on a turntable/rail, M angular stations → many virtual look-angles.
  - Walls: absorbing (no echo) default; `--metal-walls` adds delayed specular enclosure echoes.
- HDF5 writer matching the torch_loader layout: `/samples/NN/{x, x_hat, y}` complex64 `(F,T)`, `{geometry, eps}` float32 `(H,W,D)` + `meta` (scene, layout, seed).
- Keep ~seconds/scene on CPU for the 50 cm chamber.

## Verification
- Known-timings test: single point scatterer at known distance → echo peak at correct two-way delay.
- HDF5 round-trip through `datasets/torch_loader.py`.
- Determinism with fixed seed.