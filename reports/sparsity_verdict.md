# MicroTomo — Transceiver Sparsity Verdict

- Chamber: **10cm** (100 mm cube) · wall: **absorbing**
- Seeds: `42..42` (1 samples/config) · freq 57.0-64.0 GHz (16 steps) · grid 12
- Sim: reflection SFCW surface scatterers (sim/forward_surface.py) → DAS (recon/das.py) → chamfer vs true surface scatterers.
- **Resolution caveat:** reconstruction grid `12` → voxel ≈ 8.3 mm, so metrics are floored at ~1 voxel for tiny PoC chambers. Sensor-sparsity effects show up when the voxel is well below the RF resolution (range floor c/2B ≈ 2.1 cm at 7 GHz BW).

### Interpreting the numbers
At the 10 cm PoC chamber the median error is grid-limited (≈ one voxel), so all configs look green. Run the real check at the 50 cm chamber with a finer grid to expose the aperture/bandwidth limits:
```
python scripts/sparsity_study.py --transceivers 2,3,4,6,8 --static --scan --chamber 50cm --n-samples 3 --n-freq 64 --grid 32 --n-stations 12 --seed 1
```

Thresholds: **green** = median chamfer ≤ 1.5 cm · **yellow** ≤ 3 cm · **red** above.
Cost score: 1 pt per RF channel (+1.5 for motor/mechanics in the scan mode). Complexity: 1 pt per channel (+2 for motion control & pose sync). For real modules read cost as ~EUR 50-100 per 60 GHz channel (e.g. Acconeer A121, Infineon BGT60); the scan mode adds a stepper + controller.

> **Trust flag (T7 cross-validation):** the 60 GHz gprMax FDTD gate is not executable on this hardware (60 GHz forces dx ≈ 0.17 mm ≈ 27M cells), and scaled-carrier FDTD is not representative (50 mm box ≈ 1.7 λ at 10 GHz). Per the spec fallback the analytical numbers below are **flagged untrusted** until a GPU/60 GHz cross-validation exists — treat them as a self-consistent relative comparison only. See `reports/analytical_vs_gprmax.md`.

## Verdict matrix

| #Tx | Mode | Stations | mean (cm) | median (cm) | max (cm) | %≤1cm | cost | complexity | verdict |
|---|---|---|---|---|---|---|---|---|---|
| 2 | static | 1 | 0.73 | 0.65 | 3.68 | 0.835 | 2.0 | 2.0 | green |
| 2 | scan | 4 | 0.72 | 0.62 | 3.34 | 0.832 | 3.5 | 4.0 | green |
| 3 | static | 1 | 0.72 | 0.66 | 3.06 | 0.907 | 3.0 | 3.0 | green |
| 3 | scan | 4 | 0.78 | 0.71 | 3.71 | 0.833 | 4.5 | 5.0 | green |
| 4 | static | 1 | 0.72 | 0.62 | 3.34 | 0.832 | 4.0 | 4.0 | green |
| 4 | scan | 4 | 0.72 | 0.62 | 3.34 | 0.832 | 5.5 | 6.0 | green |

_(study wall time: 1.5s; rerun with `--seed N` to reproduce)_
