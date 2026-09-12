# MicroTomo — Transceiver Sparsity Verdict

- Chamber: **10cm** (100 mm cube) · wall: **absorbing**
- Seeds: `42..43` (2 samples/config) · freq 57.0-64.0 GHz (32 steps) · grid 16
- Sim: reflection SFCW surface scatterers (sim/forward_surface.py) → DAS (recon/das.py) → chamfer vs true surface scatterers.
- **Resolution caveat:** reconstruction grid `16` → voxel ≈ 6.2 mm, so metrics are floored at ~1 voxel for tiny PoC chambers. Sensor-sparsity effects show up when the voxel is well below the RF resolution (range floor c/2B ≈ 2.1 cm at 7 GHz BW).

### Interpreting the numbers
At the 10 cm PoC chamber the median error is grid-limited (≈ one voxel), so all configs look green. Run the real check at the 50 cm chamber with a finer grid to expose the aperture/bandwidth limits:
```
python scripts/sparsity_study.py --transceivers 2,3,4,6,8 --static --scan --chamber 50cm --n-samples 3 --n-freq 64 --grid 32 --n-stations 12 --seed 1
```

Thresholds: **green** = median chamfer ≤ 1.5 cm · **yellow** ≤ 3 cm · **red** above.
Cost score: 1 pt per RF channel (+1.5 for motor/mechanics in the scan mode). Complexity: 1 pt per channel (+2 for motion control & pose sync). For real modules read cost as ~EUR 50-100 per 60 GHz channel (e.g. Acconeer A121, Infineon BGT60); the scan mode adds a stepper + controller.

## Verdict matrix

| #Tx | Mode | Stations | mean (cm) | median (cm) | max (cm) | %≤1cm | cost | complexity | verdict |
|---|---|---|---|---|---|---|---|---|---|
| 2 | static | 1 | 0.76 | 0.66 | 3.80 | 0.807 | 2.0 | 2.0 | green |
| 2 | scan | 6 | 0.61 | 0.50 | 3.65 | 0.976 | 3.5 | 4.0 | green |
| 3 | static | 1 | 0.65 | 0.53 | 3.54 | 0.947 | 3.0 | 3.0 | green |
| 3 | scan | 6 | 0.61 | 0.50 | 3.65 | 0.976 | 4.5 | 5.0 | green |
| 4 | static | 1 | 0.66 | 0.57 | 3.58 | 0.935 | 4.0 | 4.0 | green |
| 4 | scan | 6 | 0.58 | 0.46 | 3.88 | 0.995 | 5.5 | 6.0 | green |
| 6 | static | 1 | 0.61 | 0.50 | 3.65 | 0.976 | 6.0 | 6.0 | green |
| 6 | scan | 6 | 0.61 | 0.50 | 3.65 | 0.976 | 7.5 | 8.0 | green |
| 8 | static | 1 | 0.60 | 0.51 | 4.11 | 0.990 | 8.0 | 8.0 | green |
| 8 | scan | 6 | 0.46 | 0.38 | 4.37 | 0.995 | 9.5 | 10.0 | green |

_(study wall time: 95.5s; rerun with `--seed N` to reproduce)_
