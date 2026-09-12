# Analytical (SFCW reflect) vs gprMax (FDTD) — validation gate

- Attempted: PEC slab 20x20x1 mm in 50 mm cube, PML walls, carrier **10 GHz**, band 6-14 GHz (scattered = filled − empty), gprMax dx=0.75 mm.

## Verdict: gate **NOT EXECUTABLE** on this machine

The 60 GHz spec gate cannot be meaningfully evaluated here, for two hard reasons:

**1. Grid budget.** At 60 GHz the gaussian pulse's significant spectrum reaches ~3×fc, so gprMax requires dx ≤ c/(30·60 GHz) ≈ 0.17 mm → ~27M cells in the 50 mm box. CPU-infeasible; even the repository's own water benchmarks already punt to 10 GHz for this reason.

**2. Scaled-carrier is not representative.** At the 10 GHz fallback the wavelength is 30 mm, so the database box (50 mm = 1.7 λ) and any probe inside are electrically small. The PEC sphere sits on a Mie resonance (ka≈0.84) and even the 20 mm PEC slab is a resonant patch (0.67 λ): measured scattered fields show multi-lobe, delayed responses (gprMax peak ~110 ps past the geometric echo) that a first-order single-reflection model does not and should not reproduce. The 60 GHz study regime (10 cm objects = ~20 λ, optically large, specular) cannot be emulated at a scaled carrier inside a 50 mm box.

## Impact on the study numbers (spec fallback)

- envelope correlation = 0.000 (scaled-carrier diagnostic only)
- peak-range error = 1.62 cm (scaled-carrier diagnostic only)
- box = 1.67 λ

**The 10 cm and 50 cm analytical numbers are flagged UNTRUSTED pending a 60 GHz cross-validation.** This requires a GPU gprMax run (or a smaller dedicated full-wave solver) and is deliberately not faked here. Within-model checks that DO hold are the exact two-way time-of-flight tests in `tests/test_forward_surface.py`.
