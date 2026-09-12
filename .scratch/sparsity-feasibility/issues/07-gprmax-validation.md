# T7 — PoC gprMax cross-validation gate

- **Blocks:** the "trust the 50 cm numbers" claim (report flag)
- **Depends on:** T2, T3

## Goal
Cross-validate the analytical reflection model against gprMax full-wave FDTD in the 10×10×10 cm PoC box on shared phantoms, and write the agreement result into the verdict report (trusted / flagged).

## Deliverables
- PoC harness reusing `scripts/run_benchmark.py` pattern: gprMax input from T2 phantom at 10 cm box, 60 GHz SFCW-ish/gaussian, transceivers at box faces.
- Comparison script: analytical vs gprMax channel responses — mean envelope correlation + peak range error.
- Gate thresholds (spec): correlation ≥ 0.9 and peak range error ≤ 0.5 cm on ≥ 1 shared phantom.
- `reports/analytical_vs_gprmax.md` result.

## Verification
- Gate runs herless on CPU in minutes (grid 0.5–1 mm → ≤ 200³ cells).
- Report includes pass/fail + numbers.