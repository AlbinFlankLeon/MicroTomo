# MicroTomo — Solver Weighted Evaluation Scorecard

> **Purpose**: Rank candidate solvers for the 60 GHz microwave tomography simulation pipeline.
> **Method**: Weighted multi-criteria scoring. Each criterion scored 0–5, multiplied by category weight.
> **Total possible**: 100 points. Target to proceed: ≥ 60 for primary solver, ≥ 40 for augmentation tier.

---

## Scoring Rules

- **0** = Not supported / unusable
- **1** = Barely functional, major gaps
- **2** = Works but painful; significant limitations
- **3** = Acceptable; meets minimum requirements
- **4** = Good; exceeds minimum with minor gaps
- **5** = Excellent; fully meets or exceeds all requirements

---

## Category 1 — 60 GHz Fidelity (Weight: 25) [plan: 25%]

The solver must produce field solutions that match analytical references (Mie scattering) and capture the dominant physics at 60 GHz in air.

| # | Criterion | Description | Score | Notes |
|---|-----------|-------------|-------|-------|
| 1.1 | Mie validation (PEC) | S21 magnitude within ±1 dB of analytical Mie for PEC cylinder @ 60 GHz (mini domain) | /5 | |
| 1.2 | Debye dispersion | Correct water relaxation model (ε∞=5.2, εs=78.3, τ=5.56ps) — S21 attenuation matches reference | /5 | |
| 1.3 | Grid convergence | S21 converges ≤0.5 dB when dx halved (0.5→0.25mm) | /5 | |
| 1.4 | Multi-material support | Air, water, fat, glass, PEC, plastics (εr 2.1–80) all representable | /5 | |
| 1.5 | UWB frequency sweep | Supports 5–60 GHz sweep or broadband excitation in single run | /5 | |
| 1.6 | Reflection/SAR mode | Correct near-field scattering in reflection geometry (source + Rx on same side) | /5 | |

**Subtotal**: ___ / 30

---

## Category 2 — Performance & Scalability (Weight: 25)

Must handle full 200mm domain (216M cells @ 0.33mm dx) within hardware budget.

| # | Criterion | Description | Score | Notes |
|---|-----------|-------------|-------|-------|
| 2.1 | Mini-domain runtime | ≤60s for PEC cylinder mini domain (100k cells) on single CPU core | /5 | |
| 2.2 | Full-domain feasibility | Can run 200mm domain within 128 GB RAM + RX 5700 XT 8 GB VRAM budget | /5 | |
| 2.3 | GPU acceleration | CUDA/OpenCL support for FDTD time-stepping | /5 | |
| 2.4 | Memory efficiency | Peak RAM ≤ 8× cell count × bytes/cell (with domain decomposition) | /5 | |
| 2.5 | Batch throughput | Can queue ≥10 phantom variations sequentially without manual intervention | /5 | |

**Subtotal**: ___ / 25

---

## Category 3 — Dataset Pipeline & Integration (Weight: 20) [plan: 20%]

Must fit into the Python/PyTorch training pipeline with minimal friction.

| # | Criterion | Description | Score | Notes |
|---|-----------|-------------|-------|-------|
| 3.1 | Python API | Scriptable via Python (subprocess or native API) — no GUI required | /5 | |
| 3.2 | HDF5 export | S-parameters + field snapshots exportable to HDF5 with metadata | /5 | |
| 3.3 | PyTorch DataLoader | Output tensor-compatible; can feed `torch_loader.py` directly | /5 | |
| 3.4 | Setup complexity | Installable in ≤3 commands; no conflicting system dependencies | /5 | |

**Subtotal**: ___ / 20

---

## Category 4 — Cost & Portability (Weight: 15) [plan: 15%]

Must be reliable for 1k–10k sample generation campaigns.

| # | Criterion | Description | Score | Notes |
|---|-----------|-------------|-------|-------|
| 4.1 | Reproducibility | Identical input → identical output (deterministic or seedable) | /5 | |
| 4.2 | Error handling | Graceful failure on ill-conditioned phantoms (divergence, NaN) with clear diagnostics | /5 | |
| 4.3 | Community / docs | Active maintenance, ≥100 GitHub stars, readable documentation | /5 | |

**Subtotal**: ___ / 15

---

## Category 5 — UWB vs CW Versatility (Weight: 15) [plan: 15%]

Bonus category for Tier 2/3 tools used alongside the primary solver.

| # | Criterion | Description | Score | Notes |
|---|-----------|-------------|-------|-------|
| 5.1 | Differentiability | Supports adjoint/automatic differentiation for inverse problems | /5 | |
| 5.2 | Ray-tracing speed | RT forward model runs in ≤1s per sample for coarse pre-training data | /5 | |

**Subtotal**: ___ / 15

---

## Candidate Solvers

| Solver | Tier | Category 1 | Category 2 | Category 3 | Category 4 | Category 5 | **Total** | Pass? |
|--------|------|------------|------------|------------|------------|------------|-----------|-------|
| **gprMax** | 1 (GPU FDTD) | /30 | /25 | /20 | /15 | /10 | **/100** | |
| **openEMS** | 1 (GPU FDTD) | /30 | /25 | /20 | /15 | /10 | **/100** | |
| **TorchFDTD** | 2 (AI-native) | /30 | /25 | /20 | /15 | /10 | **/100** | |
| **FEniCSx** | 2 (FEM CW) | /30 | /25 | /20 | /15 | /10 | **/100** | |
| **Sionna RT** | 3 (Ray tracing) | /30 | /25 | /20 | /15 | /10 | **/100** | |

**Pass threshold**: ≥ 60 for primary solver, ≥ 40 for augmentation.

---

## How to Score

1. Run `scripts/run_benchmark.py --phantom <name> --domain <domain>` for each solver.
2. Record raw metrics (runtime, RAM, S21 error vs Mie) in `reports/`.
3. Map raw metrics to 0–5 scores using the criteria descriptions above.
4. Multiply each category subtotal by the weight (already baked into subtotal max).
5. Sum all subtotals → total score. Compare to threshold.
6. Log the decision in `docs/decisions.md`.

---

*Created: Phase 5 prep — 2026-09-10*
*Update this file after each benchmark round with filled scores + notes.*
