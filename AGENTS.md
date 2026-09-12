# Project: MicroTomo — Homemade Microwave Tomography Simulation

> Simulation-first approach: validate solver throughput + accuracy before any hardware commitment.
> All local models (qwen3:8b + OmniRoute fallback). Privacy-first. GPU-aware.

---

## Project Goal

Simulate radar data for a homemade 60 GHz circular ring microwave tomography machine.
Generate large-scale synthetic 3D mock data for AI training.
Air-coupled, 3D reconstruction, reflection/SAR mode dominates at 60 GHz.

## Key Physics

| Parameter | Value |
|---|---|
| Frequency | 60 GHz (lambda=5mm air); variable 5-60 GHz UWB |
| Coupling medium | Air (no oil/water matching) |
| Materials | Air, Water, Fat, Glass, PEC, Plastics (eps 2.1-80) |
| FDTD grid dx | 0.33mm (air), 0.15mm (water/plastic) at 60 GHz |
| ROI diameter | 150-200 mm |
| Dataset target | 1k-10k samples with domain randomization |
| Water attenuation | ~20-30 dB/cm @60 GHz → transmission fails, reflection only |

## Directory Layout

```
MicroTomo/
├── AGENTS.md            ← THIS FILE — project bible for any model/agent
├── phantoms/
│   └── generator.py     ← random ellipsoids, eps samplers, domain randomization
├── sim/
│   ├── gprmax/          ← gprMax FDTD (primary GPU FDTD, CUDA+OpenCL)
│   ├── torchfdtd/       ← TorchFDTD/jaxfdtd (differentiable, batchable GPU)
│   ├── fenics/          ← FEniCSx FEM (fast CW 60 GHz frequency-domain)
│   └── sionna/          ← Sionna RT (ray tracing augmentation, coarse pre-training)
├── datasets/
│   ├── h5/              ← HDF5 output (S-params, field snapshots, phantom metadata)
│   ├── raw/             ← raw solver output (VTK, CSV, etc.)
│   └── torch_loader.py  ← PyTorch DataLoader for training
├── recon/
│   ├── das.py           ← Delay-and-Sum beamforming (baseline)
│   ├── dbim.py          ← Distorted Born Iterative Method (full-wave inverse)
│   └── unet.py          ← U-Net deep learning reconstruction
├── config/
│   └── benchmarks.yaml  ← canonical benchmark definitions (Phase 1 output)
├── scripts/
│   ├── run_benchmark.py ← run a single benchmark across tools
│   ├── batch_generate.py← mass dataset generation with joblib/ray
│   └── install_check.py ← Phase 2: automated install viability checker
├── tests/
│   └── test_mie.py      ← Mie scattering validation (PEC sphere/cylinder)
├── docs/
│   ├── scorecard.md     ← weighted evaluation scorecard (Phase 5)
│   └── decisions.md     ← architecture decisions log
└── README.md
```

## Phase Tracker

| Phase | Description | Status | Owner |
|---|---|---|---|
| 1 | Canonical 60 GHz Benchmark Definition | **COMPLETE** — `config/benchmarks.yaml` | Agent |
| 2 | Desk Research + Install Viability (Arch/Windows) | **COMPLETE** — `scripts/install_check.py` + `reports/` | Agent |
| 3 | Hello Phantom Mini-Domain Prototype (top 3 tools) | **COMPLETE** — gprMax: PEC 26.5s, water 147s, random 85.7s | Agent |
| 4 | Scalability & Modularity Stress Test | **PARTIAL** — Fast batch generator DONE; gprMax-vs-analytical validation + full-domain scaling PENDING | Agent |
| 5 | Recommendation & AI Pipeline Blueprint | **IN PROGRESS** — scorecard skeleton + torch loader done; scoring pending | Agent |

## Dataset Generation (2 tracks)

| Track | Tool | Speed | Use |
|---|---|---|---|
| Bulk | `scripts/batch_generate.py` (analytical Born point-cloud, SFCW multistatic ring) | ~2.2 samples/s core; 1000 in ~1 min with -j 8 | Pre-training data, 1k-100k samples |
| Full-wave | gprMax FDTD (`scripts/run_benchmark.py` + notebooks) | 26-150s/sample | High-fidelity truth, fine-tuning, validation |

Both write the same HDF5 layout documented in `datasets/torch_loader.py`.

## Phase 3 Results (2026-09-10)

| Benchmark | Wall time | Output | Notes |
|---|---|---|---|
| PEC cylinder (r=5mm, 60 GHz gaussian) | 26.5s | 85KB | Mie ref: S21=+4.40dB; grid-valid |
| Water vial (r=15mm, Debye, 10 GHz) | 147s | 255KB | 10 GHz used for 0.5mm grid stability |
| Random spheres (seed=42, 10 GHz sine) | 85.7s | 255KB | Domain randomization working |

Credentialed gprMax findings (full details in `docs/decisions.md`):
- **Comments must be `##`** — single `#` is parsed as a command.
- **Grid check at maxfreq = 4x center** — 60 GHz + eps>1.5 fails 0.5mm grid. 60 GHz needs dx~0.33mm; water needs 0.15mm.
- `#material: er se mr sm name` — permeability (mr) must be >= 1.
- Debye: `#add_dispersion_debye: poles d_er tau_seconds name`.
- No negative coords; everything in [0, domain].
- Water eps_eff ~19 @ 60 GHz, 20-30 dB/cm loss → reflection/SAR only.

## Evaluation Criteria (Weighted)

| Criterion | Weight | Description |
|---|---|---|
| 60 GHz Fidelity | 25% | PEC, dispersive/lossy dielectrics at 57-64 GHz, correct loss, antenna model, S-param extraction |
| Throughput & Scalability | 25% | GPU (CUDA+ROCm), MPI/cluster, samples/hr on weak (RTX 4060) vs strong (A100) |
| Dataset Pipeline | 20% | Pythonic sweep, HDF5 export, PyTorch DataLoader, domain randomization |
| Cost & Portability | 15% | 0-cost, Arch+Windows, build complexity |
| UWB vs CW Versatility | 15% | Both 60 GHz CW and 5-60 GHz UWB without full rebuild |

Validation: Error vs analytical Mie scattering <3 dB / 10 deg for S21 mag/phase.

## Candidate Tools (Ranked by Priority)

### Tier 1 — GPU FDTD (Primary)
1. **gprMax** — FDTD, Python API, CUDA+OpenCL (AMD), dispersive Debye, proven GPR/breast datasets. Fastest batch. Risk: PML stability, staircasing.
2. **openEMS** — FDTD, python-openEMS, MPI, mmWave patch antenna arrays at 60 GHz. Less Pythonic batch. Arch AUR native.
3. **MEEP** (MIT) — FDTD, Python, subpixel smoothing, best dispersion control. CPU+MPI only (no GPU). Use as accuracy reference.

### Tier 2 — AI-Native / Frequency-Domain
4. **TorchFDTD/jaxfdtd** — Pure Python/PyTorch/JAX FDTD, CUDA-native, differentiable, batchable. Validate vs MEEP.
5. **FEniCSx** (FEM) — Frequency-domain Helmholtz at 60 GHz. 1 solve vs 20k FDTD steps → 10-100x faster for CW.

### Tier 3 — Fast Approximate / Hybrid
6. **Sionna RT** (NVIDIA) — Ray tracing, CUDA, 1000x faster. Coarse pre-training → fine-tune with full-wave.

## Sparsity Study (2026-09-12) — transceiver-count feasibility

Realignment: Phases 3/4/5 (Mie/FDTD/RX-pipeline scorecards) are shelved in
favor of one product question: *can a sparse 60 GHz transceiver rig map a
produce surface contour in a 10 cm PoC chamber?*

**Single seam**: `scripts/sparsity_study.py --static --scan --chamber 50cm
--n-samples 3 --n-freq 64 --grid 32 --n-stations 12 --seed 1` →
`reports/sparsity_verdict.{md,json}`. 3D view: `./launch_gui.sh --scene 42
--transceivers 8`.

Delivered per ticket (all tests green: `pytest tests/`, 46 tests):
- **Math**: full derivation + motivations in `docs/math-signal-model.md`
  (SFCW surface-scatterer forward = delay-phase sum; DAS = its adjoint;
  range resolution ~2.1 cm at 7 GHz bandwidth; fits are honest, trust-flagged).
- `phantoms/veggie.py` — seeded produce library (ellipsoid / rounded-cylinder /
  pepper-with-stem + optional pedestal); surface scatterers, closed mesh, voxels.
- `sim/forward_surface.py` + `sim/transceivers.py` — reflection-mode SFCW
  57–64 GHz (two-way TOF, illumination culling, diffuse+specular return,
  direct coupling + metal-wall echo), static ring + turntable-scan layouts;
  HDF5 matches `datasets/torch_loader.py`.
- `recon/das.py` — delay-and-sum surface cloud (threshold + top-M).
- `metrics/contour_metrics.py` — chamfer (mean/median/max), %≤1cm, detect_feature.
- `gui/main.py` — scene view (seed or `.npy`, transceiver ring, recon overlay).

**T7 cross-validation (`reports/analytical_vs_gprmax.md`): the spec gate is
NOT EXECUTABLE on this hardware.** 60 GHz full-wave FDTD needs dx ≈ 0.17 mm
(~27M cells, CPU-infeasible); scaled-carrier (10 GHz) FDTD is not
representative — the 50 mm box and every probe are sub-wavelength resonators
(Mie ka≈0.84 sphere, 0.67 λ patch slab), which a first-order point-scatterer
model rightly does not match. Per the spec fallback, **the study numbers are
flagged UNTRUSTED until a GPU/60 GHz cross-validation exists** — they are a
self-consistent relative comparison only. Exact two-way-delay physics IS
validated in `tests/test_forward_surface.py`.

## Hardware Assumptions

- **Weak**: RX 5700 XT 8GB or RTX 4060, 32GB RAM — FEniCSx CW + downsampled FDTD (dx 0.5mm)
- **Strong**: A100/7900XTX, 64GB+ RAM — full FDTD dx 0.2mm
- **Cluster**: MPI + joblib/ray for distributed generation
- No hardware committed until simulation validates solver throughput.

## How to Resume This Project

Any model picking up work here should:

1. **Read this AGENTS.md** — full context in one file.
2. **Check `docs/decisions.md`** — log of all architecture decisions made so far.
3. **Check `config/benchmarks.yaml`** — canonical benchmark definitions (created in Phase 1).
4. **Query GraphRAG** — `graphrag_query("MicroTomo microwave tomography simulation")` for accumulated project knowledge.
5. **Check Phase Tracker** above — continue from the first "pending" phase.
6. **Store new decisions** via `graphrag_store()` — keep the knowledge graph current.

## Commands

```bash
# Check installed tools
python3 scripts/install_check.py

# Run canonical benchmark
python3 scripts/run_benchmark.py --tool gprmax --phantom pec_cylinder --freq 60e9

# Generate dataset batch
python3 scripts/batch_generate.py --samples 100 --tool gprmax --output datasets/h5/

# Run Mie validation
python3 -m pytest tests/test_mie.py -v

# Switch RAG mode (if using LightRAG)
# ./start-opencode-new.sh rag local    # private, no rate limits
# ./start-opencode-new.sh rag hybrid   # faster, uses cloud LLM
```
