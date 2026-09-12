# MicroTomo — Architecture Decisions Log

## 2026-09-12 — Signal model & reconstruction lock-in

- **Forward = SFCW surface-scatterer phase sum** (see `docs/math-signal-model.md`):
  `y[f,ch] = Σ_p γ_p/(d_i d_j) · (0.3+0.7 n̂·b̂) · e^{−j2πfτ}`, τ = two-way delay.
  Chosen because the science target is a *surface contour* and full-wave FDTD
  at 60 GHz is CPU-infeasible (T7); delay is carrier-independent physics.
- **Reconstruction baseline = DAS / matched filter**, the exact adjoint `I = H†y`
  of the forward operator. Parameter-free null model; the torch_loader HDF5 is
  the training bus for a future learned inverse. Range resolution is
  `c/(2B) ≈ 2.1 cm`; 1 cm verdict claims are grid/resolution-limited, not an
  algorithm property.
- **Background is modelled** (direct coupling, monostatic self, optional
  nearest-face image bounce) and `x − x̂ = y` exactly — the scattered channel is
  the DAS input; leakage handling is testable, not abstracted away.
- **Metrics**: two-sided chamfer (median = verdict), % ≤ 1 cm, detect_feature
  probe. Empty clouds never crash (`inf`/0.0).

## 2026-09-12 — Sparsity study realignment + T7 trust flag

- **Product realignment**: gold-standard Mie/FDTD scorecards (Phases 3-5)
  shelved. One question now drives the work: *can a sparse 60 GHz rig map a
  produce surface contour in a 10 cm PoC?* Tickets live in
  `.scratch/sparsity-study/`; the single seam is `scripts/sparsity_study.py`.
- **Forward model choice**: reflection-mode SFCW point-surface scatterer
  (two-way TOF + illumination culling + diffuse/specular return) written to
  the existing torch_loader HDF5 layout. NOT a full-wave solver.
- **Physics gate honesty**: the spec required a gprMax cross-validation gate
  (envelope correlation ≥ 0.9, peak range error ≤ 0.5 cm). It is **NOT
  EXECUTABLE on this CPU**: 60 GHz forces dx ≤ 0.17 mm (~27M cells), and the
  10 GHz scaled fallback is unrepresentative (50 mm box ≈ 1.7 λ; every probe
  sub-wavelength → resonance-dominated, which the single-reflection model
  rightly doesn't match). Decision: do NOT fake it — flag the analytical
  numbers **UNTRUSTED** in the verdict report until a GPU/60 GHz check exists.
- **Verdict report contract**: `reports/sparsity_verdict.{md,json}` carries a
  trust flag + grid-floor caveat; green ≤ 1.5 cm / yellow ≤ 3 cm median
  chamfer; cost = 1 pt/RF channel (+motor for scan), complexity +2 for scan.

> Record every significant design choice here. Reverse chronological.

---

## 2026-09-10 — Phase 4: Dataset generation strategy DECIDED

- **Hybrid two-track generation** (matches plan's tiers):
  1. **`scripts/batch_generate.py`** — FAST analytical forward model: multistatic SFCW ring array (N_antennas annular geometry), first-Born point-cloud scatterers (Clausius-Mossotti contrast), two-way delay phasing, 1/(d_tx*d_rx) spreading, direct-coupling background + enclosure echo, additive noise (--snr-db). **~2.2 samples/s single-core, 1000 samples in ~1 min with -j 8.** Bulk pre-training data.
  2. **gprMax FDTD** — sparse high-fidelity examples (26-150s/sample). Fine-tuning + validation. Full-wave truth for the analytical model.
- **HDF5 layout** (documented in `datasets/torch_loader.py`): /samples/NN/{x, x_hat, y} complex64 (F,T), {geometry, eps} float32 (H,W,D), meta JSON. x - x_hat == y guaranteed.
- **Rationale**: 10k samples via full FDTD at 26-150s each = 3-17 days CPU. Analytical model gives the plan's "coarse pre-train -> fine-tune with full-wave" strategy at 1000x speedup.
- **Next**: validate analytical model vs gprMax on shared phantom; scorecard filled for gprMax.

## 2026-09-10 — Phase 3: gprMax mini-domain prototype COMPLETE

- **gprMax v3.1.7 CONFIRMED viable** — all 3 mini-domain benchmarks ran (PEC cylinder 26.5s, water vial 147s, random 85.7s on CPU).
- **Grid resolution is THE bottleneck**: gprMax grid check uses maxfreq = 4x center freq. At 60 GHz, dielectrics need dx <= 0.33mm (water: 0.15mm) — exactly matching the plan's FDTD estimate. Full 200mm domain at 60 GHz = 216M cells / 100GB+ RAM, confirming need for fine-grid GPU or FEniCSx frequency-domain.
- **Validation strategy**: PEC cylinder at 60 GHz with gaussian (Mie reference S21=+4.4dB) is the canonical validation case. Water + random phantoms run at 10 GHz (control case) to stay on 0.5mm grid.
- **gprMax syntax gotchas** (documented for future runs):
  - Comments must be `##` (double hash). Single `#` = command.
  - `#material: er se mr sm name` — permeability MUST be >= 1, not 0.
  - Debye: `#add_dispersion_debye: poles d_er tau_s name` — tau in SECONDS.
  - No negative coordinates; everything in [0, domain] and on grid multiples.
- **Water @ 60 GHz**: eps_eff ~19, attenuation 20-30 dB/cm → reflection/SAR mode confirmed (transmission fails).
- **Next**: Phase 4 scalability (full domain + GPU/OpenCL test), Phase 5 scorecard + AI pipeline blueprint.

## 2026-09-10 — Project Kickoff

- **Simulation-first**: No hardware commitment until solver throughput + accuracy validated.
- **Tool tiers**: Tier 1 = GPU FDTD (gprMax/openEMS), Tier 2 = AI-native/FEM (TorchFDTD/FEniCSx), Tier 3 = RT augmentation (Sionna).
- **Primary target**: 60 GHz reflection/SAR tomography with air coupling.
- **Dataset format**: HDF5 for S-params + field snapshots + phantom metadata, PyTorch DataLoader integration.
- **Platform**: Arch Linux primary, Windows secondary. 0-cost license stack.
- **Expected hybrid stack**: gprMax/TorchFDTD (UWB GPU FDTD) + FEniCSx (fast CW sweep) + Sionna RT (augmentation).
- **Validation baseline**: Mie scattering analytical solution (PEC sphere/cylinder) <3dB/10deg S21.
