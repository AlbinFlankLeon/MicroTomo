# MicroTomo — Transceiver Sparsity Feasibility Study (R1 spec)

> Feature slug: `sparsity-feasibility` — realigned from the original 60 GHz internal-imaging plan.
> Status: **draft v2 for user approval** (GUI + 3D visualization added) · Date: 2026-09-11 · Source: GRILL (inline run, sub-agents offline)

---

## Problem Statement

The user wants to build a homemade 60 GHz microwave box that 3D-models the **surface contour** of vegetables,
but will only buy **2–8 transceivers** (a very sparse aperture). High-cost, high-complexity arrays are out of
reach, so the real question is: *with how few transceivers, and with what mechanical motion, can we still get
~1 cm contour accuracy in a 50×50×50 cm box?*

The project docs (`AGENTS.md`, `decisions.md`, `scorecard.md`) were built for a different goal — dense antenna
**ring**, **internal dielectric imaging**, 150–200 mm ROI, ML reconstruction. That plan is wrong for this user.
The simulation must be realigned to answer the sparsity-vs-cost/complexity question **before any hardware is ordered**.

## Solution

A modular simulation + metrics stack that, from a single study entry point:

1. **Generates randomized produce-like targets** (elliptical potato/apple/cucumber shapes; mostly-water
   dielectric; optional low-permittivity pedestal) — a phantom/structure generator tool.
2. **Models 60 GHz reflection-only sensing** with a fast analytical surface-scatterer forward model
   (multistatic SFCW around 60 GHz, two-way time-of-flight, spreading + specular-ish echo amplitude,
   wall echo, noise), usable at **any chamber size up to 50 cm³** in seconds per scene.
3. **Cross-validates the analytical model against gprMax full-wave** FDTD inside a reduced **10×10×10 cm
   proof-of-concept box** (the largest full-wave box this PC can run in minutes).
4. **Reconstructs the surface contour with delay-and-sum beamforming** (`recon/das.py`) — no ML in this stack.
5. **Sweeps transceiver count × mounting** against a fixed phantom registry: static mounts vs a simple
   mechanical scan (turntable/rail — one motor is far cheaper than two RF channels).
6. **Emits a verdict report**: contour error per configuration, a cost/complexity rating per configuration,
   and a recommendation (green/yellow/red) on what to order.

The study measures feasibility; it does not assume it.

---

## User Stories

1. As a hobbyist, I want a tool that generates vegetable-like 3D shapes inside the simulation, so I can test
   whether the system would even see them.
2. As a hobbyist, I want the generator to randomize shapes/sizes/tilt per seed, so one run explores many
   "what if" targets instead of one hand-made object.
3. As a hobbyist, I want to simulate a 50×50×50 cm chamber with only 2 transceivers and see the resulting
   surface contour, so I know the worst case before buying anything.
4. As a hobbyist, I want the same scene simulated with 3, 4, 6 and 8 transceivers, so I can watch how the
   contour quality improves with each additional sensor.
5. As a hobbyist, I want to compare "8 bolted transceivers" against "2 transceivers on a turntable", so I
   can trade a cheap motor against expensive RF hardware.
6. As a hobbyist, I want a delay-and-sum reconstruction that turns raw echoes into a 3D point cloud of the
   surface, so I can see what is detectable without any machine learning.
7. As a hobbyist, I want the reconstruction compared against the true surface with an error number in
   centimeters (~1 cm target), so I have a verdict, not vibes.
8. As a hobbyist, I want a report that lists each transceiver configuration with its error, its hardware
   cost/complexity, and a green/yellow/red verdict, so I can decide what to order.
9. As a hobbyist, I want the fast analytical model checked against the gprMax full-wave solver on the same
   phantom, so I can trust the fast results that drive the study.
10. As a hobbyist, I want the proof-of-concept carried out in a 10×10×10 cm box so full-wave runs finish in
    minutes on this PC, while all scaling logic stays valid for the 50 cm chamber.
11. As a hobbyist, I want everything seeded and headless, so I can rerun the whole study overnight and get
    the same report.
12. As a hobbyist, I want a stress configuration with metallic (reflective) chamber walls, so I know whether
    multipath clutter could wreck the contour reading.
13. As a hobbyist, I want per-frequency phase-history data in the existing HDF5 layout, so future ML or
    other reconstructors can reuse the same data without re-simulation (even though ML is out of this stack).
14. As a hobbyist, I want known-size test objects (e.g. a 1 cm sphere / 1 cm notch) in the registry, so the
    ~1 cm resolution claim is checked head-on.
15. As a hobbyist, I want a desktop GUI with a 3D viewport showing the chamber, transceiver positions, the
    phantom, and the reconstructed contour on top of it, so I can *see* what the sim produces instead of
    reading CSV columns.
16. As a hobbyist, I want to pick a configuration (transceiver count, static vs scan, chamber size) and run a
    single scene from the GUI, so I can poke at the simulation interactively before trusting batch runs.
17. As a hobbyist, I want an A-scan/echo view per channel in the GUI, so I can sanity-check the raw signals
    feeding the reconstruction.
18. As a hobbyist, I want the GUI based on an existing, proven open-source setup rather than a bespoke
    windowing first draft, so I'm not reinventing a radar-data viewer. (GitHub research done — see
    Implementation Decisions → GUI base.)

---

## Implementation Decisions

### Realigned scope (replaces the old docs' assumptions)
- **Frequency: 60 GHz, fixed.** Band swept 57–64 GHz (7 GHz) as stepped-frequency CW. Rationale: user decision; λ=5 mm supports cm-scale contour features; water-rich produce are near-mirrors at 60 GHz (strong surface echo), which is *favorable* for contour-only imaging.
- **Imaging mode: reflection only. Internal/dielectric imaging is dropped.** The reconstruction target is the **surface** (contour), not an ε-volume. `geometry` = surface voxels/point cloud; `eps` = body shell value (kept for HDF5 compatibility).
- **No ML in this stack.** `torch_loader.py` stays for future data reuse, but no training pipeline is built.
- **Chamber: 50×50×50 cm target, 10×10×10 cm PoC.** Domain configs added to `config/benchmarks.yaml` (see below).
- **Transceivers: 2–8, abstract.** Point omnidirectional Tx/Rx with an optional beamwidth knob. No specific chip hardwired; the cost column keeps candidate modules in mind (Acconeer A121, Infineon BGT60LTR11AIP, TI IWR family) qualitatively.
- **Two mounting modes, both studied**: (a) **static** fixed positions in the box, (b) **scanned** — the N transceivers on a rotating platform (turntable) producing many virtual look-angles = synthetic aperture.

### Physics reality (written into tests and report, not papered over)
- **Full-wave 50 cm³ is impossible on this PC** (≈3.4 G cells @ λ/2, >100 GB RAM). Hence the two-track quality ladder: fast analytical at full size + full-wave truth at PoC size.
- **Grid floor @ 60 GHz:** dx ≥ ~2.5 mm is the hard Nyquist limit (λ/2 ≈ 2.5 mm in air); fidelity target 0.5–1 mm. "Lower quality" means coarser grid/simpler scatterer sets — never below the floor.
- **Range-resolution floor:** with 7 GHz bandwidth, FMCW range resolution ≈ c/(2B) ≈ **2.1 cm**. Cross-range resolution comes from aperture/synthetic aperture and can beat 1 cm. The "~1 cm" target is therefore at the edge for range and plausible for cross-range — **the study measures the achieved contour error instead of assuming it**.
- **Walls:** absorbing (anechoic) baseline; `--metal-walls` stress runs add specular wall echoes.

### Existing assets reused (do not rewrite from scratch)
- `scripts/batch_generate.py` — analytical forward-model pattern (Born point-cloud, SFCW, HDF5 writer) is **adapted** to reflection-mode surface scatterers. Keep HDF5 layout: `/samples/NN/{x, x_hat, y}` complex64 (F,T), `{geometry, eps}` float32 (H,W,D), `meta`.
- `phantoms/generator.py` — extended into the structure generator (veggie-like library + seeded randomization), emitting both the analytical scatterer set *and* gprMax input.
- `scripts/run_benchmark.py` + `tests/test_mie.py` — Mie/PEC ground check stays as the gprMax sanity anchor.
- `config/benchmarks.yaml` — add `poc_10cm` and `chamber_50cm` domain definitions; add `veggie_library` phantom definitions.

### New modules
- `scripts/sparsity_study.py` — **the single seam / study entry point.** CLI:
  `--transceivers 2,3,4,6,8 --static|--scan --chamber 10cm|50cm --phantom veggie_library --seed N --metal-walls`
  Runs: generate → forward → reconstruct → metrics → verdict table. Headless, seeded, self-contained, writes `reports/sparsity_verdict.md` (+ `.json`).
- `gui/` — PyQt6 + PyVista/pyvistaqt desktop app (pattern from GPRForce, code written fresh):
  - 3D viewport: chamber wireframe, transceiver positions + scan path, phantom mesh, reconstructed contour.
  - A-scan/echo viewer per channel; HDF5 scene loader; single-scene run button bound to the study modules.
  - `gui/main.py` entry point.
- `recon/das.py` — delay-and-sum beamformer: back-projects multistatic phase history onto a voxel/point grid; output = surface point cloud + occupancy grid.
- `metrics/contour_metrics.py` — Chamfer distance (mean/median/max), % of true surface within 1 cm, and known-feature detection checks (1 cm sphere/notch test objects).
- `sim/gprmax/` — PoC validation harness (adapted from `run_benchmark.py`): runs shared phantoms in the 10 cm box, compares analytical vs gprMax echoes.

### Validation / seam gate
- **The one testing seam** = `sparsity_study.py` end-to-end on a trivial config: 2 static transceivers × 2 seeded samples in the 10 cm chamber → verdict table + metric unit tests pass. This exercises every module through one CLI.
- gprMax-vs-analytical agreement threshold (PoC box): mean envelope correlation ≥ 0.9 and peak range error ≤ 0.5 cm on ≥1 shared phantom. If it fails, the 50 cm analytical numbers are flagged untrusted in the report (and the cause investigated).

### GUI base — GitHub research (2026-09-11)
Three existing setups found and evaluated:

| Repo | Stack | What it does | License | Verdict |
|---|---|---|---|---|
| `YinchuanLi05/GPRForce` (48★) | PyQt6 + **PyVista/pyvistaqt** + h5py + scipy/matplotlib | End-to-end desktop platform for gprMax data: parses `.in` models, loads `.out`/`.npy` echoes, **3D geometry preview**, 2D material view, A-scan/B-scan views, parameterized processing pipeline (DC removal, time-zero, dewow, gain, filtering) | **No license file** (all-rights-reserved by default) | **Architecture blueprint** — right stack and module layout, but code must NOT be copied wholesale; clean reimplementation of the pattern |
| `tomsiwek/gprMax-Designer` (24★) | PyQt5/matplotlib | GUI to *draw/build* gprMax models (boxes, cylinders, polygons) in a 2D model view, material assignment | GPL-3.0 | Good idea source for interactive geometry editing, but GPL copyleft would infect the project; 2D-only, no 3D viewport |
| `thliebig/AppCSXCAD` (openEMS, 11★) | C++/Qt/VTK | 3D geometry editor for openEMS FDTD models | GPL (family) | Powerful 3D CAD but C++, overkill, GPL |

**Decision:** build a **MicroTomo GUI (`gui/`)** as a clean PyQt6 + PyVista/pyvistaqt app, **architecturally copied from GPRForce's proven module split** (main window / 3D viewport / A-scan view / controls / IO / processing), with **all code written fresh** — because GPRForce is unlicensed, GPL base is undesirable, and fresh code also matches our HDF5/dataset formats exactly. The 3D viewport renders: chamber wireframe, transceiver positions (+ scan path), phantom mesh, and the reconstructed contour point cloud. It reads the same HDF5 layout as `torch_loader.py`, so every artifact the study writes is viewable. gprMax-Designer's "draw a shape → becomes a phantom" interaction is a stretch goal, not a v1 requirement.
- v1 GUI seams: open a generated HDF5 scene → 3D viewport + A-scans; single-scene run button wired to the study modules.
- Corollary: the `sparsity_study.py` CLI remains the headless truth; the GUI is a viewer + one-scene runner, not a second implementation of the physics.

---

## Metrics & Acceptance Criteria

1. **Analytical-vs-gprMax gate** (PoC box): passes as defined above → unlocks the 50 cm study numbers.
2. **Sparsity sweep runs headless**: `< ~1 h` overnight for the full matrix, seconds per scene for the analytical model.
3. **Verdict table** (`reports/sparsity_verdict.md`): contour error (mean/median/max cm, % within 1 cm) per configuration rows `{2,3,4,6,8} × {static, scan}`, columns for cost/complexity (qualitative) and green/yellow/red.
4. **Known-feature check**: a 1 cm sphere in the registry is detected (reconstruction peak within 1.5 cm of truth) at the recommended configuration minimum.
5. **Determinism**: `--seed` reruns reproduce identical reports.
6. **Unit tests** (`tests/`): Chamfer computation against hand-made ground truth; DAS sanity (single small sphere → echo peak back-projects to true position); phantom generator determinism and shape bounds.

---

## Out of Scope (this iteration)
- ML training / learned reconstruction (explicitly dropped by the user for this stack; `torch_loader.py` remains data-compatible for later).
- Internal/dielectric imaging and transmission mode.
- Sionna RT, FEniCSx, TorchFDTD integration.
- Any hardware procurement or real-measurement campaign.
- Full-wave FDTD of the 50 cm chamber on this PC.
- Windows port; non-seeded "production" (study) runs.

## Open Questions / Risks
- **The 1 cm target vs the 2.1 cm range-resolution floor** — the study's core question; the verdict may legitimately be "contour error ≈ 2–3 cm in range unless bandwidth or aperture grows." Explicitly handled as a measured output, not hidden.
- **Wall/echo realism at 60 GHz** — absorbing-wall baseline may understate multipath; `--metal-walls` stress runs bound it.
- **Antenna pattern realism** — point-omnidirectional model is optimistic; beamwidth knob lets later iterations degrade it.
- **Sub-agent infra** — engineer-workflow sub-agents were offline (model-registry mismatch, fixed in agent files for next session); stages GRILL/SPEC were run inline. BUILD will retry sub-agents first.

---

## Deliverables
1. This spec + tickets under `.scratch/sparsity-feasibility/`.
2. Extended `phantoms/generator.py` (veggie library + seeds).
3. Reflected-mode analytical forward model (adapted in `batch_generate.py` or a dedicated module).
4. `recon/das.py` delay-and-sum reconstruction.
5. `metrics/contour_metrics.py` + unit tests.
6. `scripts/sparsity_study.py` (the seam) + `config/benchmarks.yaml` updates.
7. PoC gprMax validation harness + results in `reports/`.
8. `reports/sparsity_verdict.md` + `.json` (the user-facing answer).
9. `gui/` desktop GUI with 3D viewport (chamber + transceivers + phantom + contour), A-scan views, and a
   single-scene run button (PyQt6 + PyVista, fresh code on GPRForce's proven architecture).