# MicroTomo — Signal Model & Reconstruction Math

Canonical explainer for what runs in the background when you press **Run
simulation** in the editor. References the exact code that implements it.
Companion to `docs/decisions.md` (2026-09-12 ADR) and the T7 trust flag in
`reports/analytical_vs_gprmax.md`.

---

## 0. The pipeline at a glance

```
edit scene ──► scatterers (P,6) ──► forward model ──► y (F, T²)
                                      + background x̂ ──► x = x̂ + y + noise
                                                     │
                                       DAS adjoint ───┼──► I(q) on voxel grid
                                                     └──► top-M ──► recon cloud (N,3)
                                          │
                       chamfer / %≤1cm ────┘   (metrics vs the true surface)
```

Modules: `sim/forward_surface.py`, `recon/das.py`, `phantoms/veggie.py`,
`metrics/contour_metrics.py`, orchestrated by `gui/state.py`.

---

## 1. Forward model — SFCW surface-scatterer response

Shot: **57–64 GHz**, `F` stepped frequencies `f` (48 by default), `T`
transceivers → `T²` channels, channel index `ch = tx·T + rx`.

Each surface point `p` (position `r_p`, outward unit normal `n̂_p`) contributes
to channel `(i,j)` a phasor

```
y[f, ch] = Σ_p γ_p · (1 / d_i·d_j) · v_p(i,j) · exp(−j 2πf τ)
```

where `d_i = |r_p − r_i|`, `d_j = |r_p − r_j|`, and the round-trip delay

```
τ = (d_i + d_j) / c
```

is the **only physical quantity that enters the phase** — SFCW measures the
range phase at each frequency step; the scatterer is a delay tap. This makes
the model exactly linear in the scene and identical in structure to the radar's
receiver: phases accrue from time of flight, not from RCS amplitude tables.

**Terms, in code order** (`forward_surface.py:80-116`):
- **Spherical spreading** `1/(d_i·d_j)` — two free-space `1/d` Green's-function
  legs (Tx→√p and p→Rx). Simple, closed form, no near-field correction.
- **Illumination culling** `n̂_p · û_i > 0` — the back side of the produce never
  re-radiates toward the Tx (no shadow/specular transmission; roughness makes
  forward scatter negligible at 60 GHz).
- **Visibility / return strength** `v = 0.3 + 0.7 · clip(n̂_p · b̂, 0, 1)`, with
  `b̂` = normalized bistatic bisector `û_i + û_j`. A diffuse floor of 0.3
  (Lambert-ish for rough produce) plus a specular highlight peaking when the
  normal bisects the Tx→p→Rx angle. The 0.3/0.7 split is an empirical balance,
  **not** a material-fidelity claim (see trust flag).
- **Contrast weight** `γ_p` — default 0.9 for every point (representative
  Fresnel reflection of wet vegetable surface at 60 GHz; 1.0 for PEC). Set on
  the scatterer; `SurfaceScatterModel.gamma`.

**Background `x̂`** (what a real radar sees even with nothing there):
- Direct Tx→Rx coupling `DIRECT_AMP · exp(−j 2πf d/c)` (amplitude 4.0).
- Monostatic self-echo `SELF_AMP = 1.5` when `tx == rx`.
- Metal-wall mode: nearest-face **image-receiver** single bounce,
  `WALL_AMP · WALL_REFL · exp(−j 2πf d_img/c)/d_img` (image-source method,
  `_wall_term`). Default chamber walls are absorbing → no wall term.

**Reading** `x = x̂ + y + complex-Gaussian noise` with
`σ = 10^(−SNR/20)` relative to the RMS of the (normalised) scattered field
(30 dB default). The layout guarantees `x − x̂ ≡ y` exactly — the scattered
channel is the quantity DAS consumes.

### Design motivations
- **Surface, not volume.** The science question is a surface *contour*; a
  volumetric full-wave solve is out of reach at 60 GHz on CPU (T7: dx ≤
  0.17 mm ≈ 27M cells) and scaled carriers are not representative. The scatterer
  model is `O(P·T²·F)` — seconds, deterministic, and the code spits the exact
  `torch_loader` HDF5 layout so the same bus feeds a future learned
  reconstruction.
- **Delay-driven phases.** Delay is carrier-independent *physics* for air/PEC;
  all apparent frequency content comes from the two-way delay taps, which makes
  the adjoint (DAS) the natural inverse.
- **Coupling is modelled, not ignored.** Real SFCW radars fight Tx→Rx leakage;
  modelling it (and shipping `x̂`) lets the pipeline test "can the scattered
  channel be recovered" honestly — and `x − x̂ = y` is asserted in tests.

---

## 2. Reconstruction — Delay-And-Sum (DAS), i.e. the adjoint

A voxel grid `G³` (20 default, spinner in the GUI) covers the chamber at
`q = (k+½)·L/G`. For each voxel and each channel the matched filter scores

```
I(q) = Σ_ch Σ_f  y[f, ch] · exp(+j 2πf τ_ch(q))
```

with `τ_ch(q) = (|q − r_tx| + |q − r_rx|)/c` — the same two-way delay the
forward model used, now evaluated for a hypothetical scatterer *at q*.

Linearity makes this the **adjoint** of the forward operator:

| | forward | reconstruction |
|---|---|---|
| operator | `y = H σ` , `H[p,ch,f] = e^{−jωf τp}` | `I = H† y` |
| role | phases *away* from the scene | matched filter *back* into space |

Coherently summing the phases over all `T²` channels and `F` frequencies, the
exponential aligns where `τ_ch(q)` equals every measured `τ` at once — i.e. at
the true scatterer locations — and destructively cancels elsewhere. This is
SAR back-projection / beamforming; no model fitting is involved.

Output selection (`das.py:66-77`):
```
thr = quantile(|I|, 0.96–0.98)      # brightest (target) voxels only
recon = voxels with |I| ≥ thr (+ optional top-M)
```
A fixed quantile (not an absolute level) because the echo contrast varies with
scene fill and antenna count.

### Design motivations
- **Adjoint inverse.** Because the forward is a phase-delay sum, its adjoint is
  the matched filter — cheap, robust, and parameter-free. It is deliberately a
  "hand-crafted baseline" to beat with a learned inverse on the same HDF5 data.
- **Resolution honesty.** SFCW bandwidth 7 GHz gives range resolution
  `c/(2B) ≈ 2.1 cm`; the true surface is *denser* than `|I|` can resolve, so
  both metrics (chamfer vs the sampled truth) and the verdict carry a
  resolution caveat. Multistatic triangulation sharpens beyond the single-
  look resolution, but 1 cm accuracy claims are grid/resolution-limited, not
  algorithm-limited.
- **Multistatic divide-and-conquer.** Channel loop is `T²` independent phase
  history rows; increasing `T` or `G` costs linearly (each voxel×channel is
  independent) — the loop structure in `das.py:59-66`.

---

## 3. Scene geometry (`phantoms/veggie.py`)

- **Ellipsoid** (`x²/a² + y²/b² + z²/c² = 1`): sample unit directions `u`;
  radial function `r(u) = 1/√(u_x²/a² + u_y²/b² + u_z²/c²)`; point
  `p = r·(u⊙scale)`; outward normal along the gradient `∝ (u_x/a, u_y/b, u_z/c)`.
- **Rounded cylinder**: side shell + two hemisphere caps, points split by
  relative area, normals analytic.
- **Rigid transform** to world: `p = R p_local + center`, `n = R n_local`,
  rotation `R` from zyx Euler angles (the GUI's per-object "rotz").
- A `VeggieScene` is a union of these shapes (+ optional pedestal); a run
  samples ~1500 points per object on demand (`SimState.scatter()`).

---

## 4. Metrics (`metrics/contour_metrics.py`)

- **Two-sided chamfer** `1/2(mean NN(recon→truth) + mean NN(truth→recon))` via
  `cKDTree`; report mean / median / max. Median is the robust verdict number
  (sparse recon clouds have long tails).
- **% ≤ 1 cm** `frac_within_cm` — fraction of the true surface within 1 cm of
  the recon cloud; this backs the product's "surface within 1 cm" claim.
- **detect_feature** — probes whether a known-radius sphere reconstructs as a
  cluster at its true location (detection, not raw geometric accuracy).

---

## 5. What the editor adds (`gui/state.py`)

- Editable specs → fresh `VeggieShape`s → `scatter()` sample on demand (no stale
  caches).
- Transceivers: editable `(T,3)` table; presets fill ring / corners / random.
  In **scan** mode the physical table is rotated about the chamber z-axis by
  `2πk/n_stations` per station (mirroring `ScanLayout` but honouring edits) →
  `T·n_stations` virtual positions → `virtual_positions()`.
- `run_pipeline()` executes forward → DAS → chamfer → the panel's "median …
  %≤1cm" line; the view toggle shows true / recon / both.