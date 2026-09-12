# MicroTomo GUI UX Redesign + Viewer Reliability — Spec

Slug: `gui-ux-redesign` — v0.2.0

## Problem (user-reported)
1. Interactive 3D viewer "not working at all" on the user's desktop (symptom
   category unresolved — see Open Seams).
2. Settings are confusing: cryptic labels (`eps / rotz`, `semi a`), no units,
   no explanation of what each option means.
3. No one-click, pre-run simulation configurations (presets).

## Goals
- **G1 Viewer reliability + self-diagnosis:** the QtInteractor viewport must
  (a) enable explicit trackball interaction, (b) take keyboard/mouse focus at
  startup, (c) not silently fail — wrap creation in try/except with a visible
  error banner, (d) expose a one-click "Viewer diagnostics" button that dumps
  GL vendor/renderer/version + actor list so an unreproducible environment bug
  becomes diagnosable from the user's machine in 5 seconds.
- **G2 Intuitive parameters:** every parameter gets (from a single metadata
  source, `gui/param_meta.py`): human label, unit shown in the UI, one-line
  help tooltip, sensible ranges. Sections are ordered and numbered
  (1 preset, 2 chamber, 3 objects, 4 transceivers, 5 advanced, 6 reconstruction
  & run). Advanced-only params (snr, seed, scatter density, freq count) live
  behind a "show advanced" toggle.
- **G3 Simulation presets** (`gui/presets.py`): Quick test / Standard / High
  detail / Max quality. Selecting one sets grid, freq count, scatter density,
  tx count/scheme, SNR, and shows a description + expected runtime/quality so
  the user knows what they're buying before running.

## Out of scope
- Changing the physics pipeline (`sim/*`, `recon/*`, `metrics/*`) — 45 tests
  keep them pinned. Presets only set existing SimState fields.
- Windows/macOS installers.

## Non-goals / constraints
- Headless dev machine has NO GL: viewer interaction cannot be automated here;
  verification happens via (a) unit tests on the viewer-config helper with a
  stub plotter, (b) the user launching `./launch_gui.sh` and reporting.
- Keep `SimState` serialization unchanged (compat with saved configs).

## Acceptance
- A1: every SimState field surfaced in the UI has meta (label/unit/help) in
  param_meta; unit-visible label renders unit text; advanced params hidden
  until toggled.
- A2: presets: 4 presets, each fully valid (0<grid≤64, freq 4..128, positive
  counts), each carries human description + expected-runtime string; applying
  a preset sets the state and the UI reflects it.
- A3: viewer: `configure_viewer(plotter)` helper called on startup applies
  trackball + focus; creation failure shows a banner (not a black void); a
  Diagnostics button exists and prints a diagnostics block.
- A4: full suite passes (45 existing + new tests); release rebuilt to v0.2.0.

## Open seams (user gates surfaced, not blocking — build proceeded)
- Exact viewer symptom (crash / black / no-mouse): Diagnostics button + banner
  make the next report decisive; interaction enabled explicitly either way.
- Preset count/expected-runtime wording: defaults chosen, adjustable later.