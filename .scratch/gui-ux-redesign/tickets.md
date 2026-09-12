# Tickets — gui-ux-redesign

Frontier order; each ticket = test-first commit.

| # | Ticket | Blocked by | Notes |
|---|--------|-----------|-------|
| 01 | **param_meta.py**: `Param` dataclass (label, unit, help, advanced, decimals/step) + `PARAM_META` covering every panel-surfaced SimState field + object dims; tests: completeness, units, advanced subset | — | pure logic, no Qt |
| 02 | **presets.py**: `SimPreset` + 4 presets + `apply_preset(state,name)`; tests: validity ranges, state mutation, descriptions/runtime strings | 01 | pure logic |
| 03 | **panel.py**: numbered sections, unit-labeled rows from meta, tooltips, advanced toggle, preset combo (no preset → "Custom"), results styling | 01, 02 | Qt, but marshalable headless (existing tests) |
| 04 | **main_window.py**: `configure_viewer(plotter)` (trackball + focus), try/except banner, Diagnostics button → `viewer_diagnostics()`; tests w/ stub plotter | — | hardens viewer, makes env bugs reportable |
| 05 | **release**: bump VERSION→0.2.0, full suite, `build_release.sh`, re-upload assets, push | 03, 04 | |