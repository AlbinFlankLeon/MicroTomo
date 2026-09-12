# T6 — Sparsity study CLI + verdict report

- **Blocks:** none (final headless product)
- **Depends on:** T2, T3, T4, T5

## Goal
`scripts/sparsity_study.py` — the single seam from the spec: sweep `--transceivers 2,3,4,6,8` × `--static|--scan`, chambers 10 cm/50 cm, against the veggie registry; produce `reports/sparsity_verdict.md` (+ `.json`) with per-config contour error, cost/complexity, green/yellow/red verdict.

## Deliverables
- CLI as spec'd: `--transceivers 2,3,4,6,8 --static --scan --chamber 10cm|50cm --phantom veggie_library --seed N --metal-walls`.
- Cost/complexity row (qualitative per config: RF channel count, motor/stage for scan, sync complexity).
- Verdict logic: green = median error ≤ 1.5 cm; yellow ≤ 3 cm; red otherwise (initial thresholds; report notes them).
- Headless full matrix < ~1 h at 10 cm chamber; seconds/scene at 50 cm analytical.
- Deterministic per seed.

## Verification
- Tiny end-to-end run: `--transceivers 2 --static --chamber 10cm --n-samples 2 --seed 42` → verdict table + JSON in `reports/`.
- Everything through this CLI (spec's single seam).