#!/usr/bin/env python3
"""MicroTomo transceiver-sparsity feasibility study (T6) — the single seam.

Sweeps transceiver count x mounting mode against the veggie registry and
writes a verdict report: contour error per configuration + cost/complexity
+ green/yellow/red recommendation.

Usage:
    python scripts/sparsity_study.py --transceivers 2,3,4,6,8 --static --scan \
        --chamber 10cm --n-samples 2 --seed 42 --out reports
    python scripts/sparsity_study.py --transceivers 2 --static \
        --chamber 10cm --n-samples 1 --seed 42 --metal-walls

Pipeline per scene: veggie generate -> reflection forward -> DAS recon ->
contour metrics vs the true surface scatterers. All seeded & headless.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from sim.forward_surface import SurfaceScatterModel  # noqa: E402
from sim.transceivers import build_layout  # noqa: E402
from recon.das import das_beamform  # noqa: E402
from metrics.contour_metrics import chamfer, frac_within_cm  # noqa: E402

CHAMBER_DATA = {
    "10cm": 0.10,
    "50cm": 0.50,
}
F_MIN, F_MAX = 57.0, 64.0          # GHz
VERDICT_GREEN_CM = 1.5             # median chamfer
VERDICT_YELLOW_CM = 3.0

# Cost / complexity rubric (documented in the report)
COST_PER_TRX = 1.0                 # one RF channel point
COST_SCAN_SURCHARGE = 1.5          # motor + mechanics for the turntable
COMPLEXITY_PER_TRX = 1.0
COMPLEXITY_SCAN_SURCHARGE = 2.0    # motion control + pose synchronisation


@dataclass
class Config:
    n_trx: int
    mode: str
    chamber_m: float
    wall: str
    n_stations: int

    def cost_score(self) -> tuple[float, float]:
        c = self.n_trx * COST_PER_TRX + (COST_SCAN_SURCHARGE if self.mode == "scan" else 0.0)
        k = self.n_trx * COMPLEXITY_PER_TRX + (COMPLEXITY_SCAN_SURCHARGE if self.mode == "scan" else 0.0)
        return c, k

    def describe(self) -> str:
        return (f"{self.n_trx} transceivers, {'static' if self.mode == 'static' else 'turntable scan'
                + f' ({self.n_stations} stations)'}")


def run_config(cfg: Config, seed: int, n_freq: int, grid: int, n_scatter: int,
               rng_forward_seed: int) -> dict:
    """Simulate one config for one seeded scene; return per-scene metrics."""
    from phantoms.veggie import generate_scene

    scene = generate_scene(seed=seed, chamber="10cm" if cfg.chamber_m == 0.10 else "50cm",
                           n_shapes=1, pedestal=False)
    rng = np.random.default_rng(rng_forward_seed)
    pts6 = scene.surface_points(n_scatter, rng)
    truth = pts6[:, :3]                                   # true surface cloud
    scatter = pts6                                        # scatterers drive the echo

    layout = build_layout(cfg.mode, cfg.n_trx, cfg.chamber_m, n_stations=cfg.n_stations)
    positions = layout.positions
    freqs = np.linspace(F_MIN, F_MAX, n_freq) * 1e9

    model = SurfaceScatterModel(chamber_m=cfg.chamber_m, freqs_hz=freqs,
                                scatter=scatter, gamma=None, wall_mode=cfg.wall,
                                snr_db=30.0, seed=seed)
    _, _, y = model.phase_history(positions, write_noise=True)

    pts, _, _ = das_beamform(y, positions, freqs, cfg.chamber_m,
                             grid=grid, threshold_quantile=0.96, top_m=2000)
    ch = chamfer(pts, truth)
    coverage = frac_within_cm(pts, truth, threshold_cm=1.0)
    return {
        "n_points_recon": int(len(pts)),
        "chamfer_mean_cm": ch["mean"] * 100.0,
        "chamfer_median_cm": ch["median"] * 100.0,
        "chamfer_max_cm": ch["max"] * 100.0,
        "frac_within_1cm": float(coverage),
    }


def verdict(median_cm: float | None) -> str:
    if median_cm is None:
        return "n/a"
    if median_cm <= VERDICT_GREEN_CM:
        return "green"
    if median_cm <= VERDICT_YELLOW_CM:
        return "yellow"
    return "red"


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="MicroTomo sparsity feasibility study")
    p.add_argument("--transceivers", default="2,3,4,6,8", help="comma list")
    p.add_argument("--static", action="store_true", default=True, help=argparse.SUPPRESS)
    p.add_argument("--scan", action="store_true", help="include turntable-scan configs")
    p.add_argument("--chamber", choices=sorted(CHAMBER_DATA), default="10cm")
    p.add_argument("--metal-walls", action="store_true")
    p.add_argument("--n-samples", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n-freq", type=int, default=32)
    p.add_argument("--grid", type=int, default=20)
    p.add_argument("--n-scatter", type=int, default=1200)
    p.add_argument("--n-stations", type=int, default=6, help="scan stations (study budget)")
    p.add_argument("--out", type=str, default="reports", help="output dir")
    args = p.parse_args(argv)

    trxs = [int(x) for x in args.transceivers.replace(" ", "").split(",")]
    modes = ["static"] + (["scan"] if args.scan else [])
    chamber_m = CHAMBER_DATA[args.chamber]
    wall = "metal" if args.metal_walls else "absorbing"
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    t0 = time.time()
    for cfg_n in trxs:
        for mode in modes:
            cfg = Config(cfg_n, mode, chamber_m, wall, args.n_stations)
            scenes: list[dict] = []
            for s in range(args.n_samples):
                scenes.append(run_config(cfg, seed=args.seed + s, n_freq=args.n_freq,
                                         grid=args.grid, n_scatter=args.n_scatter,
                                         rng_forward_seed=args.seed + 100 * s))
            med = float(np.median([sc["chamfer_median_cm"] for sc in scenes]))
            cost, complexity = cfg.cost_score()
            row = {
                "n_transceivers": cfg.n_trx,
                "mode": cfg.mode,
                "n_stations": cfg.n_stations if mode == "scan" else 1,
                "chamfer_mean_cm": round(float(np.mean([sc["chamfer_mean_cm"] for sc in scenes])), 2),
                "chamfer_median_cm": round(med, 2),
                "chamfer_max_cm": round(float(np.max([sc["chamfer_max_cm"] for sc in scenes])), 2),
                "frac_within_1cm": round(float(np.mean([sc["frac_within_1cm"] for sc in scenes])), 3),
                "cost_score": round(cost, 1),
                "complexity_score": round(complexity, 1),
                "verdict": verdict(med),
            }
            rows.append(row)
            print(f"  {cfg.describe():>40s}  median {row['chamfer_median_cm']:5.2f} cm "
                  f"-> {row['verdict']}")

    report = build_report(args, rows, chamber_m, wall, time.time() - t0)
    md_path = out_dir / "sparsity_verdict.md"
    json_path = out_dir / "sparsity_verdict.json"
    md_path.write_text(report)
    json_path.write_text(json.dumps(
        {"meta": {"chamber": args.chamber, "seed": args.seed,
                  "n_samples": args.n_samples, "wall": wall,
                  "thresholds_cm": {"green": VERDICT_GREEN_CM, "yellow": VERDICT_YELLOW_CM}},
         "rows": rows}, indent=2))
    print(f"\nverdict report -> {md_path}")
    print(f"json data      -> {json_path}")
    return 0


def build_report(args, rows: list[dict], chamber_m: float, wall: str, elapsed: float) -> str:
    lines = [
        "# MicroTomo — Transceiver Sparsity Verdict",
        "",
        f"- Chamber: **{args.chamber}** ({chamber_m * 1000:.0f} mm cube) · wall: **{wall}**",
        f"- Seeds: `{args.seed}..{args.seed + args.n_samples - 1}` ({args.n_samples} samples/config)"
        f" · freq {F_MIN}-{F_MAX} GHz ({args.n_freq} steps) · grid {args.grid}",
        f"- Sim: reflection SFCW surface scatterers (sim/forward_surface.py) → "
        f"DAS (recon/das.py) → chamfer vs true surface scatterers.",
        f"- **Resolution caveat:** reconstruction grid `{args.grid}` → voxel ≈ "
        f"{chamber_m * 1000 / args.grid:.1f} mm, so metrics are floored at ~1 voxel "
        f"for tiny PoC chambers. Sensor-sparsity effects show up when the voxel is "
        f"well below the RF resolution (range floor c/2B ≈ 2.1 cm at 7 GHz BW).",
        "",
        "### Interpreting the numbers",
        "At the 10 cm PoC chamber the median error is grid-limited (≈ one voxel), so "
        "all configs look green. Run the real check at the 50 cm chamber with a finer "
        "grid to expose the aperture/bandwidth limits:",
        "```",
        f"python scripts/sparsity_study.py --transceivers 2,3,4,6,8 --static --scan "
        f"--chamber 50cm --n-samples 3 --n-freq 64 --grid 32 --n-stations 12 --seed 1",
        "```",
        "",
        "Thresholds: **green** = median chamfer ≤ 1.5 cm · **yellow** ≤ 3 cm · **red** above.",
        "Cost score: 1 pt per RF channel (+1.5 for motor/mechanics in the scan mode). "
        "Complexity: 1 pt per channel (+2 for motion control & pose sync). "
        "For real modules read cost as ~EUR 50-100 per 60 GHz channel "
        "(e.g. Acconeer A121, Infineon BGT60); the scan mode adds a stepper + controller.",
        "",
        "> **Trust flag (T7 cross-validation):** the 60 GHz gprMax FDTD gate is not "
        "executable on this hardware (60 GHz forces dx ≈ 0.17 mm ≈ 27M cells), and "
        "scaled-carrier FDTD is not representative (50 mm box ≈ 1.7 λ at 10 GHz). Per "
        "the spec fallback the analytical numbers below are **flagged untrusted** until a "
        "GPU/60 GHz cross-validation exists — treat them as a self-consistent relative "
        "comparison only. See `reports/analytical_vs_gprmax.md`.",
        "",
        "## Verdict matrix",
        "",
        "| #Tx | Mode | Stations | mean (cm) | median (cm) | max (cm) | %≤1cm | cost | complexity | verdict |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(
            f"| {r['n_transceivers']} | {r['mode']} | {r['n_stations']} | "
            f"{r['chamfer_mean_cm']:.2f} | {r['chamfer_median_cm']:.2f} | "
            f"{r['chamfer_max_cm']:.2f} | {r['frac_within_1cm']:.3f} | "
            f"{r['cost_score']:.1f} | {r['complexity_score']:.1f} | {r['verdict']} |")
    lines += ["", f"_(study wall time: {elapsed:.1f}s; rerun with `--seed N` to reproduce)_"]
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    sys.exit(main())