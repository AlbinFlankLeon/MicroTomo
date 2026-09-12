#!/usr/bin/env python3
"""MicroTomo Phantom Generator — random phantoms for domain randomization.

Generates gprMax input files with parametric phantoms:
- PEC cylinder (Mie validation)
- Water vial with Debye dispersion
- Plastic/glass layered stack
- Random ellipsoids with permittivity sampling

Usage:
    python phantoms/generator.py --type pec_cylinder --domain mini --output test.in
    python phantoms/generator.py --type random --domain mini --output test.in --seed 42
    python phantoms/generator.py --list
"""
import argparse
import os
import sys
import yaml
import numpy as np
from pathlib import Path

# Load benchmark config
CONFIG_DIR = Path(__file__).parent.parent / "config"
with open(CONFIG_DIR / "benchmarks.yaml") as f:
    BENCHMARKS = yaml.safe_load(f)


def pec_cylinder_input(domain="mini", output_path=None):
    """Generate gprMax input for PEC cylinder (Mie validation)."""
    d = BENCHMARKS["domains"][domain]
    L = d["diameter_mm"]
    dx = d["grid_dx_mm"]

    lines = [
        "# PEC Cylinder — MicroTomo Canonical Benchmark",
        f"# Domain: {L}mm, dx={dx}mm",
        f"# Phantom: PEC cylinder r=5mm",
        "",
        f"#python: import numpy as np",
        "",
        f"#domain: {L}e-3 {L}e-3 {L}e-3",
        f"#dx_dy_dz: {dx}e-3 {dx}e-3 {dx}e-3",
        "",
        "#time_window: 5e-9",
        "",
        "#waveform: gaussian 1 60e9 mywave",
        "",
        "#hertzian_dipole: z 10e-3 {y_center} 0 mywave",
        "",
        f"#box: {L/2-3}e-3 {L/2-5}e-3 0 {L/2+3}e-3 {L/2+5}e-3 {L}e-3 pec",
        "",
        f"#cylinder: 0 0 0 {L}e-3 5e-3 pec",
        "",
        "#rx: 40e-3 {y_rx} 0",
        "",
    ]
    content = "\n".join(lines)
    if output_path:
        Path(output_path).write_text(content)
    return content


def water_vial_input(domain="mini", output_path=None):
    """Generate gprMax input for water vial (Debye dispersion)."""
    d = BENCHMARKS["domains"][domain]
    L = d["diameter_mm"]
    dx = d["dx_dy_dz"][0] if isinstance(d.get("dx_dy_dz"), list) else d["grid_dx_mm"]

    lines = [
        "# Water Vial — MicroTomo Canonical Benchmark",
        f"# Domain: {L}mm, dx={dx}mm",
        f"# Phantom: Water cylinder r=15mm (Debye model @60GHz)",
        "",
        f"#domain: {L}e-3 {L}e-3 {L}e-3",
        f"#dx_dy_dz: {dx}e-3 {dx}e-3 {dx}e-3",
        "",
        "#time_window: 5e-9",
        "",
        "#waveform: gaussian 1 60e9 mywave",
        "",
        "#hertzian_dipole: z 10e-3 {y_center} 0 mywave",
        "",
        "#material: 78.3 1 1 0 water_debye",
        "#debye: 1 73.1 21.4e9 water_debye",
        "",
        "#cylinder: 0 0 0 {L}e-3 15e-3 water_debye",
        "",
        "#rx: 40e-3 {y_rx} 0",
    ]
    content = "\n".join(lines)
    if output_path:
        Path(output_path).write_text(content)
    return content


def random_ellipsoid(domain="mini", output_path=None, seed=None):
    """Generate gprMax input with random ellipsoids (domain randomization)."""
    rng = np.random.default_rng(seed)
    d = BENCHMARKS["domains"][domain]
    L = d["diameter_mm"]
    dx = d["grid_dx_mm"]

    n_objects = rng.integers(1, 5)
    eps_range = (2.1, 80.0)

    lines = [
        "# Random Ellipsoids — Domain Randomization",
        f"# Domain: {L}mm, dx={dx}mm, seed={seed}",
        "",
        f"#domain: {L}e-3 {L}e-3 {L}e-3",
        f"#dx_dy_dz: {dx}e-3 {dx}e-3 {dx}e-3",
        "",
        "#time_window: 5e-9",
        "#waveform: gaussian 1 60e9 mywave",
        "#hertzian_dipole: z 10e-3 {y_center} 0 mywave",
        "",
    ]

    for i in range(n_objects):
        eps = rng.uniform(*eps_range)
        rx = rng.uniform(-L / 4, L / 4)
        ry = rng.uniform(-L / 4, L / 4)
        rz = rng.uniform(-L / 4, L / 4)
        a = rng.uniform(2, L / 6)
        b = rng.uniform(2, L / 6)
        c = rng.uniform(2, L / 6)
        lines.append(
            f"#material: {eps:.2f} 1 1 0 obj{i}"
        )
        lines.append(
            f"#ellipsoid: {rx}e-3 {ry}e-3 {rz}e-3 {a}e-3 {b}e-3 {c}e-3 obj{i}"
        )

    lines.append("")
    lines.append("#rx: 40e-3 {y_rx} 0")
    content = "\n".join(lines)
    if output_path:
        Path(output_path).write_text(content)
    return content


GENERATORS = {
    "pec_cylinder": pec_cylinder_input,
    "water_vial": water_vial_input,
    "random": random_ellipsoid,
}


def main():
    parser = argparse.ArgumentParser(description="MicroTomo Phantom Generator")
    parser.add_argument("--type", choices=list(GENERATORS.keys()), required=True)
    parser.add_argument("--domain", default="mini", choices=["mini", "full"])
    parser.add_argument("--output", "-o", help="Output gprMax input file path")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--list", action="store_true", help="List available phantom types")
    args = parser.parse_args()

    if args.list:
        print("Available phantom types:")
        for name, func in GENERATORS.items():
            print(f"  {name:20s} — {func.__doc__.strip().split(chr(10))[0]}")
        return

    gen = GENERATORS[args.type]
    kwargs = {"domain": args.domain, "output_path": args.output}
    if args.type == "random":
        kwargs["seed"] = args.seed
    content = gen(**kwargs)
    if not args.output:
        print(content)


if __name__ == "__main__":
    main()
