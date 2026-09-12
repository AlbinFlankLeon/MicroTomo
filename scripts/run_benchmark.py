#!/usr/bin/env python3
"""MicroTomo Benchmark Runner -- run gprMax simulations + compare to Mie.

Usage:
    python scripts/run_benchmark.py --phantom pec_cylinder --domain mini
    python scripts/run_benchmark.py --phantom water_vial --domain mini
    python scripts/run_benchmark.py --phantom random --domain mini --seed 42
"""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

# Project root
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

sys.path.insert(0, str(PROJECT_ROOT / "tests"))
from test_mie import mie_pec_cylinder


def make_pec_cylinder_mini(output_path):
    """Mini domain PEC cylinder, all coords 0..50mm, cylinder along z."""
    content = """## PEC Cylinder Benchmark - Mini Domain (50mm cube)
## MicroTomo Phase 3 - gprMax validation vs Mie

#domain: 50e-3 50e-3 50e-3
#dx_dy_dz: 0.5e-3 0.5e-3 0.5e-3
#time_window: 3e-9

#waveform: gaussian 1 60e9 mywave

#hertzian_dipole: z 10e-3 25e-3 25e-3 mywave

#cylinder: 25e-3 25e-3 0 25e-3 25e-3 50e-3 5e-3 pec

#rx: 40e-3 25e-3 25e-3
"""
    Path(output_path).write_text(content)
    return content


def make_water_vial_mini(output_path):
    """Mini domain water vial with Debye dispersion."""
    content = """## Water Vial Benchmark - Mini Domain (50mm cube)
## MicroTomo Phase 3 - gprMax validation
## NOTE: Using 10 GHz for water to avoid grid resolution issues at 60 GHz.
## At 60 GHz, water eps_eff ~ 19 requires dx < 0.15mm (37M cells).

#domain: 50e-3 50e-3 50e-3
#dx_dy_dz: 0.5e-3 0.5e-3 0.5e-3
#time_window: 10e-9

#material: 5.2 0 1 0 water_debye
#add_dispersion_debye: 1 73.1 5.56e-12 water_debye

#waveform: gaussian 1 10e9 mywave

#hertzian_dipole: z 10e-3 25e-3 25e-3 mywave

#cylinder: 25e-3 25e-3 0 25e-3 25e-3 50e-3 15e-3 water_debye

#rx: 40e-3 25e-3 25e-3
"""
    Path(output_path).write_text(content)
    return content


def make_random_mini(output_path, seed=42):
    """Mini domain with random dielectric objects.

    NOTE: Using 10 GHz CW (gprMax maxfreq = 4x center = 40 GHz).
    At 60 GHz, any eps>1.5 fails the grid check on 0.5mm grid
    (lambda at 240 GHz in eps=4 is only 0.6mm). This validates the
    plan's core constraint: 60 GHz needs dx <= 0.33mm.
    """
    rng = np.random.default_rng(seed)
    n = rng.integers(2, 5)
    objs = []
    for i in range(n):
        eps = rng.uniform(2.1, 8.0)  # resolvable at 40 GHz maxfreq
        x = rng.uniform(15e-3, 35e-3)
        y = rng.uniform(15e-3, 35e-3)
        r = rng.uniform(2e-3, 8e-3)
        objs.append(f"#material: {eps:.2f} 0 1 0 obj{i}")
        objs.append(f"#sphere: {x:.4f} {y:.4f} 25e-3 {r:.4f} obj{i}")

    content = f"""## Random Phantom - Mini Domain (50mm cube)
## MicroTomo Phase 3 - domain randomization test, seed={seed}
## 10 GHz CW: freq case 3 from benchmarks.yaml (versatility test).

#domain: 50e-3 50e-3 50e-3
#dx_dy_dz: 0.5e-3 0.5e-3 0.5e-3
#time_window: 10e-9

#waveform: sine 1 10e9 mywave
#hertzian_dipole: z 10e-3 25e-3 25e-3 mywave

{chr(10).join(objs)}

#rx: 40e-3 25e-3 25e-3
"""
    Path(output_path).write_text(content)
    return content


GENERATORS = {
    "pec_cylinder": lambda p, **kw: make_pec_cylinder_mini(p),
    "water_vial": lambda p, **kw: make_water_vial_mini(p),
    "random": lambda p, **kw: make_random_mini(p, seed=kw.get("seed", 42)),
}


def run_gprmax(input_file, n_runs=1, use_gpu=False):
    """Run gprMax and return timing + result."""
    cmd = [sys.executable, "-m", "gprMax", input_file, "-n", str(n_runs)]
    if use_gpu:
        cmd.append("-gpu")

    print(f"Running: {' '.join(cmd)}")
    t0 = time.perf_counter()

    result = subprocess.run(
        cmd, capture_output=True, text=True, timeout=600,
        cwd=str(Path(input_file).parent),
    )
    elapsed = time.perf_counter() - t0

    # Find output file
    in_stem = Path(input_file).stem
    out_file = Path(input_file).parent / f"{in_stem}.out"
    out_exists = out_file.exists()
    out_size = out_file.stat().st_size if out_exists else 0

    # Parse timing from stdout
    sim_time = None
    for line in result.stdout.split("\n"):
        if "Simulation completed in" in line:
            try:
                t_part = line.split("[")[1].split("]")[0]
                parts = t_part.split(":")
                sim_time = int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])
            except Exception:
                pass

    # Parse model info
    model_info = {}
    for line in result.stdout.split("\n"):
        if "cells" in line.lower() and "domain" in line.lower():
            model_info["grid"] = line.strip()
        if "iterations" in line.lower():
            model_info["iterations"] = line.strip()
        if "time step" in line.lower():
            model_info["timestep"] = line.strip()

    print(f"  Wall time: {elapsed:.1f}s")
    if sim_time:
        print(f"  Simulation time: {sim_time:.1f}s")
    print(f"  Output: {out_file} ({out_size} bytes)")

    return {
        "returncode": result.returncode,
        "wall_time_s": elapsed,
        "sim_time_s": sim_time,
        "output_file": str(out_file),
        "output_size": out_size,
        "model_info": model_info,
        "stdout": result.stdout[-3000:],
        "stderr": result.stderr[-1000:],
    }


def compare_to_mie(result, phantom_type):
    """Compare gprMax output to Mie analytical (for PEC cylinder only)."""
    if phantom_type != "pec_cylinder":
        return {"status": "skipped", "reason": "not a PEC cylinder"}

    # Mie analytical reference
    freq = 60e9
    radius = 5e-3
    mie_mag_db, mie_phase_deg = mie_pec_cylinder(freq, radius, n_terms=100)
    print(f"\nMie analytical: S21 = {mie_mag_db:+.2f} dB, phase = {mie_phase_deg:+.1f}°")

    # Try to read gprMax .out file (HDF5)
    out_path = result["output_file"]
    if not Path(out_path).exists():
        return {"status": "no_output", "mie_ref": (mie_mag_db, mie_phase_deg)}

    try:
        import h5py
        with h5py.File(out_path, "r") as f:
            # gprMax stores Ez field at receiver
            ez_key = None
            for key in f.keys():
                if "ez" in key.lower() or "rx" in key.lower():
                    ez_key = key
                    break

            if ez_key is None:
                # List all keys
                keys = list(f.keys())
                print(f"HDF5 keys: {keys}")
                return {"status": "no_ez_field", "keys": keys, "mie_ref": (mie_mag_db, mie_phase_deg)}

            ez = f[ez_key][:]
            print(f"Field shape: {ez.shape}, max: {np.max(np.abs(ez)):.6e}")

            # Simple S21 estimation from time-domain field
            peak = np.max(np.abs(ez))
            s21_linear = peak  # simplified, need proper normalization
            s21_db = 20 * np.log10(max(s21_linear, 1e-30))

            print(f"gprMax S21 estimate: {s21_db:+.2f} dB")
            error_db = abs(s21_db - mie_mag_db)
            print(f"Error vs Mie: {error_db:.2f} dB (target: <3 dB)")

            return {
                "status": "compared",
                "mie_mag_db": mie_mag_db,
                "mie_phase_deg": mie_phase_deg,
                "fdtd_mag_db": s21_db,
                "error_db": error_db,
                "pass": error_db < 3.0,
            }
    except Exception as e:
        return {"status": "error", "error": str(e), "mie_ref": (mie_mag_db, mie_phase_deg)}


def main():
    parser = argparse.ArgumentParser(description="MicroTomo Benchmark Runner")
    parser.add_argument("--phantom", default="pec_cylinder",
                        choices=["pec_cylinder", "water_vial", "random"])
    parser.add_argument("--domain", default="mini", choices=["mini", "full"])
    parser.add_argument("--gpu", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("-n", type=int, default=1)
    args = parser.parse_args()

    work_dir = PROJECT_ROOT / "sim" / "gprmax"
    work_dir.mkdir(parents=True, exist_ok=True)

    input_file = work_dir / f"bench_{args.phantom}_{args.domain}.in"
    print(f"Generating: {input_file}")
    gen = GENERATORS[args.phantom]
    gen(str(input_file), seed=args.seed)
    print(input_file.read_text())
    print("=" * 60)

    result = run_gprmax(str(input_file), n_runs=args.n, use_gpu=args.gpu)

    if result["returncode"] == 0:
        comparison = compare_to_mie(result, args.phantom)
        result["mie_comparison"] = comparison

    # Save results
    results_file = work_dir / f"bench_{args.phantom}_{args.domain}.json"
    with open(results_file, "w") as f:
        json.dump({
            "tool": "gprmax",
            "phantom": args.phantom,
            "domain": args.domain,
            "gpu": args.gpu,
            **result,
        }, f, indent=2, default=str)
    print(f"\nResults saved: {results_file}")


if __name__ == "__main__":
    main()
