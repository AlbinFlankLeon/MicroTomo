#!/usr/bin/env python3
"""
MicroTomo Phase 2 — Install Viability Checker

Checks all candidate simulation tools on the current system.
Outputs a Go/No-Go table (plain text + Markdown) and saves a JSON
report to reports/install_check.json.

Usage:
    python3 scripts/install_check.py
    python3 scripts/install_check.py --markdown   # also write reports/install_check.md
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


# ── Helpers ──────────────────────────────────────────────────────────────────

def run(cmd: str, timeout: int = 30) -> tuple[bool, str]:
    """Run *cmd* in a shell and return (success, combined_output)."""
    try:
        r = subprocess.run(
            cmd, shell=True, capture_output=True, text=True, timeout=timeout
        )
        return r.returncode == 0, (r.stdout + r.stderr).strip()
    except subprocess.TimeoutExpired:
        return False, "TIMEOUT"
    except Exception as exc:  # noqa: BLE001
        return False, str(exc)


def _first_line(text: str) -> str:
    return text.split("\n", 1)[0].strip() if text else ""


# ── Hardware probes ──────────────────────────────────────────────────────────

def detect_cpu() -> dict:
    """Return CPU info dict: name, cores_logical, cores_physical, arch."""
    info: dict = {
        "name": "unknown",
        "cores_logical": 1,
        "cores_physical": 1,
        "arch": platform.machine(),
    }
    system = platform.system()

    if system == "Linux":
        # Prefer lscpu
        ok, out = run("lscpu 2>/dev/null")
        if ok:
            for line in out.splitlines():
                if line.startswith("Model name:") and info["name"] == "unknown":
                    info["name"] = line.split(":", 1)[1].strip()
                elif line.startswith("CPU(s):") and info["cores_logical"] == 1:
                    try:
                        info["cores_logical"] = int(line.split(":", 1)[1].strip())
                    except ValueError:
                        pass
                elif line.startswith("Core(s) per socket:") and info["cores_physical"] == 1:
                    try:
                        val = int(line.split(":", 1)[1].strip())
                        sockets = 1
                        # try to read socket count
                        for sub in out.splitlines():
                            if sub.startswith("Socket(s):"):
                                sockets = int(sub.split(":", 1)[1].strip())
                        info["cores_physical"] = val * sockets
                    except ValueError:
                        pass

    elif system == "Darwin":
        ok, out = run("sysctl -n machdep.cpu.brand_string 2>/dev/null")
        if ok:
            info["name"] = out.strip()
        ok, out = run("sysctl -n hw.ncpu 2>/dev/null")
        if ok:
            try:
                info["cores_logical"] = int(out.strip())
            except ValueError:
                pass
        ok, out = run("sysctl -n hw.physicalcpu 2>/dev/null")
        if ok:
            try:
                info["cores_physical"] = int(out.strip())
            except ValueError:
                pass

    else:
        # Fallback: use os.cpu_count()
        info["cores_logical"] = os.cpu_count() or 1
        info["name"] = platform.processor() or "unknown"

    return info


def detect_nvidia() -> Optional[dict]:
    """Return GPU info dict or None."""
    ok, out = run("nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader 2>/dev/null")
    if not ok or not out:
        return None
    parts = [p.strip() for p in out.split(",")]
    return {
        "vendor": "NVIDIA",
        "name": parts[0] if len(parts) >= 1 else "unknown",
        "vram_mb": int(parts[1].replace("MiB", "").strip()) if len(parts) >= 2 else 0,
        "driver": parts[2] if len(parts) >= 3 else "unknown",
    }


def detect_amd() -> Optional[dict]:
    """Return AMD GPU info dict or None."""
    ok, out = run("rocminfo 2>/dev/null | grep 'Marketing Name'")
    if ok and out:
        name = _first_line(out)
        return {"vendor": "AMD/ROCm", "name": name}
    # Fallback: OpenCL
    ok, out = run("clinfo 2>/dev/null | grep -i 'AMD'")
    if ok and out:
        name = _first_line(out)
        return {"vendor": "AMD/OpenCL", "name": name}
    return None


def detect_opencl() -> list[dict]:
    """Return list of OpenCL device dicts."""
    ok, out = run("clinfo --machine-readable 2>/dev/null")
    if not ok or not out:
        return []
    devices: list[dict] = []
    current: dict = {}
    for line in out.splitlines():
        if line.startswith("device_name"):
            current["name"] = line.split("\t", 1)[-1] if "\t" in line else ""
        elif line.startswith("device_vendor"):
            current["vendor"] = line.split("\t", 1)[-1] if "\t" in line else ""
        elif line.startswith("device_type"):
            current["type"] = line.split("\t", 1)[-1] if "\t" in line else ""
            if current.get("name"):
                devices.append(current)
                current = {}
    return devices


def detect_mpi() -> dict:
    """Check MPI availability."""
    info: dict = {"binary": False, "mpi4py": False, "version": None}

    # Binary check
    ok, out = run("mpirun --version 2>/dev/null | head -1")
    if not ok:
        ok, out = run("mpiexec --version 2>/dev/null | head -1")
    if ok:
        info["binary"] = True
        info["version"] = out.strip()

    # mpi4py import test
    ok, out = run(f'{sys.executable} -c "from mpi4py import MPI; print(MPI.Get_library_version())" 2>&1')
    if ok:
        info["mpi4py"] = True

    return info


# ── Tool checker ─────────────────────────────────────────────────────────────

@dataclass
class ToolCheck:
    name: str
    import_name: str
    description: str
    install_hint: str
    python_api: bool = False
    cuda_support: bool = False
    opencl_support: bool = False
    mpi_support: bool = False
    # Results filled by check()
    installed: bool = False
    version: Optional[str] = None
    import_ok: bool = False
    gpu_ok: bool = False
    notes: str = ""


def check_tool(tool: ToolCheck, has_cuda: bool, has_amd: bool, has_opencl: bool) -> ToolCheck:
    """Probe one tool: pip show, python import, GPU availability."""
    # 1. pip show
    ok, out = run(f"{sys.executable} -m pip show {tool.import_name} 2>/dev/null")
    if ok:
        tool.installed = True
        for line in out.splitlines():
            if line.startswith("Version:"):
                tool.version = line.split(":", 1)[1].strip()
            elif line.startswith("Location:"):
                tool.notes = line.split(":", 1)[1].strip()

    # 2. Python import test
    snippet = (
        f"import {tool.import_name} as m; "
        f"print(getattr(m, '__version__', getattr(m, 'VERSION', 'ok')))"
    )
    ok, out = run(f'{sys.executable} -c "{snippet}" 2>&1', timeout=60)
    if ok:
        tool.import_ok = True
        if not tool.version or tool.version == "unknown":
            tool.version = out.strip()[:40]

    # 3. GPU reachable?
    if has_cuda and tool.cuda_support:
        tool.gpu_ok = True
    elif has_amd and tool.opencl_support:
        tool.gpu_ok = True
    elif has_opencl and tool.opencl_support:
        tool.gpu_ok = True

    return tool


# ── Formatters ───────────────────────────────────────────────────────────────

def go_nogo(t: ToolCheck) -> str:
    if t.import_ok:
        return "GO"
    if t.installed:
        return "PARTIAL"
    return "NO-GO"


def print_plain(tools: list[ToolCheck], cpu: dict, nvidia: Optional[dict],
                amd: Optional[dict], mpi: dict) -> str:
    """Build plain-text report."""
    lines: list[str] = []
    w = 78
    lines.append("=" * w)
    lines.append("  MicroTomo Phase 2 — Install Viability Check")
    lines.append("=" * w)

    # System
    lines.append("")
    lines.append("SYSTEM")
    lines.append("-" * w)
    lines.append(f"  CPU     : {cpu['name']}  ({cpu['cores_logical']} logical / "
                 f"{cpu['cores_physical']} physical cores, {cpu['arch']})")
    if nvidia:
        lines.append(f"  GPU (NV): {nvidia['name']}  {nvidia.get('vram_mb', '?')} MiB  "
                     f"driver {nvidia.get('driver', '?')}")
    else:
        lines.append("  GPU (NV): —")
    if amd:
        lines.append(f"  GPU (AMD): {amd['name']}")
    else:
        lines.append("  GPU (AMD): —")
    lines.append(f"  MPI     : {'v' + mpi['version'] if mpi['version'] else 'not found'}"
                 f"  |  mpi4py: {'yes' if mpi['mpi4py'] else 'no'}")

    # System deps
    lines.append("")
    lines.append("SYSTEM DEPENDENCIES")
    lines.append("-" * w)
    for prog, desc in [("ffmpeg", "Media processing"), ("gmsh", "Mesh generation"),
                       ("clinfo", "OpenCL detection"), ("cmake", "Build system")]:
        p = shutil.which(prog)
        lines.append(f"  {desc:<22s} {prog:<12s} {'OK  ' + p if p else 'NOT FOUND'}")

    # Tool table
    lines.append("")
    lines.append("TOOL STATUS")
    lines.append("-" * w)
    hdr = f"  {'Tool':<12s} {'Installed':>10s} {'Version':<18s} {'Import':>8s} {'GPU':>6s} {'Go/No-Go'}"
    lines.append(hdr)
    lines.append("  " + "-" * (w - 2))

    go_count = 0
    for t in tools:
        g = go_nogo(t)
        if g == "GO":
            go_count += 1
        lines.append(
            f"  {t.name:<12s} {str(t.installed):>10s} {(t.version or '—'):<18s} "
            f"{str(t.import_ok):>8s} {str(t.gpu_ok):>6s} {g}"
        )

    lines.append("  " + "-" * (w - 2))
    lines.append(f"  Result: {go_count}/{len(tools)} tools ready")

    # Verdict
    lines.append("")
    if go_count == 0:
        lines.append("  *** NO-GO: No tools installed. Install candidates first.")
        lines.append("  See docs/decisions.md for install commands.")
    elif go_count < 2:
        lines.append(f"  ** PARTIAL: Only {go_count} tool ready. Install more for cross-validation.")
    else:
        lines.append(f"  ** GO: {go_count} tools ready — proceed to Phase 3 (canonical benchmark).")

    lines.append("")
    lines.append("=" * w)
    return "\n".join(lines)


def print_markdown(tools: list[ToolCheck], cpu: dict, nvidia: Optional[dict],
                   amd: Optional[dict], mpi: dict) -> str:
    """Build Markdown report."""
    md: list[str] = []
    md.append("# MicroTomo — Install Viability Report\n")

    md.append("## System\n")
    md.append(f"| Item | Value |")
    md.append(f"|------|-------|")
    md.append(f"| CPU | {cpu['name']} |")
    md.append(f"| Cores | {cpu['cores_logical']} logical / {cpu['cores_physical']} physical |")
    md.append(f"| Arch | {cpu['arch']} |")
    nv = f"{nvidia['name']} ({nvidia.get('vram_mb', '?')} MiB, driver {nvidia.get('driver', '?')})" if nvidia else "—"
    md.append(f"| NVIDIA | {nv} |")
    amd_s = amd['name'] if amd else "—"
    md.append(f"| AMD | {amd_s} |")
    mpi_v = mpi['version'] or '—'
    md.append(f"| MPI | {mpi_v} | mpi4py: {'yes' if mpi['mpi4py'] else 'no'} |")

    md.append("\n## Tool Status\n")
    md.append("| Tool | Installed | Version | Import | GPU | Go/No-Go |")
    md.append("|------|-----------|---------|--------|-----|----------|")
    go_count = 0
    for t in tools:
        g = go_nogo(t)
        if g == "GO":
            go_count += 1
        md.append(f"| {t.name} | {t.installed} | {t.version or '—'} | "
                  f"{t.import_ok} | {t.gpu_ok} | **{g}** |")

    md.append(f"\n**Result:** {go_count}/{len(tools)} tools ready\n")
    if go_count == 0:
        md.append("> **NO-GO** — No tools installed.")
    elif go_count < 2:
        md.append(f"> **PARTIAL** — Only {go_count} tool ready.")
    else:
        md.append(f"> **GO** — {go_count} tools ready, proceed to Phase 3.")

    return "\n".join(md)


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> int:
    ap = argparse.ArgumentParser(description="MicroTomo install viability checker")
    ap.add_argument("--markdown", action="store_true", help="Also write reports/install_check.md")
    args = ap.parse_args()

    # 1. System detection
    cpu = detect_cpu()
    nvidia = detect_nvidia()
    amd = detect_amd()
    ocl = detect_opencl()
    mpi = detect_mpi()

    # 2. Define candidate tools
    tools: list[ToolCheck] = [
        ToolCheck("gprMax", "gprmax",
                  "FDTD solver, Python API, CUDA+OpenCL",
                  "pip install gprMax[cuda]  # or  pip install gprMax",
                  python_api=True, cuda_support=True, opencl_support=True),
        ToolCheck("torchfdtd", "torchfdtd",
                  "PyTorch FDTD, differentiable, CUDA-native",
                  "pip install torchfdtd",
                  python_api=True, cuda_support=True),
        ToolCheck("jaxfdtd", "jaxfdtd",
                  "JAX-based FDTD, differentiable, GPU",
                  "pip install jaxfdtd",
                  python_api=True, cuda_support=True),
        ToolCheck("FEniCSx", "dolfinx",
                  "FEM, frequency-domain Helmholtz, MPI",
                  "docker run -v $(pwd):/home/fenics/shared quay.io/fenics/fenics-dolfinx:stable",
                  python_api=True, mpi_support=True),
        ToolCheck("Sionna", "sionna",
                  "NVIDIA ray-tracing channel modeler, CUDA",
                  "pip install sionna",
                  python_api=True, cuda_support=True),
        ToolCheck("Meep", "meep",
                  "FDTD, subpixel smoothing, CPU+MPI",
                  "conda install -c conda-forge pymeep",
                  python_api=True, mpi_support=True),
    ]

    # 3. Probe each tool
    has_cuda = nvidia is not None
    has_amd = amd is not None
    has_opencl = len(ocl) > 0
    for t in tools:
        check_tool(t, has_cuda, has_amd, has_opencl)

    # 4. Build reports
    plain = print_plain(tools, cpu, nvidia, amd, mpi)
    print(plain)

    # 5. Save JSON
    report_dir = Path("reports")
    report_dir.mkdir(exist_ok=True)

    json_path = report_dir / "install_check.json"
    result = {
        "python": sys.version.split()[0],
        "cpu": cpu,
        "nvidia": nvidia,
        "amd": amd,
        "opencl_devices": ocl,
        "mpi": mpi,
        "tools": {
            t.name: {
                "import_name": t.import_name,
                "installed": t.installed,
                "version": t.version,
                "import_ok": t.import_ok,
                "gpu_ok": t.gpu_ok,
                "python_api": t.python_api,
                "cuda_support": t.cuda_support,
                "opencl_support": t.opencl_support,
                "mpi_support": t.mpi_support,
                "notes": t.notes,
            }
            for t in tools
        },
    }
    json_path.write_text(json.dumps(result, indent=2) + "\n")
    print(f"JSON report saved: {json_path}")

    # 6. Optionally save Markdown
    if args.markdown:
        md_path = report_dir / "install_check.md"
        md_path.write_text(print_markdown(tools, cpu, nvidia, amd, mpi) + "\n")
        print(f"Markdown report saved: {md_path}")

    # 7. Exit code
    go_count = sum(1 for t in tools if go_nogo(t) == "GO")
    return 0 if go_count >= 2 else 1


if __name__ == "__main__":
    sys.exit(main())
