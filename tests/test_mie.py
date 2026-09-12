#!/usr/bin/env python3
"""Mie Scattering Validation — Analytical solution for PEC cylinder.

Computes the analytical Mie scattering series for a PEC (perfect electric
conductor) cylinder at 60 GHz (TM polarization). Used as ground truth for
validating FDTD/FEM solvers.

Usage:
    python tests/test_mie.py
    python -m pytest tests/test_mie.py -v
"""
import numpy as np
from math import factorial


def mie_pec_cylinder(frequency_hz, radius_m, n_terms=50):
    """Compute TM-mode Mie scattering for a PEC cylinder.

    Args:
        frequency_hz: Frequency in Hz (e.g. 60e9)
        radius_m: Cylinder radius in meters (e.g. 5e-3)
        n_terms: Number of series terms (default 50, higher = more accurate)

    Returns:
        (s21_mag_db, s21_phase_deg) — scattering parameter in dB and degrees
    """
    c0 = 299792458.0
    k0 = 2.0 * np.pi * frequency_hz / c0
    ka = k0 * radius_m

    # TM-mode Mie coefficients for PEC cylinder
    # b_n = -J_n(ka) / H_n^(1)(ka)
    from scipy.special import jv, hankel1

    total_scattered = 0.0 + 0.0j
    for n in range(-n_terms, n_terms + 1):
        Jn_ka = jv(n, ka)
        Hn_ka = hankel1(n, ka)
        if abs(Hn_ka) < 1e-30:
            continue
        b_n = -Jn_ka / Hn_ka
        total_scattered += b_n

    # Normalized scattering amplitude (far-field)
    # For bistatic S21 in dB relative to incident field
    sigma_bistatic = (2.0 / (np.pi * k0)) * abs(total_scattered) ** 2

    # Convert to dB (normalized to lambda)
    s21_mag = sigma_bistatic / (2.0 * radius_m)  # normalize
    s21_mag_db = 10.0 * np.log10(max(s21_mag, 1e-30))

    # Phase from the complex sum
    s21_phase_deg = np.degrees(np.angle(total_scattered))

    return s21_mag_db, s21_phase_deg


def run_mie_validation():
    """Run the canonical PEC cylinder validation case."""
    freq = 60e9  # 60 GHz
    radius = 5e-3  # 5 mm

    print("=" * 70)
    print("MicroTomo — Mie Scattering Validation (PEC Cylinder)")
    print("=" * 70)
    print(f"Frequency : {freq / 1e9:.1f} GHz")
    print(f"Wavelength: {299792458 / freq * 1000:.3f} mm")
    print(f"Radius    : {radius * 1000:.1f} mm")
    print(f"ka        : {2 * np.pi * freq / 299792458 * radius:.4f}")
    print("-" * 70)

    for n_terms in [10, 20, 50, 100]:
        mag_db, phase_deg = mie_pec_cylinder(freq, radius, n_terms)
        print(f"  n_terms={n_terms:3d}: S21 = {mag_db:+.2f} dB, "
              f"phase = {phase_deg:+.1f}°")

    print("-" * 70)
    print("CONVERGENCE CHECK: If values stabilize across n_terms, result is valid.")
    print("FDTD solver should match within ±3 dB / ±10° of the converged value.")
    print("=" * 70)

    return mie_pec_cylinder(freq, radius, n_terms=100)


if __name__ == "__main__":
    run_mie_validation()
