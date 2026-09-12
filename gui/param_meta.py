#!/usr/bin/env python3
"""Parameter metadata for the MicroTomo GUI — single source of truth for how
every simulation parameter is presented to the user.

Each entry carries:
  label    — human-readable name shown in the UI
  unit     — physical unit appended to the label (or None)
  help     — one-line "what does this do" tooltip / helper text
  advanced — hidden behind the panel's "show advanced" toggle
  decimals — spinbox decimals (for float widgets)
  step     — suggested widget step
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Param:
    label: str
    help: str
    unit: str | None = None
    advanced: bool = False
    decimals: int | None = None
    step: float | None = None


# Parameters that matter only when you want to polish a run — hidden by
# default so the main form stays approachable.
ADVANCED_FIELDS = {"seed", "snr_db", "n_scatter_per_object", "n_freq"}

PARAM_META: dict[str, Param] = {
    # ---- chamber / environment ------------------------------------------
    "chamber_m": Param(
        label="Chamber size",
        unit="cm",
        help="Diameter of the cylindrical measurement chamber that objects "
             "and transceivers live inside. Larger chambers spread the "
             "antennas further from the object.",
    ),
    "wall": Param(
        label="Chamber boundary",
        help="How microwaves behave at the chamber wall. 'absorbing' stops "
             "reflections (cleanest signal); 'reflecting' keeps echoes in "
             "(more realistic, slightly harder to reconstruct).",
    ),
    # ---- transceivers ----------------------------------------------------
    "mode": Param(
        label="Transceiver scheme",
        help="'static' uses one fixed ring of antennas; 'scan' rotates an "
             "array around the chamber for more measurement angles (slower).",
    ),
    "n_tx": Param(
        label="Transmitting antennas",
        unit="tx",
        help="Number of antennas that transmit, one after another. More "
             "antennas = more measurements = better reconstruction, slower run.",
    ),
    "n_stations": Param(
        label="Scan stations",
        unit="positions",
        help="How many ring positions the scanner visits in 'scan' mode. "
             "More stations fill in more angles around the object.",
    ),
    # ---- measurement physics ---------------------------------------------
    "n_freq": Param(
        label="Frequencies",
        unit="freqs",
        help="How many microwave frequencies each transmitter uses. More "
             "frequencies improve detail and robustness but slow the run.",
        advanced=True,
    ),
    "n_scatter_per_object": Param(
        label="Surface points / object",
        unit="pts",
        help="How many points sample each object's surface. Denser sampling "
             "gives a smoother, more accurate forward signal — at a cost in "
             "runtime and memory.",
        advanced=True,
    ),
    "snr_db": Param(
        label="Signal-to-noise ratio",
        unit="dB",
        help="How clean the measured signal is. Lower values add realistic "
             "measurement noise and make reconstruction harder.",
        advanced=True,
    ),
    "seed": Param(
        label="Random seed",
        help="Seeds the noise generator. Fixed seed = reproducible runs; "
             "change it to sample different noise realisations.",
        advanced=True,
    ),
    # ---- reconstruction ----------------------------------------------------
    "grid": Param(
        label="Reconstruction grid",
        unit="pts/axis",
        help="Grid resolution used when reconstructing the volume. Higher = "
             "finer detail but much slower. 16–24 is a good starting range.",
    ),
}


def get_object_meta() -> dict[str, Param]:
    """Metadata for the object-level editor fields (shared across kinds)."""
    return {
        "center": Param(
            label="Centre position",
            unit="mm",
            help="Where the object sits inside the chamber (X, Y, Z). "
                 "Keep everything within the chamber diameter and clear of "
                 "the wall.",
        ),
        "eps": Param(
            label="Dielectric permittivity",
            unit="εr",
            help="Relative permittivity of the object material. Higher "
                 "values reflect/scatter microwaves more strongly. "
                 "Water ≈ 80, tissue/veg ≈ 40–60, plastic ≈ 2–4.",
        ),
        "tilt_deg": Param(
            label="Rotation",
            unit="deg",
            help="Rotation of the object around the vertical axis, in degrees.",
        ),
    }


# Per-object-kind dimension rows: presence in the local coordinate frame.
_DIM_LABELS = {
    "ellipsoid": ("Semi-axis a", "Semi-axis b", "Semi-axis c"),
    "cylinder": ("Radius", "Half length", "Axis a"),
    "pepper": ("Body a", "Body b", "Body c"),
}

DIM_META: dict[str, list[Param]] = {
    kind: [Param(label=lbl, unit="mm", help="Extent of the object along this axis.") for lbl in labels]
    for kind, labels in _DIM_LABELS.items()
}


def dim_meta_for(kind: str) -> list[Param]:
    """Dimension metadata for one object kind (units in mm)."""
    return DIM_META.get(kind) or DIM_META["ellipsoid"]