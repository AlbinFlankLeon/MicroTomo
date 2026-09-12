#!/usr/bin/env python3
"""Simulation presets — one-click, pre-run configurations.

Each preset bundles a sensible set of parameters and tells the user what to
expect (runtime + quality). Selecting one fills the panel; the user can still
tweak individual knobs afterwards ("Custom" when diverged).
"""
from __future__ import annotations

import copy
from dataclasses import asdict, dataclass

from gui.state import SimState

# Which fields a preset is allowed to touch (everything the pipeline exposes).
PRESET_FIELDS = ("grid", "n_freq", "n_scatter_per_object",
                 "n_tx", "mode", "n_stations", "snr_db")


@dataclass(frozen=True)
class SimPreset:
    name: str
    description: str
    expected: str            # e.g. "≈ 3 s per run · coarse detail"
    grid: int
    n_freq: int
    n_scatter_per_object: int
    n_tx: int
    mode: str
    n_stations: int
    snr_db: float


PRESETS: dict[str, SimPreset] = {
    "quick": SimPreset(
        name="Quick test",
        description="A few seconds to sanity-check the scene and viewer.",
        expected="≈ 1–3 s per run · coarse detail",
        grid=12, n_freq=8, n_scatter_per_object=600,
        n_tx=4, mode="static", n_stations=8, snr_db=30.0,
    ),
    "standard": SimPreset(
        name="Standard",
        description="Balanced quality/speed — the default starting point.",
        expected="≈ 5–20 s per run · good detail",
        grid=20, n_freq=48, n_scatter_per_object=1500,
        n_tx=4, mode="static", n_stations=24, snr_db=30.0,
    ),
    "high": SimPreset(
        name="High detail",
        description="Fine reconstruction; noticeably slower.",
        expected="≈ 30–90 s per run · fine detail",
        grid=32, n_freq=96, n_scatter_per_object=3000,
        n_tx=8, mode="scan", n_stations=32, snr_db=40.0,
    ),
    "max": SimPreset(
        name="Max quality",
        description="The longest, highest-fidelity reconstruction.",
        expected="several minutes per run · highest fidelity",
        grid=48, n_freq=128, n_scatter_per_object=6000,
        n_tx=8, mode="scan", n_stations=48, snr_db=60.0,
    ),
}

# Order shown in the UI combo box.
PRESET_ORDER = ("quick", "standard", "high", "max")


def preset_names() -> list[str]:
    return list(PRESET_ORDER)


def get_preset(preset_key: str) -> SimPreset:
    return PRESETS[preset_key]


def apply_preset(state: SimState, preset_key: str) -> SimState:
    """Return a NEW state with the preset's parameters applied.

    Objects, chamber size and other user geometry are preserved untouched.
    """
    p = get_preset(preset_key)
    out = copy.deepcopy(state)
    for field in PRESET_FIELDS:
        setattr(out, field, getattr(p, field))
    return out


def state_matches_preset(state: SimState, preset_key: str) -> bool:
    """True if the state still equals exactly one preset (for 'Custom' label)."""
    p = get_preset(preset_key)
    return all(getattr(state, f) == getattr(p, f) for f in PRESET_FIELDS)


def preset_summary(preset_key: str) -> str:
    p = get_preset(preset_key)
    return f"{p.description}  {p.expected}"


def preset_to_dict(p: SimPreset) -> dict:
    d = asdict(p)
    d.pop("description", None)
    d.pop("expected", None)
    return d