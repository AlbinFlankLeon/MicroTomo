#!/usr/bin/env python3
"""Tests for gui.presets — one-click pre-run simulation configurations that
set a sensible set of parameters and tell the user what to expect."""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from gui.state import SimState  # noqa: E402


@pytest.fixture(scope="module")
def presets():
    from gui.presets import PRESETS, apply_preset, preset_names

    return PRESETS, apply_preset, preset_names


def test_four_presets_exist(presets):
    PRESETS, _, names = presets
    assert set(names()) == {"quick", "standard", "high", "max"}
    assert len(PRESETS) == 4


def test_each_preset_explains_itself(presets):
    PRESETS, _, _ = presets
    for name, p in PRESETS.items():
        assert p.name.strip()
        assert p.description.strip(), name
        assert p.expected.strip(), name
        assert p.expected != p.description, name


def test_preset_values_all_valid(presets):
    PRESETS, _, _ = presets
    for name, p in PRESETS.items():
        assert 0 < p.grid <= 64, name
        assert 4 <= p.n_freq <= 128, name
        assert p.n_scatter_per_object > 0, name
        assert 1 <= p.n_tx <= 16, name
        assert p.mode in {"static", "scan"}, name
        assert p.n_stations > 0, name
        assert p.snr_db > 0, name


def test_apply_returns_new_state_and_preserves_rest(presets):
    PRESETS, apply, _ = presets
    st = SimState()
    st.objects[0].center = [0.03, 0.03, 0.05]  # sentinel: geometry preserved
    st.grid = 11                                # sentinel: source stays untouched
    out = apply(st, "standard")
    assert out is not st
    p = PRESETS["standard"]
    assert (out.grid, out.n_freq, out.n_scatter_per_object) == (
        p.grid, p.n_freq, p.n_scatter_per_object)
    assert (out.n_tx, out.mode, out.n_stations, out.snr_db) == (
        p.n_tx, p.mode, p.n_stations, p.snr_db)
    assert out.objects[0].center == [0.03, 0.03, 0.05]
    # source untouched (geometry AND overwritten field)
    assert st.grid == 11
    assert st.objects[0].center == [0.03, 0.03, 0.05]


def test_apply_unknown_preset_raises(presets):
    _, apply, _ = presets
    with pytest.raises(KeyError):
        apply(SimState(), "nope")


def test_quick_is_fast_standard_is_baseline(presets):
    PRESETS, _, _ = presets
    assert PRESETS["quick"].grid < PRESETS["standard"].grid
    assert PRESETS["standard"].grid < PRESETS["max"].grid
    assert PRESETS["quick"].n_freq < PRESETS["standard"].n_freq