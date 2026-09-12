#!/usr/bin/env python3
"""Tests for gui.param_meta — every panel-surfaced parameter must have a
human label, a unit where relevant, and a one-line help so the UI can render
clear, explained settings instead of cryptic labels."""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))


@pytest.fixture(scope="module")
def meta():
    from gui.param_meta import PARAM_META, DIM_META, get_object_meta

    return PARAM_META, DIM_META, get_object_meta


def test_all_panel_state_fields_have_meta(meta):
    PARAM_META, _, _ = meta
    required = {
        "chamber_m", "mode", "n_tx", "n_stations",
        "n_freq", "grid", "seed", "snr_db", "n_scatter_per_object", "wall",
    }
    assert required <= set(PARAM_META)


def test_every_meta_has_label_and_help(meta):
    PARAM_META, _, _ = meta
    for key, p in PARAM_META.items():
        assert p.label.strip(), key
        assert p.help.strip(), key
        assert p.help != p.label, f"{key}: help must explain, not echo the label"


def test_units_correct(meta):
    PARAM_META, _, _ = meta
    assert PARAM_META["chamber_m"].unit == "cm"
    assert PARAM_META["snr_db"].unit == "dB"
    assert PARAM_META["grid"].unit == "pts/axis"
    assert PARAM_META["n_freq"].unit == "freqs"
    assert PARAM_META["wall"].unit is None


def test_advanced_subset_exact(meta):
    PARAM_META, _, _ = meta
    advanced = {k for k, p in PARAM_META.items() if p.advanced}
    assert advanced == {"seed", "snr_db", "n_scatter_per_object", "n_freq"}


def test_non_advanced_core_fields(meta):
    PARAM_META, _, _ = meta
    for k in ("chamber_m", "mode", "n_tx", "n_stations", "grid", "wall"):
        assert not PARAM_META[k].advanced


def test_object_meta_labels_and_units(meta):
    PARAM_META, DIM_META, get_object_meta = meta
    om = get_object_meta()
    assert om["center"].label and om["center"].unit == "mm"
    assert om["eps"].unit == "εr"           # dimensionless permittivity marker
    assert om["tilt_deg"].unit == "deg"
    for kind, dims in DIM_META.items():
        assert dims, f"no dim meta for {kind}"
        for d in dims:
            assert d.unit == "mm", f"{kind}:{d.label}"


def test_object_meta_covers_all_kinds(meta):
    _, DIM_META, _ = meta
    from gui.state import ObjectSpec

    kinds = set(ObjectSpec().kind for _ in [1])
    from gui.state import default_state

    kinds |= {o.kind for o in default_state().objects}
    # all supported kinds must have dim metadata
    from gui.panel import KIND_CHOICES

    assert set(kinds) | set(KIND_CHOICES) <= set(DIM_META) | {None}
    assert set(DIM_META) == set(KIND_CHOICES)