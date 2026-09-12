#!/usr/bin/env python3
"""Tests for contour metrics (sparsity study T5)."""
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from metrics.contour_metrics import chamfer, detect_feature, frac_within_cm  # noqa: E402

TRUTH = np.array([[0.0, 0.0, 0.0], [0.01, 0.0, 0.0], [0.0, 0.01, 0.0], [0.0, 0.0, 0.01]])


def test_identical_clouds_zero():
    res = chamfer(TRUTH, TRUTH)
    assert res["mean"] == pytest.approx(0.0, abs=1e-12)
    assert res["median"] == pytest.approx(0.0, abs=1e-12)
    assert res["max"] == pytest.approx(0.0, abs=1e-12)


def test_constant_offset_one_cm():
    # 2 well-separated points so a 1 cm shift creates no accidental matches
    a = np.array([[0.0, 0.0, 0.0], [0.0, 0.03, 0.04]])
    off = a + np.array([0.01, 0.0, 0.0])
    res = chamfer(off, a)
    assert res["mean"] == pytest.approx(0.01, abs=1e-6)
    assert res["median"] == pytest.approx(0.01, abs=1e-6)
    assert res["max"] == pytest.approx(0.01, abs=1e-6)


def test_frac_within_threshold():
    a = np.array([[0.0, 0.0, 0.0], [0.0, 0.03, 0.04]])
    off = a + np.array([0.01, 0.0, 0.0])
    assert frac_within_cm(off, a, threshold_cm=1.0) == pytest.approx(1.0)
    assert frac_within_cm(off, a, threshold_cm=0.5) == pytest.approx(0.0)


def test_partial_coverage():
    # recon covers only one of four points within 0.4 cm
    recon = TRUTH[:1] + np.array([0.003, 0.0, 0.0])
    frac = frac_within_cm(recon, TRUTH, threshold_cm=0.4)
    assert frac == pytest.approx(1.0 / 4.0, abs=1e-6)


def test_empty_clouds_do_not_crash():
    res = chamfer(np.empty((0, 3)), TRUTH)
    assert res["mean"] == np.inf
    assert frac_within_cm(np.empty((0, 3)), TRUTH) == 0.0
    feat = detect_feature(np.empty((0, 3)), [0, 0, 0], 0.02)
    assert feat["detected"] is False


def test_detect_feature_hit_and_miss():
    # probe at origin, radius 2 cm: cloud clustered at origin -> hit
    recon = np.random.default_rng(0).normal(0, 0.003, size=(200, 3))
    assert detect_feature(recon, [0, 0, 0], 0.02)["detected"] is True
    # cloud far away -> miss
    far = recon + np.array([0.10, 0.0, 0.0])
    assert detect_feature(far, [0, 0, 0], 0.02)["detected"] is False