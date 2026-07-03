"""ml-monitoring-lite のドリフト検出テスト。"""

import numpy as np
import pandas as pd
import pytest

from src.drift import (
    drift_report,
    generate_reference_and_current,
    ks_test,
    psi,
    psi_level,
)


def test_psi_near_zero_for_same_distribution():
    rng = np.random.default_rng(0)
    a = rng.normal(0, 1, 5000)
    b = rng.normal(0, 1, 5000)
    assert psi(a, b) < 0.1


def test_psi_large_for_shifted_distribution():
    rng = np.random.default_rng(0)
    a = rng.normal(0, 1, 5000)
    b = rng.normal(3, 1, 5000)
    assert psi(a, b) > 0.25


def test_psi_empty_and_constant_safe():
    assert psi(np.array([]), np.array([1.0, 2.0])) == 0.0
    assert psi(np.array([5.0, 5.0, 5.0]), np.array([5.0, 5.0])) == 0.0


def test_psi_ignores_nan():
    a = np.array([0.0, 1.0, np.nan, 2.0, 3.0])
    b = np.array([0.0, 1.0, 2.0, 3.0])
    # NaN を落として計算でき、例外にならない
    assert psi(a, b) >= 0.0


def test_ks_detects_difference():
    rng = np.random.default_rng(1)
    a = rng.normal(0, 1, 2000)
    b = rng.normal(2, 1, 2000)
    stat, pval = ks_test(a, b)
    assert stat > 0.3
    assert pval < 0.01


def test_ks_similar_distributions_high_pvalue():
    rng = np.random.default_rng(1)
    a = rng.normal(0, 1, 3000)
    b = rng.normal(0, 1, 3000)
    _, pval = ks_test(a, b)
    assert pval > 0.01


def test_ks_empty_safe():
    assert ks_test(np.array([]), np.array([1.0])) == (0.0, 1.0)


def test_psi_level_thresholds():
    assert psi_level(0.05) == "stable"
    assert psi_level(0.15) == "moderate"
    assert psi_level(0.4) == "significant"


def test_report_no_drift_when_identical():
    ref, cur = generate_reference_and_current(n=3000, drift=0.0, seed=5)
    rep = drift_report(ref, cur)
    assert len(rep) == 4
    # ドリフトなしなら重大アラートは出ないはず
    assert rep["psi"].max() < 0.25


@pytest.mark.parametrize("seed", [0, 1, 2, 5, 11])
def test_report_no_false_alert_at_zero_drift(seed):
    # KS の偽陽性でアラートが点灯しないこと（PSI 主体判定）
    ref, cur = generate_reference_and_current(n=2000, drift=0.0, seed=seed)
    rep = drift_report(ref, cur)
    assert not rep["alert"].any()


def test_report_alerts_when_drifted():
    ref, cur = generate_reference_and_current(n=3000, drift=1.5, seed=5)
    rep = drift_report(ref, cur)
    assert rep["alert"].any()
    assert rep["psi"].max() > 0.25


def test_report_reproducible():
    a1, b1 = generate_reference_and_current(n=500, drift=0.5, seed=9)
    a2, b2 = generate_reference_and_current(n=500, drift=0.5, seed=9)
    pd.testing.assert_frame_equal(a1, a2)
    pd.testing.assert_frame_equal(b1, b2)
