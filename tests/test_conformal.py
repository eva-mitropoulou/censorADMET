import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from conformal import MondrianConformal, SplitConformal, _finite_quantile_level


def test_finite_quantile_level():
    assert _finite_quantile_level(99, 0.1) <= 1.0
    assert _finite_quantile_level(9, 0.1) == 1.0  # small n -> full


def test_split_conformal_restores_coverage():
    rng = np.random.default_rng(0)
    n = 4000
    mu = rng.normal(5, 1.5, n)
    # heteroscedastic truth but a MISCALIBRATED constant sigma guess
    true_sigma = 0.5 + 0.5 * (mu > 5)
    y = mu + rng.normal(0, 1, n) * true_sigma
    sigma_guess = np.full(n, 0.3)   # too small -> raw coverage << 0.9
    ex = np.ones(n, bool)
    cal, test = np.arange(2000), np.arange(2000, n)
    sc = SplitConformal(alpha=0.1).calibrate(mu[cal], sigma_guess[cal], y[cal], ex[cal])
    cov = sc.coverage(mu[test], sigma_guess[test], y[test], ex[test])
    assert 0.86 <= cov["coverage"] <= 0.95, cov   # conformal restores ~0.9


def test_split_conformal_only_uses_exact():
    rng = np.random.default_rng(1)
    n = 1000
    mu = rng.normal(5, 1, n); sg = np.ones(n); y = mu + rng.normal(0, 1, n)
    ex = rng.random(n) < 0.5
    y[~ex] = np.nan   # censored rows have no target
    sc = SplitConformal(alpha=0.1).calibrate(mu, sg, y, ex)
    assert sc.q is not None and np.isfinite(sc.q)


def test_mondrian_group_conditional_coverage():
    rng = np.random.default_rng(2)
    n = 6000
    grp = rng.integers(0, 2, n)
    mu = rng.normal(5, 1, n)
    # group 1 has much larger noise; a global q under-covers group 1
    noise = np.where(grp == 1, 2.0, 0.5)
    y = mu + rng.normal(0, 1, n) * noise
    sg = np.ones(n)
    ex = np.ones(n, bool)
    cal, test = np.arange(3000), np.arange(3000, n)
    mc = MondrianConformal(alpha=0.1).calibrate(mu[cal], sg[cal], y[cal], ex[cal], grp[cal])
    # per-group coverage should each be ~0.9
    for g in (0, 1):
        m = test[grp[test] == g]
        cov = mc.coverage(mu[m], sg[m], y[m], ex[m], grp[m])
        assert 0.84 <= cov["coverage"] <= 0.96, (g, cov)


def test_uncalibrated_falls_back():
    # too few exact rows -> fall back to gaussian z-interval, no crash
    sc = SplitConformal(alpha=0.1)
    sc.calibrate(np.zeros(3), np.ones(3), np.zeros(3), np.ones(3, bool))
    lo, hi = sc.interval(np.array([0.0]), np.array([1.0]))
    assert lo[0] < hi[0]
