import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from baselines import (  # noqa: E402
    XGBoostAFTConcentration,
    conc_to_p,
    p_bounds_to_conc_bounds,
    p_to_conc,
)


def test_conc_roundtrip():
    p = np.array([4.0, 5.0, 6.0, 7.5, 9.0])
    assert np.allclose(conc_to_p(p_to_conc(p)), p, atol=1e-9)
    # pX = 9 <=> 1 nM ; pX = 6 <=> 1000 nM
    assert abs(p_to_conc(np.array([9.0]))[0] - 1.0) < 1e-9
    assert abs(p_to_conc(np.array([6.0]))[0] - 1000.0) < 1e-6


def test_bounds_direction_flip():
    # exact p=6 -> conc interval degenerate at 1000
    L, U = p_bounds_to_conc_bounds(np.array([6.0]), np.array([6.0]))
    assert np.isclose(L[0], 1000.0) and np.isclose(U[0], 1000.0)

    # right-censored potency: pX > 5 (lower_p=5, upper_p=+inf)
    #   => concentration < 10^(9-5)=1e4 : left-censored conc (L_x=floor, U_x=1e4)
    L, U = p_bounds_to_conc_bounds(np.array([5.0]), np.array([np.inf]))
    assert L[0] <= 1e-3 and np.isclose(U[0], 1e4)

    # left-censored potency: pX < 5 (lower_p=-inf, upper_p=5)
    #   => concentration > 1e4 : right-censored conc (L_x=1e4, U_x=+inf)
    L, U = p_bounds_to_conc_bounds(np.array([-np.inf]), np.array([5.0]))
    assert np.isclose(L[0], 1e4) and np.isposinf(U[0])


def test_bounds_are_ordered():
    rng = np.random.default_rng(0)
    lo = rng.uniform(3, 6, 100)
    hi = lo + rng.uniform(0, 2, 100)
    L, U = p_bounds_to_conc_bounds(lo, hi)
    assert np.all(U >= L)


def test_aft_recovers_signal_on_synthetic():
    # y (pX) linear in features; exact labels; AFT in conc space should correlate.
    rng = np.random.default_rng(1)
    n, d = 1500, 10
    X = rng.standard_normal((n, d)).astype(np.float32)
    w = rng.standard_normal(d)
    p = X @ w * 0.5 + 6.0 + 0.2 * rng.standard_normal(n)
    lo = p.copy(); hi = p.copy()      # all exact
    tr, va, te = np.split(rng.permutation(n), [1000, 1250])
    m = XGBoostAFTConcentration(num_boost_round=120, seed=0, n_jobs=4)
    m.fit(X[tr], lo[tr], hi[tr], X[va], lo[va], hi[va])
    pred = m.predict(X[te])
    from scipy.stats import spearmanr
    rho = spearmanr(pred, p[te]).correlation
    assert rho > 0.6, f"AFT failed to recover signal (rho={rho})"
    assert np.isfinite(pred).all()


def test_aft_handles_censoring():
    rng = np.random.default_rng(2)
    n, d = 1200, 8
    X = rng.standard_normal((n, d)).astype(np.float32)
    w = rng.standard_normal(d)
    p = X @ w * 0.4 + 6.0 + 0.2 * rng.standard_normal(n)
    lo = p.copy(); hi = p.copy()
    # right-censor top 30% of potencies: pX > c
    c = np.quantile(p, 0.7)
    rc = p > c
    lo[rc] = c; hi[rc] = np.inf
    tr, va = np.split(rng.permutation(n), [1000])
    m = XGBoostAFTConcentration(num_boost_round=120, seed=0, n_jobs=4)
    m.fit(X[tr], lo[tr], hi[tr], X[va], lo[va], hi[va])
    pred = m.predict(X)
    assert np.isfinite(pred).all()
    # censored (truly high-potency) rows should predict higher pX on average
    assert pred[rc].mean() > pred[~rc].mean()


def test_ensemble_mixture_total_variance():
    # deterministic components; check law-of-total-variance aggregation
    import types
    comps = [(np.array([5.0, 6.0]), np.array([1.0, 1.0])),
             (np.array([7.0, 6.0]), np.array([1.0, 1.0]))]
    # emulate DeepEnsemble.predict math directly
    mus = np.stack([c[0] for c in comps]); sigs = np.stack([c[1] for c in comps])
    mu = mus.mean(0); sigma = np.sqrt((sigs ** 2).mean(0) + mus.var(0))
    assert np.allclose(mu, [6.0, 6.0])
    # row0: within-var 1, between-var = var([5,7])=1 => sigma=sqrt(2)
    assert np.isclose(sigma[0], np.sqrt(2.0))
    assert np.isclose(sigma[1], 1.0)
