import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from hidden_truth import (  # noqa: E402
    build_hidden_truth,
    hidden_truth_recovery,
    learn_censoring_mechanism,
)


def _real_like(seed=0):
    rng = np.random.default_rng(seed)
    n = 2000
    y = rng.normal(6, 1, n)
    cc = np.array(["exact"] * n, dtype=object)
    lo = y.copy(); hi = y.copy()
    # 20% right-censored at cutoffs ~5, 10% left-censored at ~4.5
    ridx = rng.choice(n, 400, replace=False)
    cc[ridx] = "right_censored_p"; lo[ridx] = 5.0; hi[ridx] = np.inf
    remaining = np.setdiff1d(np.arange(n), ridx)
    lidx = rng.choice(remaining, 200, replace=False)
    cc[lidx] = "left_censored_p"; lo[lidx] = -np.inf; hi[lidx] = 4.5
    ex = cc == "exact"
    return lo, hi, ex, cc


def test_learn_mechanism():
    lo, hi, ex, cc = _real_like()
    m = learn_censoring_mechanism(lo, hi, ex, cc)
    assert abs(m["p_censored"] - 0.3) < 0.02
    assert abs(m["p_right"] - (400 / 600)) < 0.02
    assert len(m["right_thresholds"]) == 400
    assert np.allclose(m["right_thresholds"], 5.0)


def test_build_hidden_truth_realistic():
    lo, hi, ex, cc = _real_like()
    m = learn_censoring_mechanism(lo, hi, ex, cc)
    y_star = np.random.default_rng(1).normal(6, 1, 1500)
    L, U, exm, rec = build_hidden_truth(y_star, m, seed=0)
    # some rows re-censored; every re-censored row must have its truth on the
    # censored side of the recorded bound
    assert rec.sum() > 0
    for i in np.where(rec)[0]:
        if np.isposinf(U[i]):          # right-censored: y* > lower bound
            assert y_star[i] > L[i]
        elif np.isneginf(L[i]):        # left-censored: y* < upper bound
            assert y_star[i] < U[i]
    # exact rows keep their truth
    assert np.allclose(L[exm], y_star[exm])


def test_hidden_truth_recovery_scoring():
    y_star = np.array([5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
    mu = y_star + 0.5   # over-predicts by 0.5
    rec = np.array([True, True, True, True, False, False])
    r = hidden_truth_recovery(mu, y_star, rec)
    assert r["n_hidden"] == 4
    assert abs(r["hidden_mae"] - 0.5) < 1e-9
    assert abs(r["hidden_bias"] - 0.5) < 1e-9   # positive => over-predicts


def test_informative_mnar_mode_censors_high_potency_more():
    lo, hi, ex, cc = _real_like()
    m = learn_censoring_mechanism(lo, hi, ex, cc)
    rng = np.random.default_rng(3)
    y_star = rng.normal(6, 1.5, 3000)
    L, U, exm, rec = build_hidden_truth(y_star, m, seed=0, informative=True, mnar_strength=1.0)
    # mean potency of re-censored (right) rows should exceed that of exact rows
    right = rec & np.isposinf(U)
    if right.sum() > 20 and exm.sum() > 20:
        assert y_star[right].mean() > y_star[exm].mean()
