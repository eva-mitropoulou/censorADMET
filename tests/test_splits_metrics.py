import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from metrics import (  # noqa: E402
    accuracy_metrics,
    decision_metrics,
    exceedance_calibration,
    exceedance_prob,
    interval_coverage,
    interval_violation_rate,
    predictive_quantile,
)
from splits import _grouped_kfold, make_splits  # noqa: E402


# ---- a tiny fake dataset object with a .meta and .n ---- #
class _DS:
    def __init__(self, meta, exact_mask=None):
        self.meta = meta
        self.n = len(meta)
        self.exact_mask = (np.ones(len(meta), dtype=bool) if exact_mask is None
                           else np.asarray(exact_mask, dtype=bool))


def _fake_meas(n=600, seed=0):
    rng = np.random.default_rng(seed)
    exact = rng.random(n) < 0.6
    return _DS(pd.DataFrame({
        "assay_chembl_id": rng.integers(0, 20, n).astype(str),
        "document_chembl_id": rng.integers(0, 15, n).astype(str),
        "source_name": rng.choice(["PubChem BioAssays", "Scientific Literature", "DrugMatrix"], n),
        "value_p": rng.normal(5, 1, n),
        "split_group_scaffold": rng.integers(0, 40, n).astype(str),
        "standardized_smiles": [f"C{i%50}" for i in range(n)],
    }), exact_mask=exact)


def test_grouped_kfold_no_leakage():
    groups = np.array([str(i % 10) for i in range(200)])
    for tr, te in _grouped_kfold(groups, k=5, seed=0):
        assert not (set(groups[tr]) & set(groups[te])), "group appears in both splits"
        assert len(tr) + len(te) <= 200


def test_make_splits_covers_kinds():
    ds = _fake_meas()
    splits = make_splits(ds, k=4, seed=1)
    for kind in ("random", "assay", "document", "source", "threshold"):
        assert kind in splits, f"{kind} missing"
        for name, tr, te in splits[kind]:
            assert len(tr) > 0 and len(te) > 0


def test_assay_source_splits_hold_out_groups():
    ds = _fake_meas()
    splits = make_splits(ds, k=4, seed=2)
    for name, tr, te in splits["assay"]:
        a = ds.meta["assay_chembl_id"].to_numpy()
        assert not (set(a[tr]) & set(a[te]))
    for name, tr, te in splits["source"]:
        s = ds.meta["source_name"].to_numpy()
        assert not (set(s[tr]) & set(s[te]))


def test_threshold_split_excludes_censored_from_test():
    # censored rows (finite value_p = threshold) must never land in a test band.
    from splits import threshold_split
    rng = np.random.default_rng(0)
    n = 600
    vp = rng.uniform(4, 8, n)
    exact = rng.random(n) < 0.6

    class _DSt:
        pass
    ds = _DSt()
    ds.meta = pd.DataFrame({"value_p": vp})
    ds.exact_mask = exact
    ds.n = n
    folds = threshold_split(ds, n_bands=3)
    assert folds, "expected threshold folds"
    for name, tr, te in folds:
        assert exact[te].all(), "a censored row leaked into a threshold test band"


def test_accuracy_only_on_exact():
    y = np.array([5.0, 6.0, np.nan, 4.0])
    mu = np.array([5.1, 5.9, 99.0, 3.8])
    ex = np.array([True, True, False, True])
    m = accuracy_metrics(mu, y, ex)
    assert m["n_exact"] == 3
    assert m["mae"] < 0.2   # ignores the censored row's absurd prediction


def test_violation_rate_direction():
    # right-censored y>5 (lo=5, hi=inf): mu=3 violates (below bound)
    mu = np.array([3.0, 7.0])
    lo = np.array([5.0, 5.0]); hi = np.array([np.inf, np.inf])
    ex = np.array([False, False])
    r = interval_violation_rate(mu, lo, hi, ex)
    assert r["violation_rate"] == 0.5
    assert r["right_violation_rate"] == 0.5


def test_exceedance_prob_monotone_and_mixture():
    mu = np.array([5.0]); s = np.array([1.0])
    p_lo = exceedance_prob((mu, s), 4.0)   # t below mu -> high P(Y>t)
    p_hi = exceedance_prob((mu, s), 6.0)   # t above mu -> low
    assert p_lo[0] > 0.5 > p_hi[0]
    # mixture of two components averages the exceedance
    pm = exceedance_prob([(np.array([4.0]), s), (np.array([6.0]), s)], 5.0)
    assert abs(pm[0] - 0.5) < 1e-6


def test_predictive_quantile_gaussian():
    mu = np.array([0.0]); s = np.array([1.0])
    q = predictive_quantile((mu, s), 0.5)
    assert abs(q[0]) < 0.1     # median ~ 0
    q90 = predictive_quantile((mu, s), 0.9)
    assert 1.0 < q90[0] < 1.6  # ~1.2816


def test_predictive_quantile_per_row_resolution():
    # Heteroscedastic: a tight-sigma row alongside a wide-sigma row. A single
    # shared grid would give the tight row coarse resolution; per-row grids fix it.
    from scipy.stats import norm
    mu = np.array([0.0, 100.0]); s = np.array([0.01, 50.0])
    q50 = predictive_quantile((mu, s), 0.5)
    assert abs(q50[0] - 0.0) < 0.01, q50           # tight row median accurate
    q90 = predictive_quantile((mu, s), 0.9)
    # tight row 0.9-quantile ~ 0.01*1.2816
    assert abs(q90[0] - 0.01 * norm.ppf(0.9)) < 5e-3, q90[0]


def test_predictive_quantile_student_t_heavy_tail():
    # Student-t df=4 extreme quantile must not truncate at an 8-sigma grid edge.
    from scipy.stats import t as scit
    q = predictive_quantile((np.array([0.0]), np.array([1.0])), 0.999, dist="student_t", df=4.0)
    true = scit.ppf(0.999, 4.0)   # ~7.17
    assert abs(q[0] - true) < 0.5, (q[0], true)


def test_decision_metrics_resolvable():
    # 6 exact rows around threshold t=5 (>= min-resolvable guard of 5)
    y = np.array([6.0, 6.5, 7.0, 4.0, 3.5, 3.0])
    mu = np.array([6.2, 6.1, 6.8, 4.2, 3.9, 3.1]); s = np.full(6, 0.5)
    lo = y.copy(); hi = y.copy(); ex = np.array([True] * 6)
    d = decision_metrics((mu, s), y, lo, hi, ex, t=5.0, prob_cut=0.5)
    assert d["n_resolvable"] == 6
    assert d["false_safe_rate"] == 0.0        # all true-actives flagged
    assert d["false_liability_rate"] == 0.0   # all true-inactives not flagged


def test_decision_uses_censored_known_side():
    # right-censored y>7 with threshold 5 => truth known True (7>=5)
    y = np.array([np.nan]); lo = np.array([7.0]); hi = np.array([np.inf])
    ex = np.array([False])
    mu = np.array([7.5]); s = np.array([1.0])
    d = decision_metrics((mu, s), y, lo, hi, ex, t=5.0, prob_cut=0.5)
    # only 5 resolvable required; replicate to pass the min-count guard
    y = np.full(6, np.nan); lo = np.full(6, 7.0); hi = np.full(6, np.inf)
    ex = np.zeros(6, bool); mu = np.full(6, 7.5); s = np.full(6, 1.0)
    d = decision_metrics((mu, s), y, lo, hi, ex, t=5.0)
    assert d["n_resolvable"] == 6
    assert d["n_positive"] == 6
    assert d["false_safe_rate"] == 0.0   # all correctly flagged as above


def test_interval_coverage_wellspecified():
    rng = np.random.default_rng(0)
    n = 3000
    mu_true = rng.normal(5, 2, n)
    y = mu_true + rng.normal(0, 1.0, n)
    ex = np.ones(n, bool)
    cov = interval_coverage((mu_true, np.ones(n)), y, ex, level=0.9)
    assert 0.86 < cov["coverage"] < 0.94, cov


def test_exceedance_calibration_wellspecified():
    rng = np.random.default_rng(1)
    n = 4000
    mu = rng.normal(5, 1.5, n)
    y = mu + rng.normal(0, 1.0, n)
    ex = np.ones(n, bool)
    cal = exceedance_calibration((mu, np.ones(n)), y, y.copy(), y.copy(), ex, t=5.0)
    assert cal["ece"] < 0.05, cal
