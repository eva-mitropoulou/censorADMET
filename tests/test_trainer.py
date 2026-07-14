import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from heads import HeteroscedasticMLP  # noqa: E402
from satisficing_losses import LatentDistribution, satisficing_deficit  # noqa: E402
from satisficing_trainer import (  # noqa: E402
    ConstraintSpec,
    SatisficingTrainer,
    TrainConfig,
    _cvar,
    default_direction_constraints,
)

G = LatentDistribution("gaussian")


def _make_censored_data(n=2000, d=8, seed=0, cens_frac=0.5):
    """Latent y = x·w + noise; right-censor the top cens_frac of an assay-cutoff
    style: values above a per-row threshold become right-censored (y > c)."""
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, d)).astype(np.float32)
    w = rng.standard_normal(d).astype(np.float32)
    y = X @ w + 0.3 * rng.standard_normal(n).astype(np.float32) + 5.0
    # censoring threshold: a fixed cutoff so a fraction of rows are right-censored
    c = np.quantile(y, 1.0 - cens_frac)
    lower = np.where(y > c, c, y).astype(np.float32)
    upper = np.where(y > c, np.inf, y).astype(np.float32)
    exact = ~(y > c)
    return X, y, lower, upper, exact, c


def test_trainer_reduces_violation_below_budget():
    X, y, lo, hi, ex, c = _make_censored_data(n=2500, seed=1)
    eps = 0.03
    specs = default_direction_constraints(lo, hi, ex, eps)
    model = HeteroscedasticMLP(X.shape[1], hidden=(64, 64))
    cfg = TrainConfig(tau=0.8, nu=0.0, epochs=120, batch_size=512, lr=3e-3,
                      rho_init=1.0, seed=0)
    tr = SatisficingTrainer(model, G, cfg).fit(X, lo, hi, ex, specs)

    # measure achieved right-direction violation on full data
    mu, sigma = tr.predict(X)
    d = satisficing_deficit(
        torch.tensor(mu), torch.tensor(sigma),
        torch.tensor(lo), torch.tensor(hi), G, cfg.tau
    )
    right = torch.isposinf(torch.tensor(hi))
    achieved = float(d[right].mean())
    # should be at or below budget (allow small slack); definitely far below the
    # ~ (0.8-0.5)^2 = 0.09 you'd get from an unconstrained mean-only fit.
    assert achieved <= eps + 0.02, f"violation {achieved} exceeds budget {eps}"


def test_pareto_frontier_monotone():
    """Looser budget eps -> the model is allowed MORE violation, and in exchange
    should achieve BETTER exact-value accuracy (lower MAE on exact rows)."""
    X, y, lo, hi, ex, c = _make_censored_data(n=2500, seed=2)
    results = {}
    for eps in (0.02, 0.20):
        torch.manual_seed(0)
        specs = default_direction_constraints(lo, hi, ex, eps)
        model = HeteroscedasticMLP(X.shape[1], hidden=(64, 64))
        cfg = TrainConfig(tau=0.85, nu=0.0, epochs=150, batch_size=512, lr=3e-3, seed=0)
        tr = SatisficingTrainer(model, G, cfg).fit(X, lo, hi, ex, specs)
        mu, sigma = tr.predict(X)
        mae = float(np.mean(np.abs(mu[ex] - y[ex])))
        d = satisficing_deficit(torch.tensor(mu), torch.tensor(sigma),
                                torch.tensor(lo), torch.tensor(hi), G, cfg.tau)
        viol = float(d[torch.isposinf(torch.tensor(hi))].mean())
        results[eps] = (mae, viol)
    mae_tight, viol_tight = results[0.02]
    mae_loose, viol_loose = results[0.20]
    # tighter budget => not-lower violation, and typically higher exact MAE
    assert viol_tight <= viol_loose + 1e-3, (results,)
    assert mae_loose <= mae_tight + 0.05, f"loosening budget should not hurt accuracy: {results}"


def test_anchor_pulls_toward_reference():
    X, y, lo, hi, ex, c = _make_censored_data(n=1500, seed=3)
    ref = np.full(X.shape[0], 42.0, dtype=np.float32)  # absurd constant reference
    specs = default_direction_constraints(lo, hi, ex, eps=0.5)  # loose -> anchor dominates
    model = HeteroscedasticMLP(X.shape[1], hidden=(32,))
    cfg = TrainConfig(tau=0.8, nu=50.0, epochs=80, batch_size=512, lr=3e-3, seed=0)
    tr = SatisficingTrainer(model, G, cfg).fit(X, lo, hi, ex, specs, anchor_mu=ref)
    mu, _ = tr.predict(X)
    # strong anchor should drag predictions toward 42
    assert np.mean(mu) > 20.0, f"anchor did not pull predictions up: mean={np.mean(mu)}"


def test_cvar_reduction_properties():
    d = torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.9, 1.0])
    # CVaR of worst 10% ~ the max; CVaR at alpha=1 == mean
    assert _cvar(d, 0.1) >= _cvar(d, 1.0)
    assert abs(float(_cvar(d, 1.0)) - float(d.mean())) < 1e-5
    assert float(_cvar(d, 0.1)) >= float(d.max()) - 0.11


def test_cvar_constraint_controls_tail():
    """A CVaR constraint should shrink the worst-tail deficit more than a mean
    constraint at the same budget (it directly penalises the tail)."""
    X, y, lo, hi, ex, c = _make_censored_data(n=2500, seed=5)
    right = np.isposinf(hi)
    tail = {}
    for red in ("mean", "cvar"):
        torch.manual_seed(0)
        spec = ConstraintSpec("right", torch.tensor(right), eps=0.03,
                              reduction=red, alpha=0.1)
        model = HeteroscedasticMLP(X.shape[1], hidden=(64, 64))
        cfg = TrainConfig(tau=0.85, nu=0.0, epochs=120, batch_size=512, lr=3e-3, seed=0)
        tr = SatisficingTrainer(model, G, cfg).fit(X, lo, hi, ex, [spec])
        mu, sigma = tr.predict(X)
        d = satisficing_deficit(torch.tensor(mu), torch.tensor(sigma),
                                torch.tensor(lo), torch.tensor(hi), G, cfg.tau)
        dr = d[torch.tensor(right)]
        tail[red] = float(torch.quantile(dr, 0.95))
    assert tail["cvar"] <= tail["mean"] + 1e-3, f"CVaR did not control tail: {tail}"


def test_rho_does_not_run_away_when_feasible():
    # Once feasible, rho must not keep doubling toward rho_max (the old shrink-
    # ratio test fired on any tiny non-shrink). With a loose budget the model is
    # quickly feasible and rho should stay small.
    X, y, lo, hi, ex, c = _make_censored_data(n=1500, seed=11)
    specs = default_direction_constraints(lo, hi, ex, eps=0.5)   # very loose
    model = HeteroscedasticMLP(X.shape[1], hidden=(32,))
    cfg = TrainConfig(tau=0.8, nu=0.0, epochs=60, batch_size=512, lr=3e-3,
                      rho_init=1.0, rho_growth=2.0, rho_max=1e4, seed=0)
    tr = SatisficingTrainer(model, G, cfg).fit(X, lo, hi, ex, specs)
    assert tr._rho < 1e3, f"rho ran away to {tr._rho} despite feasibility"


def test_al_consistency_linearized_penalty_reaches_budget():
    # The linearized (unbiased-gradient) AL should still drive violation to the
    # budget on a moderately tight constraint -- the bias fix must not break
    # convergence.
    X, y, lo, hi, ex, c = _make_censored_data(n=2500, seed=12)
    eps = 0.05
    specs = default_direction_constraints(lo, hi, ex, eps)
    model = HeteroscedasticMLP(X.shape[1], hidden=(64, 64))
    cfg = TrainConfig(tau=0.85, nu=0.0, epochs=150, batch_size=512, lr=3e-3, seed=0)
    tr = SatisficingTrainer(model, G, cfg).fit(X, lo, hi, ex, specs)
    # final full-data right-direction constraint should be near/under budget
    assert tr.history, "no history recorded"
    final_G = tr.history[-1]["G"]
    assert min(final_G.values()) <= eps + 0.03, final_G


def test_gradless_batch_does_not_crash():
    # Regression: a batch/fold with NO exact rows and all-feasible constraints
    # yields a loss with no grad path; fit() must skip it, not crash with
    # "element 0 of tensors does not require grad".
    rng = np.random.default_rng(0)
    n, d = 200, 8
    X = rng.standard_normal((n, d)).astype(np.float32)
    # ALL rows right-censored (no exact rows at all), loose budget so constraints
    # quickly become feasible and t_coef -> 0.
    lo = rng.uniform(4, 5, n).astype(np.float32)
    hi = np.full(n, np.inf, np.float32)
    ex = np.zeros(n, bool)
    specs = default_direction_constraints(lo, hi, ex, eps=0.9)  # very loose
    model = HeteroscedasticMLP(d, hidden=(16,))
    cfg = TrainConfig(tau=0.5, nu=0.0, epochs=15, batch_size=64, seed=0)
    # should complete without raising
    tr = SatisficingTrainer(model, G, cfg).fit(X, lo, hi, ex, specs)
    mu, sigma = tr.predict(X)
    assert np.isfinite(mu).all()


def test_dual_variables_nonnegative():
    X, y, lo, hi, ex, c = _make_censored_data(n=1200, seed=7)
    specs = default_direction_constraints(lo, hi, ex, eps=0.05)
    model = HeteroscedasticMLP(X.shape[1], hidden=(32,))
    cfg = TrainConfig(epochs=40, batch_size=512, seed=0)
    tr = SatisficingTrainer(model, G, cfg).fit(X, lo, hi, ex, specs)
    assert (tr._lambda >= 0).all(), tr._lambda


def test_assay_random_effects_shrink():
    """With assay ids, the model learns per-assay intercepts; the shrinkage prior
    keeps them finite and index 0 (unknown) pinned to zero."""
    X, y, lo, hi, ex, c = _make_censored_data(n=2000, seed=9)
    rng = np.random.default_rng(0)
    n = X.shape[0]
    # 4 real assays (idx 1..4) each with a mean offset baked into y
    assay = rng.integers(1, 5, size=n)
    offsets = np.array([0.0, -2.0, 1.5, -1.0, 3.0], dtype=np.float32)
    y2 = y + offsets[assay]
    lo2 = np.where(np.isposinf(hi), lo, y2).astype(np.float32)  # keep censoring structure
    hi2 = hi.copy()
    lo2 = np.where(np.isposinf(hi2), np.quantile(y2, 0.5), y2).astype(np.float32)
    ex2 = ~np.isposinf(hi2)
    model = HeteroscedasticMLP(X.shape[1], hidden=(64,), n_assays=5)
    cfg = TrainConfig(tau=0.8, nu=0.0, epochs=100, batch_size=512, lr=3e-3,
                      assay_prior_weight=0.01, seed=0)
    specs = default_direction_constraints(lo2, hi2, ex2, eps=0.05)
    tr = SatisficingTrainer(model, G, cfg).fit(X, lo2, hi2, ex2, specs, assay_idx=assay)
    b = model.b_assay.weight.detach().cpu().numpy().ravel()
    assert np.isfinite(b).all()
    assert abs(b[0]) < 1e-6, "unknown-assay index 0 must stay pinned to zero"
    # learned offsets should correlate with true offsets on idx 1..4
    corr = np.corrcoef(b[1:], offsets[1:])[0, 1]
    assert corr > 0.5, f"assay effects did not track truth (corr={corr})"
