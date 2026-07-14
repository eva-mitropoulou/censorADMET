import math

import numpy as np
import pytest
import torch

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from satisficing_losses import (  # noqa: E402
    LatentDistribution,
    censored_interval_prob,
    exact_nll,
    interval_hinge,
    interval_log_prob,
    satisficing_deficit,
    tobit_nll,
)

G = LatentDistribution("gaussian")
T = LatentDistribution("student_t", df=4.0)


def test_interval_prob_matches_scipy_gaussian():
    from scipy.stats import norm
    mu = torch.tensor([0.0, 1.0, -0.5])
    sig = torch.tensor([1.0, 2.0, 0.5])
    lo = torch.tensor([-1.0, -float("inf"), 0.0])
    hi = torch.tensor([1.0, 2.0, float("inf")])
    q = censored_interval_prob(mu, sig, lo, hi, G).numpy()
    exp = np.array([
        norm.cdf(1) - norm.cdf(-1),                    # interval [-1,1], N(0,1)
        norm.cdf((2 - 1) / 2),                         # left-censored (-inf,2], N(1,2)
        1 - norm.cdf((0 - (-0.5)) / 0.5),              # right-censored [0,inf), N(-0.5,0.5)
    ])
    assert np.allclose(q, exp, atol=1e-5), (q, exp)


def test_student_t_log_cdf_matches_scipy():
    from scipy.stats import t as scit
    z = torch.tensor([-5.0, -1.0, 0.0, 1.0, 5.0], dtype=torch.float64)
    got = T.log_cdf_standardized(z).numpy()
    exp = scit.logcdf(z.numpy(), 4.0)
    assert np.max(np.abs(got - exp)) < 1e-5


def test_satisficing_zero_once_satisfied():
    # right-censored y > 0 (lower=0), N(mu, 1). As mu increases, q = P(Y>0) increases.
    tau = 0.8
    lo = torch.tensor([0.0]); hi = torch.tensor([float("inf")]); sig = torch.tensor([1.0])
    # mu well above bound -> q ~ 1 >= tau -> deficit 0
    d_hi = satisficing_deficit(torch.tensor([3.0]), sig, lo, hi, G, tau).item()
    # mu below bound -> q < 0.5 < tau -> deficit > 0
    d_lo = satisficing_deficit(torch.tensor([-1.0]), sig, lo, hi, G, tau).item()
    assert d_hi == 0.0
    assert d_lo > 0.0


def test_satisficing_gradient_vanishes_when_satisfied():
    # THE core property: once q>=tau, gradient is exactly zero (no extrapolation reward).
    tau = 0.8
    lo = torch.tensor([0.0]); hi = torch.tensor([float("inf")]); sig = torch.tensor([1.0])
    mu_sat = torch.tensor([4.0], requires_grad=True)   # q(P(Y>0)) ~ 1 > tau
    d = satisficing_deficit(mu_sat, sig, lo, hi, G, tau)
    d.backward()
    assert abs(float(mu_sat.grad)) < 1e-8, "satisfied row must have ~zero gradient"

    mu_unsat = torch.tensor([-0.5], requires_grad=True)  # q < tau
    d2 = satisficing_deficit(mu_unsat, sig, lo, hi, G, tau)
    d2.backward()
    assert float(mu_unsat.grad) < -1e-3, "unsatisfied right-censored row must pull mu UP (neg grad)"


def test_tobit_gradient_never_vanishes_contrast():
    # Contrast (plan §1): standard Tobit keeps a nonzero gradient even when the
    # bound is comfortably satisfied, i.e. it keeps pushing mu further out.
    lo = torch.tensor([0.0]); hi = torch.tensor([float("inf")]); sig = torch.tensor([1.0])
    mu = torch.tensor([3.0], requires_grad=True)   # already far above bound
    loss = tobit_nll(mu, sig, lo, hi, G).sum()
    loss.backward()
    assert float(mu.grad) < -1e-4, "Tobit still rewards moving mu further above the bound"


def test_tobit_finite_gradients_on_mixed_batch():
    # Regression: tobit_nll called exact_nll over the FULL batch; left-censored
    # rows carry lower=-inf, so the exact branch was +inf and torch.where's
    # backward pass hit 0*inf = NaN. Must be finite for a mixed batch.
    torch.manual_seed(0)
    n = 150
    mu = (torch.randn(n) * 3).detach().requires_grad_(True)
    sig = torch.nn.functional.softplus(torch.randn(n)) + 0.05
    lo = torch.randn(n); hi = lo.clone()          # start all-exact
    hi[:50] = float("inf")                         # right-censored
    lo[50:100] = float("-inf")                     # left-censored (the trap)
    for dist in (G, T):
        loss = tobit_nll(mu, sig, lo, hi, dist).mean()
        loss.backward()
        assert torch.isfinite(loss), f"non-finite tobit loss for {dist.dist}"
        assert torch.isfinite(mu.grad).all(), f"NaN tobit grad for {dist.dist}"
        mu.grad = None


def test_finite_gradients_on_mixed_batch():
    torch.manual_seed(0)
    n = 200
    mu = (torch.randn(n) * 5).detach().requires_grad_(True)
    sig = torch.nn.functional.softplus(torch.randn(n)) + 0.05
    lo = torch.randn(n); hi = lo.clone()
    hi[:60] = float("inf")            # right
    lo[60:120] = float("-inf")        # left
    hi[180:] = lo[180:] + 1e-7        # near-degenerate interval
    for dist in (G, T):
        d = satisficing_deficit(mu, sig, lo, hi, dist, tau=0.8).sum()
        d.backward()
        assert torch.isfinite(mu.grad).all(), f"NaN grad for {dist.dist}"
        mu.grad = None


def test_narrow_interval_not_misclassified_as_exact():
    # A genuine narrow interval-censored row [5.00, 5.01] must NOT be treated as
    # exact (torch.isclose would have). Its interval probability must be < 1 and
    # its deficit must respond to the model.
    lo = torch.tensor([5.00]); hi = torch.tensor([5.01]); sig = torch.tensor([1.0])
    from satisficing_losses import censored_interval_prob
    q = censored_interval_prob(torch.tensor([5.005]), sig, lo, hi, G).item()
    assert 0.0 < q < 0.02, f"narrow interval prob should be small, got {q}"
    # deficit should be positive (q << tau) and have a gradient
    mu = torch.tensor([5.005], requires_grad=True)
    d = satisficing_deficit(mu, sig, lo, hi, G, tau=0.8)
    d.backward()
    assert d.item() > 0
    assert torch.isfinite(mu.grad).all()


def test_explicit_exact_mask_overrides_bounds():
    # If caller marks a row exact, it is exact even if bounds differ slightly;
    # if caller marks a row NON-exact, equal bounds are treated as a degenerate
    # interval (deficit 0 because q -> the point mass is measure zero -> treated
    # as censored with prob ~0, so deficit ~tau^2). We just require no crash and
    # that the mask is honored for the exact case (deficit 0).
    lo = torch.tensor([5.0, 5.0]); hi = torch.tensor([5.0, 5.0]); sig = torch.tensor([1.0, 1.0])
    em = torch.tensor([True, True])
    d = satisficing_deficit(torch.tensor([2.0, 2.0]), sig, lo, hi, G, tau=0.8, exact_mask=em)
    assert torch.allclose(d, torch.zeros(2)), "explicitly-exact rows must have zero deficit"


def test_float32_collapsed_interval_not_silently_satisfied():
    # A non-exact interval row whose bounds collapse to equal under float32 must
    # NOT fall through all branches to log-prob 0 (=> q=1 => deficit 0). It should
    # be treated as a near-degenerate interval (tiny prob, large deficit).
    lo = torch.tensor([6.5000001], dtype=torch.float32)
    hi = torch.tensor([6.5000002], dtype=torch.float32)
    assert lo.item() == hi.item(), "these should collapse under float32"
    em = torch.tensor([False])  # explicitly NOT exact
    from satisficing_losses import censored_interval_prob
    # model far from the interval -> deficit should be large, not zero
    q = censored_interval_prob(torch.tensor([2.0]), torch.tensor([1.0]), lo, hi, G, exact_mask=em).item()
    assert q < 0.5, f"collapsed interval must not be silently satisfied (q={q})"
    d = satisficing_deficit(torch.tensor([2.0]), torch.tensor([1.0]), lo, hi, G, tau=0.8, exact_mask=em).item()
    assert d > 0.3, f"collapsed-interval deficit should be large, got {d}"


def test_far_tail_interval_prob_not_floored():
    # An interval far in the right tail should give a tiny but NON-floored prob
    # (the old _log1mexp clamp floored it near exp(-27.6) ~ 1e-12).
    from satisficing_losses import censored_interval_prob
    # interval [20, 21] under N(0,1): astronomically small probability
    q = censored_interval_prob(torch.tensor([0.0], dtype=torch.float64),
                               torch.tensor([1.0], dtype=torch.float64),
                               torch.tensor([20.0], dtype=torch.float64),
                               torch.tensor([21.0], dtype=torch.float64), G).item()
    assert q < 1e-40, f"far-tail interval prob should be << 1e-12, got {q}"


def test_exact_nll_reduces_to_gaussian():
    v = exact_nll(torch.tensor([0.0]), torch.tensor([1.0]), torch.tensor([0.0]), G).item()
    assert abs(v - 0.5 * math.log(2 * math.pi)) < 1e-6


def test_interval_hinge_zero_inside():
    mu = torch.tensor([0.5, 3.0, -2.0])
    lo = torch.tensor([0.0, -float("inf"), 0.0])
    hi = torch.tensor([1.0, 2.0, float("inf")])
    h = interval_hinge(mu, lo, hi).numpy()
    assert h[0] == 0.0            # inside [0,1]
    assert h[1] == pytest.approx(1.0)   # 3.0 above upper 2.0
    assert h[2] == pytest.approx(2.0)   # -2.0 below lower 0.0
