"""Probability-satisficing censored-regression losses (CensorADMET 2.0 method core).

Motivation (see chatgpt_plan.md §1-2). A right-censored label y > c only tells us
the latent value is above c; the standard interval/Tobit NLL keeps rewarding a
right-censored prediction as mu moves further above c (its gradient never says
"the bound is satisfied, stop extrapolating"). Under heavy censoring and an
imperfect distributional model this over-prioritises bound satisfaction at the
cost of exact-value accuracy.

The satisficing formulation instead asks the model to assign at least a required
probability tau to the OBSERVED interval, and stops rewarding it once that is
met:

    q_i(theta) = P_theta(L_i <= Y_i <= U_i | x_i)          (interval probability)
    d_i        = [tau - q_i]_+^2                            (satisficing deficit)

Exact labels keep a proper exact-value NLL. The censored deficits are then used
as CONSTRAINTS (see satisficing_trainer.py), not as an arbitrary weighted term.

All probabilities are computed in log-space from the log-CDF so that gradients
are finite for strongly one-sided rows (this repo previously hit a NaN-gradient
trap from evaluating CDFs at +/-inf bounds inside torch.where; every function
here sanitises infinite bounds before forming standardized scores).

Distributions supported for the latent Y: Gaussian and Student-t (heavy-tailed,
plan §3). Both expose log_pdf (exact) and log_cdf (for tail/interval mass).
"""
from __future__ import annotations

import math

import numpy as np


def _torch():
    import torch

    return torch


LOG_SQRT_2PI = 0.5 * math.log(2.0 * math.pi)


# --------------------------------------------------------------------------- #
# Numerically stable primitives                                               #
# --------------------------------------------------------------------------- #
def _log1mexp(x, floor: float = -1e-10):
    """Stable log(1 - exp(x)) for x <= 0 (Maechler 2012).

    x is clamped at `floor` (default -1e-10) so its gradient -exp(x)/expm1(x)
    ~ -1/x stays finite (~1e10) instead of overflowing to +/-inf -> NaN as x->0.
    This clamp only engages for inputs closer to 0 than 1e-10 (a probability
    fraction < ~1e-10). For interval rows near-degenerate cases are routed through
    a density approximation (see interval_log_prob), so the clamp is not exercised
    on used interval probabilities. NOTE: the right-/left-censored Student-t
    survival path (out=_log1mexp(logF)) can still hit the floor when the bound is
    hundreds of sigma into the tail (for df=4 only beyond ~400 sigma); there the
    survival prob is frozen at ~1e-10 instead of its true tinier value. This has
    negligible effect on the satisficing deficit ([tau - 1e-10]^2 vs
    [tau - 1e-12]^2 at any realistic tau) and keeps the gradient finite, but the
    reported tail probability itself is a floor, not exact. Inputs are log-CDF
    differences (<= 0)."""
    torch = _torch()
    x = torch.clamp(x, max=floor)
    return torch.where(
        x < -0.6931471805599453,
        torch.log1p(-torch.exp(x)),
        torch.log(-torch.expm1(x)),
    )


def _gaussian_log_cdf(z):
    """log Phi(z) via torch.special.log_ndtr (stable in both tails)."""
    torch = _torch()
    return torch.special.log_ndtr(z)


def _gaussian_log_pdf(z):
    torch = _torch()
    return -0.5 * z * z - LOG_SQRT_2PI


def _student_t_log_pdf(z, nu):
    torch = _torch()
    nu_t = torch.as_tensor(float(nu), dtype=z.dtype, device=z.device)
    c = (
        torch.lgamma(0.5 * (nu_t + 1.0))
        - torch.lgamma(0.5 * nu_t)
        - 0.5 * torch.log(nu_t * math.pi)
    )
    return c - 0.5 * (nu_t + 1.0) * torch.log1p(z * z / nu_t)


def _betacf(a, b, x, n_iter: int = 80, tol: float = 1e-12):
    """Lentz continued fraction for the regularized incomplete beta (differentiable)."""
    torch = _torch()
    tiny = 1e-300
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c = torch.ones_like(x)
    d = 1.0 - qab * x / qap
    d = torch.where(torch.abs(d) < tiny, torch.full_like(d, tiny), d)
    d = 1.0 / d
    h = d.clone()
    for m in range(1, n_iter + 1):
        m2 = 2.0 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        d = torch.where(torch.abs(d) < tiny, torch.full_like(d, tiny), d)
        c = 1.0 + aa / c
        c = torch.where(torch.abs(c) < tiny, torch.full_like(c, tiny), c)
        d = 1.0 / d
        h = h * d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        d = torch.where(torch.abs(d) < tiny, torch.full_like(d, tiny), d)
        c = 1.0 + aa / c
        c = torch.where(torch.abs(c) < tiny, torch.full_like(c, tiny), c)
        d = 1.0 / d
        delta = d * c
        h = h * delta
        if bool(torch.all(torch.abs(delta - 1.0) < tol)):
            break
    return h


def _betainc(a, b, x):
    """Regularized incomplete beta I_x(a,b), differentiable, elementwise."""
    torch = _torch()
    x = torch.clamp(x, min=1e-300, max=1.0 - 1e-16)
    lbeta = torch.lgamma(a) + torch.lgamma(b) - torch.lgamma(a + b)
    ln_front = a * torch.log(x) + b * torch.log1p(-x) - lbeta
    front = torch.exp(ln_front)
    thresh = (a + 1.0) / (a + b + 2.0)
    direct = front * _betacf(a, b, x) / a
    mirror = 1.0 - front * _betacf(b, a, 1.0 - x) / b
    return torch.where(x < thresh, direct, mirror)


def _student_t_log_cdf(z, nu):
    """log F_t(z; nu) via the regularized incomplete beta (stable, differentiable)."""
    torch = _torch()
    nu_t = torch.as_tensor(float(nu), dtype=torch.float64, device=z.device)
    z = z.to(torch.float64)
    x = nu_t / (nu_t + z * z)
    a = 0.5 * nu_t * torch.ones_like(z)
    b = 0.5 * torch.ones_like(z)
    ib = _betainc(a, b, x)  # I_x(nu/2, 1/2)
    cdf = torch.where(z > 0, 1.0 - 0.5 * ib, 0.5 * ib)
    return torch.log(torch.clamp(cdf, min=1e-300))


# --------------------------------------------------------------------------- #
# Distribution wrapper                                                        #
# --------------------------------------------------------------------------- #
class LatentDistribution:
    """Location-scale latent distribution with stable log_pdf / log_cdf.

    dist: "gaussian" or "student_t"; df used only for student_t.
    """

    def __init__(self, dist: str = "gaussian", df: float = 4.0):
        self.dist = dist
        self.df = float(df)

    def log_pdf_standardized(self, z):
        if self.dist in ("gaussian", "normal"):
            return _gaussian_log_pdf(z)
        return _student_t_log_pdf(z, self.df)

    def log_cdf_standardized(self, z):
        if self.dist in ("gaussian", "normal"):
            return _gaussian_log_cdf(z)
        return _student_t_log_cdf(z, self.df).to(z.dtype)


# --------------------------------------------------------------------------- #
# Interval log-probability and its pieces                                     #
# --------------------------------------------------------------------------- #
def _exact_mask(lower, upper, exact_mask):
    """Authoritative exact-row detection. Prefer the caller-supplied mask (the
    dataset already snaps exact rows so lower==upper exactly); otherwise fall
    back to STRICT equality. We deliberately avoid torch.isclose here: with its
    default atol/rtol a genuine narrow interval-censored row (e.g. [5.00, 5.01])
    would be misclassified as exact, corrupting q_i and the deficit."""
    torch = _torch()
    if exact_mask is not None:
        return exact_mask.to(torch.bool)
    return torch.isfinite(lower) & torch.isfinite(upper) & (lower == upper)


def interval_log_prob(mu, sigma, lower, upper, dist: LatentDistribution, exact_mask=None):
    """log P(L <= Y <= U) for each row, computed in log-space and robust to
    +/-inf bounds. Returns a per-row tensor of log-probabilities (<= 0).

    Branches:
      exact  (L==U):     log pdf(y)               (a density; may be > 0, handled by caller)
      right  (U=+inf):   log( 1 - F(zL) ) = log F(-zL)   [gaussian symmetry] / 1 - F for t
      left   (L=-inf):   log F(zU)
      interval:          log( F(zU) - F(zL) ) via log-difference, computed from
                         whichever tail is smaller for numerical stability.
    """
    torch = _torch()
    sigma = torch.clamp(sigma, min=1e-6)
    lo = lower.to(torch.float64)
    hi = upper.to(torch.float64)
    mu64 = mu.to(torch.float64)
    sg64 = sigma.to(torch.float64)

    exact = _exact_mask(lower, upper, exact_mask)
    right = torch.isfinite(lower) & torch.isposinf(upper) & (~exact)
    left = torch.isneginf(lower) & torch.isfinite(upper) & (~exact)
    # interval covers all finite non-exact rows with upper >= lower. Using >=
    # (not >) means a non-exact row whose bounds collapse to equal under the
    # float32 cast (distinct in float64, equal in float32) is still an interval
    # row: it is caught by the near-degenerate density branch below rather than
    # falling through every branch to a spurious log-prob of 0 (=> q=1, deficit 0,
    # a silently "satisfied" constraint).
    interval = torch.isfinite(lower) & torch.isfinite(upper) & (~exact) & (upper >= lower)

    # sanitize inf bounds to a finite dummy BEFORE forming z (avoid nan in the
    # unselected torch.where branch during backprop).
    lo_s = torch.where(torch.isfinite(lo), lo, torch.zeros_like(lo))
    hi_s = torch.where(torch.isfinite(hi), hi, torch.zeros_like(hi))
    zL = (lo_s - mu64) / sg64
    zU = (hi_s - mu64) / sg64

    out = torch.zeros_like(mu64)

    if bool(exact.any()):
        # exact interval prob is not used as a probability (density); the caller
        # routes exact rows to the exact NLL. Fill with log pdf for completeness.
        out = torch.where(exact, dist.log_pdf_standardized(zL) - torch.log(sg64), out)
    if bool(right.any()):
        # P(Y>=L) = 1 - F(zL). For gaussian use symmetry log F(-zL); for t use log(1-F).
        if dist.dist in ("gaussian", "normal"):
            out = torch.where(right, dist.log_cdf_standardized(-zL), out)
        else:
            logF = dist.log_cdf_standardized(zL)
            out = torch.where(right, _log1mexp(logF), out)
    if bool(left.any()):
        out = torch.where(left, dist.log_cdf_standardized(zU), out)
    if bool(interval.any()):
        # For a stable log-difference, work on whichever side gives the smaller
        # log-mass being subtracted. If the interval sits in the RIGHT tail
        # (zL > 0) use the survival function: log[SF(zL) - SF(zU)] =
        # logSF(zL) + log1mexp(logSF(zU) - logSF(zL)). Otherwise use the CDF:
        # logFU + log1mexp(logFL - logFU). Both avoid catastrophic cancellation.
        logFU = dist.log_cdf_standardized(zU)
        logFL = dist.log_cdf_standardized(zL)
        cdf_side = logFU + _log1mexp(logFL - logFU)
        if dist.dist in ("gaussian", "normal"):
            logSFL = dist.log_cdf_standardized(-zL)
            logSFU = dist.log_cdf_standardized(-zU)
        else:
            logSFL = _log1mexp(logFL)
            logSFU = _log1mexp(logFU)
        sf_side = logSFL + _log1mexp(logSFU - logSFL)
        use_sf = interval & (zL > 0)
        # NEAR-DEGENERATE intervals: when the standardized width is tiny the
        # log-difference is ill-conditioned (and its gradient explodes). Use the
        # midpoint-density approximation  P ~ pdf(z_mid) * width  =>
        # log P ~ log pdf(z_mid) - log sigma + log(width). This is accurate to
        # O(width^2) and has a well-behaved gradient. Threshold: width/sigma < 1e-3.
        width = (hi_s - lo_s)
        z_mid = (0.5 * (lo_s + hi_s) - mu64) / sg64
        narrow = interval & ((zU - zL) < 1e-3)
        dens_side = dist.log_pdf_standardized(z_mid) - torch.log(sg64) + torch.log(torch.clamp(width, min=1e-300))
        out = torch.where(interval & (~use_sf) & (~narrow), cdf_side, out)
        out = torch.where(use_sf & (~narrow), sf_side, out)
        out = torch.where(narrow, dens_side, out)
    return out


def censored_interval_prob(mu, sigma, lower, upper, dist: LatentDistribution, exact_mask=None):
    """P(L <= Y <= U) in [0,1] for censored/interval rows (exact rows -> ~1 placeholder).
    Used to form the satisficing deficit q_i."""
    torch = _torch()
    exact = _exact_mask(lower, upper, exact_mask)
    logp = interval_log_prob(mu, sigma, lower, upper, dist, exact_mask=exact)
    prob = torch.exp(torch.clamp(logp, max=0.0))
    # exact rows are not censored constraints; set their prob to 1 so deficit is 0.
    prob = torch.where(exact, torch.ones_like(prob), prob)
    return prob.to(mu.dtype)


# --------------------------------------------------------------------------- #
# Exact-value NLL (proper log-likelihood on exactly observed rows)            #
# --------------------------------------------------------------------------- #
def exact_nll(mu, sigma, y, dist: LatentDistribution):
    torch = _torch()
    sigma = torch.clamp(sigma, min=1e-6)
    z = (y - mu) / sigma
    return (-dist.log_pdf_standardized(z) + torch.log(sigma)).to(mu.dtype)


# --------------------------------------------------------------------------- #
# Satisficing deficit                                                         #
# --------------------------------------------------------------------------- #
def satisficing_deficit(mu, sigma, lower, upper, dist: LatentDistribution, tau: float,
                        exact_mask=None):
    """d_i = [tau - q_i]_+^2 for censored rows; 0 for exact rows.

    q_i is the probability mass the model assigns to the observed interval. Once
    q_i >= tau the row contributes zero deficit and zero gradient: the model must
    believe the true value is on the correct side of the bound with probability
    tau, but gets no reward for pushing further. Pass the dataset's authoritative
    exact_mask so narrow interval-censored rows are not mistaken for exact."""
    torch = _torch()
    q = censored_interval_prob(mu, sigma, lower, upper, dist, exact_mask=exact_mask)
    return torch.clamp(tau - q, min=0.0) ** 2


# --------------------------------------------------------------------------- #
# Reference losses for the controlled comparison (plan §10)                   #
# --------------------------------------------------------------------------- #
def tobit_nll(mu, sigma, lower, upper, dist: LatentDistribution, exact_mask=None):
    """Standard interval-censored NLL: exact -> -log pdf; censored -> -log P(interval).
    The 'full interval likelihood' baseline whose over-extrapolation the method targets."""
    torch = _torch()
    exact = _exact_mask(lower, upper, exact_mask)
    losses = torch.zeros_like(mu)
    if bool(exact.any()):
        # sanitize the target BEFORE exact_nll: for non-exact rows `lower` may be
        # -inf, and exact_nll computed over the full batch would produce +inf at
        # those positions; torch.where discards them in the forward pass but the
        # backward pass hits the 0*inf = NaN trap. Feed a finite dummy there.
        y_safe = torch.where(exact, lower, torch.zeros_like(lower))
        losses = torch.where(exact, exact_nll(mu, sigma, y_safe, dist), losses)
    cens = ~exact
    if bool(cens.any()):
        logp = interval_log_prob(mu, sigma, lower, upper, dist, exact_mask=exact)
        losses = torch.where(cens, -logp.to(mu.dtype), losses)
    return losses


def interval_hinge(mu, lower, upper):
    """Distance-to-interval hinge (point prediction only), a distribution-free
    interval-consistency baseline: 0 inside [L,U], linear outside."""
    torch = _torch()
    below = torch.where(torch.isfinite(lower), torch.clamp(lower - mu, min=0.0), torch.zeros_like(mu))
    above = torch.where(torch.isfinite(upper), torch.clamp(mu - upper, min=0.0), torch.zeros_like(mu))
    return below + above
