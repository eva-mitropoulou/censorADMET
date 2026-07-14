"""Semi-synthetic hidden-truth benchmark with REALISTIC censoring (plan §7-8).

Motivation & honest scope note. Plan §7 proposed fitting Bayesian dose-response
curves to raw concentration/response points to obtain a gold-standard "hidden
truth" and reconstruct censoring. We verified this is INFEASIBLE from ChEMBL 36
for these ADMET targets: the `activity_supp` table holds toxicology/histopathology
readouts (DrugMatrix-style), not enzyme/channel dose-response point series, and
our targets (hERG, CYPs, ABCB1) have ~zero supp-linked dose-response points. We
therefore replace it with a defensible semi-synthetic construction that still
gives a KNOWN latent truth AND a realistic (data-derived) censoring mechanism:

  1. Take the high-confidence EXACT measurements of an endpoint as the latent
     truth y* (these are genuine potencies; nothing is invented).
  2. Learn the endpoint's real censoring mechanism from its actually-censored
     rows: the empirical distribution of censoring THRESHOLDS (the finite bound
     of each left/right-censored measurement) and the direction mix.
  3. Re-censor a copy of the exact rows by drawing a threshold from that learned
     distribution and censoring y* against it -- so a row with y* below a drawn
     right-tail cutoff becomes right-censored at that cutoff, etc. The censoring
     thresholds and their prevalence match the real assay, so the reconstruction
     is realistic rather than an arbitrary synthetic rate.

The benefit over the v1 "synthetic censoring" arm: because y* is known for EVERY
re-censored row, we can score HIDDEN-TRUTH recovery (MAE/coverage against y*) on
the censored rows themselves -- the quantity that is unobservable on real data.

Also supports an informative (MNAR) mode: make the censoring probability depend
on y* itself (high-potency rows more likely censored), to stress-test methods
under non-ignorable censoring."""
from __future__ import annotations

import numpy as np


def learn_censoring_mechanism(lower_p, upper_p, exact_mask, censoring_class):
    """Return the empirical censoring model of a real endpoint:
    { 'p_left', 'p_right', 'left_thresholds', 'right_thresholds', 'p_censored' }.
    Thresholds are the finite bounds of the real censored rows."""
    cc = np.asarray(censoring_class)
    lo = np.asarray(lower_p, float); hi = np.asarray(upper_p, float)
    ex = np.asarray(exact_mask, bool)
    left = cc == "left_censored_p"     # Y < U : finite upper bound
    right = cc == "right_censored_p"   # Y > L : finite lower bound
    n = len(cc)
    n_cens = int(left.sum() + right.sum())
    return {
        "p_censored": n_cens / max(n, 1),
        "p_left": left.sum() / max(n_cens, 1),
        "p_right": right.sum() / max(n_cens, 1),
        "left_thresholds": hi[left][np.isfinite(hi[left])],
        "right_thresholds": lo[right][np.isfinite(lo[right])],
        "n_exact": int(ex.sum()),
    }


def build_hidden_truth(y_star, mechanism, seed=0, informative=False, mnar_strength=1.0):
    """Given latent truths y_star (from exact rows) and a learned `mechanism`,
    produce (lower, upper, exact_mask, is_recensored) by re-censoring.

    Each row is censored with prob p_censored (in informative mode, modulated by
    y_star's rank so high-potency rows are censored more). A censored row draws a
    threshold from the empirical left/right threshold pool; it is only actually
    censored if y_star falls on the censored side of the drawn threshold,
    otherwise it stays exact (this mirrors how a fixed assay cutoff censors only
    compounds beyond it). Returns arrays aligned to y_star."""
    rng = np.random.default_rng(seed)
    y = np.asarray(y_star, float)
    n = len(y)
    lower = y.copy(); upper = y.copy()
    exact = np.ones(n, bool)
    recensored = np.zeros(n, bool)

    p = np.full(n, mechanism["p_censored"])
    if informative and n > 1:
        # higher potency (larger y) -> more likely to be censored (MNAR)
        r = (np.argsort(np.argsort(y)) / (n - 1))          # rank in [0,1]
        p = np.clip(mechanism["p_censored"] * (1 + mnar_strength * (r - 0.5) * 2), 0, 1)

    pool_left = mechanism["left_thresholds"]
    pool_right = mechanism["right_thresholds"]
    has_left = len(pool_left) > 0
    has_right = len(pool_right) > 0
    if not (has_left or has_right):
        return lower, upper, exact, recensored

    draw = rng.random(n) < p
    for i in np.where(draw)[0]:
        # choose a direction per the real mix (fall back to whichever pool exists)
        go_right = (rng.random() < mechanism["p_right"]) if (has_left and has_right) else has_right
        if go_right and has_right:
            thr = rng.choice(pool_right)
            if y[i] > thr:            # truly above the cutoff -> right-censored
                lower[i] = thr; upper[i] = np.inf
                exact[i] = False; recensored[i] = True
        elif has_left:
            thr = rng.choice(pool_left)
            if y[i] < thr:            # truly below the cutoff -> left-censored
                lower[i] = -np.inf; upper[i] = thr
                exact[i] = False; recensored[i] = True
    return lower.astype(np.float32), upper.astype(np.float32), exact, recensored


def hidden_truth_recovery(mu, y_star, recensored_mask):
    """Score recovery of the KNOWN latent truth on the RE-CENSORED rows (the rows
    whose truth is hidden from the model). This is the quantity that cannot be
    measured on real data. Returns MAE/RMSE/bias on those rows."""
    mu = np.asarray(mu, float); y = np.asarray(y_star, float)
    m = np.asarray(recensored_mask, bool) & np.isfinite(y) & np.isfinite(mu)
    if m.sum() < 3:
        return {"hidden_mae": np.nan, "n_hidden": int(m.sum())}
    e = mu[m] - y[m]
    return {
        "hidden_mae": float(np.mean(np.abs(e))),
        "hidden_rmse": float(np.sqrt(np.mean(e ** 2))),
        "hidden_bias": float(np.mean(e)),      # + => over-predicts hidden potency
        "n_hidden": int(m.sum()),
    }
