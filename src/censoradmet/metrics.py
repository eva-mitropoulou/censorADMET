"""Evaluation metrics for CensorADMET 2.0 (plan §13-15).

Three tiers:

  (1) Accuracy on the subset with a trustworthy target. On real censored data we
      never see the latent value for a censored row, so accuracy metrics (MAE,
      RMSE, Spearman) are computed on EXACT rows only -- reporting them over
      censored rows (scoring a prediction against a bound) was a documented sin
      of the v1 pipeline.

  (2) Censored-consistency: interval-violation rate (fraction of censored rows
      whose point prediction lands on the wrong side of the bound) and the
      satisficing deficit -- these are legitimately computed on censored rows.

  (3) Decision-focused (plan §14): for a decision threshold t_e (e.g. "flag as a
      liability if pIC50 > t_e"), the false-safe rate (truly-active called safe)
      and false-liability rate, plus decision regret, using the KNOWN side of a
      censored label as partial ground truth. Exceedance probabilities
      P(Y > t_e) come from the predictive distribution and are assessed for
      calibration.

The predictive distribution can be a single (mu, sigma) Gaussian/Student-t OR an
ENSEMBLE mixture (list of (mu_m, sigma_m)); exceedance/quantiles are averaged
over the mixture components rather than moment-matched to a single Gaussian
(plan §15 -- moment matching throws away the heavy tail the ensemble captures).
"""
from __future__ import annotations

import numpy as np
from scipy import stats


# --------------------------------------------------------------------------- #
# Tier 1: accuracy on exact rows                                              #
# --------------------------------------------------------------------------- #
def accuracy_metrics(mu, y, exact_mask):
    mu = np.asarray(mu); y = np.asarray(y); ex = np.asarray(exact_mask, dtype=bool)
    ex = ex & np.isfinite(y)
    if ex.sum() < 3:
        return {"mae": np.nan, "rmse": np.nan, "spearman": np.nan, "n_exact": int(ex.sum())}
    e = mu[ex] - y[ex]
    sp = stats.spearmanr(mu[ex], y[ex]).correlation
    return {
        "mae": float(np.mean(np.abs(e))),
        "rmse": float(np.sqrt(np.mean(e ** 2))),
        "spearman": float(sp),
        "n_exact": int(ex.sum()),
    }


# --------------------------------------------------------------------------- #
# Tier 2: censored-consistency                                                #
# --------------------------------------------------------------------------- #
def interval_violation_rate(mu, lower, upper, exact_mask):
    """Fraction of CENSORED rows whose point prediction lands outside [L,U]."""
    mu = np.asarray(mu); lo = np.asarray(lower); hi = np.asarray(upper)
    ex = np.asarray(exact_mask, dtype=bool)
    cens = ~ex
    if cens.sum() == 0:
        return {"violation_rate": np.nan, "n_censored": 0}
    below = mu < lo
    above = mu > hi
    viol = (below | above) & cens
    # direction-resolved
    right = cens & np.isposinf(hi)   # y > L : violated if mu < L
    left = cens & np.isneginf(lo)    # y < U : violated if mu > U
    return {
        "violation_rate": float(viol.sum() / cens.sum()),
        "right_violation_rate": float((mu[right] < lo[right]).mean()) if right.any() else np.nan,
        "left_violation_rate": float((mu[left] > hi[left]).mean()) if left.any() else np.nan,
        "n_censored": int(cens.sum()),
    }


# --------------------------------------------------------------------------- #
# Predictive distribution helpers (single or ensemble mixture)                #
# --------------------------------------------------------------------------- #
def _as_components(pred):
    """Normalize pred to a list of (mu, sigma) mixture components (equal weight)."""
    if isinstance(pred, tuple):
        return [(np.asarray(pred[0], float), np.asarray(pred[1], float))]
    return [(np.asarray(m, float), np.asarray(s, float)) for (m, s) in pred]


def exceedance_prob(pred, t, dist="gaussian", df=4.0):
    """P(Y > t) under the (mixture) predictive distribution, per row."""
    comps = _as_components(pred)
    probs = []
    for mu, sigma in comps:
        sigma = np.clip(sigma, 1e-6, None)
        z = (t - mu) / sigma
        if dist == "gaussian":
            probs.append(stats.norm.sf(z))
        else:
            probs.append(stats.t.sf(z, df))
    return np.mean(probs, axis=0)


def predictive_quantile(pred, q, dist="gaussian", df=4.0, n_grid=1024):
    """Mixture predictive quantile via PER-ROW inversion of the mixture CDF.

    A single shared grid across all rows is wrong for a heteroscedastic head:
    tight-sigma rows would get catastrophically coarse resolution. We build a
    grid from EACH row's own mixture support. For the Student-t branch we widen
    the span with a df-dependent tail factor because mu +/- k*sigma truncates the
    heavy t tails (for df=4, the 0.9999 quantile is ~13.03 sigma, not 8)."""
    comps = _as_components(pred)
    mus = np.stack([c[0] for c in comps])                 # (M, N)
    sigs = np.stack([np.clip(c[1], 1e-6, None) for c in comps])
    n = mus.shape[1]
    # tail span: how many sigmas to cover the requested extreme quantile
    if dist == "gaussian":
        span = max(8.0, abs(stats.norm.ppf(min(q, 1 - q) / 2)) + 4.0)
    else:
        span = max(8.0, abs(stats.t.ppf(min(q, 1 - q) / 2, df)) + 4.0)
    out = np.empty(n)
    for i in range(n):
        lo = float((mus[:, i] - span * sigs[:, i]).min())
        hi = float((mus[:, i] + span * sigs[:, i]).max())
        if hi <= lo:
            out[i] = mus[:, i].mean()
            continue
        grid = np.linspace(lo, hi, n_grid)
        if dist == "gaussian":
            cdf = stats.norm.cdf(grid[:, None], mus[:, i], sigs[:, i]).mean(axis=1)
        else:
            cdf = stats.t.cdf((grid[:, None] - mus[:, i]) / sigs[:, i], df).mean(axis=1)
        j = int(np.searchsorted(cdf, q))
        out[i] = grid[min(max(j, 0), n_grid - 1)]
    return out


# --------------------------------------------------------------------------- #
# Tier 3: decision-focused metrics                                            #
# --------------------------------------------------------------------------- #
def decision_metrics(pred, y, lower, upper, exact_mask, t, decide="above",
                     dist="gaussian", df=4.0, prob_cut=0.5):
    """Decision at threshold t. "above" => we FLAG a compound as a hit/liability
    when we predict Y > t.

    Ground truth per row (using the known side of censoring where possible):
      exact:                 truth = (y > t)
      right-censored (Y>L):  if L >= t -> truth known True;  else unknown -> skip
      left-censored  (Y<U):  if U <= t -> truth known False; else unknown -> skip
    A decision is made from the predictive exceedance prob P(Y>t) >= prob_cut.

    Returns false-safe rate (truth True predicted safe), false-liability rate
    (truth False predicted liability), balanced accuracy, and decision regret
    (mean |P(Y>t) - 1{truth}| over resolvable rows)."""
    y = np.asarray(y, float); lo = np.asarray(lower, float); hi = np.asarray(upper, float)
    ex = np.asarray(exact_mask, bool)
    p_exc = exceedance_prob(pred, t, dist=dist, df=df)
    pred_flag = p_exc >= prob_cut

    truth = np.full(len(y), np.nan)
    truth[ex] = (y[ex] > t).astype(float)
    right = (~ex) & np.isposinf(hi)
    left = (~ex) & np.isneginf(lo)
    # resolvable censored rows
    truth[right & (lo >= t)] = 1.0     # known above threshold
    truth[left & (hi <= t)] = 0.0      # known below threshold

    resolvable = np.isfinite(truth)
    if resolvable.sum() < 5:
        return {"n_resolvable": int(resolvable.sum())}
    tr = truth[resolvable].astype(bool)
    pf = pred_flag[resolvable]
    pe = p_exc[resolvable]

    pos = tr.sum(); neg = (~tr).sum()
    false_safe = float(((~pf) & tr).sum() / pos) if pos > 0 else np.nan   # missed a true hit
    false_liab = float((pf & (~tr)).sum() / neg) if neg > 0 else np.nan   # flagged a true non-hit
    tpr = float((pf & tr).sum() / pos) if pos > 0 else np.nan
    tnr = float(((~pf) & (~tr)).sum() / neg) if neg > 0 else np.nan
    bal_acc = np.nanmean([tpr, tnr])
    regret = float(np.mean(np.abs(pe - tr.astype(float))))
    return {
        "threshold": float(t),
        "false_safe_rate": false_safe,
        "false_liability_rate": false_liab,
        "balanced_accuracy": float(bal_acc),
        "decision_regret": regret,
        "n_resolvable": int(resolvable.sum()),
        "n_positive": int(pos), "n_negative": int(neg),
    }


def exceedance_calibration(pred, y, lower, upper, exact_mask, t, n_bins=10,
                           dist="gaussian", df=4.0):
    """Reliability of P(Y>t): bin rows by predicted exceedance prob and compare to
    empirical frequency of truth, over resolvable rows. Returns ECE + per-bin."""
    y = np.asarray(y, float); lo = np.asarray(lower, float); hi = np.asarray(upper, float)
    ex = np.asarray(exact_mask, bool)
    p = exceedance_prob(pred, t, dist=dist, df=df)
    truth = np.full(len(y), np.nan)
    truth[ex] = (y[ex] > t).astype(float)
    right = (~ex) & np.isposinf(hi); left = (~ex) & np.isneginf(lo)
    truth[right & (lo >= t)] = 1.0
    truth[left & (hi <= t)] = 0.0
    r = np.isfinite(truth)
    if r.sum() < 20:
        return {"ece": np.nan, "n": int(r.sum())}
    pr = p[r]; tr = truth[r]
    edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0; bins = []
    for b in range(n_bins):
        m = (pr >= edges[b]) & (pr < edges[b + 1] if b < n_bins - 1 else pr <= edges[b + 1])
        if m.sum() == 0:
            continue
        conf = pr[m].mean(); acc = tr[m].mean()
        ece += (m.sum() / r.sum()) * abs(conf - acc)
        bins.append({"bin": b, "conf": float(conf), "freq": float(acc), "n": int(m.sum())})
    return {"ece": float(ece), "n": int(r.sum()), "bins": bins}


# --------------------------------------------------------------------------- #
# Interval calibration (coverage / width) using predictive quantiles          #
# --------------------------------------------------------------------------- #
def interval_coverage(pred, y, exact_mask, level=0.9, dist="gaussian", df=4.0):
    """Central-`level` predictive-interval coverage & mean width on EXACT rows."""
    ex = np.asarray(exact_mask, bool) & np.isfinite(np.asarray(y, float))
    if ex.sum() < 10:
        return {"coverage": np.nan, "width": np.nan, "n": int(ex.sum())}
    a = (1 - level) / 2
    comps = _as_components(pred)
    comps_ex = [(m[ex], s[ex]) for (m, s) in comps]
    qlo = predictive_quantile(comps_ex, a, dist=dist, df=df)
    qhi = predictive_quantile(comps_ex, 1 - a, dist=dist, df=df)
    yy = np.asarray(y, float)[ex]
    cov = float(np.mean((yy >= qlo) & (yy <= qhi)))
    return {"coverage": cov, "width": float(np.mean(qhi - qlo)), "n": int(ex.sum())}
