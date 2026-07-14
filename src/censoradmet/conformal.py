"""Split-conformal calibration for censored predictive intervals (plan §15).

The heteroscedastic / ensemble heads give a predictive (mu, sigma) but their raw
90% intervals are miscalibrated under distribution shift (the smoke test showed
coverage well below nominal on assay-held-out splits). We wrap any predictor with
SPLIT CONFORMAL prediction to restore finite-sample marginal coverage, using only
the EXACT calibration rows (a censored row has no observed target to score).

Two variants:
  * Scalar CQR-style: score s_i = |y_i - mu_i| / sigma_i on exact calibration rows;
    the (1-alpha) empirical quantile q gives intervals [mu +- q*sigma] with
    guaranteed >= 1-alpha marginal coverage on exchangeable exact rows.
  * Mondrian (group-conditional): compute a separate q per group (e.g. censoring
    direction, assay-type, or potency band) so coverage holds WITHIN each group,
    not just marginally -- important because censored and exact rows differ.

Only exact rows enter calibration and coverage scoring; we never score a
prediction against a censoring bound.
"""
from __future__ import annotations

import numpy as np


def _finite_quantile_level(n, alpha):
    """Conformal quantile level with finite-sample correction: ceil((n+1)(1-a))/n."""
    return min(1.0, np.ceil((n + 1) * (1 - alpha)) / max(n, 1))


class SplitConformal:
    def __init__(self, alpha=0.1):
        self.alpha = alpha
        self.q = None

    def calibrate(self, mu_cal, sigma_cal, y_cal, exact_cal):
        mu = np.asarray(mu_cal, float); sg = np.clip(np.asarray(sigma_cal, float), 1e-6, None)
        y = np.asarray(y_cal, float); ex = np.asarray(exact_cal, bool) & np.isfinite(y)
        if ex.sum() < 10:
            self.q = None
            return self
        scores = np.abs(y[ex] - mu[ex]) / sg[ex]
        lvl = _finite_quantile_level(ex.sum(), self.alpha)
        self.q = float(np.quantile(scores, lvl, method="higher"))
        return self

    def interval(self, mu, sigma):
        mu = np.asarray(mu, float); sg = np.clip(np.asarray(sigma, float), 1e-6, None)
        if self.q is None:
            # fall back to a Gaussian z-interval if uncalibrated
            from scipy.stats import norm
            z = norm.ppf(1 - self.alpha / 2)
            return mu - z * sg, mu + z * sg
        return mu - self.q * sg, mu + self.q * sg

    def coverage(self, mu, sigma, y, exact_mask):
        lo, hi = self.interval(mu, sigma)
        ex = np.asarray(exact_mask, bool) & np.isfinite(np.asarray(y, float))
        if ex.sum() == 0:
            return {"coverage": np.nan, "width": float(np.mean(hi - lo)), "n": 0}
        yy = np.asarray(y, float)[ex]
        cov = float(np.mean((yy >= lo[ex]) & (yy <= hi[ex])))
        return {"coverage": cov, "width": float(np.mean(hi[ex] - lo[ex])), "n": int(ex.sum()),
                "q": self.q}


class MondrianConformal:
    """Group-conditional split conformal: one conformal quantile per group key."""

    def __init__(self, alpha=0.1):
        self.alpha = alpha
        self.q_by_group = {}
        self.q_global = None

    def calibrate(self, mu_cal, sigma_cal, y_cal, exact_cal, groups_cal):
        mu = np.asarray(mu_cal, float); sg = np.clip(np.asarray(sigma_cal, float), 1e-6, None)
        y = np.asarray(y_cal, float); ex = np.asarray(exact_cal, bool) & np.isfinite(y)
        groups = np.asarray(groups_cal)
        scores_all = np.abs(y - mu) / sg
        if ex.sum() >= 10:
            lvl = _finite_quantile_level(ex.sum(), self.alpha)
            self.q_global = float(np.quantile(scores_all[ex], lvl, method="higher"))
        for g in np.unique(groups[ex]):
            m = ex & (groups == g)
            if m.sum() >= 10:
                lvl = _finite_quantile_level(m.sum(), self.alpha)
                self.q_by_group[g] = float(np.quantile(scores_all[m], lvl, method="higher"))
        return self

    def interval(self, mu, sigma, groups):
        mu = np.asarray(mu, float); sg = np.clip(np.asarray(sigma, float), 1e-6, None)
        groups = np.asarray(groups)
        q = np.array([self.q_by_group.get(g, self.q_global if self.q_global is not None else np.nan)
                      for g in groups], dtype=float)
        # rows with no group/global calibration fall back to a wide z-interval
        if np.isnan(q).any():
            from scipy.stats import norm
            q = np.where(np.isnan(q), norm.ppf(1 - self.alpha / 2), q)
        return mu - q * sg, mu + q * sg

    def coverage(self, mu, sigma, y, exact_mask, groups):
        lo, hi = self.interval(mu, sigma, groups)
        ex = np.asarray(exact_mask, bool) & np.isfinite(np.asarray(y, float))
        if ex.sum() == 0:
            return {"coverage": np.nan, "n": 0}
        yy = np.asarray(y, float)[ex]
        return {"coverage": float(np.mean((yy >= lo[ex]) & (yy <= hi[ex]))),
                "width": float(np.mean(hi[ex] - lo[ex])), "n": int(ex.sum()),
                "n_groups": len(self.q_by_group)}
