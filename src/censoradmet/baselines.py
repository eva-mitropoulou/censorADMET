"""Baseline / competitor models for the controlled comparison (plan §10-12).

Included:
  * XGBoostAFTConcentration -- XGBoost accelerated-failure-time run in the
    NATURAL concentration domain (nM), not shifted p-space. Rationale (plan §11):
    AFT models a positive "survival time" T and internally fits a distribution on
    log(T). Concentration X_nM is exactly such a positive quantity, and pX =
    9 - log10(X_nM) already contains one log. Running AFT on a shifted p-value
    therefore fits a distribution on log(pX_shifted) -- a DOUBLE log that distorts
    the scale. We instead map the observed p-interval to a concentration interval
    (monotone-decreasing: X = 10^(9 - pX)), fit AFT on X, and convert the median
    prediction back to p-scale. Right/left censoring flips between the two
    domains and is handled explicitly.

  * DeepEnsemble -- trains K heteroscedastic heads (via SatisficingTrainer) with
    different seeds and exposes the ensemble predictive as a MIXTURE (list of
    (mu, sigma)) so downstream exceedance / quantile metrics integrate over the
    mixture instead of moment-matching to one Gaussian (plan §15).
"""
from __future__ import annotations

import numpy as np

P_ANCHOR = 9.0  # pX = P_ANCHOR - log10(X_nM)   (X in nM)


# --------------------------------------------------------------------------- #
# p-scale <-> concentration conversion                                        #
# --------------------------------------------------------------------------- #
def p_to_conc(p):
    """pX -> concentration in nM. Monotone DECREASING."""
    return np.power(10.0, P_ANCHOR - np.asarray(p, dtype=float))


def conc_to_p(x):
    x = np.clip(np.asarray(x, dtype=float), 1e-30, None)
    return P_ANCHOR - np.log10(x)


def p_bounds_to_conc_bounds(lower_p, upper_p, conc_floor=1e-6, conc_ceil=1e15):
    """Map a p-interval [L_p, U_p] to a concentration interval [L_x, U_x].

    Because X = 10^(9 - pX) is decreasing:
        L_x = 10^(9 - U_p)   (upper p -> lower conc)
        U_x = 10^(9 - L_p)   (lower p -> upper conc)
    A -inf p-lower (left-censored potency) => +inf conc-upper (right-censored
    concentration); a +inf p-upper (right-censored potency) => 0 conc-lower
    (left-censored concentration, floored to conc_floor for AFT positivity)."""
    lower_p = np.asarray(lower_p, dtype=float)
    upper_p = np.asarray(upper_p, dtype=float)
    L_x = np.where(np.isposinf(upper_p), conc_floor, np.power(10.0, P_ANCHOR - upper_p))
    U_x = np.where(np.isneginf(lower_p), np.inf, np.power(10.0, P_ANCHOR - lower_p))
    L_x = np.clip(L_x, conc_floor, conc_ceil)
    U_x = np.where(np.isfinite(U_x), np.clip(U_x, conc_floor, conc_ceil), np.inf)
    return L_x, U_x


# --------------------------------------------------------------------------- #
# Concentration-space XGBoost-AFT                                             #
# --------------------------------------------------------------------------- #
class XGBoostAFTConcentration:
    def __init__(self, params: dict | None = None, num_boost_round: int = 400,
                 early_stopping_rounds: int = 40, seed: int = 0, n_jobs: int = 8):
        self.params = params or {
            "objective": "survival:aft",
            "eval_metric": "aft-nloglik",
            "aft_loss_distribution": "normal",
            "aft_loss_distribution_scale": 1.0,
            "tree_method": "hist",
            "learning_rate": 0.05,
            "max_depth": 6,
            "min_child_weight": 1.0,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
        }
        self.num_boost_round = num_boost_round
        self.early_stopping_rounds = early_stopping_rounds
        self.seed = seed
        self.n_jobs = n_jobs
        self.model = None

    def _dmatrix(self, X, lower_p=None, upper_p=None):
        import xgboost as xgb
        dm = xgb.DMatrix(np.asarray(X, dtype=np.float32))
        if lower_p is not None:
            L_x, U_x = p_bounds_to_conc_bounds(lower_p, upper_p)
            dm.set_float_info("label_lower_bound", L_x.astype(float))
            dm.set_float_info("label_upper_bound", U_x.astype(float))
        return dm

    def fit(self, X, lower_p, upper_p, X_val=None, lower_p_val=None, upper_p_val=None):
        import xgboost as xgb
        params = dict(self.params, seed=self.seed, nthread=self.n_jobs)
        dtrain = self._dmatrix(X, lower_p, upper_p)
        evals = [(dtrain, "train")]
        esr = None
        if X_val is not None:
            dval = self._dmatrix(X_val, lower_p_val, upper_p_val)
            evals.append((dval, "valid"))
            esr = self.early_stopping_rounds
        self.model = xgb.train(params, dtrain, num_boost_round=self.num_boost_round,
                               evals=evals, early_stopping_rounds=esr, verbose_eval=False)
        return self

    def predict(self, X):
        """Return predicted potency pX (median). AFT predicts the concentration T;
        convert back to p-scale."""
        import xgboost as xgb
        dm = xgb.DMatrix(np.asarray(X, dtype=np.float32))
        best_it = getattr(self.model, "best_iteration", None)
        if best_it is not None:
            conc = self.model.predict(dm, iteration_range=(0, int(best_it) + 1))
        else:
            conc = self.model.predict(dm)
        return conc_to_p(conc)

    def fit_scale_on_valid(self, X_val, y_val_p):
        """Calibrate the predictive sd on exact validation rows: set sigma_p to
        the std of (pred_pX - y_pX). Optional; call after fit() when a calibrated
        baseline UQ is wanted instead of the fixed hyperparameter."""
        y_val_p = np.asarray(y_val_p, float)
        m = np.isfinite(y_val_p)
        if m.sum() >= 10:
            resid = self.predict(X_val)[m] - y_val_p[m]
            self._calibrated_sigma_p = float(np.std(resid))
        return self

    def predict_dist(self, X):
        """(mu, sigma) in p-space. AFT gives a point prediction (median conc).
        For the predictive sd we use, by default, the FIXED training hyperparameter
        aft_loss_distribution_scale converted to p-units (sigma_p = scale/ln10);
        note this is a constant that boosting does NOT update, so it is a rough
        homoscedastic UQ. If fit_scale_on_valid() was called, the calibrated
        residual sd is used instead."""
        mu = self.predict(X)
        sig = getattr(self, "_calibrated_sigma_p", None)
        if sig is None:
            scale = float(self.params.get("aft_loss_distribution_scale", 1.0))
            sig = scale / np.log(10.0)
        sigma_p = np.full(len(mu), sig, dtype=float)
        return mu, sigma_p


# --------------------------------------------------------------------------- #
# Genuine Tobit full-interval-likelihood MLP (prior-art baseline, plan §10)   #
# --------------------------------------------------------------------------- #
class MLPTobit:
    """Heteroscedastic MLP trained by minimising the full interval-censored NLL
    (tobit_nll): exact rows -> -log pdf, censored rows -> -log P(interval). This
    is the standard 'censor-aware' baseline whose over-extrapolation the
    satisficing method targets; we train it honestly (same architecture / budget
    as our method) rather than approximating it via an eps=0 constraint."""

    def __init__(self, make_model, dist, lr=1e-3, weight_decay=1e-5, epochs=120,
                 batch_size=1024, grad_clip=5.0, seed=0, device="cpu",
                 censored_weight=1.0, anchor_weight=0.0):
        self.make_model = make_model
        self.dist = dist
        self.lr = lr; self.weight_decay = weight_decay; self.epochs = epochs
        self.batch_size = batch_size; self.grad_clip = grad_clip
        self.seed = seed; self.device = device
        # censored_weight w downweights (w<1) the censored-row NLL term relative to
        # the exact-row term. w=1 is the standard Tobit; sweeping w gives the
        # "weighted Tobit" frontier -- the obvious way to trade accuracy for bound
        # consistency with a scalar knob, which we compare against the satisficing
        # budget as a baseline (does an interpretable a-priori budget beat an
        # opaque loss weight?).
        self.censored_weight = float(censored_weight)
        self.anchor_weight = float(anchor_weight)
        self.model = None

    def fit(self, X, lower, upper, exact_mask=None, anchor_mu=None):
        import torch

        from satisficing_losses import _exact_mask, tobit_nll
        torch.manual_seed(self.seed)
        dev = torch.device(self.device)
        self.model = self.make_model().to(dev)
        Xt = torch.as_tensor(np.asarray(X, np.float32), device=dev)
        lo = torch.as_tensor(np.asarray(lower, np.float32), device=dev)
        hi = torch.as_tensor(np.asarray(upper, np.float32), device=dev)
        em = _exact_mask(lo, hi, torch.as_tensor(np.asarray(exact_mask, bool), device=dev)
                         if exact_mask is not None else None)
        anchor = None
        if anchor_mu is not None and self.anchor_weight != 0.0:
            anchor = torch.as_tensor(np.asarray(anchor_mu, np.float32), device=dev)
            if anchor.shape[0] != Xt.shape[0]:
                raise ValueError("anchor_mu must align with the training rows")
        opt = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        n = Xt.shape[0]
        gen = torch.Generator().manual_seed(self.seed)
        w = self.censored_weight
        for _ in range(self.epochs):
            self.model.train()
            perm = torch.randperm(n, generator=gen)
            for i in range(0, n, self.batch_size):
                idx = perm[i:i + self.batch_size]
                opt.zero_grad()
                mu, sg = self.model(Xt[idx])
                per_row = tobit_nll(mu, sg, lo[idx], hi[idx], self.dist, exact_mask=em[idx])
                if w == 1.0:
                    loss = per_row.mean()
                else:
                    # weight censored rows by w, exact rows by 1
                    exb = em[idx]
                    wts = torch.where(exb, torch.ones_like(per_row), torch.full_like(per_row, w))
                    loss = (per_row * wts).sum() / wts.sum().clamp(min=1.0)
                if anchor is not None:
                    loss = loss + self.anchor_weight * ((mu - anchor[idx]) ** 2).mean()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                opt.step()
        return self

    def predict_dist(self, X):
        import torch
        self.model.eval()
        dev = torch.device(self.device)
        with torch.no_grad():
            mu, sg = self.model(torch.as_tensor(np.asarray(X, np.float32), device=dev))
        return mu.cpu().numpy(), sg.cpu().numpy()

    def predict(self, X):
        return self.predict_dist(X)[0]


# --------------------------------------------------------------------------- #
# Deep ensemble (mixture predictive)                                          #
# --------------------------------------------------------------------------- #
class DeepEnsemble:
    """Train K SatisficingTrainer members with different seeds; expose the
    mixture predictive [(mu_k, sigma_k)]."""

    def __init__(self, make_model, dist, base_config, k: int = 5):
        self.make_model = make_model
        self.dist = dist
        self.base_config = base_config
        self.k = k
        self.members = []

    def fit(self, X, lower, upper, exact_mask, constraints, anchor_mu=None, assay_idx=None):
        from dataclasses import replace

        from satisficing_trainer import SatisficingTrainer
        self.members = []
        for m in range(self.k):
            cfg = replace(self.base_config, seed=self.base_config.seed + m)
            model = self.make_model()
            tr = SatisficingTrainer(model, self.dist, cfg)
            tr.fit(X, lower, upper, exact_mask, constraints, anchor_mu=anchor_mu, assay_idx=assay_idx)
            self.members.append(tr)
        return self

    def predict_mixture(self, X, assay_idx=None):
        return [tr.predict(X, assay_idx) for tr in self.members]

    def predict(self, X, assay_idx=None):
        """Ensemble mean prediction (for point metrics)."""
        comps = self.predict_mixture(X, assay_idx)
        mus = np.stack([c[0] for c in comps])
        # predictive mean = mean of component means; predictive sigma via total
        # variance = mean(sigma^2) + var(mu) (law of total variance).
        sigs = np.stack([c[1] for c in comps])
        mu = mus.mean(axis=0)
        sigma = np.sqrt((sigs ** 2).mean(axis=0) + mus.var(axis=0))
        return mu, sigma
