"""Controlled experiment matrix (plan §10-14).

One experiment = (endpoint, split-kind, fold, treatment, backbone, seed). Each
produces one row of metrics. Treatments span the comparison axis:

  T0  exact-only          : ignore censored rows entirely (accuracy ceiling / violation floor)
  T1  tobit                : full interval likelihood (the "censor-aware" prior art)
  T2  satisficing@eps      : our method at a given violation budget (per eps)
  T3  weighted_tobit       : scalar-weighted interval likelihood comparator
  T4  soft_satisficing     : same-deficit scalar-penalty comparator
  T5  aft_conc             : concentration-space XGBoost-AFT competitor
  T6  ensemble_satisficing : deep ensemble of the satisficing head (UQ arm)

Backbones: "ecfp_mlp" (Morgan -> heteroscedastic MLP) is the default; the AFT
treatment uses XGBoost on the same features. (A Chemprop D-MPNN backbone hook is
provided but off by default — enabled by backbone="dmpnn" where available.)

The runner is deliberately side-effect-light: run_cell(...) returns a dict; the
orchestrator collects rows into a DataFrame. Metrics come from metrics.py and are
computed on the correct row subsets (accuracy on exact rows, violation on
censored rows, decision/exceedance on resolvable rows).
"""
from __future__ import annotations

from dataclasses import replace

import numpy as np

from baselines import DeepEnsemble, MLPSoftSatisficing, MLPTobit, XGBoostAFTConcentration
from heads import HeteroscedasticMLP
from metrics import (
    accuracy_metrics,
    decision_metrics,
    exceedance_calibration,
    interval_coverage,
    interval_violation_rate,
)
from satisficing_losses import LatentDistribution
from satisficing_trainer import (
    SatisficingTrainer,
    TrainConfig,
    default_direction_constraints,
)


def _threshold_for(ds, train_idx):
    """Decision threshold t_e for exceedance metrics: the median exact potency on
    train (a data-driven "active/inactive" cut). Endpoint-agnostic and leakage-safe."""
    vp = ds.meta["value_p"].to_numpy(dtype=float)
    ex = ds.exact_mask & np.isfinite(vp)
    vals = vp[train_idx][ex[train_idx]] if np.any(ex[train_idx]) else vp[ex]
    return float(np.median(vals)) if len(vals) else 6.0


def _build_features(ds, train_idx, use_context, text_dim=16):
    """Return (X_full, assay_idx, enc). X_full concatenates chemistry with context
    features (context fit on train only)."""
    X = ds.X
    enc = None
    if use_context and ds.granularity == "measurement":
        enc = ContextEncoder(text_dim=text_dim).fit(ds.meta, train_idx)
        Xc = enc.transform(ds.meta)
        if Xc.shape[1] > 0:
            X = np.concatenate([X, Xc], axis=1).astype(np.float32)
    return X, enc


def _make_mlp(in_dim, n_assays=0, hidden=(256, 128), homoscedastic=False):
    return lambda: HeteroscedasticMLP(in_dim, hidden=hidden, n_assays=n_assays,
                                      homoscedastic=homoscedastic, dropout=0.0)


def run_cell(ds, split_name, train_idx, test_idx, treatment, seed=0,
             eps=0.05, tau=0.85, epochs=120, dist_name="gaussian", df=4.0,
             n_jobs=8, ensemble_k=5, device="cpu", anchor_cache=None,
             anchor_weight=1.0, return_constraint_diagnostics=False):
    """Run one experiment cell. Returns a metrics dict (or {'error':...}).

    anchor_cache: optional path (str) to a directory where the exact-only anchor
    for this (endpoint, split, fold, seed) is cached, so satisficing / ensemble /
    every-eps cells that share it do not each retrain it."""
    dist = LatentDistribution(dist_name, df=df)
    y = ds.meta["value_p"].to_numpy(dtype=float)
    lo, hi, ex = ds.lower, ds.upper, ds.exact_mask
    t_e = _threshold_for(ds, train_idx)

    base = dict(endpoint=ds.endpoint_id, granularity=ds.granularity, split=split_name,
                treatment=treatment, seed=seed, eps=eps, tau=tau, dist=dist_name,
                n_train=int(len(train_idx)), n_test=int(len(test_idx)), threshold=t_e)

    try:
        # ---------------- feature setup ----------------
        X, enc = _build_features(ds, train_idx, use_context=False)

        cfg = TrainConfig(tau=tau, nu=anchor_weight, epochs=epochs, batch_size=1024, lr=1e-3,
                          seed=seed, device=device)

        # ---------------- treatments ----------------
        if treatment == "exact_only":
            tr = SatisficingTrainer(_make_mlp(X.shape[1])(), dist, replace(cfg, nu=0.0))
            tr.fit(X[train_idx], lo[train_idx], hi[train_idx], ex[train_idx], constraints=[])
            pred = tr.predict(X[test_idx])

        elif treatment == "tobit":
            # genuine full interval-censored NLL (prior-art censor-aware baseline)
            m = MLPTobit(_make_mlp(X.shape[1]), dist, epochs=epochs, seed=seed, device=device)
            m.fit(X[train_idx], lo[train_idx], hi[train_idx], ex[train_idx])
            pred = m.predict_dist(X[test_idx])

        elif treatment == "weighted_tobit":
            # weighted interval-NLL: censored-row term scaled by `eps` (reused as the
            # weight w here). Sweeping w traces the weighted-Tobit frontier, the
            # scalar-knob baseline for the satisficing budget.
            m = MLPTobit(_make_mlp(X.shape[1]), dist, epochs=epochs, seed=seed,
                         device=device, censored_weight=eps)
            m.fit(X[train_idx], lo[train_idx], hi[train_idx], ex[train_idx])
            pred = m.predict_dist(X[test_idx])

        elif treatment == "weighted_tobit_anchor":
            # Fair sensitivity comparator: the same exact-only minimal-deviation
            # anchor and weight used by satisficing, paired with weighted interval NLL.
            anchor = _get_anchor(ds, X, lo, hi, ex, train_idx, dist, cfg,
                                 split_name, seed, anchor_cache)
            m = MLPTobit(_make_mlp(X.shape[1]), dist, epochs=epochs, seed=seed,
                         device=device, censored_weight=eps, anchor_weight=anchor_weight)
            m.fit(X[train_idx], lo[train_idx], hi[train_idx], ex[train_idx], anchor_mu=anchor)
            pred = m.predict_dist(X[test_idx])

        elif treatment == "soft_satisficing":
            # Holds the exact loss, anchor, predictive family, and direction-wise
            # deficit fixed; only the constrained formulation becomes a penalty.
            anchor = _get_anchor(ds, X, lo, hi, ex, train_idx, dist, cfg,
                                 split_name, seed, anchor_cache)
            m = MLPSoftSatisficing(
                _make_mlp(X.shape[1]), dist, lambda_=eps, anchor_weight=anchor_weight,
                tau=tau, epochs=epochs, seed=seed, device=device,
            )
            m.fit(X[train_idx], lo[train_idx], hi[train_idx], ex[train_idx], anchor_mu=anchor)
            pred = m.predict_dist(X[test_idx])

        elif treatment == "satisficing":
            anchor = (_get_anchor(ds, X, lo, hi, ex, train_idx, dist, cfg, split_name, seed, anchor_cache)
                      if anchor_weight != 0 else None)
            specs = default_direction_constraints(lo[train_idx], hi[train_idx], ex[train_idx], eps=eps)
            tr = SatisficingTrainer(_make_mlp(X.shape[1])(), dist, cfg)
            tr.fit(X[train_idx], lo[train_idx], hi[train_idx], ex[train_idx], specs, anchor_mu=anchor)
            pred = tr.predict(X[test_idx])
            if return_constraint_diagnostics:
                scored = _score(base, pred, pred, y, lo, hi, ex, test_idx, t_e, dist_name, df)
                return {**scored, **_constraint_diagnostics(
                    tr, specs, X[train_idx], lo[train_idx], hi[train_idx], ex[train_idx], tau
                )}

        elif treatment == "aft_conc":
            # carve a validation slice from train for early stopping
            rng = np.random.default_rng(seed)
            tr_perm = rng.permutation(train_idx)
            n_val = max(50, int(0.15 * len(tr_perm)))
            va_idx, fit_idx = tr_perm[:n_val], tr_perm[n_val:]
            m = XGBoostAFTConcentration(seed=seed, n_jobs=n_jobs)
            m.fit(X[fit_idx], lo[fit_idx], hi[fit_idx], X[va_idx], lo[va_idx], hi[va_idx])
            pred = m.predict_dist(X[test_idx])

        elif treatment == "ensemble_satisficing":
            anchor = _get_anchor(ds, X, lo, hi, ex, train_idx, dist, cfg, split_name, seed, anchor_cache)
            specs = default_direction_constraints(lo[train_idx], hi[train_idx], ex[train_idx], eps=eps)
            ens = DeepEnsemble(_make_mlp(X.shape[1]), dist, cfg, k=ensemble_k)
            ens.fit(X[train_idx], lo[train_idx], hi[train_idx], ex[train_idx], specs, anchor_mu=anchor)
            mix = ens.predict_mixture(X[test_idx])
            mu = np.mean([c[0] for c in mix], axis=0)
            sigma = np.sqrt(np.mean([c[1] ** 2 for c in mix], axis=0) + np.var([c[0] for c in mix], axis=0))
            pred = (mu, sigma)
            pred_for_prob = mix   # mixture for exceedance/calibration
            return _score(base, pred, pred_for_prob, y, lo, hi, ex, test_idx, t_e, dist_name, df)

        else:
            return {**base, "error": f"unknown treatment {treatment}"}

        return _score(base, pred, pred, y, lo, hi, ex, test_idx, t_e, dist_name, df)

    except Exception as e:  # never let one cell kill the matrix
        import traceback
        return {**base, "error": f"{type(e).__name__}: {e}", "traceback": traceback.format_exc()[-800:]}


def _get_anchor(ds, X, lo, hi, ex, train_idx, dist, cfg, split_name, seed, anchor_cache):
    """Return the exact-only anchor for this (endpoint, split, fold, seed),
    reading/writing a race-safe .npy cache when anchor_cache is given so that all
    satisficing / ensemble / per-eps cells sharing the same fold reuse ONE anchor
    instead of retraining it (a pure speedup; the anchor is deterministic in seed
    and does not depend on eps)."""
    if not anchor_cache:
        return _exact_anchor(X, lo, hi, ex, train_idx, dist, cfg)
    import hashlib
    import os
    from pathlib import Path
    key = hashlib.sha256(
        f"{ds.endpoint_id}|{split_name}|{seed}|{dist.dist}|{cfg.epochs}|{cfg.tau}|"
        f"{train_idx.tobytes()}|{X.shape[1]}".encode()
    ).hexdigest()[:20]
    cdir = Path(anchor_cache); cdir.mkdir(parents=True, exist_ok=True)
    path = cdir / f"anchor_{key}.npy"
    if path.exists():
        try:
            return np.load(path)
        except Exception:
            pass  # corrupt/partial -> recompute
    anchor = _exact_anchor(X, lo, hi, ex, train_idx, dist, cfg)
    # atomic write (unique tmp then rename) so concurrent workers don't collide
    tmp = cdir / f".anchor_{key}.{os.getpid()}.tmp.npy"
    try:
        np.save(tmp, anchor)
        os.replace(tmp, path)
    except Exception:
        if tmp.exists():
            tmp.unlink(missing_ok=True)
    return anchor


def _exact_anchor(X, lo, hi, ex, train_idx, dist, cfg, assay_idx=None):
    """Train an exact-only reference model and return its predictions on the TRAIN
    rows, ORDERED to match X[train_idx] (the array the trainer receives), for the
    minimal-deviation anchor (plan §2.3).

    IMPORTANT: the trainer is called with X[train_idx] and indexes anchor_mu with
    train-relative batch positions in [0, n_train). The anchor must therefore be a
    train-length array in the SAME order as X[train_idx] -- NOT a full-dataset
    array indexed by dataset row (that misaligns the anchor for every split whose
    train_idx != arange(n_train))."""
    # The anchor is the POPULATION-level exact-only reference (no assay effects):
    # a plain heteroscedastic MLP on chemistry+context. The constrained model is
    # what adds the per-assay offsets, and the minimal-deviation term keeps it
    # close to this population reference. assay_idx is intentionally unused here.
    ref = SatisficingTrainer(_make_mlp(X.shape[1])(), dist, replace(cfg, nu=0.0, epochs=max(40, cfg.epochs // 2)))
    ref.fit(X[train_idx], lo[train_idx], hi[train_idx], ex[train_idx], constraints=[])
    return ref.predict(X[train_idx])[0].astype(np.float32)


def _score(base, pred_point, pred_prob, y, lo, hi, ex, test_idx, t_e, dist_name, df):
    """Compute the full metric tier on the test rows and return a flat dict."""
    mu = pred_point[0] if isinstance(pred_point, tuple) else pred_point
    acc = accuracy_metrics(mu, y[test_idx], ex[test_idx])
    viol = interval_violation_rate(mu, lo[test_idx], hi[test_idx], ex[test_idx])
    # slice the predictive distribution to test rows for prob metrics
    prob_sliced = _slice_pred(pred_prob, test_idx, full_len=len(y))
    dec = decision_metrics(prob_sliced, y[test_idx], lo[test_idx], hi[test_idx], ex[test_idx],
                           t=t_e, dist=dist_name, df=df)
    cov = interval_coverage(prob_sliced, y[test_idx], ex[test_idx], level=0.9, dist=dist_name, df=df)
    cal = exceedance_calibration(prob_sliced, y[test_idx], lo[test_idx], hi[test_idx], ex[test_idx],
                                 t=t_e, dist=dist_name, df=df)
    out = dict(base)
    for k, v in acc.items():
        out[f"acc_{k}"] = v
    for k, v in viol.items():
        out[f"viol_{k}"] = v
    for k, v in dec.items():
        out[f"dec_{k}"] = v
    out["cov90"] = cov.get("coverage")
    out["cov90_width"] = cov.get("width")
    out["exceed_ece"] = cal.get("ece")
    return out


def _constraint_diagnostics(trainer, specs, X, lower, upper, exact_mask, tau):
    """Return final full-data G values plus interpretable censored-row summaries."""
    from scipy.special import ndtr

    final = trainer.history[-1]
    mu, sigma = trainer.predict(X)
    sigma = np.clip(sigma, 1e-6, None)
    censored = ~np.asarray(exact_mask, bool)
    z_lo = (np.asarray(lower, float) - mu) / sigma
    z_hi = (np.asarray(upper, float) - mu) / sigma
    q = np.ones_like(mu, dtype=float)
    right = censored & np.isposinf(upper)
    left = censored & np.isneginf(lower)
    interval = censored & np.isfinite(lower) & np.isfinite(upper)
    q[right] = ndtr(-z_lo[right])
    q[left] = ndtr(z_hi[left])
    q[interval] = np.clip(ndtr(z_hi[interval]) - ndtr(z_lo[interval]), 0.0, 1.0)
    qc = q[censored]
    result = {
        "constraint_fraction_q_ge_tau": float(np.mean(qc >= tau)) if len(qc) else np.nan,
        "constraint_mean_probability_shortfall": float(np.mean(np.maximum(tau - qc, 0.0))) if len(qc) else np.nan,
        "constraint_mean_probability": float(np.mean(qc)) if len(qc) else np.nan,
    }
    for spec in specs:
        result[f"constraint_G_{spec.name}"] = float(final["G"].get(spec.name, np.nan))
        result[f"constraint_lambda_{spec.name}"] = float(final["lambda"].get(spec.name, np.nan))
    return result


def _slice_pred(pred, idx, full_len):
    """pred is (mu,sigma) or a mixture list, already aligned to TEST rows in most
    treatments. When it is full-length (mu covers all rows) slice to idx; when it
    already matches len(idx) leave it. Detect by length."""
    def _sl(mu, sigma):
        mu = np.asarray(mu); sigma = np.asarray(sigma)
        if len(mu) == full_len:
            return mu[idx], sigma[idx]
        return mu, sigma
    if isinstance(pred, tuple):
        return _sl(pred[0], pred[1])
    return [_sl(m, s) for (m, s) in pred]
