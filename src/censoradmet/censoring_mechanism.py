"""Censoring-mechanism analysis (plan §8).

Question: is censoring in these ADMET assays ignorable (censoring depends only on
observed covariates x, e -- "MAR") or informative (censoring depends on the
latent value itself even after conditioning on x, e -- "MNAR")? If informative,
a likelihood that treats the censoring indicator as ignorable is mis-specified,
which is itself a publishable finding and motivates the assay-aware terms.

Two analyses:

  (1) Predictability of censoring from context. Fit a classifier for the
      censoring event (censored vs exact) and for the direction (left/right)
      using ONLY chemistry x and assay context e. High AUC => censoring is
      strongly structured by protocol/assay (assay cutoffs), i.e. the censoring
      threshold is a property of the assay, supporting the assay-aware design.

  (2) Informativeness test (a practical MNAR probe). On assays that contain BOTH
      exact and censored measurements, compare the fitted potency distribution of
      exact rows against the implied bound distribution: if censored rows'
      thresholds sit in a systematically different potency region than exact
      rows AFTER conditioning on chemistry, the censoring is informative. We
      implement the Little (1988)-style test as a group-wise comparison of the
      residual of a chemistry-only model between soon-to-be-censored and exact
      rows (a permutation test on the mean residual gap).

Everything is fit on train rows and evaluated on held-out rows; the permutation
test uses a fixed RNG seed passed in (Math.random is unavailable in this env's
workflow scripts but this is plain Python -- np.random.default_rng(seed))."""
from __future__ import annotations

import numpy as np


def censoring_labels(censoring_class):
    """Return (is_censored, is_left, is_right) boolean arrays."""
    cc = np.asarray(censoring_class)
    exact = cc == "exact"
    left = cc == "left_censored_p"
    right = cc == "right_censored_p"
    return (~exact), left, right


def predict_censoring(X_chem, X_ctx, censoring_class, train_idx, test_idx, seed=0):
    """Fit logistic classifiers (chem-only, ctx-only, chem+ctx) for the censoring
    EVENT and report test AUC for each. A large ctx-only AUC is direct evidence
    that assay context (not chemistry) governs censoring -- the paper's premise."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score

    is_cens, _, _ = censoring_labels(censoring_class)
    y = is_cens.astype(int)
    feats = {
        "chem": X_chem,
        "ctx": X_ctx,
        "chem+ctx": np.concatenate([X_chem, X_ctx], axis=1) if X_ctx.shape[1] else X_chem,
    }
    out = {}
    ytr = y[train_idx]
    if ytr.sum() == 0 or ytr.sum() == len(ytr):
        return {"note": "train has a single class; AUC undefined", "base_rate": float(y.mean())}
    yte = y[test_idx]
    for name, F in feats.items():
        if F.shape[1] == 0:
            continue
        clf = LogisticRegression(max_iter=300, C=1.0)
        clf.fit(F[train_idx], ytr)
        if len(np.unique(yte)) < 2:
            out[name] = np.nan
        else:
            p = clf.predict_proba(F[test_idx])[:, 1]
            out[name] = float(roc_auc_score(yte, p))
    out["base_rate"] = float(y.mean())
    out["n_test"] = int(len(test_idx))
    return out


def informativeness_test(X_chem, value_p, censoring_class, lower_p, upper_p,
                         train_idx, test_idx, n_perm=2000, seed=0):
    """Valid ignorable-censoring (MAR) probe via a Little (1988)-style test.

    Idea. Fit a chemistry-only regression of potency on EXACT train rows. If
    censoring is ignorable given chemistry (MAR/MCAR w.r.t. x), then the
    censoring INDICATOR should carry no additional information about the model's
    signed residual beyond x. We test this like-for-like.

    Construction (right-censored branch, the dominant direction here). For a
    right-censored row Y >= L we do NOT observe Y, but whenever the chemistry
    prediction p_hat < L we KNOW the signed residual (Y - p_hat) is strictly
    positive and at least (L - p_hat) -- a valid, observed lower bound on the
    residual. For an exact row the signed residual (y - p_hat) is fully observed.
    We compare the distribution of a COMPARABLE statistic between the two groups:
    r_censored = max(0, L - p_hat)   (observed positive-residual lower bound)
    r_exact    = max(0, y - p_hat)   (its exact analogue -- same clipping, same units)
    r_censored is a LOWER BOUND on the censored rows' true positive residual, so
    this test is conservative: a significant positive difference is strong
    evidence that censored rows carry larger residuals to a chemistry-only model
    than exact rows do. HONEST CAVEAT: because the threshold L is a property of
    the assay design, a positive result establishes that censoring co-varies with
    the chemistry-residual (i.e. censoring is not ignorable given x ALONE); it
    does not by itself separate "the assay simply censors high-potency compounds"
    from residual MNAR after also conditioning on assay context. We therefore
    report it as a censoring-informativeness diagnostic, and pair it with
    predict_censoring (which measures how much of the censoring is explained by
    assay CONTEXT vs chemistry). Interpret alongside that, not in isolation.

    Returns the observed mean difference and a two-sided permutation p-value."""
    from sklearn.linear_model import Ridge

    cc = np.asarray(censoring_class)
    vp = np.asarray(value_p, dtype=float)
    lo = np.asarray(lower_p, dtype=float)

    exact_tr = train_idx[cc[train_idx] == "exact"]
    if len(exact_tr) < 50:
        return {"note": "too few exact train rows", "n_exact_train": int(len(exact_tr))}
    reg = Ridge(alpha=1.0)
    reg.fit(X_chem[exact_tr], vp[exact_tr])

    rc_te = test_idx[cc[test_idx] == "right_censored_p"]
    ex_te = test_idx[cc[test_idx] == "exact"]
    if len(rc_te) < 15 or len(ex_te) < 15:
        return {"note": "too few test rows in a group",
                "n_right_test": int(len(rc_te)), "n_exact_test": int(len(ex_te))}

    pred_rc = reg.predict(X_chem[rc_te])
    pred_ex = reg.predict(X_chem[ex_te])
    # SAME functional for both groups: clipped positive residual to the chem model.
    r_censored = np.clip(lo[rc_te] - pred_rc, 0.0, None)   # observed lower bound on (Y - p_hat)_+
    r_exact = np.clip(vp[ex_te] - pred_ex, 0.0, None)      # exact (y - p_hat)_+

    observed = float(np.mean(r_censored) - np.mean(r_exact))

    rng = np.random.default_rng(seed)
    combined = np.concatenate([r_censored, r_exact])
    n_rc = len(r_censored)
    count = 0
    for _ in range(n_perm):
        perm = rng.permutation(combined)
        stat = np.mean(perm[:n_rc]) - np.mean(perm[n_rc:])
        if abs(stat) >= abs(observed):
            count += 1
    pval = (count + 1) / (n_perm + 1)
    return {
        "observed_mean_diff": observed,
        "perm_pvalue": float(pval),
        # informative (non-ignorable) if censored rows carry SYSTEMATICALLY larger
        # positive residuals than exact rows after conditioning on chemistry.
        "informative": bool(pval < 0.05 and observed > 0),
        "statistic": "clipped_positive_residual",
        "n_right_test": int(len(rc_te)),
        "n_exact_test": int(len(ex_te)),
    }
