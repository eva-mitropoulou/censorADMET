"""Result aggregation and endpoint-level statistics.

Consumes the tidy all_results.parquet emitted by run_matrix.py and produces the
paper's headline comparisons with HONEST, non-pseudoreplicated inference.

Key statistical discipline:
  * The inference unit is the ENDPOINT (or endpoint x split-kind), NOT the
    individual (endpoint, fold, seed) cell. Folds and seeds are correlated
    replicates within an endpoint; treating them as independent inflates n and
    fabricates tiny p-values. We therefore aggregate cell metrics to one value
    per endpoint (mean over folds x seeds) BEFORE any paired test.
  * Method-vs-method comparisons use a paired test across endpoints (Wilcoxon
    signed-rank), with Benjamini-Hochberg FDR across the family of comparisons.
  * We also fit a mixed-effects model (metric ~ treatment + (1|endpoint)) via
    statsmodels when available, as a hierarchical cross-check that respects the
    endpoint grouping (plan §17).

Nothing here fabricates numbers: if a comparison has too few endpoints it is
reported as such rather than forced.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


# --------------------------------------------------------------------------- #
# Aggregation                                                                 #
# --------------------------------------------------------------------------- #
METRIC_COLS = [
    "acc_mae", "acc_rmse", "acc_spearman",
    "viol_violation_rate", "viol_left_violation_rate", "viol_right_violation_rate",
    "cov90", "cov90_width", "exceed_ece",
    "dec_false_safe_rate", "dec_false_liability_rate", "dec_balanced_accuracy",
    "dec_decision_regret",
]


def _treatment_label(row):
    """Distinguish eps-swept treatments by their eps in the label."""
    t = row["treatment"]
    if t in ("satisficing", "ensemble_satisficing"):
        return f"{t}@{row['eps']:.2f}"
    return t


def load_results(parquet_path):
    df = pd.read_parquet(parquet_path)
    if "error" in df.columns:
        n_err = df["error"].notna().sum()
        if n_err:
            print(f"[analysis] dropping {n_err} errored cells")
        df = df[df["error"].isna()].copy()
    df["treatment_label"] = df.apply(_treatment_label, axis=1)
    return df


def aggregate_to_endpoint(df, by=("endpoint", "split_kind", "treatment_label")):
    """Mean each metric over folds x seeds -> one row per (endpoint, split, treatment).
    This is the inference unit; folds/seeds are correlated replicates."""
    metrics = [m for m in METRIC_COLS if m in df.columns]
    agg = df.groupby(list(by))[metrics].mean().reset_index()
    counts = df.groupby(list(by)).size().reset_index(name="n_cells")
    return agg.merge(counts, on=list(by))


# --------------------------------------------------------------------------- #
# Paired method comparison across endpoints                                   #
# --------------------------------------------------------------------------- #
def paired_comparison(agg, metric, treatment_a, treatment_b, split_kind=None):
    """Paired Wilcoxon of `metric` between two treatments across endpoints (the
    pairing unit). Positive `median_delta` means A > B on that metric."""
    from scipy.stats import wilcoxon
    d = agg if split_kind is None else agg[agg["split_kind"] == split_kind]
    a = d[d["treatment_label"] == treatment_a].set_index("endpoint")[metric]
    b = d[d["treatment_label"] == treatment_b].set_index("endpoint")[metric]
    common = a.index.intersection(b.index)
    a, b = a.loc[common], b.loc[common]
    pair = pd.DataFrame({"a": a, "b": b}).dropna()
    if len(pair) < 3:
        return {"metric": metric, "a": treatment_a, "b": treatment_b,
                "n_endpoints": int(len(pair)), "note": "too few endpoints"}
    delta = pair["a"] - pair["b"]
    try:
        stat, p = wilcoxon(pair["a"], pair["b"])
    except ValueError:
        stat, p = np.nan, 1.0
    return {
        "metric": metric, "a": treatment_a, "b": treatment_b,
        "split_kind": split_kind or "all",
        "n_endpoints": int(len(pair)),
        "median_delta": float(delta.median()),
        "mean_delta": float(delta.mean()),
        "wilcoxon_stat": float(stat) if np.isfinite(stat) else None,
        "p_value": float(p),
    }


def bh_fdr(pvals, alpha=0.05):
    """Benjamini-Hochberg FDR. Returns (rejected_bool_array, qvalues)."""
    p = np.asarray(pvals, dtype=float)
    n = len(p)
    order = np.argsort(p)
    ranked = p[order]
    q = ranked * n / (np.arange(n) + 1)
    # enforce monotonicity
    q = np.minimum.accumulate(q[::-1])[::-1]
    q_full = np.empty(n)
    q_full[order] = np.clip(q, 0, 1)
    rejected = q_full <= alpha
    return rejected, q_full


def comparison_table(agg, metric, treatments, baseline, split_kind=None):
    """Compare every treatment against `baseline` on `metric`, with BH-FDR."""
    rows = [paired_comparison(agg, metric, t, baseline, split_kind)
            for t in treatments if t != baseline]
    valid = [r for r in rows if "p_value" in r]
    if valid:
        rej, q = bh_fdr([r["p_value"] for r in valid])
        for r, rr, qq in zip(valid, rej, q):
            r["fdr_qvalue"] = float(qq)
            r["fdr_significant"] = bool(rr)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Mixed-effects hierarchical model (plan §17)                                 #
# --------------------------------------------------------------------------- #
def mixed_effects(df, metric, treatments=None, split_kind=None):
    """metric ~ C(treatment) + (1|endpoint) via statsmodels MixedLM, on the
    per-cell data (folds/seeds as replicates within the endpoint random effect).
    Returns a summary dict of fixed-effect coefficients vs the reference treatment."""
    import statsmodels.formula.api as smf

    d = df.copy()
    if split_kind is not None:
        d = d[d["split_kind"] == split_kind]
    if treatments is not None:
        d = d[d["treatment_label"].isin(treatments)]
    d = d[["endpoint", "treatment_label", metric]].dropna()
    if d["endpoint"].nunique() < 3 or len(d) < 20:
        return {"note": "insufficient data for mixed model", "n": int(len(d))}
    d = d.rename(columns={metric: "y", "treatment_label": "trt"})
    try:
        md = smf.mixedlm("y ~ C(trt)", d, groups=d["endpoint"])
        mf = md.fit(method="lbfgs", maxiter=200, disp=False)
    except Exception as e:
        return {"note": f"mixed model failed: {e}"}
    out = {"metric": metric, "split_kind": split_kind or "all",
           "n_obs": int(len(d)), "n_endpoints": int(d["endpoint"].nunique()),
           "group_var": float(mf.cov_re.iloc[0, 0]) if mf.cov_re.size else None,
           "fixed_effects": {}}
    for name, coef, p in zip(mf.params.index, mf.params.values, mf.pvalues.values):
        out["fixed_effects"][str(name)] = {"coef": float(coef), "p_value": float(p)}
    return out


# --------------------------------------------------------------------------- #
# Pareto summary (plan §2.4, §13)                                             #
# --------------------------------------------------------------------------- #
def pareto_summary(agg, split_kind=None, x="viol_violation_rate", y="acc_mae"):
    """For each endpoint, return the (violation, MAE) operating points of exact_only,
    tobit, and satisficing@eps, so the frontier can be plotted / summarised."""
    d = agg if split_kind is None else agg[agg["split_kind"] == split_kind]
    keep = d[d["treatment_label"].str.startswith(("exact_only", "tobit", "satisficing@"))]
    return keep[["endpoint", "split_kind", "treatment_label", x, y]].sort_values(
        ["endpoint", x])
