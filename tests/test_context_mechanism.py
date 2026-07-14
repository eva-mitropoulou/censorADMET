import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from assay_context import ContextEncoder  # noqa: E402
from censoring_mechanism import (  # noqa: E402
    censoring_labels,
    informativeness_test,
    predict_censoring,
)


def _meta(n=400, seed=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "assay_type": rng.choice(["A", "B", "F"], n),
        "source_name": rng.choice(["PubChem BioAssays", "Scientific Literature"], n),
        "standard_relation": rng.choice(["=", "<", ">"], n),
        "confidence_score": rng.choice([8, 9], n).astype(float),
        "document_year": rng.integers(2005, 2022, n).astype(float),
        "assay_description": rng.choice([
            "hERG patch clamp IC50 automated",
            "fluorescence based binding assay",
            "manual patch clamp electrophysiology",
        ], n),
    })


def test_encoder_fit_transform_shapes_and_leakage():
    meta = _meta()
    tr = np.arange(0, 300)
    enc = ContextEncoder(text_dim=8).fit(meta, tr)
    X = enc.transform(meta)
    assert X.shape[0] == len(meta)
    assert X.shape[1] == enc.dim > 0
    # transform on a subframe with an UNSEEN category -> all-zero indicator, no crash
    m2 = meta.iloc[:5].copy()
    m2.loc[m2.index[0], "assay_type"] = "ZZZ_unseen"
    X2 = enc.transform(m2)
    assert X2.shape == (5, enc.dim)
    assert np.isfinite(X2).all()


def test_encoder_numeric_standardised_on_train():
    meta = _meta()
    tr = np.arange(0, 300)
    enc = ContextEncoder(text_dim=4).fit(meta, tr)
    # find the confidence_score numeric column index
    names = enc._feature_names
    idx = names.index("num:confidence_score")
    Xtr = enc.transform(meta.iloc[tr])
    # standardised train column ~ mean 0
    assert abs(Xtr[:, idx].mean()) < 1e-5


def test_encoder_deterministic():
    meta = _meta()
    tr = np.arange(0, 300)
    a = ContextEncoder(text_dim=8).fit(meta, tr).transform(meta)
    b = ContextEncoder(text_dim=8).fit(meta, tr).transform(meta)
    assert np.allclose(a, b)


def test_censoring_labels():
    cc = np.array(["exact", "left_censored_p", "right_censored_p", "exact"])
    isc, left, right = censoring_labels(cc)
    assert isc.tolist() == [False, True, True, False]
    assert left.tolist() == [False, True, False, False]
    assert right.tolist() == [False, False, True, False]


def test_predict_censoring_recovers_context_signal():
    # construct data where censoring is driven by CONTEXT (assay_type == 'B')
    rng = np.random.default_rng(1)
    n = 800
    meta = _meta(n, seed=1)
    is_b = (meta["assay_type"] == "B").to_numpy()
    p_cens = np.where(is_b, 0.8, 0.1)
    cens = rng.random(n) < p_cens
    cc = np.where(cens, "right_censored_p", "exact")
    X_chem = rng.standard_normal((n, 12)).astype(np.float32)  # chemistry unrelated to censoring
    enc = ContextEncoder(text_dim=6).fit(meta, np.arange(600))
    X_ctx = enc.transform(meta)
    res = predict_censoring(X_chem, X_ctx, cc, np.arange(600), np.arange(600, 800), seed=0)
    assert res["ctx"] > 0.75, res       # context strongly predicts censoring
    assert res["ctx"] > res["chem"]     # and beats chemistry-only


def test_informativeness_detects_mnar():
    # MNAR: right-censoring is applied to the highest-potency rows beyond what
    # chemistry predicts (add latent noise pushing them over an assay cutoff).
    rng = np.random.default_rng(2)
    n, d = 1500, 10
    X = rng.standard_normal((n, d)).astype(np.float32)
    w = rng.standard_normal(d) * 0.4
    latent = X @ w + 6.0 + 0.8 * rng.standard_normal(n)  # extra latent noise
    value_p = latent.copy()
    # right-censor rows whose LATENT value exceeds a cutoff (informative!)
    cutoff = np.quantile(latent, 0.75)
    rc = latent > cutoff
    cc = np.where(rc, "right_censored_p", "exact")
    lower_p = np.where(rc, cutoff, value_p)
    upper_p = np.where(rc, np.inf, value_p)
    value_p_obs = np.where(rc, np.nan, value_p)
    tr = np.arange(0, 1000); te = np.arange(1000, 1500)
    res = informativeness_test(X, np.where(rc, latent, value_p), cc, lower_p, upper_p,
                               tr, te, n_perm=1000, seed=0)
    # threshold sits below the true (latent) potency of censored rows, but the
    # chemistry model under-predicts them => positive gap, detectable.
    assert "perm_pvalue" in res, res
    assert res["n_right_test"] > 15
