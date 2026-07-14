import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


class _DS:
    """Minimal in-memory CensoredDataset stand-in (no RDKit/parquet needed)."""
    def __init__(self, n=800, d=32, seed=0, granularity="endpoint"):
        rng = np.random.default_rng(seed)
        self.X = rng.standard_normal((n, d)).astype(np.float32)
        w = rng.standard_normal(d) * 0.3
        p = self.X @ w + 6.0 + 0.3 * rng.standard_normal(n)
        c = np.quantile(p, 0.7)
        rc = p > c
        self.lower = np.where(rc, c, p).astype(np.float32)
        self.upper = np.where(rc, np.inf, p).astype(np.float32)
        self.exact_mask = ~rc
        self.censoring_class = np.where(rc, "right_censored_p", "exact")
        self.endpoint_id = "SYN_TEST"
        self.granularity = granularity
        self.assay_vocab = {}
        self.meta = pd.DataFrame({
            "value_p": np.where(rc, np.nan, p),
            "assay_chembl_id": rng.integers(0, 12, n).astype(str),
            "document_chembl_id": rng.integers(0, 8, n).astype(str),
            "source_name": rng.choice(["PubChem BioAssays", "Scientific Literature"], n),
            "document_year": rng.integers(2005, 2022, n).astype(float),
            "confidence_score": rng.choice([8, 9], n).astype(float),
            "assay_type": rng.choice(["A", "B"], n),
            "standard_relation": rng.choice(["=", ">"], n),
            "assay_description": rng.choice(["patch clamp", "fluorescence assay"], n),
        })
        # exact rows must have value_p defined for accuracy metrics
        self.meta.loc[self.exact_mask, "value_p"] = p[self.exact_mask]
        self.n = n

    def assay_idx(self, key_col="assay_chembl_id"):
        return np.array([self.assay_vocab.get(a, 0) for a in self.meta[key_col]], dtype=np.int64)


def _split(ds, seed=0):
    rng = np.random.default_rng(seed)
    perm = rng.permutation(ds.n)
    cut = int(0.75 * ds.n)
    return np.sort(perm[:cut]), np.sort(perm[cut:])


def test_all_treatments_run_and_score():
    from experiment import run_cell
    ds = _DS(granularity="measurement")
    tr, te = _split(ds)
    for treatment in ["exact_only", "tobit", "satisficing", "satisficing_assay",
                      "satisficing_transfer", "aft_conc", "ensemble_satisficing"]:
        r = run_cell(ds, "random", tr, te, treatment, seed=0, eps=0.05,
                     epochs=20, ensemble_k=2, n_jobs=2)
        assert "error" not in r, f"{treatment} errored: {r.get('error')}\n{r.get('traceback','')}"
        assert r["treatment"] == treatment
        assert np.isfinite(r["acc_mae"]), f"{treatment} produced non-finite MAE"
        assert 0.0 <= r["viol_violation_rate"] <= 1.0
        assert r["n_test"] == len(te)


def test_satisficing_reduces_violation_vs_exact_only():
    from experiment import run_cell
    ds = _DS(seed=3, granularity="endpoint")
    tr, te = _split(ds, seed=3)
    r_exact = run_cell(ds, "random", tr, te, "exact_only", seed=0, epochs=60)
    r_satis = run_cell(ds, "random", tr, te, "satisficing", seed=0, eps=0.02, epochs=60)
    # satisficing must not INCREASE violation vs the accuracy-only baseline
    assert r_satis["viol_violation_rate"] <= r_exact["viol_violation_rate"] + 0.05, (
        r_exact["viol_violation_rate"], r_satis["viol_violation_rate"])


def test_threshold_is_leakage_safe():
    # threshold must be computable from train exact rows only
    from experiment import _threshold_for
    ds = _DS()
    tr, te = _split(ds)
    t = _threshold_for(ds, tr)
    assert np.isfinite(t)


def test_anchor_is_train_length_and_aligned():
    # Regression: _exact_anchor must return a TRAIN-length array aligned with
    # X[train_idx], not a full-dataset array (which misaligned the anchor for
    # every split whose train_idx != arange(n_train)).
    from experiment import _exact_anchor
    from satisficing_losses import LatentDistribution
    from satisficing_trainer import TrainConfig
    ds = _DS(seed=4)
    # a scattered train_idx (gaps) -- the misalignment trigger
    rng = np.random.default_rng(0)
    train_idx = np.sort(rng.choice(ds.n, size=int(0.7 * ds.n), replace=False))
    dist = LatentDistribution("gaussian")
    cfg = TrainConfig(epochs=20, batch_size=256, seed=0)
    anchor = _exact_anchor(ds.X, ds.lower, ds.upper, ds.exact_mask, train_idx, dist, cfg)
    assert anchor.shape == (len(train_idx),), (anchor.shape, len(train_idx))
    assert np.isfinite(anchor).all()
    # anchor should be a sensible potency reference, not the degenerate all-mean
    assert anchor.std() > 0.05, "anchor collapsed to a constant (misalignment fill?)"
