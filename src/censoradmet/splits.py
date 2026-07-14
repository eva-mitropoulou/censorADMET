"""Distribution-shift splits (plan §9).

Each split returns a list of (name, train_idx, test_idx) folds. The random split
is an IID row-wise reference. Scaffold, assay, document, source and threshold
splits use their respective groups so that the grouped quantity is never split
across train and test; they probe chemical, assay, provenance, or potency-regime
shift rather than treating the random split as leakage-free chemistry evaluation.

Leakage guard: for grouped splits we additionally assert that no group key
appears in both train and test.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _grouped_kfold(groups: np.ndarray, k: int, seed: int):
    """Yield (train_idx, test_idx) putting whole groups in the test fold."""
    uniq = pd.unique(pd.Series(groups))
    rng = np.random.default_rng(seed)
    uniq = rng.permutation(uniq)
    folds = np.array_split(uniq, k)
    gser = pd.Series(groups)
    for f in folds:
        test_groups = set(f.tolist())
        test_mask = gser.isin(test_groups).to_numpy()
        test_idx = np.where(test_mask)[0]
        train_idx = np.where(~test_mask)[0]
        if len(test_idx) == 0 or len(train_idx) == 0:
            continue
        assert not (set(gser.iloc[train_idx]) & set(gser.iloc[test_idx])), "group leakage"
        yield train_idx, test_idx


def scaffold_split(ds, k: int = 5, seed: int = 0):
    g = ds.meta["split_group_scaffold"].to_numpy() if "split_group_scaffold" in ds.meta else \
        ds.meta["standardized_smiles"].to_numpy()
    return [(f"scaffold_fold{i}", tr, te) for i, (tr, te) in enumerate(_grouped_kfold(g, k, seed))]


def assay_split(ds, k: int = 5, seed: int = 0):
    """Assay-held-out: whole assays go to test. Directly tests the assay-shift
    hypothesis that motivates the assay-aware model."""
    g = ds.meta["assay_chembl_id"].astype(str).to_numpy()
    return [(f"assay_fold{i}", tr, te) for i, (tr, te) in enumerate(_grouped_kfold(g, k, seed))]


def document_split(ds, k: int = 5, seed: int = 0):
    g = ds.meta["document_chembl_id"].astype(str).to_numpy()
    return [(f"document_fold{i}", tr, te) for i, (tr, te) in enumerate(_grouped_kfold(g, k, seed))]


def assay_scaffold_split(ds, k: int = 5, seed: int = 0):
    """Crossed shift: test rows must have BOTH an unseen assay AND an unseen
    scaffold relative to train. Built by grouped-k-fold on the composite
    (assay, scaffold) key, then removing from each test fold any row whose assay
    OR scaffold also appears in that fold's train set. This is the strictest,
    most prospective split: simultaneous chemistry and assay shift."""
    assay = ds.meta["assay_chembl_id"].astype(str).to_numpy()
    scaf = (ds.meta["split_group_scaffold"].astype(str).to_numpy()
            if "split_group_scaffold" in ds.meta
            else ds.meta["standardized_smiles"].astype(str).to_numpy())
    composite = np.array([f"{a}|{s}" for a, s in zip(assay, scaf)])
    out = []
    for i, (tr, te) in enumerate(_grouped_kfold(composite, k, seed)):
        tr_assays = set(assay[tr]); tr_scaf = set(scaf[tr])
        keep = np.array([(assay[j] not in tr_assays) and (scaf[j] not in tr_scaf) for j in te])
        te2 = te[keep]
        if len(te2) >= 30 and len(tr) >= 100:
            out.append((f"assay_scaffold_fold{i}", tr, te2))
    return out


def source_split(ds):
    """Source-held-out (cross-source): each source_name becomes a test fold in
    turn, trained on all other sources. Cross-source is the strongest realistic
    shift (different labs, assay tech, curation)."""
    if "source_name" not in ds.meta:
        return []
    g = ds.meta["source_name"].astype(str)
    folds = []
    sources = [s for s, c in g.value_counts().items() if c >= 50]
    for s in sources:
        test_idx = np.where((g == s).to_numpy())[0]
        train_idx = np.where((g != s).to_numpy())[0]
        if len(test_idx) >= 30 and len(train_idx) >= 100:
            folds.append((f"source={s[:24]}", train_idx, test_idx))
    return folds


def threshold_split(ds, n_bands: int = 3, seed: int = 0):
    """Threshold-held-out (plan §9): hold out EXACT compounds whose exact potency
    falls in a target band; train on everything else. Tests extrapolation across
    the potency regime -- crucial because censoring is concentrated at specific
    assay cutoffs.

    Banding uses ONLY exact rows (value_p finite AND exact_mask True): a censored
    row's finite value_p equals its censoring THRESHOLD, not its potency, so
    banding censored rows by that value would mislabel the regime. Censored rows
    are always placed in TRAIN (they carry bound information without polluting the
    held-out potency band). The test set is therefore purely exact-potency rows
    in the target band, which is exactly the extrapolation quantity of interest."""
    vp = ds.meta["value_p"].to_numpy(dtype=float) if "value_p" in ds.meta else None
    if vp is None:
        return []
    exact = np.asarray(ds.exact_mask, dtype=bool)
    exact_finite = exact & np.isfinite(vp)
    if exact_finite.sum() < 100:
        return []
    qs = np.quantile(vp[exact_finite], np.linspace(0, 1, n_bands + 1))
    folds = []
    for b in range(n_bands):
        lo_q, hi_q = qs[b], qs[b + 1]
        # test band = exact rows in [lo_q, hi_q); censored & other-band rows train
        in_band = exact_finite & (vp >= lo_q) & (vp <= hi_q if b == n_bands - 1 else vp < hi_q)
        test_idx = np.where(in_band)[0]
        train_idx = np.where(~in_band)[0]
        if len(test_idx) >= 30 and len(train_idx) >= 100:
            folds.append((f"threshold_band{b}", train_idx, test_idx))
    return folds


def random_split(ds, k: int = 5, seed: int = 0):
    """IID reference (in-distribution) split for comparison."""
    n = ds.n
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    folds = np.array_split(perm, k)
    out = []
    for i, f in enumerate(folds):
        test_idx = np.sort(f)
        train_idx = np.sort(np.setdiff1d(perm, f))
        out.append((f"random_fold{i}", train_idx, test_idx))
    return out


def compound_overlap(ds, train_idx, test_idx, key_col="split_group_connectivity_inchikey"):
    """Fraction of test rows whose parent compound also appears in train (0 = no
    chemistry overlap). Reported per grouped split so 'assay-held-out' is not
    misread as 'unseen chemistry'."""
    col = key_col if key_col in ds.meta else "standardized_smiles"
    g = ds.meta[col].astype(str).to_numpy()
    tr_keys = set(g[train_idx])
    if len(test_idx) == 0:
        return float("nan")
    return float(np.mean([g[j] in tr_keys for j in test_idx]))


SPLIT_FNS = {
    "random": random_split,
    "scaffold": scaffold_split,
    "assay": assay_split,
    "document": document_split,
    "source": source_split,
    "threshold": threshold_split,
    "assay_scaffold": assay_scaffold_split,
}


def make_splits(ds, kinds=("random", "scaffold", "assay", "document", "source", "threshold"),
                k: int = 5, seed: int = 0):
    out = {}
    for kind in kinds:
        fn = SPLIT_FNS[kind]
        try:
            if kind in ("source",):
                folds = fn(ds)
            elif kind == "threshold":
                folds = fn(ds, seed=seed)
            else:
                folds = fn(ds, k=k, seed=seed)
        except KeyError:
            folds = []
        if folds:
            out[kind] = folds
    return out
