import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

pytest.importorskip("chemprop")

from dmpnn import (  # noqa: E402
    DMPNNConfig,
    DMPNNSatisficingTrainer,
    _featurize_graphs,
)
from satisficing_losses import LatentDistribution  # noqa: E402

G = LatentDistribution("gaussian")


def _toy_smiles(n=200, seed=0):
    # a small set of real, parseable drug-like SMILES repeated with variety
    base = ["CCO", "c1ccccc1", "CC(=O)Oc1ccccc1C(=O)O", "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",
            "CC(C)Cc1ccc(cc1)C(C)C(=O)O", "OC(=O)c1ccccc1O", "Cn1cnc2c1c(=O)[nH]c(=O)n2C",
            "c1ccc2ccccc2c1", "CCN(CC)CC", "CC(=O)Nc1ccc(O)cc1"]
    rng = np.random.default_rng(seed)
    return [base[i % len(base)] for i in range(n)], rng


def test_featurize_graphs():
    graphs = _featurize_graphs(["CCO", "c1ccccc1", "not_a_smiles"])
    assert graphs[0] is not None and graphs[1] is not None
    assert graphs[2] is None


def test_dmpnn_trains_and_recovers_signal():
    smiles, rng = _toy_smiles(240, seed=1)
    graphs = _featurize_graphs(smiles)
    ok = [g is not None for g in graphs]
    graphs = [g for g, k in zip(graphs, ok) if k]
    # target correlated with molecule identity (index mod 10)
    idx = np.array([i % 10 for i in range(len(smiles))])[np.array(ok)]
    y = 5.0 + 0.3 * idx + 0.1 * rng.standard_normal(len(graphs))
    lo = y.copy(); hi = y.copy(); ex = np.ones(len(graphs), bool)
    cfg = DMPNNConfig(epochs=40, batch_size=64, lr=1e-3, seed=0)
    tr = DMPNNSatisficingTrainer(G, cfg, hidden=64, depth=2).fit(graphs, lo, hi, ex, eps=0.05)
    mu, sg = tr.predict(graphs)
    assert np.isfinite(mu).all() and np.isfinite(sg).all()
    from scipy.stats import spearmanr
    rho = spearmanr(mu, y).correlation
    assert rho > 0.5, f"D-MPNN failed to fit signal (rho={rho})"


def test_dmpnn_constraint_reduces_violation():
    smiles, rng = _toy_smiles(240, seed=2)
    graphs = _featurize_graphs(smiles)
    graphs = [g for g in graphs if g is not None]
    n = len(graphs)
    idx = np.array([i % 10 for i in range(n)])
    y = 5.0 + 0.3 * idx + 0.2 * rng.standard_normal(n)
    # right-censor the top 30%
    c = np.quantile(y, 0.7); rc = y > c
    lo = np.where(rc, c, y).astype(np.float32); hi = np.where(rc, np.inf, y).astype(np.float32)
    ex = ~rc
    cfg = DMPNNConfig(epochs=50, batch_size=64, lr=1e-3, seed=0)
    tr = DMPNNSatisficingTrainer(G, cfg, hidden=64, depth=2).fit(graphs, lo, hi, ex, eps=0.05)
    mu, sg = tr.predict(graphs)
    # censored (right) rows should mostly predict above their lower bound
    viol = np.mean(mu[rc] < lo[rc])
    assert np.isfinite(mu).all()
    assert viol < 0.6, f"constraint not reducing violation (viol={viol})"
