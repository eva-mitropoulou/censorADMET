import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from heads import AssayTransferMLP  # noqa: E402
from satisficing_losses import LatentDistribution  # noqa: E402
from satisficing_trainer import (  # noqa: E402
    SatisficingTrainer,
    TrainConfig,
    default_direction_constraints,
)

G = LatentDistribution("gaussian")


def test_forward_shapes_and_unseen_offset():
    torch.manual_seed(0)
    m = AssayTransferMLP(in_dim=16, assay_embed_dim=8, hidden=(32,), n_assays=5)
    x = torch.randn(10, 16)
    ae = torch.randn(10, 8)
    ai = torch.randint(0, 5, (10,))
    mu, sigma = m(x, assay_embed=ae, assay_idx=ai)
    assert mu.shape == (10,) and sigma.shape == (10,)
    assert torch.isfinite(mu).all() and (sigma > 0).all()
    # with a trained g_b (nonzero), an UNSEEN assay (idx 0, no residual) still gets
    # an embedding-driven offset -> mu differs from the no-embedding prediction.
    with torch.no_grad():
        m.g_b[-1].weight.fill_(0.5); m.g_b[-1].bias.fill_(0.1)
    mu_e, _ = m(x, assay_embed=ae, assay_idx=torch.zeros(10, dtype=torch.long))
    mu_ne, _ = m(x, assay_embed=None, assay_idx=torch.zeros(10, dtype=torch.long))
    assert not torch.allclose(mu_e, mu_ne), "amortized offset must apply to unseen assays"


def test_pure_amortized_has_no_residual():
    m = AssayTransferMLP(in_dim=8, assay_embed_dim=4, n_assays=5, use_residual=False)
    assert not hasattr(m, "b_res")
    assert float(m.assay_prior_penalty()) == 0.0


def _make_assay_shift_data(seed=0, n=2400, d=16, n_train_assays=8, n_test_assays=4):
    """Latent y depends on chemistry AND an assay offset that is a function of the
    assay's embedding. Test assays are DISJOINT from train assays but their offset
    follows the SAME embedding->offset law, so a transfer model can recover it and
    a lookup model cannot."""
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, d)).astype(np.float32)
    w = rng.standard_normal(d) * 0.3
    # assay embedding space (4-dim); offset = linear function of embedding
    emb_dim = 4
    beta = rng.standard_normal(emb_dim)
    n_assays_total = n_train_assays + n_test_assays
    assay_emb_table = rng.standard_normal((n_assays_total, emb_dim)).astype(np.float32)
    assay_offset = assay_emb_table @ beta  # the law
    # assign rows: first 80% train assays, last 20% test assays
    a = np.empty(n, dtype=int)
    cut = int(0.8 * n)
    a[:cut] = rng.integers(0, n_train_assays, cut)
    a[cut:] = rng.integers(n_train_assays, n_assays_total, n - cut)
    y = X @ w + 6.0 + assay_offset[a] + 0.2 * rng.standard_normal(n)
    ae = assay_emb_table[a]
    # all exact for a clean recovery test
    lo = y.astype(np.float32); hi = y.astype(np.float32); ex = np.ones(n, bool)
    tr = np.arange(cut); te = np.arange(cut, n)
    return X, y, lo, hi, ex, ae, a, tr, te, n_train_assays


def test_transfer_beats_lookup_on_unseen_assays():
    from heads import HeteroscedasticMLP
    X, y, lo, hi, ex, ae, a, tr, te, n_train_assays = _make_assay_shift_data(seed=1)
    # contiguous train-assay vocab: train assays 1..n_train, test assays -> 0 (unseen)
    idx = np.where(a < n_train_assays, a + 1, 0).astype(np.int64)

    cfg = TrainConfig(tau=0.85, nu=0.0, epochs=80, batch_size=256, lr=3e-3, seed=0,
                      weight_decay=1e-5)
    specs = default_direction_constraints(lo[tr], hi[tr], ex[tr], eps=0.5)

    # lookup random-effect model (cannot transfer to unseen assays)
    torch.manual_seed(0)
    look = HeteroscedasticMLP(X.shape[1], hidden=(64,), n_assays=n_train_assays + 1)
    tl = SatisficingTrainer(look, G, cfg).fit(X[tr], lo[tr], hi[tr], ex[tr], specs,
                                              assay_idx=idx[tr])
    mu_look, _ = tl.predict(X[te], assay_idx=idx[te])
    mae_look = np.mean(np.abs(mu_look - y[te]))

    # transfer model (amortized offset from embedding, no residual so it's pure transfer)
    torch.manual_seed(0)
    trm = AssayTransferMLP(X.shape[1], assay_embed_dim=ae.shape[1], hidden=(64,),
                           n_assays=n_train_assays + 1, use_residual=False)
    tt = SatisficingTrainer(trm, G, cfg).fit(X[tr], lo[tr], hi[tr], ex[tr], specs,
                                             assay_idx=idx[tr], assay_embed=ae[tr])
    mu_tr, _ = tt.predict(X[te], assay_idx=idx[te], assay_embed=ae[te])
    mae_tr = np.mean(np.abs(mu_tr - y[te]))

    # the transfer model should recover the unseen-assay offset far better
    assert mae_tr < mae_look - 0.1, f"transfer {mae_tr:.3f} not better than lookup {mae_look:.3f}"
