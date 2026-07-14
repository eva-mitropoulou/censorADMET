"""D-MPNN backbone for constraint-satisficing censored regression (plan §10).

A second architecture (a LEARNED molecular representation) to show the method is
backbone-agnostic, complementing the ECFP-MLP (fixed fingerprint + MLP) and
XGBoost-AFT (tree) backbones. We reuse chemprop 2.x's directed message-passing
(BondMessagePassing + mean aggregation) to produce a per-molecule embedding, then
attach the SAME heteroscedastic (mu, log_sigma) head and train with the SAME
validated satisficing losses (satisficing_losses.py) via a self-contained loop.

Design choice: this is a separate trainer rather than a plug-in to
SatisficingTrainer because a graph model consumes batched molecule graphs, not a
fixed feature matrix. The censored-likelihood math, however, is identical and
reused verbatim, so the two backbones differ ONLY in the representation -- exactly
the controlled comparison §10 asks for. The augmented-Lagrangian logic is the
linearized form (freeze full-data penalty coefficient per epoch), matching
SatisficingTrainer, so results are comparable.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn

from satisficing_losses import (
    LatentDistribution,
    exact_nll,
    satisficing_deficit,
)


def _featurize_graphs(smiles):
    """Return chemprop MolGraph objects (or None for unparseable) for a SMILES list."""
    from chemprop import featurizers
    from rdkit import Chem
    feat = featurizers.SimpleMoleculeMolGraphFeaturizer()
    graphs = []
    for smi in smiles:
        mol = Chem.MolFromSmiles(str(smi))
        if mol is None:
            graphs.append(None)
        else:
            graphs.append(feat(mol))
    return graphs


def _batch_molgraphs(molgraphs, device):
    """Collate a list of chemprop MolGraph into a device-placed BatchMolGraph."""
    from chemprop.data.collate import BatchMolGraph
    bmg = BatchMolGraph(molgraphs)
    bmg.to(device)
    return bmg


class DMPNNRegressor(nn.Module):
    def __init__(self, hidden=300, depth=3, ffn_hidden=256,
                 min_log_sigma=-4.0, max_log_sigma=3.0, homoscedastic=False):
        super().__init__()
        from chemprop import nn as cnn
        self.mp = cnn.BondMessagePassing(d_h=hidden, depth=depth)
        self.agg = cnn.MeanAggregation()
        d = self.mp.output_dim
        self.ffn = nn.Sequential(nn.Linear(d, ffn_hidden), nn.ReLU(), nn.Linear(ffn_hidden, ffn_hidden), nn.ReLU())
        self.mu_head = nn.Linear(ffn_hidden, 1)
        self.homoscedastic = homoscedastic
        if homoscedastic:
            self.log_sigma_param = nn.Parameter(torch.zeros(1))
        else:
            self.logs_head = nn.Linear(ffn_hidden, 1)
        self.min_log_sigma = float(min_log_sigma)
        self.max_log_sigma = float(max_log_sigma)

    def forward(self, bmg):
        H = self.mp(bmg)                          # per-atom hidden
        h = self.agg(H, bmg.batch)                # per-molecule embedding
        z = self.ffn(h)
        mu = self.mu_head(z).squeeze(-1)
        if self.homoscedastic:
            log_sigma = self.log_sigma_param.expand(mu.shape[0])
        else:
            log_sigma = self.logs_head(z).squeeze(-1)
        log_sigma = torch.clamp(log_sigma, self.min_log_sigma, self.max_log_sigma)
        return mu, torch.exp(log_sigma)


@dataclass
class DMPNNConfig:
    tau: float = 0.85
    nu: float = 1.0
    lr: float = 5e-4
    weight_decay: float = 0.0
    epochs: int = 60
    batch_size: int = 256
    rho: float = 5.0
    grad_clip: float = 5.0
    seed: int = 0
    device: str = "cpu"


class DMPNNSatisficingTrainer:
    """Self-contained D-MPNN trainer with the linearized satisficing penalty."""

    def __init__(self, dist: LatentDistribution, config: DMPNNConfig,
                 hidden=300, depth=3, homoscedastic=False):
        self.dist = dist
        self.cfg = config
        self.device = torch.device(config.device)
        self.model = DMPNNRegressor(hidden=hidden, depth=depth, homoscedastic=homoscedastic).to(self.device)

    def _batches(self, n, gen):
        perm = torch.randperm(n, generator=gen)
        bs = self.cfg.batch_size
        return [perm[i:i + bs].tolist() for i in range(0, n, bs)]

    def _make_bmg(self, graphs, idx):
        return _batch_molgraphs([graphs[i] for i in idx], self.device)

    def fit(self, graphs, lower, upper, exact_mask, eps=0.05, anchor_mu=None):
        """graphs: list of chemprop MolGraph (no None). lower/upper/exact_mask/anchor: arrays."""
        cfg = self.cfg
        gen = torch.Generator().manual_seed(cfg.seed)
        torch.manual_seed(cfg.seed)
        lo = torch.as_tensor(np.asarray(lower, np.float32), device=self.device)
        hi = torch.as_tensor(np.asarray(upper, np.float32), device=self.device)
        ex = torch.as_tensor(np.asarray(exact_mask, bool), device=self.device)
        anc = torch.as_tensor(np.asarray(anchor_mu, np.float32), device=self.device) if anchor_mu is not None else None
        n = len(graphs)
        opt = torch.optim.Adam(self.model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

        # constraint row masks (per direction), computed once
        right = torch.isfinite(lo) & torch.isposinf(hi) & (~ex)
        left = torch.isneginf(lo) & torch.isfinite(hi) & (~ex)
        interval = torch.isfinite(lo) & torch.isfinite(hi) & (~ex)
        masks = {"right": right, "left": left, "interval": interval}
        masks = {k: v for k, v in masks.items() if bool(v.any())}
        lam = {k: 0.0 for k in masks}

        for epoch in range(cfg.epochs):
            # freeze full-data penalty coefficient from a no-grad pass
            self.model.eval()
            with torch.no_grad():
                mu_f, sg_f = self._predict_all(graphs)
                d_full = satisficing_deficit(mu_f, sg_f, lo, hi, self.dist, cfg.tau, exact_mask=ex)
                t_coef = {}
                for k, m in masks.items():
                    G = float(d_full[m].mean())
                    t_coef[k] = max(0.0, lam[k] + cfg.rho * (G - eps))
            self.model.train()
            for idx in self._batches(n, gen):
                opt.zero_grad()
                bmg = self._make_bmg(graphs, idx)
                mu, sg = self.model(bmg)
                exb = ex[idx]
                f_exact = exact_nll(mu[exb], sg[exb], lo[idx][exb], self.dist).mean() if bool(exb.any()) else mu.new_zeros(())
                f_anchor = ((mu - anc[idx]) ** 2).mean() if anc is not None else mu.new_zeros(())
                d_b = satisficing_deficit(mu, sg, lo[idx], hi[idx], self.dist, cfg.tau, exact_mask=exb)
                penalty = mu.new_zeros(())
                idx_t = torch.as_tensor(idx, device=self.device)
                for k, m in masks.items():
                    if t_coef[k] == 0.0:
                        continue
                    mb = m[idx_t]
                    if bool(mb.any()):
                        penalty = penalty + t_coef[k] * d_b[mb].mean()
                loss = f_exact + cfg.nu * f_anchor + penalty
                if not loss.requires_grad:
                    continue  # grad-less batch (no exact rows, all constraints inactive)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), cfg.grad_clip)
                opt.step()
            # dual update on fresh full-data G
            self.model.eval()
            with torch.no_grad():
                mu_f, sg_f = self._predict_all(graphs)
                d_full = satisficing_deficit(mu_f, sg_f, lo, hi, self.dist, cfg.tau, exact_mask=ex)
                for k, m in masks.items():
                    lam[k] = max(0.0, lam[k] + cfg.rho * (float(d_full[m].mean()) - eps))
        return self

    @torch.no_grad()
    def _predict_all(self, graphs):
        self.model.eval()
        bs = self.cfg.batch_size
        mus, sgs = [], []
        for i in range(0, len(graphs), bs):
            idx = list(range(i, min(i + bs, len(graphs))))
            bmg = self._make_bmg(graphs, idx)
            mu, sg = self.model(bmg)
            mus.append(mu); sgs.append(sg)
        return torch.cat(mus), torch.cat(sgs)

    @torch.no_grad()
    def predict(self, graphs):
        mu, sg = self._predict_all(graphs)
        return mu.cpu().numpy(), sg.cpu().numpy()
