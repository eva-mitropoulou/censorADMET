"""Prediction heads for CensorADMET 2.0.

A heteroscedastic regressor that outputs a latent-value location mu(x[,a]) and a
log-scale log_sigma(x[,a]). Optionally assay-aware (plan §4): a per-assay random
intercept b_a on the mean and a per-assay log-scale offset r_a, both drawn toward
zero by a Gaussian shrinkage prior (hierarchical / empirical-Bayes). When no
assay ids are supplied it degrades to a plain heteroscedastic MLP.

Design notes:
  * log_sigma is clamped to [min_log_sigma, max_log_sigma] to avoid the classic
    heteroscedastic collapse (sigma -> 0 on easy rows dominating the loss) and to
    keep tail probabilities finite.
  * assay intercepts/offsets live in nn.Embedding tables indexed by a contiguous
    assay index; unknown/held-out assays map to index 0 which is pinned to 0 (a
    fresh assay contributes no offset, i.e. we predict the population mean).
"""
from __future__ import annotations

import torch
import torch.nn as nn


class HeteroscedasticMLP(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden=(256, 128),
        n_assays: int = 0,
        assay_embed_dim: int = 0,
        min_log_sigma: float = -4.0,
        max_log_sigma: float = 3.0,
        homoscedastic: bool = False,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.min_log_sigma = float(min_log_sigma)
        self.max_log_sigma = float(max_log_sigma)
        self.homoscedastic = bool(homoscedastic)
        self.n_assays = int(n_assays)

        layers = []
        d = in_dim
        for h in hidden:
            layers += [nn.Linear(d, h), nn.ReLU()]
            if dropout > 0:
                layers += [nn.Dropout(dropout)]
            d = h
        self.body = nn.Sequential(*layers)
        self.mu_head = nn.Linear(d, 1)
        if homoscedastic:
            # single global log-sigma parameter
            self.log_sigma_param = nn.Parameter(torch.zeros(1))
        else:
            self.logs_head = nn.Linear(d, 1)

        # assay-aware random effects (index 0 pinned to zero for unknown assays)
        if self.n_assays > 0:
            self.b_assay = nn.Embedding(self.n_assays, 1)
            self.r_assay = nn.Embedding(self.n_assays, 1)
            nn.init.zeros_(self.b_assay.weight)
            nn.init.zeros_(self.r_assay.weight)

    def forward(self, x, assay_idx=None):
        h = self.body(x)
        mu = self.mu_head(h).squeeze(-1)
        if self.homoscedastic:
            log_sigma = self.log_sigma_param.expand(mu.shape[0])
        else:
            log_sigma = self.logs_head(h).squeeze(-1)

        if self.n_assays > 0 and assay_idx is not None:
            b = self.b_assay(assay_idx).squeeze(-1)
            r = self.r_assay(assay_idx).squeeze(-1)
            # pin index 0 (unknown / held-out assay) to zero effect
            known = (assay_idx != 0).to(mu.dtype)
            mu = mu + b * known
            log_sigma = log_sigma + r * known

        log_sigma = torch.clamp(log_sigma, self.min_log_sigma, self.max_log_sigma)
        sigma = torch.exp(log_sigma)
        return mu, sigma

    def assay_prior_penalty(self, tau_b: float = 1.0, tau_r: float = 1.0):
        """Gaussian shrinkage prior on the random effects: 0.5*(b^2/tau_b^2 + r^2/tau_r^2)
        summed over assays (excluding the pinned index 0). Returns a scalar tensor."""
        if self.n_assays == 0:
            return torch.zeros((), device=self.mu_head.weight.device)
        b = self.b_assay.weight[1:]
        r = self.r_assay.weight[1:]
        return 0.5 * ((b * b).sum() / (tau_b ** 2) + (r * r).sum() / (tau_r ** 2))


class AssayTransferMLP(nn.Module):
    """Heteroscedastic MLP whose per-assay offset is an AMORTIZED function of the
    assay's embedding, so it transfers to UNSEEN assays (the key idea for the
    source- and assay-held-out splits, where a lookup-table random effect can only
    fall back to zero).

    The mean offset for a measurement from assay a is
        b_a = g_b(z_a) + b_a^res ,
    where z_a is the assay's embedding (a FIXED per-row input built from the assay
    description + aggregated context, see assay_context/AssayEmbedder), g_b is a
    small learned MLP shared across assays, and b_a^res is an OPTIONAL shrunk
    per-assay residual effect for assays SEEN in training (index 0 = unseen -> 0
    residual). At test time an unseen assay has no residual but still receives the
    amortized g_b(z_a): if its description resembles training assays, it gets an
    informed, nonzero offset. The log-scale offset is modelled the same way.

    Setting use_residual=False gives a pure amortized (fully transferable) model;
    use_residual=True is the hybrid that also captures seen-assay idiosyncrasy.
    """

    def __init__(
        self,
        in_dim: int,
        assay_embed_dim: int,
        hidden=(256, 128),
        offset_hidden=(64,),
        n_assays: int = 0,
        use_residual: bool = True,
        min_log_sigma: float = -4.0,
        max_log_sigma: float = 3.0,
        homoscedastic: bool = False,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.min_log_sigma = float(min_log_sigma)
        self.max_log_sigma = float(max_log_sigma)
        self.homoscedastic = bool(homoscedastic)
        self.assay_embed_dim = int(assay_embed_dim)
        self.n_assays = int(n_assays)
        self.use_residual = bool(use_residual) and self.n_assays > 0

        layers = []
        d = in_dim
        for h in hidden:
            layers += [nn.Linear(d, h), nn.ReLU()]
            if dropout > 0:
                layers += [nn.Dropout(dropout)]
            d = h
        self.body = nn.Sequential(*layers)
        self.mu_head = nn.Linear(d, 1)
        if homoscedastic:
            self.log_sigma_param = nn.Parameter(torch.zeros(1))
        else:
            self.logs_head = nn.Linear(d, 1)

        # amortized offset networks g_b (mean) and g_r (log-scale) over z_a
        def _mlp(out_last_zero=True):
            ls, dd = [], self.assay_embed_dim
            for h in offset_hidden:
                ls += [nn.Linear(dd, h), nn.ReLU()]
                dd = h
            last = nn.Linear(dd, 1)
            if out_last_zero:
                nn.init.zeros_(last.weight); nn.init.zeros_(last.bias)
            ls += [last]
            return nn.Sequential(*ls)

        self.g_b = _mlp()
        self.g_r = _mlp()

        if self.use_residual:
            self.b_res = nn.Embedding(self.n_assays, 1)
            self.r_res = nn.Embedding(self.n_assays, 1)
            nn.init.zeros_(self.b_res.weight)
            nn.init.zeros_(self.r_res.weight)

    def forward(self, x, assay_embed=None, assay_idx=None):
        h = self.body(x)
        mu = self.mu_head(h).squeeze(-1)
        if self.homoscedastic:
            log_sigma = self.log_sigma_param.expand(mu.shape[0])
        else:
            log_sigma = self.logs_head(h).squeeze(-1)

        if assay_embed is not None:
            # amortized offsets from the assay embedding (transfers to unseen assays)
            mu = mu + self.g_b(assay_embed).squeeze(-1)
            log_sigma = log_sigma + self.g_r(assay_embed).squeeze(-1)

        if self.use_residual and assay_idx is not None:
            known = (assay_idx != 0).to(mu.dtype)
            mu = mu + self.b_res(assay_idx).squeeze(-1) * known
            log_sigma = log_sigma + self.r_res(assay_idx).squeeze(-1) * known

        log_sigma = torch.clamp(log_sigma, self.min_log_sigma, self.max_log_sigma)
        return mu, torch.exp(log_sigma)

    def assay_prior_penalty(self, tau_b: float = 1.0, tau_r: float = 1.0):
        """Shrinkage prior on the per-assay RESIDUAL effects only (the amortized
        g_b/g_r are regularized by weight decay in the optimizer). Zero when no
        residual is used."""
        if not self.use_residual:
            return torch.zeros((), device=self.mu_head.weight.device)
        b = self.b_res.weight[1:]
        r = self.r_res.weight[1:]
        return 0.5 * ((b * b).sum() / (tau_b ** 2) + (r * r).sum() / (tau_r ** 2))
