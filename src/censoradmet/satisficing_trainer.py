"""Augmented-Lagrangian constraint-satisficing trainer (CensorADMET 2.0 core).

We solve, for a required interval-probability level tau and violation budgets eps_k:

    minimize_theta   f(theta)  =  (1/|E|) sum_{i in exact} exact_nll_i
                                   + nu * (1/n) sum_i ( mu_theta(x_i) - mu0(x_i) )^2
    subject to       G_k(theta) <= eps_k   for each constraint k

where G_k aggregates the per-row satisficing deficit d_i = [tau - q_i]_+^2 over a
subset of censored rows. The default constraint set is one budget per censoring
direction (left / right / interval); optional group budgets (per assay-type,
endpoint, ...) and CVaR-alpha tail budgets (plan §16, distributionally-robust)
are supported through ConstraintSpec.

Optimization: Powell-Hestenes-Rockafellar augmented Lagrangian for inequalities.
For g_k(theta) = G_k(theta) - eps_k <= 0 the smooth penalty is

    P(theta; lambda, rho) = (1/2rho) sum_k [ max(0, lambda_k + rho * g_k)^2 - lambda_k^2 ]

whose theta-gradient is  sum_k max(0, lambda_k + rho * g_k) * grad g_k  (so a
constraint that is satisfied with lambda_k = 0 contributes exactly zero gradient
-- the satisficing property carries all the way to the outer loop). Primal steps
are Adam on f + P over minibatches; after each epoch we recompute the full-data
G_k and take the dual step  lambda_k <- max(0, lambda_k + rho * g_k), increasing
rho when feasibility stalls. This yields, per budget eps, one Pareto operating
point (accuracy vs. violation); sweeping eps traces the frontier (plan §2.4).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np
import torch

from satisficing_losses import (
    LatentDistribution,
    exact_nll,
    satisficing_deficit,
)


# --------------------------------------------------------------------------- #
# Constraint specification                                                    #
# --------------------------------------------------------------------------- #
@dataclass
class ConstraintSpec:
    """A single satisficing constraint  G(theta) <= eps.

    name     : identifier (e.g. "right", "left", "interval", "grp:assay=B").
    row_mask : boolean tensor over training rows selecting the rows this
               constraint governs (evaluated once, stored).
    eps      : violation budget (upper bound on the aggregated deficit).
    reduction: "mean" (average deficit over the masked rows) or
               "cvar" (mean of the worst-alpha tail of the deficits;
               distributionally robust, plan §16).
    alpha    : tail fraction for CVaR (ignored for mean).
    """

    name: str
    row_mask: torch.Tensor
    eps: float
    reduction: str = "mean"
    alpha: float = 0.1


def _cvar_eta(deficits: torch.Tensor, alpha: float) -> float:
    """Rockafellar-Uryasev threshold eta* = empirical (1-alpha) quantile of the
    deficits. Computed on the FULL masked set and frozen for the epoch (see the
    linearized penalty in SatisficingTrainer.fit)."""
    if deficits.numel() == 0:
        return 0.0
    alpha = float(min(max(alpha, 1e-6), 1.0))
    return float(torch.quantile(deficits.detach(), 1.0 - alpha))


def _cvar_from_eta(deficits: torch.Tensor, alpha: float, eta: float) -> torch.Tensor:
    """CVaR_alpha(d) = eta + (1/alpha) E[(d-eta)_+] evaluated at a FIXED eta.
    With eta frozen at the full-data quantile, this is (a) an unbiased estimator
    of the full-data CVaR value when averaged over minibatches and (b) has the
    correct Danskin gradient (1/alpha)*E[1{d>eta} grad d]. Reduces to mean(d) as
    alpha->1."""
    if deficits.numel() == 0:
        return deficits.new_zeros(())
    alpha = float(min(max(alpha, 1e-6), 1.0))
    return eta + torch.clamp(deficits - eta, min=0.0).mean() / alpha


def _cvar(deficits: torch.Tensor, alpha: float) -> torch.Tensor:
    """Self-contained CVaR_alpha (computes its own RU threshold). Convenience for
    reporting/tests; the trainer uses the frozen-eta split (_cvar_eta +
    _cvar_from_eta) to keep the minibatch gradient unbiased."""
    return _cvar_from_eta(deficits, alpha, _cvar_eta(deficits, alpha))


def _reduce(deficits: torch.Tensor, spec: ConstraintSpec) -> torch.Tensor:
    """Full-data reduction (self-contained eta for CVaR). Used for reporting the
    achieved constraint value and for the dual/coefficient pass."""
    if deficits.numel() == 0:
        return deficits.new_zeros(())
    if spec.reduction == "cvar":
        return _cvar_from_eta(deficits, spec.alpha, _cvar_eta(deficits, spec.alpha))
    return deficits.mean()


def _reduce_batch(deficits: torch.Tensor, spec: ConstraintSpec, eta: float) -> torch.Tensor:
    """Differentiable MINIBATCH surrogate with the CVaR threshold frozen to the
    full-data eta. For 'mean' this is just the batch mean (an unbiased estimator
    of the full-data mean and its gradient); for 'cvar' it is the RU surrogate at
    the frozen eta (unbiased Danskin gradient). This keeps the primal gradient
    consistent with the full-data augmented-Lagrangian gradient (no Jensen bias
    from squaring a noisy minibatch constraint estimate)."""
    if deficits.numel() == 0:
        return deficits.new_zeros(())
    if spec.reduction == "cvar":
        return _cvar_from_eta(deficits, spec.alpha, eta)
    return deficits.mean()


# --------------------------------------------------------------------------- #
# Config                                                                       #
# --------------------------------------------------------------------------- #
@dataclass
class TrainConfig:
    tau: float = 0.8
    nu: float = 1.0                 # anchor weight
    lr: float = 1e-3
    weight_decay: float = 1e-5
    epochs: int = 200
    batch_size: int = 4096
    rho_init: float = 5.0           # augmented-Lagrangian penalty coefficient
    rho_max: float = 100.0          # kept modest: the DUAL variable lam does the
                                    # heavy lifting; a huge rho makes the frozen
                                    # linearized coefficient stale within an epoch
                                    # and destabilizes loose-budget runs.
    rho_growth: float = 1.5         # gentle growth when feasibility stalls
    feas_stall_tol: float = 0.5     # grow rho only if new_viol > tol*old_viol + feas_tol
    feas_tol: float = 1e-3          # a constraint with g_k <= feas_tol is "feasible"
    dual_every: int = 3             # epochs between dual updates (let primal settle)
    assay_tau_b: float = 1.0        # shrinkage sd for assay mean effects
    assay_tau_r: float = 1.0        # shrinkage sd for assay log-scale effects
    assay_prior_weight: float = 1.0
    grad_clip: float = 5.0
    device: str = "cpu"
    seed: int = 0
    verbose: bool = False
    log_every: int = 25


# --------------------------------------------------------------------------- #
# Trainer                                                                       #
# --------------------------------------------------------------------------- #
class SatisficingTrainer:
    def __init__(
        self,
        model: torch.nn.Module,
        dist: LatentDistribution,
        config: TrainConfig,
    ):
        self.model = model
        self.dist = dist
        self.cfg = config
        self.device = torch.device(config.device)
        self.model.to(self.device)
        self.history: list[dict] = []

    def _split_batches(self, n, generator):
        perm = torch.randperm(n, generator=generator)
        bs = self.cfg.batch_size
        return [perm[i : i + bs] for i in range(0, n, bs)]

    @staticmethod
    def _to_t(a, dtype, device):
        if isinstance(a, torch.Tensor):
            return a.to(device=device, dtype=dtype)
        return torch.as_tensor(np.asarray(a), dtype=dtype, device=device)

    def _call_model(self, xb, ai=None, ae=None):
        """Route arguments to the model by its type. AssayTransferMLP takes a
        per-row assay EMBEDDING (and optional residual index); HeteroscedasticMLP
        takes just the assay index."""
        from heads import AssayTransferMLP
        if isinstance(self.model, AssayTransferMLP):
            return self.model(xb, assay_embed=ae, assay_idx=ai)
        return self.model(xb, ai)

    def fit(
        self,
        X,
        lower,
        upper,
        exact_mask,
        constraints: list[ConstraintSpec],
        anchor_mu=None,
        assay_idx=None,
        assay_embed=None,
    ):
        """Fit under the given constraints.

        X          : (n, d) features.
        lower/upper : (n,) interval bounds in p-scale (+/- inf allowed).
        exact_mask  : (n,) bool, True where lower==upper (exact observation).
        constraints : list of ConstraintSpec over the n rows.
        anchor_mu   : (n,) reference-model predictions mu0 (None -> no anchor).
        assay_idx   : (n,) long assay indices (0 = unknown), or None.
        assay_embed : (n, d_e) per-row assay embeddings for AssayTransferMLP, or None.
        """
        cfg = self.cfg
        dev = self.device
        f32 = torch.float32
        gen = torch.Generator().manual_seed(cfg.seed)
        torch.manual_seed(cfg.seed)

        X = self._to_t(X, f32, dev)
        lower = self._to_t(lower, f32, dev)
        upper = self._to_t(upper, f32, dev)
        exact_mask = self._to_t(exact_mask, torch.bool, dev)
        n = X.shape[0]
        if anchor_mu is not None:
            anchor_mu = self._to_t(anchor_mu, f32, dev)
        if assay_idx is not None:
            assay_idx = self._to_t(assay_idx, torch.long, dev)
        if assay_embed is not None:
            assay_embed = self._to_t(assay_embed, f32, dev)
        self._fit_assay_embed = assay_embed  # cache for _full_constraints

        cons_masks = [self._to_t(c.row_mask, torch.bool, dev) for c in constraints]

        lam = torch.zeros(len(constraints), device=dev)
        rho = float(cfg.rho_init)
        opt = torch.optim.Adam(
            self.model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
        )

        prev_total_viol = None
        # initial full-data constraint pass (used to freeze the epoch-0 penalty
        # coefficient / CVaR threshold). See _reduce_batch for why we freeze.
        Gfull, eta_full = self._full_constraints(X, lower, upper, exact_mask, assay_idx,
                                                 constraints, cons_masks)

        for epoch in range(cfg.epochs):
            self.model.train()
            # Refresh the full-data constraint values EACH epoch so the frozen
            # penalty coefficient stays responsive (one cheap forward pass). The
            # DUAL variable lam is updated less often (every dual_every epochs)
            # to let the primal settle between multiplier steps.
            if epoch > 0:
                Gfull, eta_full = self._full_constraints(X, lower, upper, exact_mask,
                                                         assay_idx, constraints, cons_masks)
            # LINEARIZED augmented-Lagrangian: freeze the penalty coefficient
            #   t_k = max(0, lam_k + rho*(G_k_full - eps_k))
            # and the CVaR threshold eta_k from the full-data pass; the minibatch
            # then contributes the LINEAR term t_k * G_k_batch, whose gradient
            # t_k * grad G_k_batch is an unbiased estimate of the true full-data
            # AL gradient (no Jensen bias from squaring a noisy batch estimate).
            t_coef = [max(0.0, float(lam[k] + rho * (Gfull[k] - c.eps)))
                      for k, c in enumerate(constraints)]

            for idx in self._split_batches(n, gen):
                opt.zero_grad()
                xb = X[idx]
                ai = assay_idx[idx] if assay_idx is not None else None
                ae = assay_embed[idx] if assay_embed is not None else None
                mu, sigma = self._call_model(xb, ai, ae)

                lo_b, hi_b, ex_b = lower[idx], upper[idx], exact_mask[idx]

                # ---- accuracy objective on exact rows ----
                if ex_b.any():
                    f_exact = exact_nll(mu[ex_b], sigma[ex_b], lo_b[ex_b], self.dist).mean()
                else:
                    f_exact = mu.new_zeros(())

                # ---- minimal-deviation anchor ----
                if anchor_mu is not None:
                    f_anchor = ((mu - anchor_mu[idx]) ** 2).mean()
                else:
                    f_anchor = mu.new_zeros(())

                # ---- assay shrinkage prior ----
                if hasattr(self.model, "assay_prior_penalty"):
                    prior = self.model.assay_prior_penalty(cfg.assay_tau_b, cfg.assay_tau_r)
                    # scale prior by batch fraction so its weight is epoch-invariant
                    prior = prior * (xb.shape[0] / n)
                else:
                    prior = mu.new_zeros(())

                # ---- per-row deficits for this batch ----
                d_all = satisficing_deficit(mu, sigma, lo_b, hi_b, self.dist, cfg.tau,
                                            exact_mask=ex_b)

                # ---- linearized augmented-Lagrangian penalty ----
                penalty = mu.new_zeros(())
                for k, spec in enumerate(constraints):
                    if t_coef[k] == 0.0:
                        continue  # inactive constraint: exactly zero gradient
                    m = cons_masks[k][idx]
                    if not bool(m.any()):
                        continue
                    Gk_batch = _reduce_batch(d_all[m], spec, eta_full[k])
                    penalty = penalty + t_coef[k] * Gk_batch

                loss = (
                    f_exact
                    + cfg.nu * f_anchor
                    + cfg.assay_prior_weight * prior
                    + penalty
                )
                # A minibatch can contribute NO gradient path when it has no exact
                # rows, no anchor, no assay prior, and every constraint is inactive
                # (t_coef==0). This happens on small/skewed folds (e.g. a held-out
                # source with few exact rows). Skip the step rather than calling
                # backward() on a grad-less constant (which raises "element 0 ...
                # does not require grad").
                if not loss.requires_grad:
                    continue
                loss.backward()
                if cfg.grad_clip:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), cfg.grad_clip)
                opt.step()

            # ---- dual update (periodic) using the FRESH full-data G_k ----
            # Gfull was just recomputed at the top of this epoch; reuse it (the
            # primal has taken one epoch of steps under the current coefficient).
            if (epoch + 1) % cfg.dual_every == 0:
                total_viol = 0.0
                any_infeasible = False
                for k, spec in enumerate(constraints):
                    gk = Gfull[k] - spec.eps
                    lam[k] = max(0.0, float(lam[k] + rho * gk))
                    total_viol += max(0.0, gk)
                    if gk > cfg.feas_tol:
                        any_infeasible = True
                # grow rho ONLY if some constraint is still meaningfully infeasible
                # AND aggregate violation is not shrinking (additive floor guards
                # against runaway growth once feasible). Never grow when feasible.
                if (any_infeasible and prev_total_viol is not None
                        and total_viol > cfg.feas_stall_tol * prev_total_viol + cfg.feas_tol):
                    rho = min(cfg.rho_max, rho * cfg.rho_growth)
                prev_total_viol = total_viol

                if cfg.verbose and ((epoch + 1) % cfg.log_every == 0 or epoch == 0):
                    gstr = " ".join(
                        f"{c.name}={Gfull[k]:.3f}" for k, c in enumerate(constraints)
                    )
                    print(f"[ep {epoch+1}] viol_sum={total_viol:.4f} rho={rho:.1f} {gstr}")
                self.history.append(
                    {"epoch": epoch + 1, "rho": rho, "total_viol": total_viol,
                     "G": {c.name: float(Gfull[k]) for k, c in enumerate(constraints)},
                     "lambda": {c.name: float(lam[k]) for k, c in enumerate(constraints)}}
                )
        # Always record a final snapshot, so history is non-empty even if
        # epochs < dual_every (consumers do history[-1]).
        if not self.history or self.history[-1]["epoch"] != cfg.epochs:
            total_viol = sum(max(0.0, Gfull[k] - c.eps) for k, c in enumerate(constraints))
            self.history.append(
                {"epoch": cfg.epochs, "rho": rho, "total_viol": total_viol,
                 "G": {c.name: float(Gfull[k]) for k, c in enumerate(constraints)},
                 "lambda": {c.name: float(lam[k]) for k, c in enumerate(constraints)}}
            )
        self._lambda = lam.detach().cpu().numpy()
        self._rho = rho
        return self

    @torch.no_grad()
    def _full_constraints(self, X, lower, upper, exact_mask, assay_idx,
                          constraints, cons_masks):
        """Return (G_list, eta_list): full-data constraint values and, for CVaR
        constraints, the frozen RU threshold eta (0.0 for mean constraints)."""
        self.model.eval()
        mu, sigma = self._predict_dist(X, assay_idx, getattr(self, "_fit_assay_embed", None))
        d_all = satisficing_deficit(mu, sigma, lower, upper, self.dist, self.cfg.tau,
                                    exact_mask=exact_mask)
        etas = []
        for k, spec in enumerate(constraints):
            m = cons_masks[k]
            if spec.reduction == "cvar" and bool(m.any()):
                etas.append(_cvar_eta(d_all[m], spec.alpha))
            else:
                etas.append(0.0)
        out = []
        for k, spec in enumerate(constraints):
            m = cons_masks[k]
            out.append(_reduce(d_all[m], spec) if bool(m.any()) else torch.zeros((), device=mu.device))
        self.model.train()
        return [float(o) for o in out], etas

    @torch.no_grad()
    def _predict_dist(self, X, assay_idx=None, assay_embed=None):
        self.model.eval()
        X = self._to_t(X, torch.float32, self.device)
        ai = self._to_t(assay_idx, torch.long, self.device) if assay_idx is not None else None
        ae = self._to_t(assay_embed, torch.float32, self.device) if assay_embed is not None else None
        bs = self.cfg.batch_size
        mus, sigs = [], []
        for i in range(0, X.shape[0], bs):
            aib = ai[i : i + bs] if ai is not None else None
            aeb = ae[i : i + bs] if ae is not None else None
            m, s = self._call_model(X[i : i + bs], aib, aeb)
            mus.append(m)
            sigs.append(s)
        return torch.cat(mus), torch.cat(sigs)

    @torch.no_grad()
    def predict(self, X, assay_idx=None, assay_embed=None):
        """Return (mu, sigma) as numpy arrays. Unknown assays -> idx 0 (no residual);
        AssayTransferMLP still applies the amortized embedding offset."""
        mu, sigma = self._predict_dist(X, assay_idx, assay_embed)
        return mu.cpu().numpy(), sigma.cpu().numpy()


# --------------------------------------------------------------------------- #
# Convenience: default per-direction constraints                              #
# --------------------------------------------------------------------------- #
def default_direction_constraints(lower, upper, exact_mask, eps: float):
    """Build the three canonical per-direction budgets G_left, G_right, G_interval <= eps."""
    lower = torch.as_tensor(np.asarray(lower), dtype=torch.float32)
    upper = torch.as_tensor(np.asarray(upper), dtype=torch.float32)
    exact_mask = torch.as_tensor(np.asarray(exact_mask), dtype=torch.bool)
    right = torch.isfinite(lower) & torch.isposinf(upper)
    left = torch.isneginf(lower) & torch.isfinite(upper)
    interval = torch.isfinite(lower) & torch.isfinite(upper) & (~exact_mask)
    specs = []
    if bool(right.any()):
        specs.append(ConstraintSpec("right", right, eps))
    if bool(left.any()):
        specs.append(ConstraintSpec("left", left, eps))
    if bool(interval.any()):
        specs.append(ConstraintSpec("interval", interval, eps))
    return specs
