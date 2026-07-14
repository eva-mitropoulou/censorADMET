"""Hidden-truth recovery study (plan §7-8).

For each endpoint we build a semi-synthetic benchmark: exact rows are the latent
truth, then we re-censor a copy using the endpoint's REAL censoring mechanism
(learned thresholds + direction mix). We train each treatment on the re-censored
labels and score how well it recovers the KNOWN truth on the re-censored rows --
the quantity that is unobservable on real data. We do this under both the natural
(ignorable-ish) mechanism and an informative (MNAR) mechanism.

Writes results/hidden_truth/hidden_<endpoint>.csv with per-treatment hidden-truth
MAE / bias / coverage on the re-censored rows.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src" / "censoradmet"))

import numpy as np
import pandas as pd

from baselines import MLPTobit, XGBoostAFTConcentration
from data import load_endpoint
from heads import HeteroscedasticMLP
from hidden_truth import build_hidden_truth, hidden_truth_recovery, learn_censoring_mechanism
from metrics import interval_coverage
from satisficing_losses import LatentDistribution
from satisficing_trainer import SatisficingTrainer, TrainConfig, default_direction_constraints


def _train_predict(treatment, X, lo, hi, ex, tr, te, dist, cfg, eps):
    if treatment == "exact_only":
        m = SatisficingTrainer(HeteroscedasticMLP(X.shape[1], hidden=(256, 128)), dist,
                               TrainConfig(**{**cfg, "nu": 0.0}))
        m.fit(X[tr], lo[tr], hi[tr], ex[tr], constraints=[])
        return m.predict(X[te])
    if treatment == "tobit":
        m = MLPTobit(lambda: HeteroscedasticMLP(X.shape[1], hidden=(256, 128)),
                     dist, epochs=cfg["epochs"], seed=cfg["seed"])
        m.fit(X[tr], lo[tr], hi[tr], ex[tr])
        return m.predict_dist(X[te])
    if treatment == "aft_conc":
        rng = np.random.default_rng(cfg["seed"]); p = rng.permutation(tr)
        nv = max(50, int(0.15 * len(p))); va, fit = p[:nv], p[nv:]
        m = XGBoostAFTConcentration(seed=cfg["seed"], n_jobs=4)
        m.fit(X[fit], lo[fit], hi[fit], X[va], lo[va], hi[va])
        return m.predict_dist(X[te])
    # satisficing
    specs = default_direction_constraints(lo[tr], hi[tr], ex[tr], eps=eps)
    m = SatisficingTrainer(HeteroscedasticMLP(X.shape[1], hidden=(256, 128)), dist, TrainConfig(**cfg))
    m.fit(X[tr], lo[tr], hi[tr], ex[tr], specs)
    return m.predict(X[te])


def run(endpoint, seeds, epochs, eps, outdir):
    ds = load_endpoint(endpoint, feature_set="morgan", n_bits=2048)
    dist = LatentDistribution("gaussian")
    mech = learn_censoring_mechanism(ds.lower, ds.upper, ds.exact_mask, ds.censoring_class)
    y_all = ds.meta["value_p"].to_numpy(float)
    exact_idx = np.where(ds.exact_mask & np.isfinite(y_all))[0]
    y_star_full = y_all.copy()
    rows = []
    treatments = ["exact_only", "tobit", "aft_conc", "satisficing"]
    for mode in ["natural", "informative"]:
        for seed in seeds:
            rng = np.random.default_rng(seed)
            # re-censor ONLY the exact rows (their truth is known)
            ys = y_star_full[exact_idx]
            L, U, exm, rec = build_hidden_truth(
                ys, mech, seed=seed, informative=(mode == "informative"), mnar_strength=1.5)
            # assemble a dataset over the exact rows with re-censored labels
            Xe = ds.X[exact_idx]
            # train/test split (random) on these rows
            perm = rng.permutation(len(exact_idx))
            cut = int(0.8 * len(perm)); tr, te = np.sort(perm[:cut]), np.sort(perm[cut:])
            cfg = dict(tau=0.85, nu=1.0, epochs=epochs, batch_size=1024, seed=seed)
            for t in treatments:
                pred = _train_predict(t, Xe, L, U, exm, tr, te, dist, cfg, eps)
                mu = pred[0] if isinstance(pred, tuple) else pred
                # hidden-truth recovery on the re-censored TEST rows
                rec_te = rec[te]
                hr = hidden_truth_recovery(mu, ys[te], rec_te)
                # also observed-exact accuracy (rows that stayed exact in test)
                obs_ex = exm[te]
                obs_mae = float(np.mean(np.abs(mu[obs_ex] - ys[te][obs_ex]))) if obs_ex.sum() > 3 else np.nan
                rows.append(dict(endpoint=endpoint, mode=mode, seed=seed, treatment=t,
                                 **hr, obs_exact_mae=obs_mae,
                                 n_recensored_test=int(rec_te.sum())))
                print(f"[{endpoint} {mode} s{seed} {t:12s}] hidden_mae="
                      f"{hr.get('hidden_mae'):.3f} bias={hr.get('hidden_bias'):.3f} "
                      f"n_hidden={hr.get('n_hidden')}", flush=True)
    out = Path(outdir); out.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(out / f"hidden_{endpoint}.csv", index=False)
    print(f"[hidden] wrote {len(df)} rows for {endpoint}", flush=True)
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--endpoints", nargs="+", required=True)
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--eps", type=float, default=0.05)
    ap.add_argument("--outdir", default="results/hidden_truth")
    args = ap.parse_args()
    for ep in args.endpoints:
        try:
            run(ep, args.seeds, args.epochs, args.eps, args.outdir)
        except Exception as e:
            import traceback
            print(f"[hidden] {ep} FAILED: {e}\n{traceback.format_exc()[-500:]}", flush=True)


if __name__ == "__main__":
    main()
