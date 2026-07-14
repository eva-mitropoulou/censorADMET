"""Constraint-feasibility diagnostics.

For each (endpoint, split, eps) the satisficing trainer records a per-epoch
history of the full-data per-direction deficits G_k and dual multipliers lambda_k.
This script trains the satisficing model and dumps, at convergence: the target
eps, the achieved G_left/G_right/G_interval, whether each is feasible (G_k <= eps),
the final lambda_k, and the achieved test one-sided violation rate. This
demonstrates the augmented-Lagrangian trainer actually enforces the stated
constraint rather than merely producing a useful regularisation path.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src" / "censoradmet"))

import numpy as np
import pandas as pd

from data import load_measurement_endpoint
from heads import HeteroscedasticMLP
from metrics import interval_violation_rate
from satisficing_losses import LatentDistribution
from satisficing_trainer import SatisficingTrainer, TrainConfig, default_direction_constraints
from splits import make_splits


def run(endpoints, splits, epsilons, epochs, seed, outdir):
    dist = LatentDistribution("gaussian")
    rows = []
    for ep in endpoints:
        tk, st = ep.split(":")[0], ep.split(":")[1]
        ds = load_measurement_endpoint(tk, st, feature_set="morgan", n_bits=2048)
        y = ds.meta["value_p"].to_numpy(float)
        allsp = make_splits(ds, kinds=tuple(splits), k=5, seed=0)
        for sk, folds in allsp.items():
            name, tr, te = folds[0]
            for eps in epsilons:
                specs = default_direction_constraints(ds.lower[tr], ds.upper[tr], ds.exact_mask[tr], eps=eps)
                cfg = TrainConfig(tau=0.85, nu=1.0, epochs=epochs, batch_size=1024, seed=seed)
                m = SatisficingTrainer(HeteroscedasticMLP(ds.X.shape[1], hidden=(256, 128)), dist, cfg)
                m.fit(ds.X[tr], ds.lower[tr], ds.upper[tr], ds.exact_mask[tr], specs)
                h = m.history[-1] if m.history else {"G": {}, "lambda": {}}
                mu, _ = m.predict(ds.X[te])
                viol = interval_violation_rate(mu, ds.lower[te], ds.upper[te], ds.exact_mask[te])
                row = dict(endpoint=ep, split=sk, eps=eps,
                           G_right=h["G"].get("right"), G_left=h["G"].get("left"),
                           G_interval=h["G"].get("interval"),
                           lam_right=h["lambda"].get("right"), lam_left=h["lambda"].get("left"),
                           feasible=all((v is None) or (v <= eps + 1e-3) for v in h["G"].values()),
                           test_violation=viol.get("violation_rate"), rho=h.get("rho"))
                rows.append(row)
                print(f"[feas {ep} {sk} eps={eps}] G={h['G']} lam={ {k: round(v,2) for k,v in h['lambda'].items()} } "
                      f"feasible={row['feasible']} testviol={row['test_violation']:.3f}", flush=True)
    out = Path(outdir); out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out / "feasibility.csv", index=False)
    print(f"[feas] wrote {len(rows)} rows", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--endpoints", nargs="+", default=["hERG:IC50", "CYP3A4:IC50", "CYP2C9:IC50"])
    ap.add_argument("--splits", nargs="+", default=["random", "assay"])
    ap.add_argument("--eps", nargs="+", type=float, default=[0.02, 0.05, 0.10, 0.20])
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--outdir", default="results/feasibility")
    args = ap.parse_args()
    run(args.endpoints, args.splits, args.eps, args.epochs, args.seed, args.outdir)


if __name__ == "__main__":
    main()
