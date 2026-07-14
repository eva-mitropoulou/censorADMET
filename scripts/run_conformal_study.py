"""Focused UQ study (plan §15): does split-conformal restore calibrated coverage
of the censored predictive intervals under distribution shift?

For each (measurement endpoint, split-kind, fold), we split train into fit (80%)
and calibration (20%), train the satisficing model on fit, then compare:
  - RAW Gaussian 90% interval coverage on test exact rows
  - SPLIT-conformal coverage (scalar, calibrated on calib exact rows)
  - MONDRIAN-conformal coverage (per censoring-direction group)
Coverage is scored ONLY on exact test rows. Writes results/conformal/study.csv.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import numpy as np
import pandas as pd

from conformal import MondrianConformal, SplitConformal
from data import load_measurement_endpoint
from heads import HeteroscedasticMLP
from metrics import interval_coverage
from satisficing_losses import LatentDistribution
from satisficing_trainer import SatisficingTrainer, TrainConfig, default_direction_constraints
from splits import make_splits


def _direction(lo, hi, ex):
    g = np.where(ex, "exact",
                 np.where(np.isposinf(hi), "right",
                          np.where(np.isneginf(lo), "left", "interval")))
    return g


def _interval_score(lo, hi, y, exact_mask, alpha=0.1):
    """Mean central-interval score on exact rows; lower is better."""
    lo = np.asarray(lo, float)
    hi = np.asarray(hi, float)
    y = np.asarray(y, float)
    exact = np.asarray(exact_mask, bool) & np.isfinite(y)
    if exact.sum() == 0:
        return np.nan
    yy = y[exact]
    ll = lo[exact]
    hh = hi[exact]
    return float(np.mean((hh - ll) + (2.0 / alpha) * np.maximum(ll - yy, 0.0)
                         + (2.0 / alpha) * np.maximum(yy - hh, 0.0)))


def run(endpoint, splits, seeds, epochs, eps, outdir):
    tk, st = endpoint.split(":")[0], endpoint.split(":")[1]
    ds = load_measurement_endpoint(tk, st, feature_set="morgan", n_bits=2048)
    dist = LatentDistribution("gaussian")
    y = ds.meta["value_p"].to_numpy(float)
    rows = []
    all_splits = make_splits(ds, kinds=tuple(splits), k=5, seed=0)
    for sk, folds in all_splits.items():
        for fi, (name, tr, te) in enumerate(folds):
            for seed in seeds:
                rng = np.random.default_rng(seed)
                perm = rng.permutation(tr)
                ncal = max(50, int(0.2 * len(perm)))
                calib, fit = perm[:ncal], perm[ncal:]
                specs = default_direction_constraints(ds.lower[fit], ds.upper[fit], ds.exact_mask[fit], eps=eps)
                cfg = TrainConfig(tau=0.85, nu=1.0, epochs=epochs, batch_size=1024, seed=seed)
                m = SatisficingTrainer(HeteroscedasticMLP(ds.X.shape[1], hidden=(256, 128)), dist, cfg)
                m.fit(ds.X[fit], ds.lower[fit], ds.upper[fit], ds.exact_mask[fit], specs)

                mu_c, sg_c = m.predict(ds.X[calib])
                mu_t, sg_t = m.predict(ds.X[te])

                raw = interval_coverage((mu_t, sg_t), y[te], ds.exact_mask[te], level=0.9)
                z90 = 1.6448536269514722
                raw_lo, raw_hi = mu_t - z90 * sg_t, mu_t + z90 * sg_t
                sc = SplitConformal(0.1).calibrate(mu_c, sg_c, y[calib], ds.exact_mask[calib])
                sc_cov = sc.coverage(mu_t, sg_t, y[te], ds.exact_mask[te])
                sc_lo, sc_hi = sc.interval(mu_t, sg_t)
                grp_c = _direction(ds.lower[calib], ds.upper[calib], ds.exact_mask[calib])
                grp_t = _direction(ds.lower[te], ds.upper[te], ds.exact_mask[te])
                mc = MondrianConformal(0.1).calibrate(mu_c, sg_c, y[calib], ds.exact_mask[calib], grp_c)
                mc_cov = mc.coverage(mu_t, sg_t, y[te], ds.exact_mask[te], grp_t)

                rows.append(dict(endpoint=endpoint, split=sk, fold=fi, seed=seed,
                                 raw_cov=raw["coverage"], raw_width=raw["width"],
                                 raw_interval_score=_interval_score(raw_lo, raw_hi, y[te], ds.exact_mask[te]),
                                 conformal_cov=sc_cov["coverage"], conformal_width=sc_cov["width"],
                                 conformal_interval_score=_interval_score(sc_lo, sc_hi, y[te], ds.exact_mask[te]),
                                 mondrian_cov=mc_cov["coverage"], mondrian_width=mc_cov["width"],
                                 n_test_exact=raw["n"]))
                print(f"[{endpoint} {sk} f{fi} s{seed}] raw={raw['coverage']:.3f} "
                      f"conf={sc_cov['coverage']:.3f} mond={mc_cov['coverage']:.3f}", flush=True)
    out = Path(outdir); out.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(out / f"conformal_{endpoint.replace(':', '_')}.csv", index=False)
    print(f"[conformal] wrote {len(df)} rows for {endpoint}", flush=True)
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--endpoints", nargs="+", required=True)
    ap.add_argument("--splits", nargs="+", default=["random", "scaffold", "assay"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1])
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--eps", type=float, default=0.05)
    ap.add_argument("--outdir", default="results/conformal")
    args = ap.parse_args()
    for ep in args.endpoints:
        run(ep, args.splits, args.seeds, args.epochs, args.eps, args.outdir)


if __name__ == "__main__":
    main()
