"""Focused D-MPNN backbone comparison (plan §10, backbone-agnosticism).

D-MPNN on CPU is expensive, so this is a FOCUSED study (not a full grid): a few
representative endpoints x {exact_only, tobit, satisficing@eps} x {scaffold,
random} splits x a small seed set, using the learned-representation backbone. The
point is to show the SAME accuracy-vs-violation trade-off appears with a graph
model as with ECFP-MLP -- i.e. the effect is a property of the objective, not the
fingerprint. Writes results/dmpnn/dmpnn_<endpoint>.csv.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src" / "censoradmet"))

import numpy as np
import pandas as pd

from data import load_endpoint
from dmpnn import DMPNNConfig, DMPNNSatisficingTrainer, _featurize_graphs
from metrics import accuracy_metrics, interval_violation_rate
from satisficing_losses import LatentDistribution
from splits import make_splits


def run(endpoint, splits, seeds, epochs, eps_list, outdir):
    ds = load_endpoint(endpoint, feature_set="morgan", n_bits=2048)
    dist = LatentDistribution("gaussian")
    y = ds.meta["value_p"].to_numpy(float)
    smiles = ds.meta["standardized_smiles"].tolist()
    graphs = _featurize_graphs(smiles)
    ok = np.array([g is not None for g in graphs])
    if not ok.all():
        print(f"[dmpnn] {endpoint}: dropping {int((~ok).sum())} unparseable", flush=True)
    graphs = [g for g, k in zip(graphs, ok) if k]
    lo, hi, ex = ds.lower[ok], ds.upper[ok], ds.exact_mask[ok]
    yv = y[ok]

    class _V:  # lightweight ds view for make_splits (needs .meta, .exact_mask, .n)
        pass
    v = _V(); v.meta = ds.meta.iloc[ok].reset_index(drop=True); v.exact_mask = ex; v.n = len(graphs)
    all_splits = make_splits(v, kinds=tuple(splits), k=5, seed=0)

    rows = []
    for sk, folds in all_splits.items():
        name, tr, te = folds[0]  # fold 0 only (focused)
        gtr = [graphs[i] for i in tr]; gte = [graphs[i] for i in te]
        for seed in seeds:
            # exact-only anchor
            acfg = DMPNNConfig(epochs=max(30, epochs // 2), batch_size=256, seed=seed, nu=0.0)
            aref = DMPNNSatisficingTrainer(dist, acfg, hidden=200, depth=3).fit(
                gtr, lo[tr], hi[tr], ex[tr], eps=1.0)  # eps=1 -> effectively unconstrained
            anchor = aref.predict(gtr)[0]

            for treatment, eps in ([("exact_only", 1.0), ("tobit", 0.0)]
                                   + [("satisficing", e) for e in eps_list]):
                cfg = DMPNNConfig(epochs=epochs, batch_size=256, seed=seed,
                                  nu=(0.0 if treatment in ("exact_only", "tobit") else 1.0))
                m = DMPNNSatisficingTrainer(dist, cfg, hidden=200, depth=3)
                anc = None if treatment in ("exact_only", "tobit") else anchor
                # tobit ~ satisficing with eps->0; exact_only ~ eps->inf (no constraint)
                use_eps = 1e-6 if treatment == "tobit" else eps
                m.fit(gtr, lo[tr], hi[tr], ex[tr], eps=use_eps, anchor_mu=anc)
                mu, sg = m.predict(gte)
                acc = accuracy_metrics(mu, yv[te], ex[te])
                viol = interval_violation_rate(mu, lo[te], hi[te], ex[te])
                rows.append(dict(endpoint=endpoint, backbone="dmpnn", split=sk, seed=seed,
                                 treatment=treatment, eps=eps,
                                 mae=acc["mae"], rmse=acc["rmse"], spearman=acc["spearman"],
                                 violation=viol["violation_rate"]))
                print(f"[dmpnn {endpoint} {sk} s{seed} {treatment}@{eps}] "
                      f"MAE={acc['mae']:.3f} viol={viol['violation_rate']:.3f}", flush=True)
    out = Path(outdir); out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out / f"dmpnn_{endpoint}.csv", index=False)
    print(f"[dmpnn] wrote {len(rows)} rows for {endpoint}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--endpoints", nargs="+", required=True)
    ap.add_argument("--splits", nargs="+", default=["scaffold", "random"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1])
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--eps", nargs="+", type=float, default=[0.02, 0.05, 0.10, 0.20])
    ap.add_argument("--outdir", default="results/dmpnn")
    args = ap.parse_args()
    for ep in args.endpoints:
        try:
            run(ep, args.splits, args.seeds, args.epochs, args.eps, args.outdir)
        except Exception as e:
            import traceback
            print(f"[dmpnn] {ep} FAILED: {e}\n{traceback.format_exc()[-600:]}", flush=True)


if __name__ == "__main__":
    main()
