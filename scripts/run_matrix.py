"""Full controlled-experiment orchestrator (plan §10-18).

Runs the treatment x backbone x split x fold x seed x endpoint matrix in
parallel across CPU cores and writes one tidy results parquet. Designed to be
resumable: each cell writes a small json under results/cells/ keyed by a
deterministic tag; a completed tag is skipped on re-run.

Usage:
  python run_matrix.py --endpoints hERG_IC50_B CYP3A4_IC50_A ... \
      --granularity endpoint --splits scaffold random --seeds 0 1 2 \
      --eps 0.02 0.05 0.10 0.20 --epochs 150 --workers 24 --outdir results/matrix

Endpoints are the CensorADMET-v1 curated endpoints (granularity=endpoint) or
target_key:standard_type[:assay_type] triples for measurement-level runs
(granularity=measurement), which unlock assay-aware treatments + assay/document/
source splits.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import numpy as np
import pandas as pd

TREATMENTS_BASE = ["exact_only", "tobit", "aft_conc"]
TREATMENTS_SATIS = ["satisficing", "ensemble_satisficing"]  # per-eps
# measurement-only, per-eps: lookup random effect + amortized description transfer
TREATMENTS_ASSAY = ["satisficing_assay", "satisficing_transfer"]


def _cell_tag(spec):
    return "__".join(str(spec[k]) for k in
                     ["endpoint", "split_kind", "fold", "treatment", "eps", "seed", "dist"]).replace("/", "-")


def _load_ds(endpoint, granularity, feature_set, n_bits):
    from data import load_admet_endpoint, load_endpoint, load_measurement_endpoint
    if granularity == "endpoint":
        return load_endpoint(endpoint, feature_set=feature_set, n_bits=n_bits)
    if granularity == "admet":
        # broadened ADMET property endpoints curated by curate_admet.py
        return load_admet_endpoint(endpoint, feature_set=feature_set, n_bits=n_bits)
    parts = endpoint.split(":")
    tk, st = parts[0], parts[1]
    at = parts[2] if len(parts) > 2 else None
    return load_measurement_endpoint(tk, st, assay_type=at, feature_set=feature_set, n_bits=n_bits)


def _run_one(spec, outdir):
    """Worker: load ds, build the requested split fold, run the cell, dump json."""
    from experiment import run_cell
    from splits import make_splits
    tag = _cell_tag(spec)
    cell_path = Path(outdir) / "cells" / f"{tag}.json"
    if cell_path.exists():
        # a previously-errored cell is retryable (e.g. a transient cache race);
        # a successful cell is cached and skipped.
        try:
            prev = json.loads(cell_path.read_text())
            if "error" not in prev:
                return tag, "cached"
        except Exception:
            pass  # unreadable -> recompute
    try:
        ds = _load_ds(spec["endpoint"], spec["granularity"], spec["feature_set"], spec["n_bits"])
        splits = make_splits(ds, kinds=(spec["split_kind"],), k=spec["k"], seed=spec["split_seed"])
        folds = splits.get(spec["split_kind"], [])
        if spec["fold"] >= len(folds):
            return tag, "no_fold"
        name, tr, te = folds[spec["fold"]]
        anchor_cache = str(Path(outdir) / "anchor_cache")
        r = run_cell(ds, name, tr, te, spec["treatment"], seed=spec["seed"],
                     eps=spec["eps"], tau=spec["tau"], epochs=spec["epochs"],
                     dist_name=spec["dist"], n_jobs=spec["cell_jobs"],
                     ensemble_k=spec["ensemble_k"], anchor_cache=anchor_cache)
        r["split_kind"] = spec["split_kind"]
        r["fold"] = spec["fold"]
        r["feature_set"] = spec["feature_set"]
        cell_path.parent.mkdir(parents=True, exist_ok=True)
        cell_path.write_text(json.dumps(r, default=float))
        return tag, ("error" if "error" in r else "ok")
    except Exception as e:
        cell_path.parent.mkdir(parents=True, exist_ok=True)
        cell_path.write_text(json.dumps({**spec, "error": f"{type(e).__name__}: {e}",
                                         "traceback": traceback.format_exc()[-1000:]}, default=float))
        return tag, "exception"


def build_specs(args):
    specs = []
    # ensemble is expensive (k models per cell); by default run it only at the
    # single operating-point eps rather than sweeping the full budget grid.
    ens_eps = args.ensemble_eps if args.ensemble_eps else args.eps
    for endpoint in args.endpoints:
        for split_kind in args.splits:
            for fold in range(args.k):
                for seed in args.seeds:
                    # eps-free treatments
                    for tr in TREATMENTS_BASE:
                        specs.append(_spec(args, endpoint, split_kind, fold, tr, 0.0, seed))
                    # satisficing sweeps the full eps grid (the Pareto frontier)
                    for eps in args.eps:
                        specs.append(_spec(args, endpoint, split_kind, fold, "satisficing", eps, seed))
                    # ensemble at its (smaller) eps set
                    for eps in ens_eps:
                        specs.append(_spec(args, endpoint, split_kind, fold, "ensemble_satisficing", eps, seed))
                    if args.granularity in ("measurement", "admet"):
                        for tr in TREATMENTS_ASSAY:
                            for eps in args.eps:
                                specs.append(_spec(args, endpoint, split_kind, fold, tr, eps, seed))
                    # weighted-Tobit baseline: sweep the censored-row weight (passed
                    # via the eps field, reused as the weight w in experiment.py).
                    for w in args.weighted_tobit_weights:
                        specs.append(_spec(args, endpoint, split_kind, fold, "weighted_tobit", w, seed))
    return specs


def _spec(args, endpoint, split_kind, fold, treatment, eps, seed):
    return dict(endpoint=endpoint, granularity=args.granularity, split_kind=split_kind,
                fold=fold, treatment=treatment, eps=float(eps), seed=int(seed),
                tau=args.tau, epochs=args.epochs, dist=args.dist, k=args.k,
                split_seed=args.split_seed, feature_set=args.feature_set, n_bits=args.n_bits,
                cell_jobs=args.cell_jobs, ensemble_k=args.ensemble_k)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--endpoints", nargs="+", required=True)
    ap.add_argument("--granularity", choices=["endpoint", "measurement", "admet"], default="endpoint")
    ap.add_argument("--splits", nargs="+", default=["scaffold", "random"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--eps", nargs="+", type=float, default=[0.02, 0.05, 0.10, 0.20])
    ap.add_argument("--ensemble-eps", nargs="+", type=float, default=[0.05],
                    help="eps values for the (expensive) ensemble treatment; default single point")
    ap.add_argument("--weighted-tobit-weights", nargs="+", type=float, default=[],
                    help="censored-row weights for the weighted-Tobit baseline sweep (empty = skip)")
    ap.add_argument("--tau", type=float, default=0.85)
    ap.add_argument("--dist", default="gaussian")
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--split-seed", type=int, default=0)
    ap.add_argument("--feature-set", default="morgan")
    ap.add_argument("--n-bits", type=int, default=2048)
    ap.add_argument("--ensemble-k", type=int, default=5)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--cell-jobs", type=int, default=2)
    ap.add_argument("--outdir", default="results/matrix")
    args = ap.parse_args()

    Path(args.outdir).mkdir(parents=True, exist_ok=True)
    specs = build_specs(args)
    print(f"[matrix] {len(specs)} cells, {args.workers} workers", flush=True)
    t0 = time.time()
    done = {"ok": 0, "error": 0, "cached": 0, "exception": 0, "no_fold": 0}
    # cap intra-cell threads so workers don't oversubscribe
    os.environ.setdefault("OMP_NUM_THREADS", str(args.cell_jobs))
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(_run_one, s, args.outdir): s for s in specs}
        for i, fut in enumerate(as_completed(futs), 1):
            tag, status = fut.result()
            done[status] = done.get(status, 0) + 1
            if i % 20 == 0 or i == len(specs):
                el = time.time() - t0
                print(f"[matrix] {i}/{len(specs)} ({el:.0f}s) {done}", flush=True)

    # collate
    rows = []
    for p in (Path(args.outdir) / "cells").glob("*.json"):
        try:
            rows.append(json.loads(p.read_text()))
        except Exception:
            pass
    df = pd.DataFrame(rows)
    out_parquet = Path(args.outdir) / "all_results.parquet"
    df.to_parquet(out_parquet)
    print(f"[matrix] wrote {len(df)} rows -> {out_parquet} in {time.time()-t0:.0f}s", flush=True)
    print(f"[matrix] status: {done}", flush=True)


if __name__ == "__main__":
    main()
