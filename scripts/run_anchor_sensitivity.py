"""Focused anchor-ablation experiment for the frozen measurement/random setup.

Runs the primary satisficing objective with the minimal-deviation anchor removed
(nu=0) on the eight primary endpoints.  It retains the original fixed random
splits, seeds, model and epsilon so that its aggregate can be compared directly
with the released weighted-Tobit operating curve.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from data import load_measurement_endpoint
from experiment import run_cell
from splits import make_splits


def _run_one(spec: dict) -> dict:
    ds = load_measurement_endpoint(*spec["endpoint"].split(":"), feature_set="morgan", n_bits=2048)
    folds = make_splits(ds, kinds=("random",), k=5, seed=0)["random"]
    name, train_idx, test_idx = folds[spec["fold"]]
    result = run_cell(
        ds, name, train_idx, test_idx, "satisficing", seed=spec["seed"],
        eps=spec["eps"], tau=0.85, epochs=150, anchor_weight=0.0,
    )
    result.update(split_kind="random", fold=spec["fold"], feature_set="morgan", anchor_weight=0.0)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--endpoints", nargs="+", required=True)
    parser.add_argument("--eps", type=float, default=0.02)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1])
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--outdir", default="results/anchor_sensitivity")
    args = parser.parse_args()

    specs = [
        {"endpoint": endpoint, "fold": fold, "seed": seed, "eps": args.eps}
        for endpoint in args.endpoints
        for fold in range(5)
        for seed in args.seeds
    ]
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    rows = []
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(_run_one, spec) for spec in specs]
        for index, future in enumerate(as_completed(futures), 1):
            row = future.result()
            rows.append(row)
            print(f"[anchor-sensitivity] {index}/{len(specs)}", flush=True)

    frame = pd.DataFrame(rows).sort_values(["endpoint", "fold", "seed"])
    output = outdir / "anchor_free_satisficing_random.csv"
    frame.to_csv(output, index=False)
    print(f"[anchor-sensitivity] wrote {len(frame)} cells to {output}", flush=True)


if __name__ == "__main__":
    main()
