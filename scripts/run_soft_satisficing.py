"""Run the anchored same-deficit soft-penalty comparator on fixed splits."""
from __future__ import annotations

import argparse
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src" / "censoradmet"))

from data import load_measurement_endpoint
from experiment import run_cell
from splits import make_splits


def _run_one(spec: dict, anchor_cache: str) -> dict:
    dataset = load_measurement_endpoint(*spec["endpoint"].split(":"), feature_set="morgan", n_bits=2048)
    name, train_idx, test_idx = make_splits(dataset, kinds=("random",), k=5, seed=0)["random"][spec["fold"]]
    result = run_cell(dataset, name, train_idx, test_idx, "soft_satisficing", seed=spec["seed"],
                      eps=spec["lambda"], tau=0.85, epochs=150, anchor_cache=anchor_cache,
                      anchor_weight=1.0)
    result.update(split_kind="random", fold=spec["fold"], feature_set="morgan",
                  anchor_weight=1.0, penalty_lambda=spec["lambda"])
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--endpoints", nargs="+", required=True)
    parser.add_argument("--lambdas", nargs="+", type=float,
                        default=[0.1, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1])
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--outdir", default="results/soft_satisficing")
    parser.add_argument("--anchor-cache", default=None)
    args = parser.parse_args()
    specs = [
        {"endpoint": endpoint, "fold": fold, "seed": seed, "lambda": penalty_lambda}
        for endpoint in args.endpoints for fold in range(5) for seed in args.seeds
        for penalty_lambda in args.lambdas
    ]
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    anchor_cache = args.anchor_cache or str(outdir / "anchor_cache")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    rows = []
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(_run_one, spec, anchor_cache) for spec in specs]
        for number, future in enumerate(as_completed(futures), 1):
            rows.append(future.result())
            if number % 20 == 0 or number == len(futures):
                print(f"[soft-satisficing] {number}/{len(futures)}", flush=True)
    pd.DataFrame(rows).sort_values(["endpoint", "fold", "seed", "penalty_lambda"]).to_csv(
        outdir / "soft_satisficing_random.csv", index=False
    )


if __name__ == "__main__":
    main()
