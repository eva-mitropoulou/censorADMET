"""Fair weighted-Tobit sensitivity: identical minimal-deviation anchor.

Evaluates anchored weighted Tobit on the primary measurement/random grid using
the same endpoint set, fixed folds, seeds, architecture and anchor weight as
the satisficing operating curve. The censored-row likelihood weight is swept
over the frozen weighted-Tobit values.
"""
from __future__ import annotations

import argparse
import json
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
    ds = load_measurement_endpoint(*spec["endpoint"].split(":"), feature_set="morgan", n_bits=2048)
    name, train_idx, test_idx = make_splits(ds, kinds=("random",), k=5, seed=0)["random"][spec["fold"]]
    result = run_cell(
        ds, name, train_idx, test_idx, "weighted_tobit_anchor", seed=spec["seed"],
        eps=spec["weight"], tau=0.85, epochs=150, anchor_cache=anchor_cache,
        anchor_weight=1.0,
    )
    result.update(split_kind="random", fold=spec["fold"], feature_set="morgan", anchor_weight=1.0,
                  censored_weight=spec["weight"])
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--endpoints", nargs="+", required=True)
    parser.add_argument("--weights", nargs="+", type=float, default=[0.1, 0.25, 0.5, 0.75, 1.0])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1])
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--outdir", default="results/anchored_weighted_tobit")
    parser.add_argument(
        "--anchor-cache",
        default=None,
        help="Optional existing exact-only anchor cache shared with another sweep.",
    )
    args = parser.parse_args()

    specs = [
        {"endpoint": endpoint, "fold": fold, "seed": seed, "weight": weight}
        for endpoint in args.endpoints
        for fold in range(5)
        for seed in args.seeds
        for weight in args.weights
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
                print(f"[anchored-weighted-tobit] {number}/{len(futures)}", flush=True)

    frame = pd.DataFrame(rows).sort_values(["endpoint", "fold", "seed", "censored_weight"])
    output = outdir / "anchored_weighted_tobit_random.csv"
    frame.to_csv(output, index=False)
    print(f"[anchored-weighted-tobit] wrote {len(frame)} cells to {output}", flush=True)


if __name__ == "__main__":
    main()
