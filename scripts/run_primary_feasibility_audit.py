"""All-primary-run feasibility audit for satisficing at epsilon=0.05."""
from __future__ import annotations

import argparse
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from data import load_measurement_endpoint
from experiment import run_cell
from splits import make_splits


def _run_one(spec: dict, anchor_cache: str) -> dict:
    ds = load_measurement_endpoint(*spec["endpoint"].split(":"), feature_set="morgan", n_bits=2048)
    name, train_idx, test_idx = make_splits(ds, kinds=(spec["split_kind"],), k=5, seed=0)[spec["split_kind"]][spec["fold"]]
    result = run_cell(
        ds, name, train_idx, test_idx, "satisficing", seed=spec["seed"], eps=0.05,
        tau=0.85, epochs=150, anchor_cache=anchor_cache, anchor_weight=1.0,
        return_constraint_diagnostics=True,
    )
    result.update(split_kind=spec["split_kind"], fold=spec["fold"], feature_set="morgan")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--endpoints", nargs="+", required=True)
    parser.add_argument("--splits", nargs="+", default=["random", "scaffold", "assay", "document", "source"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1])
    parser.add_argument("--workers", type=int, default=64)
    parser.add_argument("--outdir", default="results/primary_feasibility_audit")
    args = parser.parse_args()

    # Grouped splits can legitimately yield fewer than k valid folds after the
    # prespecified minimum-size and leakage rules. Enumerate the exact released
    # split assignments rather than assuming five folds for every combination.
    specs = []
    for endpoint in args.endpoints:
        ds = load_measurement_endpoint(*endpoint.split(":"), feature_set="morgan", n_bits=2048)
        available = make_splits(ds, kinds=tuple(args.splits), k=5, seed=0)
        for split_kind in args.splits:
            for fold, _ in enumerate(available.get(split_kind, [])):
                for seed in args.seeds:
                    specs.append({
                        "endpoint": endpoint,
                        "split_kind": split_kind,
                        "fold": fold,
                        "seed": seed,
                    })
    if not specs:
        raise RuntimeError("No valid primary split assignments were generated.")
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    rows = []
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(_run_one, spec, str(outdir / "anchor_cache")) for spec in specs]
        for number, future in enumerate(as_completed(futures), 1):
            rows.append(future.result())
            if number % 20 == 0 or number == len(futures):
                print(f"[primary-feasibility] {number}/{len(futures)}", flush=True)

    frame = pd.DataFrame(rows).sort_values(["endpoint", "split_kind", "fold", "seed"])
    output = outdir / "primary_feasibility_audit.csv"
    frame.to_csv(output, index=False)
    print(f"[primary-feasibility] wrote {len(frame)} cells to {output}", flush=True)


if __name__ == "__main__":
    main()
