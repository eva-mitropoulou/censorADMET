"""Audit calibration-row counts for frozen conformal result rows without fitting models."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src" / "censoradmet"))

from data import load_measurement_endpoint
from splits import make_splits


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", default="results/conformal_scored")
    parser.add_argument("--output", default="results/synthesis/conformal_sample_audit.csv")
    args = parser.parse_args()
    result_rows = pd.concat(
        [pd.read_csv(path) for path in sorted(Path(args.input_dir).glob("conformal_*.csv"))],
        ignore_index=True,
    )
    datasets, folds_by_endpoint, audit = {}, {}, []
    for row in result_rows.itertuples(index=False):
        if row.endpoint not in datasets:
            target, standard_type = row.endpoint.split(":", 1)
            datasets[row.endpoint] = load_measurement_endpoint(target, standard_type, feature_set="morgan", n_bits=2048)
            folds_by_endpoint[row.endpoint] = make_splits(
                datasets[row.endpoint], kinds=("random", "scaffold", "assay"), k=5, seed=0
            )
        dataset = datasets[row.endpoint]
        _, train_idx, test_idx = folds_by_endpoint[row.endpoint][row.split][int(row.fold)]
        calibration_idx = np.random.default_rng(int(row.seed)).permutation(train_idx)[:max(50, int(0.2 * len(train_idx)))]
        n_test_exact = int(dataset.exact_mask[test_idx].sum())
        if n_test_exact != int(row.n_test_exact):
            raise RuntimeError(f"Test exact-row mismatch for {row.endpoint}/{row.split}/{row.fold}/{row.seed}")
        audit.append({
            "endpoint": row.endpoint, "split": row.split, "fold": int(row.fold), "seed": int(row.seed),
            "n_train": int(len(train_idx)), "n_calibration": int(len(calibration_idx)),
            "n_calibration_exact": int(dataset.exact_mask[calibration_idx].sum()),
            "n_test_exact": n_test_exact,
        })
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(audit).sort_values(["endpoint", "split", "fold", "seed"]).to_csv(output, index=False)


if __name__ == "__main__":
    main()
