"""Export the deterministic fold assignments used by the manuscript package.

This utility performs no fitting.  It writes one compressed NumPy archive per
endpoint and split family, containing the row order, train indices, and test
indices returned by :func:`make_splits` at the configured split seed.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "censoradmet"))

from data import load_admet_endpoint, load_endpoint, load_measurement_endpoint  # noqa: E402
from splits import make_splits  # noqa: E402


def _safe_name(value: str) -> str:
    return value.replace(":", "_").replace("/", "_")


def _row_ids(dataset):
    for key in ("row_id", "activity_id"):
        if key in dataset.meta:
            return np.asarray(dataset.meta[key].astype(str).to_numpy(), dtype=str)
    return np.asarray(np.arange(dataset.n, dtype=np.int64).astype(str), dtype=str)


def _write(dataset, kinds, folds, split_seed, output_dir: Path) -> int:
    produced = 0
    assignments = make_splits(dataset, kinds=tuple(kinds), k=folds, seed=split_seed)
    for kind, split_folds in assignments.items():
        payload = {"row_ids": _row_ids(dataset)}
        for fold, (_, train, test) in enumerate(split_folds):
            payload[f"fold_{fold}_train"] = np.asarray(train, dtype=np.int64)
            payload[f"fold_{fold}_test"] = np.asarray(test, dtype=np.int64)
        dest = output_dir / dataset.granularity / _safe_name(dataset.endpoint_id)
        dest.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(dest / f"{kind}.npz", **payload)
        produced += 1
    return produced


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "manuscript-v1.0.0.json")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data" / "fixed_splits")
    args = parser.parse_args()
    cfg = json.loads(args.config.read_text())

    count = 0
    for endpoint in cfg["aggregated_endpoints"]:
        dataset = load_endpoint(endpoint, feature_set=cfg["feature_set"], n_bits=cfg["morgan_bits"])
        count += _write(dataset, cfg["aggregated_splits"], cfg["folds"], cfg["split_seed"], args.output_dir)
    for spec in cfg["measurement_endpoints"]:
        parts = spec.split(":")
        dataset = load_measurement_endpoint(*parts, feature_set=cfg["feature_set"], n_bits=cfg["morgan_bits"])
        count += _write(dataset, cfg["measurement_splits"], cfg["folds"], cfg["split_seed"], args.output_dir)
    for endpoint in cfg.get("broadened_property_endpoints", []):
        dataset = load_admet_endpoint(endpoint, feature_set=cfg["feature_set"], n_bits=cfg["morgan_bits"])
        count += _write(dataset, cfg["broadened_property_splits"], cfg["folds"], cfg["split_seed"], args.output_dir)
    print(f"wrote {count} fixed split archives to {args.output_dir}")


if __name__ == "__main__":
    main()
