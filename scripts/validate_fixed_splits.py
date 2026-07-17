"""Validate the fixed split archives without model fitting."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split-dir", type=Path, default=Path("data/fixed_splits"))
    args = parser.parse_args()
    archives = sorted(args.split_dir.rglob("*.npz"))
    if not archives:
        raise SystemExit(f"no split archives under {args.split_dir}")
    checked_folds = 0
    for path in archives:
        with np.load(path, allow_pickle=False) as data:
            n = len(data["row_ids"])
            folds = sorted({key.rsplit("_", 1)[0] for key in data.files if key.endswith("_train")})
            for fold in folds:
                train, test = data[f"{fold}_train"], data[f"{fold}_test"]
                if train.dtype.kind not in "iu" or test.dtype.kind not in "iu":
                    raise AssertionError(f"{path}: non-integer indices")
                if train.size == 0 or test.size == 0:
                    raise AssertionError(f"{path}: empty {fold}")
                if train.min() < 0 or test.min() < 0 or train.max() >= n or test.max() >= n:
                    raise AssertionError(f"{path}: out-of-range {fold}")
                if np.intersect1d(train, test).size:
                    raise AssertionError(f"{path}: train/test leakage in {fold}")
                checked_folds += 1
    print(f"OK {len(archives)} archives; {checked_folds} non-overlapping train/test folds")


if __name__ == "__main__":
    main()
