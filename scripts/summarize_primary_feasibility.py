"""Summarize the all-primary-run satisficing feasibility audit.

The manuscript treats endpoint/split pairs as analysis units. This script first
averages fold and seed replicates within each such unit, then reports the
constraint diagnostics at the prespecified epsilon=0.05 operating point.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        default="results/primary_feasibility_audit/primary_feasibility_audit.csv",
    )
    parser.add_argument(
        "--output-dir",
        default="results/synthesis",
    )
    args = parser.parse_args()

    frame = pd.read_csv(args.input)
    if "error" in frame and frame["error"].notna().any():
        raise RuntimeError("Feasibility audit contains failed cells.")

    g_columns = [column for column in ("constraint_G_left", "constraint_G_right") if column in frame]
    if not g_columns:
        raise RuntimeError("Feasibility diagnostics are absent from the audit output.")
    frame["max_constraint_excess"] = frame[g_columns].max(axis=1) - 0.05
    frame["strictly_feasible"] = frame["max_constraint_excess"] <= 1e-3

    numeric = [
        *g_columns,
        "max_constraint_excess",
        "strictly_feasible",
        "constraint_fraction_q_ge_tau",
        "constraint_mean_probability_shortfall",
        "constraint_mean_probability",
    ]
    per_unit = (
        frame.groupby(["endpoint", "split_kind"], as_index=False)[numeric]
        .mean()
        .sort_values(["split_kind", "endpoint"])
    )
    summary = (
        per_unit.groupby("split_kind", as_index=False)[numeric]
        .mean()
        .rename(columns={"strictly_feasible": "fraction_strictly_feasible"})
    )
    overall = pd.DataFrame(
        [{
            "split_kind": "all",
            **{column: per_unit[column].mean() for column in numeric if column != "strictly_feasible"},
            "fraction_strictly_feasible": per_unit["strictly_feasible"].mean(),
            "median_max_constraint_excess": per_unit["max_constraint_excess"].median(),
            "max_max_constraint_excess": per_unit["max_constraint_excess"].max(),
        }]
    )
    summary["median_max_constraint_excess"] = summary["split_kind"].map(
        per_unit.groupby("split_kind")["max_constraint_excess"].median()
    )
    summary["max_max_constraint_excess"] = summary["split_kind"].map(
        per_unit.groupby("split_kind")["max_constraint_excess"].max()
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    per_unit.to_csv(output_dir / "primary_feasibility_per_endpoint_split.csv", index=False)
    pd.concat([summary, overall], ignore_index=True).to_csv(
        output_dir / "primary_feasibility_summary.csv", index=False
    )


if __name__ == "__main__":
    main()
