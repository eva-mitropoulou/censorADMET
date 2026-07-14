# Frozen Results

This directory contains the frozen artifacts used by the CensorADMET analyses.
The main result matrices retain per-endpoint, split, fold, seed, and treatment
metrics. The `synthesis/` files provide compact endpoint-level summaries.

Key supporting artifacts include:

- `conformal_scored/`: raw and split-conformal coverage, interval width, and
  interval score for the Gaussian satisficing configuration at `epsilon=0.05`.
- `synthesis/conformal_sample_audit.csv`: exact calibration-row and exact
  test-row counts for every conformal endpoint/split/fold/seed run.
- `synthesis/conformal_scored_summary.csv`: endpoint-aggregated coverage,
  mean width, median width, and interval score.
- `synthesis/soft_satisficing_summary.csv`: the matched same-deficit
  scalar-penalty operating curve.
- `dmpnn_tobit_audit/`: per-seed diagnostic output for the strict-deficit
  D-MPNN configuration.

Run `make verify` from the repository root to recompute the checked reported
values from these frozen artifacts. Calibration is evaluated only on exact test
rows; it does not alter the released point-prediction or decision metrics.
