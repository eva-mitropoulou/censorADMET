# CensorADMET

**CensorADMET** is an open, reproducible benchmark and method implementation for ADMET regression with inequality-labelled measurements. Instead of discarding assay statements such as `IC50 > 10 uM` or replacing them with point labels, it preserves them as interval constraints and makes the accuracy-versus-bound-consistency trade-off explicit.

The accompanying ChemRxiv preprint introduces **constraint satisficing**: an augmented-Lagrangian objective that limits the aggregate probability deficit assigned to observed censored intervals while retaining an exact-label objective.

## Why This Matters

Public ADMET data contain assay bounds as well as exact values. Common handling rules either throw away this information or treat a threshold as the unknown truth. CensorADMET evaluates these choices directly under chemical and provenance shift.

The central result is deliberately not a leaderboard claim. Plain interval likelihood substantially reduces one-sided violations of known bounds, but can incur large exact-label penalties. Constraint satisficing exposes a prespecified operating curve between these objectives.

## Included Results

The frozen result artifacts cover eight CYP, hERG, and ABCB1 measurement-level endpoints from ChEMBL 36, with random, scaffold, assay-held-out, document-held-out, and source-held-out evaluation.

| Finding | Result |
|---|---:|
| Primary satisficing violation reduction versus exact-only | 0.15--0.29 median endpoint-level reduction |
| Primary satisficing exact-label MAE cost | <= 0.04 transformed-scale units median |
| Plain Tobit bound-consistency gain | Larger, with substantially higher exact-label MAE in matched comparisons |
| Anchored weighted-Tobit comparison | Does not reach the stricter satisficing violation regime; no uniform MAE dominance where curves overlap |
| Complete feasibility audit | 37 endpoint/split units, 370 valid fold-and-seed cells; aggregate budget targeting is reported rather than claimed as per-run enforcement |
| Split-conformal coverage | 0.79--0.90, with broad intervals under shift |

`figures/` contains the two core operating-curve and endpoint-effect plots. `results/` contains the tidy per-run matrices and compact summaries used for the reported analyses.

## Repository Layout

```text
src/censoradmet/   constraint loss, augmented-Lagrangian trainer, models, splits, metrics
scripts/           reproducible experiment and validation entry points
data/              frozen ChEMBL-36 measurement-level benchmark input
results/           frozen result matrices and manuscript-value summaries
figures/           main operating-curve and paired-effect figures
tests/             unit tests for interval handling, losses, splits, and metrics
```

The repository intentionally excludes manuscript files, internal project logs, review material, model checkpoints, and intermediate workspace artifacts.

## Install

Python 3.11+ and RDKit are required. A conda environment is the most reliable way to install RDKit.

```bash
conda create -n censoradmet python=3.11 rdkit -c conda-forge
conda activate censoradmet
pip install -e .
```

## Reproduce And Verify

The repository ships the frozen input and result artifacts needed to audit the reported analyses without retraining.

```bash
make test
make verify
```

`make verify` recomputes representative manuscript values from the released result files. Full experiment drivers are in `scripts/`; they use the fixed data, split generator, seeds, and hyperparameters distributed here.

## Citation

Please cite the accompanying ChemRxiv preprint:

> Mitropoulou, E.; Giannopoulos, D. *CensorADMET: Constraint-Satisficing Regression for Censored ADMET Prediction with a Controllable Accuracy-versus-Consistency Trade-off.* ChemRxiv, 2026, preprint.

The stable ChemRxiv identifier will be added to [CITATION.cff](CITATION.cff) on publication.

## License

Code is released under the [MIT License](LICENSE). The included benchmark input is derived from ChEMBL 36; see [data/README.md](data/README.md) for attribution and data-use details.
