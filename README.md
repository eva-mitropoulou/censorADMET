# CensorADMET

**CensorADMET** is an open, reproducible benchmark and method implementation for ADMET regression with inequality-labelled measurements. Instead of discarding assay statements such as `IC50 > 10 uM` or replacing them with point labels, it preserves them as interval constraints and makes the accuracy-versus-bound-consistency trade-off explicit.

The accompanying ChemRxiv publication introduces **constraint satisficing**: an augmented-Lagrangian objective that limits the aggregate probability deficit assigned to observed censored intervals while retaining an exact-label objective.

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
| Matched soft-deficit comparator | Similar operating points can be obtained with a scalar penalty; the distinct contribution is direct deficit-budget control |
| Complete feasibility audit | 40 endpoint/split assignments and 370 fold-and-seed cells; aggregate budget targeting is reported rather than claimed as per-run enforcement |
| Split-conformal intervals | Coverage 0.79--0.90; mean widths 3.59--4.83 and median widths 3.27--3.48 transformed-scale units |

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

`make verify` recomputes representative manuscript values from the released result files. The conformal audit uses exact calibration rows only and records the fixed calibration/test sample sizes for every endpoint, split, fold, and seed. Full experiment drivers are in `scripts/`; they use the fixed data, split generator, seeds, and hyperparameters distributed here.

## Citation

Please cite the accompanying ChemRxiv publication:

> Mitropoulou, E.; Giannopoulos, D. *CensorADMET: Controllable Constraint-Satisficing Regression for Censored ADMET Data.* ChemRxiv, 2026.

The stable ChemRxiv identifier will be added to [CITATION.cff](CITATION.cff) on publication.

## Licenses And Attribution

The code is released under the [MIT License](LICENSE_CODE). The curated benchmark input is a ChEMBL 36-derived adaptation and is released separately under [CC BY-SA 3.0](LICENSE_DATA), consistent with ChEMBL's data license. See [ATTRIBUTION.md](ATTRIBUTION.md) and [data/README.md](data/README.md) for required attribution and reuse terms.
