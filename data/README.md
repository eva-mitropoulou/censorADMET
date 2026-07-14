# Benchmark Input

`measurement_records.parquet` is the frozen, measurement-level ChEMBL 36 input used by the released CYP, hERG, and ABCB1 experiments. Each row preserves its activity relation, transformed interval bounds, assay identifier, document identifier, source, publication year, and standardized molecular structure.

The data are derived from [ChEMBL 36](https://www.ebi.ac.uk/chembl/), which must be cited when the benchmark input is reused. The source database and all original metadata remain attributable to the ChEMBL team. This repository distributes only the curated research input required to audit the accompanying analyses.

The data package is not a claim of assay harmonization or experimental ground truth. Censoring thresholds are assay-specific and the released labels should be interpreted as reported inequalities on the documented transformed scale.
