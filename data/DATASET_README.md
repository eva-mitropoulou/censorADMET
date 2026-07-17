# CensorADMET curated dataset and result artifacts v1.0.0

This archive is the data companion to the CensorADMET software release. It contains curated ChEMBL 36-derived benchmark inputs, fixed split assignments, per-run result artifacts, aggregated summaries, and provenance documentation. It does not contain source-code duplication, Git metadata, virtual environments, model checkpoints, manuscript files, or raw ChEMBL database dumps.

## Contents

- `measurement_records.parquet` — one row per curated ChEMBL activity for the eight measurement-level CYP, hERG, and ABCB1 endpoints.
- `endpoints/` — ten aggregated one-row-per-compound–endpoint benchmark inputs.
- `admet_endpoints/` — curated solubility, clearance, and half-life inputs for the broadened-property cohort.
- `fixed_splits/` — compressed five-fold train/test index archives (split seed zero); each archive includes stable source row identifiers.
- `provenance/` — curation lineage, rules, schema, endpoint manifest, and audit reports.
- `results/` — released per-run matrices and manuscript summaries used for auditing.

## Curation and label representation

The inputs are derived from ChEMBL release 36 (DOI: 10.6019/CHEMBL.database.36). Exact labels carry equal lower and upper bounds. One-sided observations are represented by an infinite lower or upper bound on the transformed p-scale and a `censoring_class` identifying the known direction. Additional field definitions are in `provenance/data_dictionary.md` and `provenance/schema.json`.

## Licence, attribution, and citation

The curated ChEMBL-derived inputs, fixed splits, and derived result artifacts are distributed under CC BY-SA 3.0 Unported. Retain the ChEMBL attribution in `ATTRIBUTION.md`, cite ChEMBL 36, and cite the corresponding CensorADMET software release. The dataset is intended to be associated with software tag `v1.0.0-paper`; the manuscript/preprint is a separate scholarly object and will receive its own DOI.

## Integrity checks

Run `sha256sum -c censoradmet-dataset-v1.0.0.sha256` from the release-asset directory to verify the archive checksum. `RELEASE_MANIFEST.json` records SHA-256 checksums for its major archived inputs and result artifacts.
