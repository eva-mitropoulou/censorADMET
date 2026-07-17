# Stage 3 Data Dictionary

## endpoint_id

`target_key + standard_type + assay_type`, for example `CYP3A4_IC50_B`.

## parent_molecule_chembl_id

ChEMBL parent molecule identifier used for deduplication and later split grouping.

## standardized_smiles

RDKit-standardized canonical SMILES generated from parent SMILES when available, otherwise child SMILES.

## relation_p

Final relation on the p-scale. Allowed values are `=`, `<`, `>`, and `interval`.

## value_p

Final exact p-scale value when `relation_p = "="`; null for one-sided or interval labels.

## lower_bound_p / upper_bound_p

Final p-scale interval bounds. Exact labels have equal lower and upper bounds. One-sided censored labels use infinite bounds.

## censoring_class

`exact`, `left_censored_p`, `right_censored_p`, or `interval`.

## exact

Zero-width interval where the p-scale value is observed directly.

## left_censored_p

The p-scale value is bounded above, represented as `(-inf, upper_bound_p]`.

## right_censored_p

The p-scale value is bounded below, represented as `[lower_bound_p, +inf)`.

## interval

The p-scale value is bounded on both sides but is not exact.

## n_measurements

Number of Stage 2 activity records aggregated into the final parent-compound endpoint row.

## quality_flag

Deterministic label describing duplicate/interval compatibility status.

## split_group_* fields

Identifiers for future split generation: parent ID, InChIKey, connectivity InChIKey, scaffold, and year.


## Stage 4 Additions

- `row_id`: deterministic SHA-256 identifier for endpoint/parent/type/assay row.
- `lower_bound_is_infinite`: true when `lower_bound_p` is `-inf`.
- `upper_bound_is_infinite`: true when `upper_bound_p` is `inf`.
- `benchmark_manifest`: machine-readable release metadata.
- `endpoint_manifest`: per-endpoint row counts, censoring counts, file paths, and hashes.
- `checksum`: SHA-256 file hash.
- `package versioning`: semantic version for fixed dataset releases.
