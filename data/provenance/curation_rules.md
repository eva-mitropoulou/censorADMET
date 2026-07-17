general:
  strict_dataset_name: v1_strict
  broad_dataset_name: v1_broad
  group_by:
    - endpoint_id
    - parent_molecule_chembl_id
  keep_original_child_ids: true
  preserve_raw_records: true

record_filters:
  allowed_relations:
    - "="
    - "<"
    - "<="
    - ">"
    - ">="
  allowed_p_relations:
    - "="
    - "<"
    - "<="
    - ">"
    - ">="
  allowed_units:
    - nM
    - uM
    - µM
    - mM
    - M
    - pM
  require_positive_value_nM: true
  require_valid_p_value: true
  p_value_min: 0
  p_value_max: 14
  allowed_target_type:
    - SINGLE PROTEIN
  allowed_target_organism:
    - Homo sapiens
  minimum_confidence_score_strict: 9
  minimum_confidence_score_broad: 7
  allowed_data_validity_comment_strict:
    - null
    - Manually validated
  allowed_data_validity_comment_broad:
    - null
    - Manually validated
    - Outside typical range
  exclude_potential_duplicate_strict: true
  exclude_potential_duplicate_broad: false
  require_document_year_strict: false

structure_rules:
  use_parent_smiles_when_available: true
  fallback_to_child_smiles: true
  validate_with_rdkit: true
  use_chembl_structure_pipeline_if_available: true
  keep_only_largest_parent_fragment: true
  allow_inorganic: false
  allow_mixtures: false
  min_heavy_atoms: 5
  max_heavy_atoms: 120
  max_molecular_weight: 1200
  generate:
    - standardized_smiles
    - standard_inchikey
    - connectivity_inchikey
    - murcko_scaffold_smiles
    - heavy_atom_count
    - molecular_weight

activity_comment_rules:
  do_not_generate_labels_from_comments: true
  flag_non_null_activity_comment: true
  conflict_terms:
    - inactive
    - not active
    - inconclusive
    - non-toxic
    - toxic
    - active
  strict_action_on_comment_conflict: quarantine
  broad_action_on_comment_conflict: keep_flagged

duplicate_rules:
  exact_range_high_confidence_log_units: 0.5
  exact_range_max_strict_log_units: 1.0
  exact_range_max_broad_log_units: 2.0
  contradiction_action_strict: quarantine_group
  contradiction_action_broad: keep_flagged_if_partial
  aggregate_exact_values_with: median
  aggregate_censored_bounds_with: interval_intersection
  prefer_exact_only_if_consistent_with_censored_bounds: true
  retain_measurement_counts: true

interval_rules:
  exact:
    lower_equals_upper: true
  right_censored_p:
    p_relation:
      - ">"
      - ">="
    representation:
      lower_bound_p: p_value
      upper_bound_p: inf
  left_censored_p:
    p_relation:
      - "<"
      - "<="
    representation:
      lower_bound_p: -inf
      upper_bound_p: p_value
