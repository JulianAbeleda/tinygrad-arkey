# Primitive coverage map summary

- rows: `12`
- validation: `PASS`
- commit: `e6c4a9055`

## State Counts

- `deferred_until_target_expands`: `1`
- `open_diagnostic`: `2`
- `out_of_scope`: `1`
- `pass_research_small`: `1`
- `project_level_closed_for_bounded_build`: `1`
- `project_level_live`: `1`
- `proposed`: `1`
- `refuted_for_e2e_speed`: `1`
- `research_only_deferred`: `1`
- `separate_audit`: `1`
- `supporting_row`: `1`

## Priority Rows

1. `decode_mmvq_runtime_cache_identity` - next bounded decode diagnostic.
2. `prefill_non_matmul_overhead` - replaces the stale Tensile-as-speed-route framing.
3. `decode_mmvq_artifact_import_family` - only if artifact/import is an accepted research direction.
4. long-context / serving / alternative-quant / CUDA rows stay deferred until the target changes.
