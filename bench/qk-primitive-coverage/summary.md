# Primitive coverage map summary

- rows: `12`
- validation: `PASS`
- commit: `c4fa1c145`

## State Counts

- `closed_no_mismatch`: `1`
- `closed_no_ready_artifact_source_import_project_level`: `1`
- `deferred_until_target_expands`: `1`
- `open_diagnostic`: `1`
- `out_of_scope`: `1`
- `pass_research_small`: `1`
- `project_level_closed_for_bounded_build`: `1`
- `project_level_live`: `1`
- `refuted_for_e2e_speed`: `1`
- `research_only_deferred`: `1`
- `separate_audit`: `1`
- `supporting_row`: `1`

## Priority Rows

1. `prefill_non_matmul_overhead` - next open diagnostic if continuing prefill.
2. `decode_mmvq_contract_preservation` - large decode parity path, now project-level renderer/scheduler or source import.
3. `decode_q8_artifact_lifecycle` - small decode path, already passed as default-off research flag.
4. long-context / serving / alternative-quant / CUDA rows stay deferred until the target changes.
