# AMD Decode Flywheel Kernel Triage Dataset v1

This Phase 3D artifact preserves the v0 family split while adding normalized
mechanisms and a frozen candidate-outcome feature schema for future cost
model training.

- rows: `83`
- train rows: `45`
- holdout rows: `38`
- feature schema: `candidate_outcome_v1`
- unknown mechanism rows: `0`
- mechanism changes from v0: `26`

## Mechanisms

| mechanism | rows |
|---|---:|
| `direct_output` | 11 |
| `packed_word_lane_unroll` | 2 |
| `parts_local_policy` | 23 |
| `qk_block_dot` | 2 |
| `reduce_unroll` | 8 |
| `row_grouping` | 4 |
| `row_upcast` | 10 |
| `shared_storage` | 3 |
| `storage_cap` | 1 |
| `tile_custom` | 8 |
| `two_dim_local` | 8 |
| `vector_load` | 2 |
| `wide_load_only` | 1 |
