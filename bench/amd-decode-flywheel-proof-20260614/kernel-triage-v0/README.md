# AMD Decode Flywheel Kernel Triage Dataset

This Phase 1 artifact converts existing QK/kernel experiment artifacts into
structured candidate-outcome examples. Prompt rows expose pre-result context;
example rows retain hidden labels, reasons, and evidence for deterministic
Phase 2 triage evaluation.

- rows: `83`
- train rows: `45`
- holdout rows: `38`
- split policy: `family_split_v0`
- prompt contract: `/no_think`, strict compact JSON, `max_tokens=64`

## Labels

| label | rows |
|---|---:|
| `accept` | 9 |
| `construction_blocked` | 20 |
| `diagnostic_only` | 1 |
| `needs_rerun` | 2 |
| `raw_accept_unconfirmed` | 3 |
| `reject` | 29 |
| `tie` | 19 |

## Mechanisms

| mechanism | rows |
|---|---:|
| `direct_output` | 11 |
| `packed_word_lane_unroll` | 2 |
| `parts_local_policy` | 31 |
| `qk_block_dot` | 2 |
| `row_grouping` | 4 |
| `shared_storage` | 3 |
| `storage_cap` | 1 |
| `tile_custom` | 8 |
| `unknown` | 18 |
| `vector_load` | 2 |
| `wide_load_only` | 1 |
