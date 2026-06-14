# AMD Decode Flywheel Kernel Triage Dataset v1 Featured Plus

This Phase 3F dataset appends the targeted-outcomes train batch to the
Phase 3E featured dataset while preserving the original family-split
holdout. It is an intermediate coverage artifact, not a cost-model win.

- rows: `121`
- train rows: `83`
- holdout rows: `38`
- targeted rows added: `38`
- split policy: `family_split_v0_preserved_plus_post_phase3e_train_batch`
- real UOp/source rows: `18`

## Targeted Mechanisms

| mechanism | rows |
|---|---:|
| `direct_output` | 5 |
| `qk_block_dot` | 1 |
| `reduce_unroll` | 8 |
| `row_upcast` | 10 |
| `two_dim_local` | 8 |
| `vector_load` | 4 |
| `wide_load_only` | 2 |

## Targeted Labels

| label | rows |
|---|---:|
| `construction_blocked` | 19 |
| `diagnostic_only` | 5 |
| `raw_accept_unconfirmed` | 6 |
| `reject` | 4 |
| `tie` | 4 |
