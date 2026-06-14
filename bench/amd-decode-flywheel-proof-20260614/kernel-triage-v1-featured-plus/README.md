# AMD Decode Flywheel Kernel Triage Dataset v1 Featured Plus

This Phase 3F dataset appends the targeted-outcomes train batch to the
Phase 3E featured dataset while preserving the original family-split
holdout. It is an intermediate coverage artifact, not a cost-model win.

- rows: `136`
- train rows: `98`
- holdout rows: `38`
- targeted rows added: `53`
- split policy: `family_split_v0_preserved_plus_post_phase3e_train_batch`
- real UOp/source rows: `22`

## Targeted Mechanisms

| mechanism | rows |
|---|---:|
| `direct_output` | 5 |
| `packed_word_lane_unroll` | 5 |
| `qk_block_dot` | 5 |
| `reduce_unroll` | 8 |
| `row_upcast` | 10 |
| `tile_custom` | 1 |
| `two_dim_local` | 8 |
| `vector_load` | 6 |
| `wide_load_only` | 5 |

## Targeted Labels

| label | rows |
|---|---:|
| `construction_blocked` | 22 |
| `diagnostic_only` | 8 |
| `raw_accept_unconfirmed` | 7 |
| `reject` | 9 |
| `tie` | 7 |
