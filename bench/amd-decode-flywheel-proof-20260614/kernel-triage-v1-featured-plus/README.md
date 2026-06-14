# AMD Decode Flywheel Kernel Triage Dataset v1 Featured Plus

This Phase 3F dataset appends the targeted-outcomes train batch to the
Phase 3E featured dataset while preserving the original family-split
holdout. It is an intermediate coverage artifact, not a cost-model win.

- rows: `90`
- train rows: `52`
- holdout rows: `38`
- targeted rows added: `7`
- split policy: `family_split_v0_preserved_plus_post_phase3e_train_batch`
- real UOp/source rows: `18`

## Targeted Mechanisms

| mechanism | rows |
|---|---:|
| `qk_block_dot` | 1 |
| `vector_load` | 4 |
| `wide_load_only` | 2 |

## Targeted Labels

| label | rows |
|---|---:|
| `construction_blocked` | 2 |
| `diagnostic_only` | 5 |
