# AMD Decode Flywheel Targeted Outcomes v1

This Phase 3F artifact converts unused committed real probe/source
diagnostics into a small post-Phase-3E train batch. It is deliberately
partial: no holdout row is moved into train, no synthetic outcome is added,
and design-only contracts remain excluded.

- conclusion: `partial_real_outcome_batch_cost_model_rerun_still_gated`
- targeted train rows: `38`
- base rows: `83`
- plus rows: `121`
- real feature rows: `5`

## Mechanisms

| mechanism | rows |
|---|---:|
| `direct_output` | 5 |
| `qk_block_dot` | 1 |
| `reduce_unroll` | 8 |
| `row_upcast` | 10 |
| `two_dim_local` | 8 |
| `vector_load` | 4 |
| `wide_load_only` | 2 |

## Labels

| label | rows |
|---|---:|
| `construction_blocked` | 19 |
| `diagnostic_only` | 5 |
| `raw_accept_unconfirmed` | 6 |
| `reject` | 4 |
| `tie` | 4 |

## Excluded Sources

| source | rows | reason |
|---|---:|---|
| `bench/qk-packed-semantic-op-20260613/semantic-op-contract.json` | 8 | design_only_no_runtime_lowering; recorded in plan but not used as train labels |

## Rules

- uses only committed real probe/compile/source diagnostic artifacts
- does not duplicate any existing v1-featured row id
- does not move family-split holdout rows into train
- does not use design-only contracts as train labels
- does not authorize a Phase 3B cost-model rerun as a decision point unless the plus audit clears the coverage gate
