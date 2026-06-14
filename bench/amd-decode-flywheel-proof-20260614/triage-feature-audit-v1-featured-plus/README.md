# AMD Decode Flywheel Feature Coverage Audit

This Phase 3E artifact audits the normalized v1 schema after adding real source/compile features. It does not train a model.

- conclusion: `needs_data_and_feature_expansion`
- train rows: `88`
- holdout rows: `38`
- unseen holdout categorical values: `2`
- weak rows: `11`
- post-full-decode train rows: `9`
- real UOp/source rows: `20`
- compile-report rows: `4`
- load-width-report rows: `16`

## Highest Priority Targets

- P3 `collect_mechanism_coverage`: Add targeted candidates for holdout mechanisms with fewer than five train rows.
- P4 `reduce_unseen_categorical_gap`: Add rows or normalize feature extraction so holdout families/mechanisms/schedule names are represented before model training.
- P5 `improve_feature_extraction`: Add first-class tinygrad/UOp/profile features for weak rows instead of relying on top-level labels and candidate names.

## Label Targets

| label | train | holdout | needed train rows |
|---|---:|---:|---:|
| `construction_blocked` | 20 | 19 | 0 |
| `diagnostic_only` | 7 | 1 | 0 |
| `needs_rerun` | 2 | 0 | 3 |
| `raw_accept_unconfirmed` | 6 | 3 | 0 |
| `reject` | 27 | 9 | 0 |
| `tie` | 17 | 6 | 0 |

## Holdout Mechanism Targets

| mechanism | train | holdout | needed train rows |
|---|---:|---:|---:|
| `direct_output` | 12 | 4 | 0 |
| `packed_word_lane_unroll` | 0 | 2 | 5 |
| `qk_block_dot` | 3 | 2 | 2 |
| `reduce_unroll` | 8 | 8 | 0 |
| `row_upcast` | 10 | 10 | 0 |
| `tile_custom` | 8 | 1 | 0 |
| `two_dim_local` | 8 | 8 | 0 |
| `vector_load` | 4 | 2 | 1 |
| `wide_load_only` | 4 | 1 | 1 |

## Top Unseen Categorical Features

| feature | unseen holdout values |
|---|---|
| `v1_static_mechanism` | `packed_word_lane_unroll` |
| `v1_static_prediction_stage` | `after_microbench_before_full_decode` |

## Top Weak Reasons

| reason | rows |
|---|---:|
| `post_full_decode_training_row` | 9 |
| `mechanism_unseen_in_train` | 2 |

## Real Feature Coverage

| mechanism | rows |
|---|---:|
| `packed_word_lane_unroll` | 2 |
| `qk_block_dot` | 5 |
| `tile_custom` | 7 |
| `vector_load` | 4 |
| `wide_load_only` | 2 |
