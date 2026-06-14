# AMD Decode Flywheel Feature Coverage Audit

This Phase 3D artifact extends the data and feature audit over the normalized v1 schema. It does not train a model.

- conclusion: `needs_data_and_feature_expansion`
- train rows: `45`
- holdout rows: `38`
- unseen holdout categorical values: `15`
- weak rows: `43`
- post-full-decode train rows: `9`

## Highest Priority Targets

- P1 `collect_label_coverage`: Add train rows for labels that appear in holdout but are absent or undercovered in train.
- P3 `collect_mechanism_coverage`: Add targeted candidates for holdout mechanisms with fewer than five train rows.
- P4 `reduce_unseen_categorical_gap`: Add rows or normalize feature extraction so holdout families/mechanisms/schedule names are represented before model training.
- P5 `improve_feature_extraction`: Add first-class tinygrad/UOp/profile features for weak rows instead of relying on top-level labels and candidate names.

## Label Targets

| label | train | holdout | needed train rows |
|---|---:|---:|---:|
| `construction_blocked` | 1 | 19 | 4 |
| `diagnostic_only` | 0 | 1 | 5 |
| `needs_rerun` | 2 | 0 | 3 |
| `raw_accept_unconfirmed` | 0 | 3 | 5 |
| `reject` | 20 | 9 | 0 |
| `tie` | 13 | 6 | 0 |

## Holdout Mechanism Targets

| mechanism | train | holdout | needed train rows |
|---|---:|---:|---:|
| `direct_output` | 7 | 4 | 0 |
| `packed_word_lane_unroll` | 0 | 2 | 5 |
| `qk_block_dot` | 0 | 2 | 5 |
| `reduce_unroll` | 0 | 8 | 5 |
| `row_upcast` | 0 | 10 | 5 |
| `tile_custom` | 7 | 1 | 0 |
| `two_dim_local` | 0 | 8 | 5 |
| `vector_load` | 0 | 2 | 5 |
| `wide_load_only` | 0 | 1 | 5 |

## Top Unseen Categorical Features

| feature | unseen holdout values |
|---|---|
| `v1_static_mechanism` | `packed_word_lane_unroll, qk_block_dot, reduce_unroll, row_upcast, two_dim_local, vector_load, wide_load_only` |
| `v1_static_schedule_name` | `reduce_unroll, row_upcast, two_dim_local` |
| `v1_static_format` | `Q6_K` |
| `v1_static_prediction_stage` | `after_microbench_before_full_decode` |
| `v1_static_row_kind` | `diagnostic` |
| `v1_static_schedule_codegen_mode` | `partial` |
| `v1_static_schedule_reduction_mode` | `split_k_partial` |

## Top Weak Reasons

| reason | rows |
|---|---:|
| `mechanism_unseen_in_train` | 33 |
| `post_full_decode_training_row` | 9 |
| `label_unseen_in_train` | 4 |
