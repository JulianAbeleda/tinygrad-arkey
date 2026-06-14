# AMD Decode Flywheel Feature Coverage Audit

This Phase 3C artifact scopes the data and feature gaps that blocked the
learned cost-model triage result. It does not train a model.

- conclusion: `needs_data_and_feature_expansion`
- train rows: `45`
- holdout rows: `38`
- unseen holdout categorical values: `24`
- weak rows: `56`
- post-full-decode train rows: `9`

## Highest Priority Targets

- P1 `collect_label_coverage`: Add train rows for labels that appear in holdout but are absent or undercovered in train.
- P2 `normalize_unknown_mechanisms`: Map unknown holdout mechanisms to first-class mechanism names before treating them as learnable classes.
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
| `parts_local_policy` | 23 | 8 | 0 |
| `qk_block_dot` | 0 | 2 | 5 |
| `tile_custom` | 7 | 1 | 0 |
| `unknown` | 0 | 18 | 0 |
| `vector_load` | 0 | 2 | 5 |
| `wide_load_only` | 0 | 1 | 5 |

## Top Unseen Categorical Features

| feature | unseen holdout values |
|---|---|
| `family` | `qk_block_dot, semantic_codegen_v3, semantic_codegen_v4, semantic_schedule_v0, threeway_load` |
| `mechanism` | `packed_word_lane_unroll, qk_block_dot, unknown, vector_load, wide_load_only` |
| `schedule_name` | `direct_out, reduce_unroll4, row_upcast2, two_dim_local4` |
| `schedule_family` | `q4_k_packed_u32, q6_k_packed_u16` |
| `context_mode` | `vector_load` |
| `format` | `Q6_K` |
| `prediction_stage` | `after_microbench_before_full_decode` |
| `row_kind` | `diagnostic` |
| `schedule_codegen_mode` | `partial` |
| `schedule_format` | `Q6_K` |
| `schedule_reduction_mode` | `split_k_partial` |
| `schedule_semantic_object` | `packed_quant_gemv_schedule` |

## Top Weak Reasons

| reason | rows |
|---|---:|
| `family_unseen_in_train` | 38 |
| `no_structural_kernel_detail` | 26 |
| `mechanism_unseen_in_train` | 25 |
| `unknown_mechanism` | 18 |
| `post_full_decode_training_row` | 9 |
| `label_unseen_in_train` | 4 |
