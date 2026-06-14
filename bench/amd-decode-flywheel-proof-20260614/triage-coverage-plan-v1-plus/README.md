# AMD Decode Flywheel Phase 3F Plus Coverage Plan

This artifact turns the current featured audit into a concrete
data-collection batch. It does not add training examples by itself.

- conclusion: `collect_targeted_outcomes_before_cost_model_rerun`
- rerun Phase 3B allowed: `False`
- minimum mechanism rows: `28`
- minimum label rows: `7`

## Mechanism Batches

| mechanism | needed | batch | stage |
|---|---:|---|---|
| `packed_word_lane_unroll` | 5 | packed-load lane-unroll microbench | `after_static_before_microbench` |
| `qk_block_dot` | 4 | QK_BLOCK_DOT compile+dominant-shape microbench | `after_compile_before_microbench` |
| `reduce_unroll` | 5 | semantic schedule v1 | `after_static_before_microbench` |
| `row_upcast` | 5 | semantic schedule v1 | `after_static_before_microbench` |
| `two_dim_local` | 5 | semantic schedule v1 | `after_static_before_microbench` |
| `vector_load` | 1 | vector-load construction probe | `after_static_before_microbench` |
| `wide_load_only` | 3 | three-way load diagnostic continuation | `after_compile_before_microbench` |

## Label Batches

| label | needed | note |
|---|---:|---|
| `construction_blocked` | 2 | Natural outcome from failed construction/static candidates; record exact verifier/shape failure. |
| `raw_accept_unconfirmed` | 5 | Only record when a repeated microbench clears its predeclared bar but no full-decode confirmation exists yet; do not force this label. |

## Rules

- Rows must be real candidate outcomes, not duplicated holdout rows or synthetic labels.
- The existing family-split holdout remains valid; new post-Phase-3E outcomes should form a dated train/rolling-shadow batch.
- Full decode remains gated by the normal static, correctness, microbench, and confirmation rules.
