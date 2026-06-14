# AMD Decode Flywheel Phase 3F Plus Coverage Plan

This artifact turns the current featured audit into a concrete
data-collection batch. It does not add training examples by itself.

- conclusion: `coverage_gate_cleared_cost_model_rerun_allowed`
- rerun Phase 3B allowed: `True`
- minimum mechanism rows: `0`
- minimum label rows: `0`

## Mechanism Batches

| mechanism | needed | batch | stage |
|---|---:|---|---|

## Label Batches

| label | needed | note |
|---|---:|---|

## Rules

- Rows must be real candidate outcomes, not duplicated holdout rows or synthetic labels.
- The existing family-split holdout remains valid; new post-Phase-3E outcomes should form a dated train/rolling-shadow batch.
- Full decode remains gated by the normal static, correctness, microbench, and confirmation rules.
