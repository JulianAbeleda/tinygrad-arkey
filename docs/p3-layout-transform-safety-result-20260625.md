# P3 layout transform safety result

Date: 2026-06-25
Verdict: `P3_LAYOUT_TRANSFORM_SAFETY_PASS`

## Scope executed

This is the safety-first P3.2 slice after the static COALESCE scorer passed.

- Added `Ops.LAYOUT_TRANSFORM` as a movement op.
- Added a single validated transform name: `q4k_lane_partition`.
- The transform is intentionally identity-on-shape and identity-on-ranges in v1.
- Added movement sugar `.layout_transform(name)` for Tensor/UOp surfaces.
- Added an `apply_movement_op` case and a movement-on-INDEX rewrite gate.
- Connected static scoring through `score_layout_transform("q4k_lane_partition")`.

## Non-scope

This is not a real storage permutation yet. It is a declared layout-intent carrier for search/static scoring.
Generic `add_gpudims` REDUCE substitution and model default routing remain unchanged.

## Next step

P3.3 should make beam/search reproduce the q4k lane-partition route without using `Q4K_GEMV_SCHEDULER=4` as the user-facing selector. The safe target is still route selection, not a generic storage permutation rewrite.
