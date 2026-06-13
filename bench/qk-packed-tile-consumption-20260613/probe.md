# QK Packed Tile Consumption Probe

Construction gate for consuming a `PackedQKTile` Q4_K `u32x4_aligned` load.
This is not a speed benchmark.

## Decision

- decision: `semantic_custom_op_required`
- next path: add a first-class packed QK load/decode/dot semantic op or renderer PatternMatcher lowering
- run microbench: `False`
- run full decode: `False`

## Packed Tile

- source descriptor: `bench/qk-ansor-transition-20260612/descriptors/8b.json`
- tensor: `blk.0.ffn_gate.weight`
- shape: `12288x4096`
- legal load tiles: `u32_scalar, u32x4_aligned`
- required load tile: `u32x4_aligned`

## Rows

| mode | status | exact | device ms | key evidence |
|---|---|---:|---:|---|
| `uop_lane_gep` | `expected_fail` | `None` | n/a | UOp verification failed at 9 on Ops.GEP dtypes.uint 1 [(Ops.LOAD, dtypes.uint.vec(4), None)] (0,) |
| `uop_vector_arith` | `expected_fail` | `None` | n/a | UOp verification failed at 10 on Ops.CONST dtypes.weakint 0 [] 1 |
| `custom_q4_dot` | `pass` | `True` | n/a | uint4 load + lane extraction + nibble unpack |

## Interpretation

Current normal UOps cannot consume the packed vector load. Scalar lane
extraction through `GEP` fails verifier, and vector integer arithmetic fails
shape validation. A custom semantic kernel can load `tg_uint4`, index lanes,
unpack low/high Q4 nibbles, and accumulate an exact dot. Therefore the next
implementation should be a first-class packed QK load/decode/dot lowering or
renderer PatternMatcher rule, not another normal-UOp rewrite of v4.
