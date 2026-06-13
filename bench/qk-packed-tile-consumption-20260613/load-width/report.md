# QK Load Width Report

Generated-source parser for QK load-width evidence. This is a source-shape
check, not a hardware-counter measurement.

## Summary

- logs: `1`
- modes: `packed_tile_custom_q4_dot`
- vector load evidence: `True`
- packed-load kernel present: `False`
- packed-dot present: `False`

| log | mode | inferred load width | kernels | packed dot |
|---|---|---|---|---:|
| `bench/qk-packed-tile-consumption-20260613/load-width/probe-debug4.log` | `packed_tile_custom_q4_dot` | `vector_u32x4` | `qk_probe_tile_custom_q4_dot_36, qk_probe_tile_uop_lane_gep_sum_4, qk_probe_tile_uop_vector_arith_store_16` | `False` |
