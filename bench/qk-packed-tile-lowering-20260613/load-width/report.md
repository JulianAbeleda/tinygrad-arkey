# QK Load Width Report

Generated-source parser for QK load-width evidence. This is a source-shape
check, not a hardware-counter measurement.

## Summary

- logs: `1`
- modes: `tile_custom_partial`
- vector load evidence: `True`
- packed-load kernel present: `False`
- packed-dot present: `False`

| log | mode | inferred load width | kernels | packed dot |
|---|---|---|---|---:|
| `bench/qk-packed-tile-lowering-20260613/load-width/tile-custom-debug4.log` | `tile_custom_partial` | `vector_u32x4` | `q4k_gemv_tile_custom_partial_2_4096_1` | `False` |
