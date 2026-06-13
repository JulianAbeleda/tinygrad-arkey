# QK Load Width Report

Generated-source parser for QK load-width evidence. This is a source-shape
check, not a hardware-counter measurement.

## Summary

- logs: `2`
- modes: `baseline_partial, tile_custom_partial`
- vector load evidence: `True`
- packed-load kernel present: `False`
- packed-dot present: `False`

| log | mode | inferred load width | kernels | packed dot |
|---|---|---|---|---:|
| `bench/qk-packed-tile-lowering-analysis-20260613/source/v1_partial-debug4.log` | `baseline_partial` | `u32_scalar` | `q4k_gemv_partial_64_4096_1` | `False` |
| `bench/qk-packed-tile-lowering-analysis-20260613/source/tile_custom-debug4.log` | `tile_custom_partial` | `vector_u32x4` | `q4k_gemv_tile_custom_partial_64_4096_1` | `False` |
