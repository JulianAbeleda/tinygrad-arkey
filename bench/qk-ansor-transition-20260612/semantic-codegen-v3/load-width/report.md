# QK Load Width Report

Generated-source parser for QK load-width evidence. This is a source-shape
check, not a hardware-counter measurement.

## Summary

- logs: `2`
- modes: `baseline_partial, packed_load`
- vector load evidence: `False`
- packed-load kernel present: `True`
- packed-dot present: `False`

| log | mode | inferred load width | kernels | packed dot |
|---|---|---|---|---:|
| `bench/qk-ansor-transition-20260612/semantic-codegen-v3/load-width/8b-ffn-gate-current-debug4.log` | `baseline_partial` | `u32_scalar` | `q4k_gemv_partial_12288_4096_1` | `False` |
| `bench/qk-ansor-transition-20260612/semantic-codegen-v3/load-width/8b-ffn-gate-packed-load-debug4.log` | `packed_load` | `u32_scalar` | `q4k_gemv_packed_load_partial_12288_4096_1` | `False` |
