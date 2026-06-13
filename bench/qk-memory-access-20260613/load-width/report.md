# QK Load Width Report

Generated-source parser for QK load-width evidence. This is a source-shape
check, not a hardware-counter measurement.

## Summary

- logs: `2`
- modes: `custom_uint4, uop_vec_request`
- vector load evidence: `True`
- packed-load kernel present: `False`
- packed-dot present: `False`

| log | mode | inferred load width | kernels | packed dot |
|---|---|---|---|---:|
| `bench/qk-memory-access-20260613/load-width/uop-vec-request-debug4.log` | `uop_vec_request` | `u32_scalar` | `qk_probe_uop_vec_request_u32x4_copy_4096` | `False` |
| `bench/qk-memory-access-20260613/load-width/custom-uint4-debug4.log` | `custom_uint4` | `vector_u32x4` | `qk_probe_custom_uint4_copy_4096` | `False` |
