# QK Semantic Codegen Microbench: 8B

Each candidate is compared against the current schedule for the same tensor.
Full decode is only justified for accepted rows that are also runtime-policy
supported.

## Summary

- candidates: `3`
- raw accepts: `0`
- ties: `3`
- rejected: `0`
- invalid: `0`
- full decode ready: `0`
- next decision: `semantic_codegen_frontier_blocked`

| id | status | gain % | current GB/s | candidate GB/s | full decode | reasons |
|---|---|---:|---:|---:|---:|---|
| `001-ffn-gate-blk-4-ffn-gate-weight-packed-load-u32x4` | `tie` | 0.92 | 203.19 | 205.06 | `False` | within tie_band=0.030 |
| `002-ffn-gate-blk-5-ffn-gate-weight-packed-load-u32x4` | `tie` | 0.22 | 205.01 | 205.47 | `False` | within tie_band=0.030 |
| `003-ffn-gate-blk-6-ffn-gate-weight-packed-load-u32x4` | `tie` | -0.06 | 206.03 | 205.91 | `False` | within tie_band=0.030 |
