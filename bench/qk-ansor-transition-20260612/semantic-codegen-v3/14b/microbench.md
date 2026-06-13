# QK Semantic Codegen Microbench: 14B

Each candidate is compared against the current schedule for the same tensor.
Full decode is only justified for accepted rows that are also runtime-policy
supported.

## Summary

- candidates: `1`
- raw accepts: `0`
- ties: `1`
- rejected: `0`
- invalid: `0`
- full decode ready: `0`
- next decision: `semantic_codegen_frontier_blocked`

| id | status | gain % | current GB/s | candidate GB/s | full decode | reasons |
|---|---|---:|---:|---:|---:|---|
| `001-ffn-gate-blk-0-ffn-gate-weight-packed-load-u32x4` | `tie` | -0.31 | 367.98 | 366.84 | `False` | within tie_band=0.030 |
