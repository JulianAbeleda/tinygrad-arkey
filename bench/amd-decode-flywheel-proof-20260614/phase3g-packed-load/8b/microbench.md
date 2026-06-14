# QK Semantic Codegen Microbench: 8B

Each candidate is compared against the current schedule for the same tensor.
Full decode is only justified for accepted rows that are also runtime-policy
supported.

## Summary

- candidates: `3`
- raw accepts: `1`
- ties: `1`
- rejected: `0`
- invalid: `1`
- full decode ready: `0`
- next decision: `semantic_codegen_frontier_blocked`

| id | status | gain % | current GB/s | candidate GB/s | full decode | reasons |
|---|---|---:|---:|---:|---:|---|
| `001-ffn-gate-blk-1-ffn-gate-weight-packed-load-u32x4` | `invalid` | n/a | n/a | n/a | `False` | current schedule did not pass microbench |
| `002-ffn-gate-blk-2-ffn-gate-weight-packed-load-u32x4` | `raw_accept` | 3.59 | 205.78 | 213.16 | `False` | requires confirmation before promotion |
| `003-ffn-gate-blk-3-ffn-gate-weight-packed-load-u32x4` | `tie` | -2.72 | 212.24 | 206.46 | `False` | within tie_band=0.030 |
