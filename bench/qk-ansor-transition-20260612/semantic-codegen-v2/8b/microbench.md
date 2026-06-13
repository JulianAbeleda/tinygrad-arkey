# QK Semantic Codegen Microbench: 8B

Each candidate is compared against the current schedule for the same tensor.
Full decode is only justified for accepted rows that are also runtime-policy
supported.

## Summary

- candidates: `2`
- raw accepts: `0`
- ties: `0`
- rejected: `2`
- invalid: `0`
- full decode ready: `0`
- next decision: `semantic_codegen_frontier_blocked`

| id | status | gain % | current GB/s | candidate GB/s | full decode | reasons |
|---|---|---:|---:|---:|---:|---|
| `001-ffn-down-blk-4-ffn-down-weight-row-group2` | `reject` | -31.03 | 267.69 | 184.63 | `False` | below min_gain=0.030 |
| `002-ffn-down-blk-4-ffn-down-weight-row-group4` | `reject` | -71.54 | 265.79 | 75.64 | `False` | below min_gain=0.030 |
