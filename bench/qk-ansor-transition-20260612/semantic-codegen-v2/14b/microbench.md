# QK Semantic Codegen Microbench: 14B

Each candidate is compared against the current schedule for the same tensor.
Full decode is only justified for accepted rows that are also runtime-policy
supported.

## Summary

- candidates: `2`
- raw accepts: `0`
- ties: `0`
- rejected: `1`
- invalid: `1`
- full decode ready: `0`
- next decision: `semantic_codegen_frontier_blocked`

| id | status | gain % | current GB/s | candidate GB/s | full decode | reasons |
|---|---|---:|---:|---:|---:|---|
| `001-ffn-down-blk-5-ffn-down-weight-row-group2` | `reject` | -52.59 | 366.45 | 173.74 | `False` | below min_gain=0.030 |
| `002-ffn-down-blk-5-ffn-down-weight-row-group4` | `invalid` | n/a | 364.73 | n/a | `False` | candidate status=illegal-opt |
