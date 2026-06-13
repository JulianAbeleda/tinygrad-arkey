# QK Semantic Codegen Microbench: 14B

Each candidate is compared against the current schedule for the same tensor.
Full decode is only justified for accepted rows that are also runtime-policy
supported.

## Summary

- candidates: `4`
- accepted: `0`
- ties: `2`
- rejected: `2`
- invalid: `0`
- full decode ready: `0`
- next decision: `semantic_codegen_frontier_blocked`

| id | status | gain % | current GB/s | candidate GB/s | full decode | reasons |
|---|---|---:|---:|---:|---:|---|
| `001-ffn-gate-blk-0-ffn-gate-weight-direct-out-tensor` | `tie` | 2.41 | 357.98 | 366.62 | `True` | within tie_band=0.030 |
| `002-ffn-down-blk-5-ffn-down-weight-direct-out-tensor` | `tie` | 1.97 | 355.80 | 362.81 | `True` | within tie_band=0.030 |
| `003-attn-q-blk-0-attn-q-weight-direct-out-tensor` | `reject` | -4.29 | 109.77 | 105.06 | `True` | below min_gain=0.030 |
| `004-attn-k-blk-0-attn-k-weight-direct-out-tensor` | `reject` | -55.37 | 48.13 | 21.48 | `True` | below min_gain=0.030 |
