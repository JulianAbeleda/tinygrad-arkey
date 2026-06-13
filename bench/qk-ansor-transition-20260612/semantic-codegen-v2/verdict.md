# QK Semantic Codegen v2 Verdict

This is the 8B/14B gate for Family B v2a: exact-tensor Q4_K ffn_down
row-grouped partial GEMV. 32B is intentionally excluded unless both
target models show promise.

## Summary

- overall decision: `semantic_codegen_v2_rejected`
- microbench rows: `4`
- raw microbench accepts: `0`
- strong raw microbench accepts: `0`
- microbench invalid: `1`
- full-decode confirmed accepts: `0`
- run 32B: `False`

Reasons:

- 8B no raw accepts (2 rejects)
- 14B no raw accepts (1 rejects, 1 invalid)
- full decode and 32B skipped because the 8B/14B microbench gate produced no raw accepts
- row grouping is a rejected Family B v2a mechanism, not a runtime candidate

## Models

| model | row | status | gain % | current GB/s | candidate GB/s | reasons |
|---|---|---|---:|---:|---:|---|
| 8B | `001-ffn-down-blk-4-ffn-down-weight-row-group2` | `reject` | -31.03 | 267.69 | 184.63 | below min_gain=0.030 |
| 8B | `002-ffn-down-blk-4-ffn-down-weight-row-group4` | `reject` | -71.54 | 265.79 | 75.64 | below min_gain=0.030 |
| 14B | `001-ffn-down-blk-5-ffn-down-weight-row-group2` | `reject` | -52.59 | 366.45 | 173.74 | below min_gain=0.030 |
| 14B | `002-ffn-down-blk-5-ffn-down-weight-row-group4` | `invalid` | n/a | 364.73 | n/a | candidate status=illegal-opt |

## Interpretation

Family B v2a does not justify runtime installation. The row-grouped Q4_K
`ffn_down` mechanism regressed both target models and produced no raw
accepts. The next step is not to broaden this same row-group surface.
