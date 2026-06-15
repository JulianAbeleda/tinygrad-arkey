# QK Semantic Schedule Microbench: 8B

Each candidate is compared against the current schedule for the same tensor.
Full decode is only justified for accepted rows that are also runtime-policy
supported.

## Summary

- candidates: `16`
- raw accepts: `2`
- ties: `5`
- rejected: `1`
- invalid: `8`
- full decode ready: `2`
- next decision: `run_full_policy_benchmark`

| id | status | gain % | current GB/s | candidate GB/s | full decode | reasons |
|---|---|---:|---:|---:|---:|---|
| `001-ffn-gate-blk-7-ffn-gate-weight-direct-out` | `tie` | -1.75 | 213.69 | 209.94 | `False` | within tie_band=0.030 |
| `002-ffn-gate-blk-7-ffn-gate-weight-row-upcast2` | `tie` | -2.77 | 205.78 | 200.08 | `True` | within tie_band=0.030 |
| `003-ffn-gate-blk-7-ffn-gate-weight-reduce-unroll4` | `invalid` | n/a | 201.60 | n/a | `True` | candidate status=compile-fail |
| `004-ffn-gate-blk-7-ffn-gate-weight-two-dim-local4` | `invalid` | n/a | 194.05 | n/a | `True` | candidate status=illegal-opt |
| `005-ffn-gate-blk-8-ffn-gate-weight-direct-out` | `reject` | -3.42 | 214.07 | 206.74 | `False` | below min_gain=0.030 |
| `006-ffn-gate-blk-8-ffn-gate-weight-row-upcast2` | `tie` | 2.03 | 210.39 | 214.67 | `True` | within tie_band=0.030 |
| `007-ffn-gate-blk-8-ffn-gate-weight-reduce-unroll4` | `invalid` | n/a | 204.65 | n/a | `True` | candidate status=compile-fail |
| `008-ffn-gate-blk-8-ffn-gate-weight-two-dim-local4` | `invalid` | n/a | 206.51 | n/a | `True` | candidate status=illegal-opt |
| `009-attn-q-blk-1-attn-q-weight-direct-out` | `tie` | 2.46 | 69.06 | 70.76 | `False` | within tie_band=0.030 |
| `010-attn-q-blk-1-attn-q-weight-row-upcast2` | `raw_accept` | 4.55 | 65.55 | 68.53 | `True` | requires confirmation before promotion |
| `011-attn-q-blk-1-attn-q-weight-reduce-unroll4` | `invalid` | n/a | 71.73 | n/a | `True` | candidate status=compile-fail |
| `012-attn-q-blk-1-attn-q-weight-two-dim-local4` | `invalid` | n/a | 65.25 | n/a | `True` | candidate status=illegal-opt |
| `013-attn-q-blk-2-attn-q-weight-direct-out` | `tie` | 1.73 | 68.36 | 69.54 | `False` | within tie_band=0.030 |
| `014-attn-q-blk-2-attn-q-weight-row-upcast2` | `raw_accept` | 6.87 | 65.17 | 69.65 | `True` | requires confirmation before promotion |
| `015-attn-q-blk-2-attn-q-weight-reduce-unroll4` | `invalid` | n/a | 66.30 | n/a | `True` | candidate status=compile-fail |
| `016-attn-q-blk-2-attn-q-weight-two-dim-local4` | `invalid` | n/a | 70.62 | n/a | `True` | candidate status=illegal-opt |
