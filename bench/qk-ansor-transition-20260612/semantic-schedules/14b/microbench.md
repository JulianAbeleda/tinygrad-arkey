# QK Semantic Schedule Microbench: 14B

Each candidate is compared against the current schedule for the same tensor.
Full decode is only justified for accepted rows that are also runtime-policy
supported.

## Summary

- candidates: `14`
- accepted: `1`
- ties: `3`
- rejected: `1`
- invalid: `9`
- full decode ready: `1`
- next decision: `run_full_policy_benchmark`

| id | status | gain % | current GB/s | candidate GB/s | full decode | reasons |
|---|---|---:|---:|---:|---:|---|
| `001-ffn-gate-blk-0-ffn-gate-weight-direct-out` | `tie` | 2.69 | 360.67 | 370.36 | `False` | within tie_band=0.030 |
| `002-ffn-gate-blk-0-ffn-gate-weight-row-upcast2` | `tie` | 2.66 | 368.96 | 378.77 | `True` | within tie_band=0.030 |
| `003-ffn-gate-blk-0-ffn-gate-weight-reduce-unroll4` | `invalid` | n/a | 358.72 | n/a | `True` | candidate status=compile-fail |
| `004-ffn-gate-blk-0-ffn-gate-weight-two-dim-local4` | `invalid` | n/a | 360.55 | n/a | `True` | candidate status=illegal-opt |
| `005-ffn-down-blk-5-ffn-down-weight-row-upcast2` | `reject` | -41.89 | 361.55 | 210.11 | `True` | below min_gain=0.030 |
| `006-ffn-down-blk-5-ffn-down-weight-reduce-unroll4` | `invalid` | n/a | 352.45 | n/a | `True` | candidate status=compile-fail |
| `007-ffn-down-blk-5-ffn-down-weight-two-dim-local4` | `invalid` | n/a | 352.61 | n/a | `True` | candidate status=illegal-opt |
| `008-attn-q-blk-0-attn-q-weight-direct-out` | `tie` | -1.10 | 107.18 | 106.00 | `False` | within tie_band=0.030 |
| `009-attn-q-blk-0-attn-q-weight-row-upcast2` | `accept` | 5.56 | 101.01 | 106.63 | `True` | none |
| `010-attn-q-blk-0-attn-q-weight-reduce-unroll4` | `invalid` | n/a | 107.59 | n/a | `True` | candidate status=compile-fail |
| `011-attn-q-blk-0-attn-q-weight-two-dim-local4` | `invalid` | n/a | 104.83 | n/a | `True` | candidate status=illegal-opt |
| `012-ffn-down-blk-0-ffn-down-weight-row-upcast2` | `invalid` | n/a | n/a | n/a | `True` | candidate status=wrong |
| `013-ffn-down-blk-0-ffn-down-weight-reduce-unroll4` | `invalid` | n/a | n/a | n/a | `True` | candidate status=illegal-opt |
| `014-ffn-down-blk-0-ffn-down-weight-two-dim-local4` | `invalid` | n/a | n/a | n/a | `True` | candidate status=illegal-opt |
