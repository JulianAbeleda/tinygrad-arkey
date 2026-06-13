# QK Semantic Schedule Microbench: 8B

Each candidate is compared against the current schedule for the same tensor.
Full decode is only justified for accepted rows that are also runtime-policy
supported.

## Summary

- candidates: `14`
- accepted: `2`
- ties: `1`
- rejected: `3`
- invalid: `8`
- full decode ready: `1`
- next decision: `run_full_policy_benchmark`

| id | status | gain % | current GB/s | candidate GB/s | full decode | reasons |
|---|---|---:|---:|---:|---:|---|
| `001-ffn-gate-blk-0-ffn-gate-weight-direct-out` | `tie` | -1.84 | 205.35 | 201.57 | `False` | within tie_band=0.030 |
| `002-ffn-gate-blk-0-ffn-gate-weight-row-upcast2` | `reject` | -4.53 | 215.42 | 205.67 | `True` | below min_gain=0.030 |
| `003-ffn-gate-blk-0-ffn-gate-weight-reduce-unroll4` | `invalid` | n/a | 213.87 | n/a | `True` | candidate status=compile-fail |
| `004-ffn-gate-blk-0-ffn-gate-weight-two-dim-local4` | `invalid` | n/a | 199.92 | n/a | `True` | candidate status=illegal-opt |
| `005-ffn-down-blk-4-ffn-down-weight-row-upcast2` | `reject` | -26.58 | 266.71 | 195.82 | `True` | below min_gain=0.030 |
| `006-ffn-down-blk-4-ffn-down-weight-reduce-unroll4` | `invalid` | n/a | 270.54 | n/a | `True` | candidate status=compile-fail |
| `007-ffn-down-blk-4-ffn-down-weight-two-dim-local4` | `reject` | -23.69 | 263.44 | 201.02 | `True` | below min_gain=0.030 |
| `008-attn-q-blk-0-attn-q-weight-direct-out` | `accept` | 4.37 | 68.72 | 71.72 | `False` | none |
| `009-attn-q-blk-0-attn-q-weight-row-upcast2` | `accept` | 4.93 | 65.58 | 68.81 | `True` | none |
| `010-attn-q-blk-0-attn-q-weight-reduce-unroll4` | `invalid` | n/a | 66.09 | n/a | `True` | candidate status=compile-fail |
| `011-attn-q-blk-0-attn-q-weight-two-dim-local4` | `invalid` | n/a | 69.71 | n/a | `True` | candidate status=illegal-opt |
| `012-ffn-down-blk-0-ffn-down-weight-row-upcast2` | `invalid` | n/a | n/a | n/a | `True` | candidate status=wrong |
| `013-ffn-down-blk-0-ffn-down-weight-reduce-unroll4` | `invalid` | n/a | n/a | n/a | `True` | candidate status=illegal-opt |
| `014-ffn-down-blk-0-ffn-down-weight-two-dim-local4` | `invalid` | n/a | n/a | n/a | `True` | candidate status=illegal-opt |
