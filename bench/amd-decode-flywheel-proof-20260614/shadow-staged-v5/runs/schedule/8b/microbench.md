# QK Semantic Schedule Microbench: 8B

Each candidate is compared against the current schedule for the same tensor.
Full decode is only justified for accepted rows that are also runtime-policy
supported.

## Summary

- candidates: `32`
- raw accepts: `5`
- ties: `10`
- rejected: `1`
- invalid: `16`
- full decode ready: `2`
- next decision: `run_full_policy_benchmark`

| id | status | gain % | current GB/s | candidate GB/s | full decode | reasons |
|---|---|---:|---:|---:|---:|---|
| `001-attn-q-blk-16-attn-q-weight-direct-out` | `raw_accept` | 5.01 | 64.92 | 68.17 | `False` | requires confirmation before promotion |
| `002-attn-q-blk-16-attn-q-weight-row-upcast2` | `raw_accept` | 11.29 | 64.39 | 71.66 | `True` | requires confirmation before promotion |
| `003-attn-q-blk-16-attn-q-weight-reduce-unroll4` | `invalid` | n/a | 68.73 | n/a | `True` | candidate status=compile-fail |
| `004-attn-q-blk-16-attn-q-weight-two-dim-local4` | `invalid` | n/a | 71.22 | n/a | `True` | candidate status=illegal-opt |
| `005-attn-q-blk-17-attn-q-weight-direct-out` | `tie` | -1.09 | 70.00 | 69.24 | `False` | within tie_band=0.030 |
| `006-attn-q-blk-17-attn-q-weight-row-upcast2` | `tie` | -0.73 | 70.25 | 69.74 | `True` | within tie_band=0.030 |
| `007-attn-q-blk-17-attn-q-weight-reduce-unroll4` | `invalid` | n/a | 69.98 | n/a | `True` | candidate status=compile-fail |
| `008-attn-q-blk-17-attn-q-weight-two-dim-local4` | `invalid` | n/a | 69.09 | n/a | `True` | candidate status=illegal-opt |
| `009-attn-q-blk-18-attn-q-weight-direct-out` | `tie` | -0.19 | 68.41 | 68.28 | `False` | within tie_band=0.030 |
| `010-attn-q-blk-18-attn-q-weight-row-upcast2` | `reject` | -5.19 | 69.40 | 65.80 | `True` | below min_gain=0.030 |
| `011-attn-q-blk-18-attn-q-weight-reduce-unroll4` | `invalid` | n/a | 70.83 | n/a | `True` | candidate status=compile-fail |
| `012-attn-q-blk-18-attn-q-weight-two-dim-local4` | `invalid` | n/a | 68.03 | n/a | `True` | candidate status=illegal-opt |
| `013-ffn-gate-blk-22-ffn-gate-weight-direct-out` | `raw_accept` | 10.16 | 193.97 | 213.67 | `False` | requires confirmation before promotion |
| `014-ffn-gate-blk-22-ffn-gate-weight-row-upcast2` | `tie` | 2.79 | 198.97 | 204.53 | `True` | within tie_band=0.030 |
| `015-ffn-gate-blk-22-ffn-gate-weight-reduce-unroll4` | `invalid` | n/a | 207.43 | n/a | `True` | candidate status=compile-fail |
| `016-ffn-gate-blk-22-ffn-gate-weight-two-dim-local4` | `invalid` | n/a | 210.09 | n/a | `True` | candidate status=illegal-opt |
| `017-ffn-gate-blk-23-ffn-gate-weight-direct-out` | `tie` | 0.06 | 208.94 | 209.06 | `False` | within tie_band=0.030 |
| `018-ffn-gate-blk-23-ffn-gate-weight-row-upcast2` | `tie` | 1.64 | 203.59 | 206.92 | `True` | within tie_band=0.030 |
| `019-ffn-gate-blk-23-ffn-gate-weight-reduce-unroll4` | `invalid` | n/a | 200.50 | n/a | `True` | candidate status=compile-fail |
| `020-ffn-gate-blk-23-ffn-gate-weight-two-dim-local4` | `invalid` | n/a | 203.06 | n/a | `True` | candidate status=illegal-opt |
| `021-ffn-gate-blk-24-ffn-gate-weight-direct-out` | `tie` | -0.03 | 207.79 | 207.72 | `False` | within tie_band=0.030 |
| `022-ffn-gate-blk-24-ffn-gate-weight-row-upcast2` | `tie` | 0.03 | 208.56 | 208.63 | `True` | within tie_band=0.030 |
| `023-ffn-gate-blk-24-ffn-gate-weight-reduce-unroll4` | `invalid` | n/a | 203.76 | n/a | `True` | candidate status=compile-fail |
| `024-ffn-gate-blk-24-ffn-gate-weight-two-dim-local4` | `invalid` | n/a | 207.14 | n/a | `True` | candidate status=illegal-opt |
| `025-ffn-gate-blk-25-ffn-gate-weight-direct-out` | `tie` | 0.57 | 207.14 | 208.33 | `False` | within tie_band=0.030 |
| `026-ffn-gate-blk-25-ffn-gate-weight-row-upcast2` | `tie` | 0.27 | 196.54 | 197.08 | `True` | within tie_band=0.030 |
| `027-ffn-gate-blk-25-ffn-gate-weight-reduce-unroll4` | `invalid` | n/a | 202.07 | n/a | `True` | candidate status=compile-fail |
| `028-ffn-gate-blk-25-ffn-gate-weight-two-dim-local4` | `invalid` | n/a | 210.62 | n/a | `True` | candidate status=illegal-opt |
| `029-ffn-gate-blk-26-ffn-gate-weight-direct-out` | `raw_accept` | 3.40 | 204.53 | 211.48 | `False` | requires confirmation before promotion |
| `030-ffn-gate-blk-26-ffn-gate-weight-row-upcast2` | `raw_accept` | 4.64 | 202.27 | 211.65 | `True` | requires confirmation before promotion |
| `031-ffn-gate-blk-26-ffn-gate-weight-reduce-unroll4` | `invalid` | n/a | 195.99 | n/a | `True` | candidate status=compile-fail |
| `032-ffn-gate-blk-26-ffn-gate-weight-two-dim-local4` | `invalid` | n/a | 198.53 | n/a | `True` | candidate status=illegal-opt |
