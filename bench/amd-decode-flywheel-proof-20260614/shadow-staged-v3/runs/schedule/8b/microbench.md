# QK Semantic Schedule Microbench: 8B

Each candidate is compared against the current schedule for the same tensor.
Full decode is only justified for accepted rows that are also runtime-policy
supported.

## Summary

- candidates: `32`
- raw accepts: `4`
- ties: `10`
- rejected: `2`
- invalid: `16`
- full decode ready: `3`
- next decision: `run_full_policy_benchmark`

| id | status | gain % | current GB/s | candidate GB/s | full decode | reasons |
|---|---|---:|---:|---:|---:|---|
| `001-attn-q-blk-10-attn-q-weight-direct-out` | `tie` | 2.06 | 69.75 | 71.19 | `False` | within tie_band=0.030 |
| `002-attn-q-blk-10-attn-q-weight-row-upcast2` | `raw_accept` | 5.00 | 66.79 | 70.13 | `True` | requires confirmation before promotion |
| `003-attn-q-blk-10-attn-q-weight-reduce-unroll4` | `invalid` | n/a | 69.09 | n/a | `True` | candidate status=compile-fail |
| `004-attn-q-blk-10-attn-q-weight-two-dim-local4` | `invalid` | n/a | 51.62 | n/a | `True` | candidate status=illegal-opt |
| `005-attn-q-blk-11-attn-q-weight-direct-out` | `raw_accept` | 6.25 | 64.66 | 68.70 | `False` | requires confirmation before promotion |
| `006-attn-q-blk-11-attn-q-weight-row-upcast2` | `tie` | 0.21 | 68.29 | 68.43 | `True` | within tie_band=0.030 |
| `007-attn-q-blk-11-attn-q-weight-reduce-unroll4` | `invalid` | n/a | 67.52 | n/a | `True` | candidate status=compile-fail |
| `008-attn-q-blk-11-attn-q-weight-two-dim-local4` | `invalid` | n/a | 69.70 | n/a | `True` | candidate status=illegal-opt |
| `009-attn-q-blk-12-attn-q-weight-direct-out` | `tie` | -0.55 | 71.28 | 70.89 | `False` | within tie_band=0.030 |
| `010-attn-q-blk-12-attn-q-weight-row-upcast2` | `raw_accept` | 4.90 | 68.78 | 72.15 | `True` | requires confirmation before promotion |
| `011-attn-q-blk-12-attn-q-weight-reduce-unroll4` | `invalid` | n/a | 68.82 | n/a | `True` | candidate status=compile-fail |
| `012-attn-q-blk-12-attn-q-weight-two-dim-local4` | `invalid` | n/a | 70.18 | n/a | `True` | candidate status=illegal-opt |
| `013-ffn-gate-blk-12-ffn-gate-weight-direct-out` | `tie` | -2.30 | 211.01 | 206.15 | `False` | within tie_band=0.030 |
| `014-ffn-gate-blk-12-ffn-gate-weight-row-upcast2` | `reject` | -3.04 | 214.90 | 208.36 | `True` | below min_gain=0.030 |
| `015-ffn-gate-blk-12-ffn-gate-weight-reduce-unroll4` | `invalid` | n/a | 211.93 | n/a | `True` | candidate status=compile-fail |
| `016-ffn-gate-blk-12-ffn-gate-weight-two-dim-local4` | `invalid` | n/a | 200.23 | n/a | `True` | candidate status=illegal-opt |
| `017-ffn-gate-blk-13-ffn-gate-weight-direct-out` | `reject` | -3.21 | 206.53 | 199.90 | `False` | below min_gain=0.030 |
| `018-ffn-gate-blk-13-ffn-gate-weight-row-upcast2` | `raw_accept` | 6.16 | 204.73 | 217.35 | `True` | requires confirmation before promotion |
| `019-ffn-gate-blk-13-ffn-gate-weight-reduce-unroll4` | `invalid` | n/a | 210.66 | n/a | `True` | candidate status=compile-fail |
| `020-ffn-gate-blk-13-ffn-gate-weight-two-dim-local4` | `invalid` | n/a | 202.32 | n/a | `True` | candidate status=illegal-opt |
| `021-ffn-gate-blk-14-ffn-gate-weight-direct-out` | `tie` | -1.23 | 212.31 | 209.70 | `False` | within tie_band=0.030 |
| `022-ffn-gate-blk-14-ffn-gate-weight-row-upcast2` | `tie` | 1.21 | 207.35 | 209.86 | `True` | within tie_band=0.030 |
| `023-ffn-gate-blk-14-ffn-gate-weight-reduce-unroll4` | `invalid` | n/a | 210.87 | n/a | `True` | candidate status=compile-fail |
| `024-ffn-gate-blk-14-ffn-gate-weight-two-dim-local4` | `invalid` | n/a | 209.56 | n/a | `True` | candidate status=illegal-opt |
| `025-ffn-gate-blk-15-ffn-gate-weight-direct-out` | `tie` | -1.85 | 212.12 | 208.20 | `False` | within tie_band=0.030 |
| `026-ffn-gate-blk-15-ffn-gate-weight-row-upcast2` | `tie` | -2.13 | 213.61 | 209.05 | `True` | within tie_band=0.030 |
| `027-ffn-gate-blk-15-ffn-gate-weight-reduce-unroll4` | `invalid` | n/a | 213.47 | n/a | `True` | candidate status=compile-fail |
| `028-ffn-gate-blk-15-ffn-gate-weight-two-dim-local4` | `invalid` | n/a | 211.54 | n/a | `True` | candidate status=illegal-opt |
| `029-ffn-gate-blk-16-ffn-gate-weight-direct-out` | `tie` | -1.70 | 216.19 | 212.51 | `False` | within tie_band=0.030 |
| `030-ffn-gate-blk-16-ffn-gate-weight-row-upcast2` | `tie` | 0.44 | 211.27 | 212.20 | `True` | within tie_band=0.030 |
| `031-ffn-gate-blk-16-ffn-gate-weight-reduce-unroll4` | `invalid` | n/a | 205.42 | n/a | `True` | candidate status=compile-fail |
| `032-ffn-gate-blk-16-ffn-gate-weight-two-dim-local4` | `invalid` | n/a | 210.37 | n/a | `True` | candidate status=illegal-opt |
