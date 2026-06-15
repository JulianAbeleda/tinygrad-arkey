# QK Semantic Schedule Microbench: 8B

Each candidate is compared against the current schedule for the same tensor.
Full decode is only justified for accepted rows that are also runtime-policy
supported.

## Summary

- candidates: `32`
- raw accepts: `4`
- ties: `7`
- rejected: `5`
- invalid: `16`
- full decode ready: `2`
- next decision: `run_full_policy_benchmark`

| id | status | gain % | current GB/s | candidate GB/s | full decode | reasons |
|---|---|---:|---:|---:|---:|---|
| `001-attn-q-blk-13-attn-q-weight-direct-out` | `reject` | -3.80 | 70.26 | 67.59 | `False` | below min_gain=0.030 |
| `002-attn-q-blk-13-attn-q-weight-row-upcast2` | `tie` | 0.84 | 70.36 | 70.95 | `True` | within tie_band=0.030 |
| `003-attn-q-blk-13-attn-q-weight-reduce-unroll4` | `invalid` | n/a | 70.40 | n/a | `True` | candidate status=compile-fail |
| `004-attn-q-blk-13-attn-q-weight-two-dim-local4` | `invalid` | n/a | 69.57 | n/a | `True` | candidate status=illegal-opt |
| `005-attn-q-blk-14-attn-q-weight-direct-out` | `reject` | -10.50 | 72.11 | 64.54 | `False` | below min_gain=0.030 |
| `006-attn-q-blk-14-attn-q-weight-row-upcast2` | `tie` | 1.35 | 68.06 | 68.98 | `True` | within tie_band=0.030 |
| `007-attn-q-blk-14-attn-q-weight-reduce-unroll4` | `invalid` | n/a | 66.17 | n/a | `True` | candidate status=compile-fail |
| `008-attn-q-blk-14-attn-q-weight-two-dim-local4` | `invalid` | n/a | 63.97 | n/a | `True` | candidate status=illegal-opt |
| `009-attn-q-blk-15-attn-q-weight-direct-out` | `raw_accept` | 3.09 | 67.71 | 69.80 | `False` | requires confirmation before promotion |
| `010-attn-q-blk-15-attn-q-weight-row-upcast2` | `tie` | -2.19 | 72.05 | 70.47 | `True` | within tie_band=0.030 |
| `011-attn-q-blk-15-attn-q-weight-reduce-unroll4` | `invalid` | n/a | 70.34 | n/a | `True` | candidate status=compile-fail |
| `012-attn-q-blk-15-attn-q-weight-two-dim-local4` | `invalid` | n/a | 68.10 | n/a | `True` | candidate status=illegal-opt |
| `013-ffn-gate-blk-17-ffn-gate-weight-direct-out` | `tie` | 0.94 | 203.80 | 205.71 | `False` | within tie_band=0.030 |
| `014-ffn-gate-blk-17-ffn-gate-weight-row-upcast2` | `reject` | -5.85 | 208.38 | 196.19 | `True` | below min_gain=0.030 |
| `015-ffn-gate-blk-17-ffn-gate-weight-reduce-unroll4` | `invalid` | n/a | 205.30 | n/a | `True` | candidate status=compile-fail |
| `016-ffn-gate-blk-17-ffn-gate-weight-two-dim-local4` | `invalid` | n/a | 212.73 | n/a | `True` | candidate status=illegal-opt |
| `017-ffn-gate-blk-18-ffn-gate-weight-direct-out` | `raw_accept` | 4.49 | 195.01 | 203.77 | `False` | requires confirmation before promotion |
| `018-ffn-gate-blk-18-ffn-gate-weight-row-upcast2` | `raw_accept` | 4.37 | 197.71 | 206.35 | `True` | requires confirmation before promotion |
| `019-ffn-gate-blk-18-ffn-gate-weight-reduce-unroll4` | `invalid` | n/a | 202.75 | n/a | `True` | candidate status=compile-fail |
| `020-ffn-gate-blk-18-ffn-gate-weight-two-dim-local4` | `invalid` | n/a | 207.20 | n/a | `True` | candidate status=illegal-opt |
| `021-ffn-gate-blk-19-ffn-gate-weight-direct-out` | `tie` | 0.59 | 210.19 | 211.43 | `False` | within tie_band=0.030 |
| `022-ffn-gate-blk-19-ffn-gate-weight-row-upcast2` | `raw_accept` | 11.23 | 192.60 | 214.23 | `True` | requires confirmation before promotion |
| `023-ffn-gate-blk-19-ffn-gate-weight-reduce-unroll4` | `invalid` | n/a | 204.75 | n/a | `True` | candidate status=compile-fail |
| `024-ffn-gate-blk-19-ffn-gate-weight-two-dim-local4` | `invalid` | n/a | 209.28 | n/a | `True` | candidate status=illegal-opt |
| `025-ffn-gate-blk-20-ffn-gate-weight-direct-out` | `tie` | -2.89 | 209.65 | 203.60 | `False` | within tie_band=0.030 |
| `026-ffn-gate-blk-20-ffn-gate-weight-row-upcast2` | `tie` | -0.94 | 205.95 | 204.02 | `True` | within tie_band=0.030 |
| `027-ffn-gate-blk-20-ffn-gate-weight-reduce-unroll4` | `invalid` | n/a | 210.30 | n/a | `True` | candidate status=compile-fail |
| `028-ffn-gate-blk-20-ffn-gate-weight-two-dim-local4` | `invalid` | n/a | 209.27 | n/a | `True` | candidate status=illegal-opt |
| `029-ffn-gate-blk-21-ffn-gate-weight-direct-out` | `reject` | -8.34 | 211.92 | 194.25 | `False` | below min_gain=0.030 |
| `030-ffn-gate-blk-21-ffn-gate-weight-row-upcast2` | `reject` | -5.73 | 213.15 | 200.94 | `True` | below min_gain=0.030 |
| `031-ffn-gate-blk-21-ffn-gate-weight-reduce-unroll4` | `invalid` | n/a | 205.92 | n/a | `True` | candidate status=compile-fail |
| `032-ffn-gate-blk-21-ffn-gate-weight-two-dim-local4` | `invalid` | n/a | 200.89 | n/a | `True` | candidate status=illegal-opt |
