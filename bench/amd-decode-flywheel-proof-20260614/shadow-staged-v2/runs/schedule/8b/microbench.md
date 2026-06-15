# QK Semantic Schedule Microbench: 8B

Each candidate is compared against the current schedule for the same tensor.
Full decode is only justified for accepted rows that are also runtime-policy
supported.

## Summary

- candidates: `40`
- raw accepts: `7`
- ties: `12`
- rejected: `1`
- invalid: `20`
- full decode ready: `4`
- next decision: `run_full_policy_benchmark`

| id | status | gain % | current GB/s | candidate GB/s | full decode | reasons |
|---|---|---:|---:|---:|---:|---|
| `001-attn-q-blk-3-attn-q-weight-direct-out` | `tie` | 0.13 | 69.30 | 69.39 | `False` | within tie_band=0.030 |
| `002-attn-q-blk-3-attn-q-weight-row-upcast2` | `tie` | -2.52 | 70.18 | 68.41 | `True` | within tie_band=0.030 |
| `003-attn-q-blk-3-attn-q-weight-reduce-unroll4` | `invalid` | n/a | 65.62 | n/a | `True` | candidate status=compile-fail |
| `004-attn-q-blk-3-attn-q-weight-two-dim-local4` | `invalid` | n/a | 71.03 | n/a | `True` | candidate status=illegal-opt |
| `005-attn-q-blk-4-attn-q-weight-direct-out` | `tie` | 0.14 | 69.01 | 69.11 | `False` | within tie_band=0.030 |
| `006-attn-q-blk-4-attn-q-weight-row-upcast2` | `tie` | 0.70 | 70.36 | 70.85 | `True` | within tie_band=0.030 |
| `007-attn-q-blk-4-attn-q-weight-reduce-unroll4` | `invalid` | n/a | 70.07 | n/a | `True` | candidate status=compile-fail |
| `008-attn-q-blk-4-attn-q-weight-two-dim-local4` | `invalid` | n/a | 65.60 | n/a | `True` | candidate status=illegal-opt |
| `009-attn-q-blk-5-attn-q-weight-direct-out` | `reject` | -5.25 | 70.88 | 67.16 | `False` | below min_gain=0.030 |
| `010-attn-q-blk-5-attn-q-weight-row-upcast2` | `tie` | -1.60 | 69.59 | 68.48 | `True` | within tie_band=0.030 |
| `011-attn-q-blk-5-attn-q-weight-reduce-unroll4` | `invalid` | n/a | 66.96 | n/a | `True` | candidate status=compile-fail |
| `012-attn-q-blk-5-attn-q-weight-two-dim-local4` | `invalid` | n/a | 69.86 | n/a | `True` | candidate status=illegal-opt |
| `013-attn-q-blk-6-attn-q-weight-direct-out` | `raw_accept` | 4.06 | 67.16 | 69.89 | `False` | requires confirmation before promotion |
| `014-attn-q-blk-6-attn-q-weight-row-upcast2` | `raw_accept` | 12.30 | 62.46 | 70.14 | `True` | requires confirmation before promotion |
| `015-attn-q-blk-6-attn-q-weight-reduce-unroll4` | `invalid` | n/a | 69.51 | n/a | `True` | candidate status=compile-fail |
| `016-attn-q-blk-6-attn-q-weight-two-dim-local4` | `invalid` | n/a | 73.58 | n/a | `True` | candidate status=illegal-opt |
| `017-attn-q-blk-7-attn-q-weight-direct-out` | `raw_accept` | 5.25 | 68.92 | 72.54 | `False` | requires confirmation before promotion |
| `018-attn-q-blk-7-attn-q-weight-row-upcast2` | `raw_accept` | 4.07 | 68.52 | 71.31 | `True` | requires confirmation before promotion |
| `019-attn-q-blk-7-attn-q-weight-reduce-unroll4` | `invalid` | n/a | 69.95 | n/a | `True` | candidate status=compile-fail |
| `020-attn-q-blk-7-attn-q-weight-two-dim-local4` | `invalid` | n/a | 67.49 | n/a | `True` | candidate status=illegal-opt |
| `021-attn-q-blk-8-attn-q-weight-direct-out` | `tie` | 1.74 | 70.10 | 71.32 | `False` | within tie_band=0.030 |
| `022-attn-q-blk-8-attn-q-weight-row-upcast2` | `raw_accept` | 3.59 | 66.38 | 68.76 | `True` | requires confirmation before promotion |
| `023-attn-q-blk-8-attn-q-weight-reduce-unroll4` | `invalid` | n/a | 69.00 | n/a | `True` | candidate status=compile-fail |
| `024-attn-q-blk-8-attn-q-weight-two-dim-local4` | `invalid` | n/a | 66.84 | n/a | `True` | candidate status=illegal-opt |
| `025-attn-q-blk-9-attn-q-weight-direct-out` | `raw_accept` | 9.31 | 64.32 | 70.31 | `False` | requires confirmation before promotion |
| `026-attn-q-blk-9-attn-q-weight-row-upcast2` | `tie` | 0.95 | 70.29 | 70.96 | `True` | within tie_band=0.030 |
| `027-attn-q-blk-9-attn-q-weight-reduce-unroll4` | `invalid` | n/a | 68.88 | n/a | `True` | candidate status=compile-fail |
| `028-attn-q-blk-9-attn-q-weight-two-dim-local4` | `invalid` | n/a | 71.01 | n/a | `True` | candidate status=illegal-opt |
| `029-ffn-gate-blk-9-ffn-gate-weight-direct-out` | `tie` | -2.80 | 211.04 | 205.14 | `False` | within tie_band=0.030 |
| `030-ffn-gate-blk-9-ffn-gate-weight-row-upcast2` | `raw_accept` | 4.45 | 205.62 | 214.78 | `True` | requires confirmation before promotion |
| `031-ffn-gate-blk-9-ffn-gate-weight-reduce-unroll4` | `invalid` | n/a | 215.39 | n/a | `True` | candidate status=compile-fail |
| `032-ffn-gate-blk-9-ffn-gate-weight-two-dim-local4` | `invalid` | n/a | 211.17 | n/a | `True` | candidate status=illegal-opt |
| `033-ffn-gate-blk-10-ffn-gate-weight-direct-out` | `tie` | 0.70 | 207.95 | 209.40 | `False` | within tie_band=0.030 |
| `034-ffn-gate-blk-10-ffn-gate-weight-row-upcast2` | `tie` | 0.68 | 208.87 | 210.29 | `True` | within tie_band=0.030 |
| `035-ffn-gate-blk-10-ffn-gate-weight-reduce-unroll4` | `invalid` | n/a | 207.15 | n/a | `True` | candidate status=compile-fail |
| `036-ffn-gate-blk-10-ffn-gate-weight-two-dim-local4` | `invalid` | n/a | 205.53 | n/a | `True` | candidate status=illegal-opt |
| `037-ffn-gate-blk-11-ffn-gate-weight-direct-out` | `tie` | 2.06 | 209.28 | 213.60 | `False` | within tie_band=0.030 |
| `038-ffn-gate-blk-11-ffn-gate-weight-row-upcast2` | `tie` | 2.39 | 207.40 | 212.36 | `True` | within tie_band=0.030 |
| `039-ffn-gate-blk-11-ffn-gate-weight-reduce-unroll4` | `invalid` | n/a | 204.25 | n/a | `True` | candidate status=compile-fail |
| `040-ffn-gate-blk-11-ffn-gate-weight-two-dim-local4` | `invalid` | n/a | 214.67 | n/a | `True` | candidate status=illegal-opt |
