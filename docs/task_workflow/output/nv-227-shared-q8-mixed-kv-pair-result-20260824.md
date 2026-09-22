# NV shared-Q8 mixed Q4-K/Q6-V pair promotion

Date: 2026-08-24
GPU: RTX 5090 (`NV sm_120`), Qwen3-8B-Q4_K_M decode

## Outcome

The shared-Q8 mixed dual producer is promoted for the eight eligible Q4-K /
Q6-V blocks. It is bit-exact, removes eight launches, has fully accounted
production topology, and measured a positive `3.126 us/token` full-wall
midpoint recovery. `TINYGRAD_SHARED_Q8_Q4Q6_KV_PAIR_DISABLE=1` restores the
two separate consumers without changing the nine existing Q4/Q4 pairs.

The earlier no-go label applied a stricter requirement that the candidate beat
each individual drifting control. The promotion policy now accepts a positive
full-wall midpoint when exactness and topology accounting are complete. No
measurement changed; only its adjudication changed.

## Evidence

The native 500-pass x 9 gate compares one direct cooperative Q4 consumer plus
one direct Q6 consumer against the mixed producer:

```text
2048 fp32 words mismatched       0
control median             4.536576 us/pair
candidate median           3.468160 us/pair
recovery                   1.068416 us/pair
eight-pair ceiling         8.547328 us/token
candidate resources        40 registers, 32 B smem, one barrier, no spills
```

The production profile changes only the intended population:

| row | control | candidate | delta |
| --- | ---: | ---: | ---: |
| nodes | 462 | 454 | -8 |
| node sum | 4270.768 us | 4264.640 us | -6.128 us |
| device union | 4258.000 us | 4246.500 us | -11.500 us |

The full-wall bracket is token-exact:

```text
control A          4476.361 us/token
candidate          4479.870 us/token
control C          4489.632 us/token
control midpoint   4482.996 us/token
recovery              3.126 us/token
```

The candidate is slower than opening control A by `3.510 us`, but faster than
the A/C midpoint by `3.126 us`. The node-sum and union changes independently
confirm the accounted direction; only the smaller measured wall recovery is
booked.

## Current conservative booking

```text
4451.459844 - 3.126000 = 4448.333844 us/token
throughput                       = 224.803271 tok/s
remaining to 227                = 43.047500 us/token
                                = 2.196729 tok/s
```

Loader census: default has 17 shared-Q8 leases, nine Q4/Q4 pair producers and
eight mixed producers. Rollback retains the nine Q4/Q4 pairs and removes all
eight mixed producers. Focused tests: `34 passed`.

Evidence is in
`docs/task_workflow/evidence/nv-227-shared-q8-mixed-kv-pair-20260824/` and
`docs/task_workflow/evidence/nv-ranked-parity-campaign-20260824/06-shared-q8-mixed-pair-loader-census.json`.

Verdict: `PROMOTED_SHARED_Q8_MIXED_Q4Q6_PAIR_BOOK_3_126_US`.
