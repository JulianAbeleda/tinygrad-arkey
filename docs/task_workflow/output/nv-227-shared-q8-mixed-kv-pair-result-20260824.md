# NV 227 push: shared-Q8 mixed Q4-K/Q6-V pair result

Date: 2026-08-24
Base checkpoint: `ae72610ba`
Verdict: `NO_GO_WALL`

## Result

The shared-Q8 mixed dual producer is exact and removes the intended eight
launches, but its production wall effect is below resolution and it is not
promoted. The conservative endpoint remains `4.515396 ms/token = 221.465
tok/s`, with `110.109 us/token` (`5.535 tok/s`) left to 227.

## Native gate

The 500-pass x 9 CUDA-event gate compares one direct cooperative Q4 consumer
plus one direct Q6 consumer against one Q4/Q6 producer.

```text
2048 fp32 words mismatched       0
control median             4.536576 us/pair
candidate median           3.468160 us/pair
recovery                   1.068416 us/pair
eight-pair ceiling         8.547328 us/token
candidate resources        40 registers, 32 B smem, one barrier, no spills
```

## Production profile

All tokens match and the route changes only the intended population:

| row | control | candidate | delta |
| --- | ---: | ---: | ---: |
| nodes | 462 | 454 | -8 |
| node sum | 4270.768 us | 4264.640 us | -6.128 us |
| device union | 4258.000 us | 4246.500 us | -11.500 us |

The candidate graph contains eight
`q4k_q6k_warp_coop_q8_dp4a_pair_direct_1024_4096` calls. The nine promoted
Q4/Q4 shared pair producers remain unchanged.

## Wall bracket

```text
control A  4.476361 ms/token
candidate  4.479870 ms/token
control C  4.489632 ms/token
midpoint   4.482996 ms/token
midpoint recovery 3.126 us/token
```

The candidate loses to Control A by `3.510 us/token`, so the strict verdict is
`NO_GO_WALL` despite the positive midpoint. Re-bracketing is not justified:
the production node-sum effect is only `6.128 us/token` and cannot materially
close the 227 gap.

No route policy enables the mixed producer. The emitter, closed admission
flag, microgate, and qualification variant remain as a reproducible research
record.
