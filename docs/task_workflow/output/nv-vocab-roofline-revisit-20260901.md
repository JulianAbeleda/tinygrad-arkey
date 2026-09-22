# NV vocabulary roofline revisit

Date: 2026-09-01

## Byte floor

The Q6_K vocabulary matrix contains exactly `510,504,960` bytes:

```text
151936 rows * 16 Q6_K blocks/row * 105 halfwords/block * 2 bytes
```

The same-machine streaming-read authority is `1701.3 GB/s` (the 4 GiB result in
`extra/llm_research/microbench/README.md`).  The corresponding practical weight-only
roofline is therefore:

```text
510,504,960 / 1.7013e12 = 300.064 us
```

The `1792 GB/s` sheet ceiling gives `284.880 us`, but the streaming benchmark proves
that this is not the appropriate investment denominator.  The machine leaves about
`15.184 us` between sheet and measured streaming ceilings even for a pure streaming
load kernel.

## Current position

Common-protocol retained bodies:

| route | time | gap to measured roof | effective weight BW | fraction of measured roof |
| --- | ---: | ---: | ---: | ---: |
| tinygrad direct-FP16 Q6 body | 317.402 us | 17.338 us | 1608.4 GB/s | 94.54% |
| llama Q6/Q8 body | 300.930 us | 0.866 us | 1696.4 GB/s | 99.71% |
| llama Q8 producer + Q6/Q8 body | 301.602 us | 0.866 us above body-plus-producer floor | n/a | n/a |

The clean body gap is `16.472 us`.  Thus llama has already recovered about 95% of
tinygrad's practical roofline headroom, and only `0.866 us` separates llama's body from
the measured streaming floor.  Tinygrad cannot legitimately claim the additional
`15.184 us` down to the sheet ceiling as kernel opportunity.

NCU's older exact-image replay reports `515.70 MB` total DRAM traffic for tinygrad,
including `5.18 MB` writes, at `330.53 us` and 88.42% peak sustained DRAM throughput.
That counter domain is useful for diagnosing extra traffic, but the common semantic
roofline uses the compulsory `510,504,960` weight bytes so it remains comparable across
the two implementations.

## 2026-09-01 legacy cooperative ceiling rerun

Command shape: `q6k_vocab_coop_ceiling_cuda <NACC> <ROW_GROUPS> <XSH> 32` on sm_120.

| NACC | row groups | shared x | us | GB/s |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 1 | 0 | 331.26 | 1541 |
| 2 | 1 | 0 | 329.32 | 1550 |
| 4 | 1 | 0 | **327.68** | **1558** |
| 8 | 2 | 0 | 346.89 | 1472 |
| 8 | 4 | 0 | 342.31 | 1491 |
| 16 | 1 | 0 | 360.74 | 1415 |
| 1 | 1 | 1 | 501.59 | 1018 |
| 2 | 1 | 1 | 518.50 | 985 |
| 4 | 1 | 1 | 525.53 | 971 |
| 8 | 1 | 1 | 525.81 | 971 |
| 8 | 2 | 1 | 367.30 | 1390 |
| 8 | 4 | 1 | 369.53 | 1381 |
| 16 | 1 | 1 | 608.85 | 838 |

This probe deliberately reproduces the legacy FP16 cooperative grammar and writes
`(N,16)` partials.  It is not a performance proxy for llama's full-logit Q8_1 route.
It does prove that accumulator count, block grouping, and naive shared-FP16 activation
staging do not recover the remaining roofline distance.

## Decision

The remaining target is `17.338 us` to the measured practical roofline and `16.472 us`
to llama.  The target is real because llama reaches within `0.866 us` of that roof on
the same compulsory weight stream.

Do not invest further in the legacy row-tile/accumulator/shared-FP16 axes.  The next
causal gate must reproduce llama's distinct execution grammar: Q8_1 activation records,
four warps per output row, 128-thread CTAs, lower register pressure, and the Q6/Q8
integer-dot/dequant schedule.  Admission requires at least 8 us isolated recovery before
recurrent-quality and strict-token wall tests.
