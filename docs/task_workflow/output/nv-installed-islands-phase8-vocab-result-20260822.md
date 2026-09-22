# NV installed-island Phase 8 vocab and sampler tail

Date: 2026-08-22
Branch: `nvidia-bringup-20260731`
HEAD: `6570abc025514273faa100c66b979e531585a1e1`

Evidence: `docs/task_workflow/evidence/nv-installed-islands-20260822/phase8/`

## Findings, ordered by wall severity

`MEASURED` The frozen census "vocab main + tail" row is `+67.612 us`
(tinygrad `371.520 us`, 5 nodes; llama `303.908 us`, 2 nodes). It is not a
launch/dispatch problem. It decomposes as:

```text
vocab main body            +10.012 us   (313.248 - 303.236)
r_32_4_1187 reduction      +39.296 us   (no llama counterpart)
r_128_16_8_1187 reduction  +11.040 us   (no llama counterpart)
E_1187_32_4 x2             +7.936 us    (no llama counterpart)
llama vocab_quant          -0.672 us    (tinygrad has no separate node)
sum                        +67.612 us
```

`MEASURED` The vocab main body is DRAM-bandwidth-bound on both sides and is
effectively at parity:

```text
tinygrad exact body  318.306 us   (510.5 MB weight -> 1.604 TB/s)
llama body           303.236 us   (510.5 MB weight -> 1.684 TB/s)
body delta           +15.070 us   (tinygrad ~5% less efficient, within 10%)
```

The 510,504,960-byte Q6_K weight is the only material traffic. tinygrad runs at
~89.6% of the ~1.79 TB/s GDDR7 peak; llama runs at ~94.1%. The `+10.012 us`
installed delta is this streaming-efficiency gap, not arithmetic.

`MEASURED` The remaining `+58.272 us` is structural: llama folds argmax into
the vocab GEMV epilogue (2 nodes), while tinygrad materializes four separate
reduction/support kernels after the vocab GEMV. The dominant one is a
single-warp reduction:

```text
r_32_4_1187   grid [1,1,1] block [32,1,1]  -> 39.296 us installed
              607,744 bytes / 37.985 us body = 16.0 GB/s effective (<1% peak)
```

`MEASURED` `r_32_4_1187` reduces the full 151,936 fp32 logit vector to 128
partials with exactly one 32-thread block. It is latency/occupancy-bound, not
bandwidth-bound: 16 GB/s effective versus a ~1.79 TB/s device peak.
`r_128_16_8_1187` uses 2048 threads (128 blocks x 16) and is already 58 GB/s,
but still 10.432 us of pure body.

## Exact-body / clean-HCQ / production decomposition

`MEASURED` Every tail node above 2 us/token was replayed with its exact
production cubin (nsys 2000 reps, body `B`) and clean chained-HCQ drain
(slope `C`). `P` is the frozen per-token production command interval,
`D = C - B`, `R = P - C`, identity residual zero in every row.

| kernel | P | B | C | D | R | role |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `q6k_gen_coop_151936_4096_inkernel` | 313.248 | 318.306 | 312.475 | -5.831 | 0.773 | vocab main |
| `r_32_4_1187` | 39.296 | 37.985 | 38.658 | 0.673 | 0.638 | vocab tail |
| `r_128_16_8_1187` | 11.040 | 10.432 | 10.786 | 0.354 | 0.254 | vocab tail |
| `E_1187_32_4` (e1ccc426) | 4.928 | 1.216 | 1.737 | 0.521 | 3.191 | vocab tail |
| `E_1187_32_4` (76d37a73) | 3.008 | 1.504 | 1.969 | 0.465 | 1.039 | vocab tail |
| `E_16_4_2_8_16_2_4_4` | 7.200 | 1.664 | 2.155 | 0.491 | 5.045 | positional |
| `r_32_32_4_32_4` | 4.384 | 3.840 | 4.297 | 0.457 | 0.087 | sampler |
| `E` | 2.944 | 0.480 | 0.955 | 0.475 | 1.989 | sampler |
| `E_2` | 2.656 | 0.480 | 0.973 | 0.493 | 1.683 | sampler |
| `r_16_8` | 1.184 | not measured (<2 us) | - | - | - | sampler |

`MEASURED` The negative `D` on the vocab main is expected: its ~315 us DRAM
body hides the clean launch entirely. The tail is split into two regimes:

```text
body-bound reductions   r_32_4_1187 + r_128_16_8_1187 + r_32_32_4_32_4
                        = 52.257 us body (89-97% of their own P)

install-bound elements  E, E_2, E_1187_32_4, E_16_4_2_8_16_2_4_4
                        = 12.328 us body / 17.855 us install across the tail
```

## Installed boundary

`MEASURED` Boundary defined as final-norm output ready to sampler feedback
ready.

```text
tinygrad post-norm tail   vocab main 313.248 + 9 reduction/sampler nodes 76.640
                          = 389.888 us
llama post-norm tail      vocab 303.236 + vocab_quant 0.672
                          + binbcast 0.800 + get_rows 1.984 + final_norm 2.688
                          = 309.380 us
```

The final-norm node itself is outside this boundary on both sides.

`INFERRED` `E_16_4_2_8_16_2_4_4` (7.200 us, `R = 5.045 us`) carries a large
production-conditioned residual but is mapped to rope/store by position and
shape only; its semantic role is not confirmed by metadata, so its residual is
not attributed to a mechanism here.

## Real bytes and reduction traffic

| kernel | reads | writes | effective BW |
| --- | ---: | ---: | ---: |
| vocab main | 510.5 MB | 0.6 MB | 1.604 TB/s |
| r_32_4_1187 | 607,744 B | 512 B | 16.0 GB/s |
| r_128_16_8_1187 | 607,744 B | 512 B | 58.3 GB/s |
| E_1187_32_4 | ~610 KB | ~4 KB | install-bound |

## Verdicts

```text
vocab main      BODY_PARITY     (5% DRAM-efficiency gap, within 10% bar)
vocab row       BODY_DOMINANT   (+67.61 is ~86% structural tail body + ~14% main body)
r_32_4_1187     BODY_DOMINANT   (single-warp reduction topology)
r_128_16_8_1187 BODY_DOMINANT   (2048-thread reduction topology)
E/E_2/E_1187    INSTALL_DOMINANT (small bodies, R dominates)
```

No row is `DISPATCH_DOMINANT`. The clean dispatch `D` is under 0.7 us per
kernel everywhere and is fully hidden for the 315 us vocab main.

## Decision

`MEASURED` The legal tail ceiling is `~58.27 us/token`, well above the
`20 us/token` close threshold, so vocab is not closed for the 240 campaign.
The closed candidate (top-1 fusion into the current main route) stays closed:
the prior wall-negative result is not invalidated by this decomposition. The
new mechanism is a changed main-output/reduction topology, not naive fusion:
the `r_32_4_1187` single-warp reduction is the highest-leverage target.

`UNMEASURED` Whether a two-stage or wider-block reduction topology reduces the
39.296 us installed cost while preserving the exact sampler contract. This is
the one implementation scope handed to Phase 11.

## Ledger snapshot

```text
node_sum   = 4677.920 us (tinygrad) / 3878.254 us (llama)
union      = 4671.500 us (tinygrad) / 3878.254 us (llama PDL-off)
overlap    = 6.420 us (tinygrad) / 0 us (llama PDL-off)
wall       = 4771.423 us (fresh control)
host_gap   = unmeasured single-domain
useful_body = unmeasured
booked_recovery = 0.000 us
remaining_to_240 = 604.756 us
```
