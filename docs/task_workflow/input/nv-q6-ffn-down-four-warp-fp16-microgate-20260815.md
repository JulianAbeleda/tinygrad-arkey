# NV Q6 FFN-down four-warp fp16 microgate (2026-08-15)

Date: 2026-08-15
Branch: `nvidia-bringup-20260731` (HEAD `498306f47`)
Status: **measurement record. Research-only microgate; no production route changed.**

This is the test requested after the 240 audit: before promoting any Q6
geometry work, build the four-warp fp16 Q6_K FFN-down direct emitter and
microgate it against the installed coop route.

## 1. Candidate vs control (arithmetic)

| side | kernel | threads/row | warps/row | Q8 provider |
| --- | --- | ---: | ---: | --- |
| control | `q6k_gen_coop_4096_12288_inkernel_epi_ffnresadd` (row_tile 2) | 16 | 0.5 | 0 |
| candidate | `q6k_four_warp_fp16_direct_4096_12288_epi_ffnresadd` | 128 | 4 | 0 |

Both arms consume the fp16 activation directly and keep the M2b residual add
in-kernel. The candidate partitions 48 K-blocks as 4 warps x 2 sub-groups x 6
blocks, 16 position lanes, cross-warp shared-memory reduce (the exact Q4
four-warp pattern applied to Q6).

Correctness vs the `q6_k_reference` oracle: candidate max-abs error
`2.67e-5`, control `4.48e-5`, candidate-vs-control `3.24e-5`. PASS at the
microgate atol `2e-2`.

## 2. Timing (device time is the real signal)

The single-node graph replay is launch-bound: A/B/A replay medians are
`130.39 us` candidate vs `130.64 us` control (`-0.25 us`, wall-noise).

`DEBUG=2` device-time trace removes the launch floor:

| side | steady-state `tm` | DRAM GB/s |
| --- | ---: | ---: |
| control | 30.9 us | ~7550 |
| candidate | 25.7 us | ~9630 |

Candidate is `-5.2 us/node` (`-17%`) and `+27%` DRAM bandwidth. The installed
FFN-down node census is 34.92 us/node vs llama 28.75 us/node; the candidate at
25.7 us is already below llama on this node, so the four-warp geometry fully
closes the Q6 FFN-down excess (18 nodes x ~5-6 us ~= ~95-110 us).

## 3. Verdict

Worth promoting. The lever is the same mechanism the Q4 promotion already
landed (`765f03f30`), applied to Q6's larger geometry gap. Promotion follows
the Q4 template: production emitter, route-policy JSON, loader wiring, unit
test, then a reverse wall bracket to book the real tok/s delta.
