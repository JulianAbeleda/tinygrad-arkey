# NV Q4 FFN-down: DP4A SASS one-to-one census vs llama (2026-08-14)

Date: 2026-08-14
Branch: `nvidia-bringup-20260731`
Status: **measurement record.** The DP4A datapath is arithmetic-identical to
llama, but the emitted kernel is not one-to-one. This record isolates the SASS
reasons for the remaining `9.91 -> 8.43 us` standalone gap.

## 0. Question

Is the installed `sum_dp4a=True` Q4_K x Q8_1 consumer a one-to-one match for
llama's `mul_mat_vec_q<Q4_K>`? Answer at two levels:

- **Arithmetic: yes.** Byte-for-byte on the decode path (see
  `nv-q4k-q8-substrate-arithmetic-trace-20260812.md`).
- **Kernel/SASS: no.** Same 48 DP4A per thread, but different memory load
  widths, integer-vs-float scale arithmetic, and loop structure. That is the
  residual perf gap, not the datapath itself.

## 1. Identical substrate

Both kernels launch the same shape: 4096 rows x 12288 K, one row per block,
128 threads/block (4 warps). For 96 values/thread both execute **48 int8 DP4A
per thread**: ours is 12 source iterations x 4 `__dp4a`; llama is 6 outer
iterations x 8 `dp4a` (`QR4_K=2` x 4). The Q4 nibble order, scale/min decode,
signedness, and `dp4a(0x01010101, q8)` correction sum are term-identical
(reference: `nv-q4k-q8-substrate-arithmetic-trace-20260812.md`).

## 2. Measured kernel divergence

SASS dump: `cuobjdump -sass -fun ...`. Static instruction lines count both
real SASS (PTX-only sections excluded): ours 200, llama 360. Resource usage:

| kernel | REG | SHARED | block | resident blocks/SM | occupancy |
| --- | ---: | ---: | ---: | ---: | ---: |
| ours `q4k_q8_mmvq_direct_4096_12288` | 40 | 1408 | 128 | 12 | 100% |
| llama `mul_mat_vec_q<Q4_K,1,0,0>` | 56 | 1408 | 32x4 | 9 | 75% |

Occupancy is arithmetic from RTX 5090 limits (1536 threads/SM, 65536
registers/SM): ours `5120 regs/block`, llama `7168 regs/block`. llama wins
while at lower occupancy, so the gap is instruction/memory efficiency, not
occupancy.

## 3. Per-thread global load traffic

Hot-loop SASS body, static load mix:

| kernel body | LDG.E | LDG.E.CONSTANT | LDG.E.U16 | LDG.E.U16.CONSTANT | bytes/body | trips | bytes/thread |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ours (8 IDP) | 16 | 0 | 2 | 0 | 68 | 6 | 408 |
| llama (16 IDP) | 8 | 6 | 4 | 10 | 84 | 3 | 252 |

llama reads roughly 62% of our per-thread global traffic for the same 96
values. It does this with `LDG.E.U16.CONSTANT` / `LDG.E.CONSTANT` for packed
scales, mins, and weight metadata, while ours uses sixteen full-width generic
`LDG.E` and only two `LDG.E.U16`.

## 4. Float-pipe vs integer scale arithmetic

Normalized to one IDP in the hot loop:

| metric per IDP | ours | llama |
| --- | ---: | ---: |
| FMUL | 1.00 | 0.125 |
| FFMA | 1.00 | 0.625 |
| I2FP | 1.50 | 0.50 |
| IMAD | 0.125 | 0.625 |
| LDG | 2.25 | 1.75 |

llama multiplies `dot * scale` and `dot * min` in integer (`IMAD`) and
converts once to float; the generated tinygrad consumer converts each DP4A
and each scale/min to float first, then does `FMUL`/`FFMA`. That makes ours
float-pipe heavy on a kernel whose lever is integer DP4A.

## 5. Loop structure

- ours: 8-IDP body, 6 trips, with the `<4 / >=4` scale branch emitted inside
  the body (`ISETP` plus predicated `@P1` / `@!P1` decode).
- llama: 16-IDP body, 3 trips, with the decode branches hoisted out of the
  steady-state IDP loop.

llama still pays the same DP4A work, but it does so with fewer loop-control
and float instructions in the way.

## 6. Verdict

The DP4A datapath claim is confirmed and now bounded precisely:

- **One-to-one on datapath arithmetic:** yes.
- **One-to-one on emitted SASS:** no.
- Residual is memory load width (408 vs 252 bytes/thread), float-vs-integer
  scale arithmetic, and 8-IDP-vs-16-IDP loop structure, not DP4A throughput.

This does not change the production gate: the in-loop DP4A route still loses
to the no-provider fp16 geometry route because the standalone
`q8_1_llama_provider_12288` node costs more than the datapath saves. The
provider-fold into the w1w3 producer epilogue remains the gating move
(`nv-q4-down-dp4a-datapath-verification-20260814.md`, section 3). If the fold
lands, these three SASS deltas are the next tuning surface, in order:
U16/CONSTANT metadata loads, integer IMAD scale application, then loop shape.

## 7. Artifacts and limitation

- `/tmp/dp4a_sum_wall` and `/tmp/q4k_dp4a_sum.cu` (ours)
- `scratchpad/llama_cuda_quantized_oracle_dump/libggml-cuda.so.0.14.36.sm_120a.cubin`
  (llama pinned cubin)
- `ncu` profiling was attempted but blocked by `ERR_NVGPUCTRPERM` on this
  driver; this record therefore uses SASS static census plus source-level
  arithmetic rather than hardware counter dynamic counts.

## 8. References

- `nv-q4-down-dp4a-datapath-verification-20260814.md` (the 9.91 vs 8.43 us
  lever measurement)
- `nv-q4k-q8-substrate-arithmetic-trace-20260812.md` (byte-for-byte decode)
- `nv-q4-down-fp16-geometry-four-warp-verdict-20260814.md` (geometry half)
- llama.cpp `ac4cddeb0` pinned source: `vecdotq.cuh`, `mmvq.cu`
