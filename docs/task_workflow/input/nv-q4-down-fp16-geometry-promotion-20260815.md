# NV Q4 FFN-down four-warp fp16 geometry promotion (2026-08-15)

Date: 2026-08-15
Branch: `nvidia-bringup-20260731` (HEAD `39d1a1d56`)
Status: **promoted to NV sm_120.** The Q4_K FFN-down critical-path lever books
as the geometry-only fp16-FMA route; the DP4A/Q8 successor is measured
NO_GO_WALL and stays closed.

## 1. What was decided

The occupancy proof (`nv-ffn-down-gap-occupancy-proof-20260814.md`) reduced the
2.29x Q4 FFN-down gap to thread geometry: llama runs 4 warps/row (128 threads,
66.3% occupancy, 77.2% DRAM) against our 1 warp/row (32 threads, 38.8%, 54.5%).
Two candidate spellings exist:

| spelling | datapath | result |
| --- | --- | --- |
| `fp16_fma` four-warp | fp16 FMA, no Q8 | **WALL_PASS -100.3 us at d512, +2.01%, token-exact** |
| `scalar_q8_packet` DP4A + folded Q8 | DP4A, producer-folded Q8_1 | **NO_GO_WALL +45 us, -0.87%** |

The DP4A route reintroduces the Q8_1 quantization (folded into the w1w3
producer) that our fp16-direct path already skips; that producer cost cancels
the geometry/datapath win. The geometry-only route is the booked lever.

## 2. The gate that promoted it

Reverse wall bracket at d512 on Qwen3-8B-Q4_K_M / RTX 5090, fresh process per
arm, `DEV=NV`:

| arm | ms/token | tok/s |
| --- | ---: | ---: |
| control bracket median | 5.0931 | ~196.4 |
| candidate (18 Q4 down blocks) | 4.9928 | ~200.3 |
| delta | -100.3 us | +2.01% |

All token hashes equal across arms. Semantic gate: tokens/argmax/top-k/
top-k-order equal, full-logit relative L2 5.29e-4 inside the 1e-3 gate.
Topology: program count unchanged (the installed `epi_ffnresadd` GEMV swaps 1:1
to the `q4k_fp16_mmvq_direct` consumer, zero added transport/materialize
nodes).

Precision note: the historical 1e-2 max-abs logit atol is exceeded (1.17e-2)
and is reported non-authoritative; every authoritative gate passes and the
token stream is bitwise identical. The fp32 reduction reorder is the only
numeric change.

## 3. Production confirmation (after promotion)

Fresh production census after the policy promotion:

| metric | before | after |
| --- | ---: | ---: |
| kernels/token | 596 | 596 (unchanged) |
| token sha | `227ad3ce...` | `227ad3ce...` (exact) |
| census tok/s | 193.49 | 198.86 |

The Q4 down family swaps `q4k_g3_lanemap_gemv_epi_ffnresadd_4096_12288` (18 ->
0) for `q4k_fp16_mmvq_direct_4096_12288_epi_ffnresadd` (0 -> 18), 22.05 us
median vs the installed 26.85 us. Q6 down is unchanged.

## 4. Wiring

New route policy `decode-q4k-ffn-down-fp16-geometry-route-policy.json` targets
NV sm_120. The loader flag is ANDed with `_ffn_resadd_promoted` (the control
topology this route swaps against), and the `Q4KFFNDownMMVQAdmission(index,
fp16_fma=True)` is installed only on the Q4_K 4096x12288 ffn_down role.

## Evidence

- reverse bracket: `/tmp/q4k_fp16_geom_timing_20260815.json` (WALL_PASS)
- prior gate: `/tmp/fp16_qualify_18.json` (semantic PASS, topology PASS)
- DP4A NO-GO: `/tmp/q4k_scalar_q8_all18_timing.json` (NO_GO_WALL)
- production census: `/tmp/census_prod_promoted_20260815.json`
- geometry proof: `nv-ffn-down-gap-occupancy-proof-20260814.md`
