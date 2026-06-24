# DEFINING "rocBLAS quality" vs "our own" — the measurable axes for the fp16 GEMM gap

Before searching for why tinygrad's FMA matmul isn't at rocBLAS quality, define both on measurable axes (gfx1100,
fp16 NN GEMM, shape 12288x4096x512). Peaks: FP16 packed = 122 TFLOPS; FP32 / fp16-in-fp32-acc ~ 61 TFLOPS.

## The 7 quality axes
1. **Throughput** (TFLOPS) — the outcome.
2. **Roofline fraction** — % of the RELEVANT peak. Key correction: the relevant peak for the winning kernel is the
   packed-FMA roofline (fp16 ~122 if 2x-packed, or ~61 if fp32-rate accumulate), NOT a "WMMA peak" (rocBLAS doesn't
   use WMMA).
3. **Operand reuse / arithmetic intensity** — global bytes read per FLOP; does it stage operands for reuse, or
   re-read? Measured by L2-hit% and achieved-vs-minimum global bytes.
4. **Occupancy** — waves resident/CU (VGPR-limited; LDS-limited).
5. **Compute-issue efficiency** — % of cycles issuing useful compute (FMA/WMMA) vs address/overhead ALU + stalls.
6. **Latency hiding** — software pipelining / prefetch (overlap global-load with compute).
7. **Compute strategy** — packed-FMA vs WMMA.

## ROCBLAS QUALITY (the bar) — measured / disassembled (verified)
| axis | rocBLAS |
|---|---|
| Throughput | **66-77 TFLOPS** |
| Roofline frac | near the achievable fp16/FMA roofline |
| Operand reuse | **operand-LDS-staging** (1644 ds_load_u16, 9KB LDS/wg) -> high reuse, minimal global re-read |
| Occupancy | VGPR 45-185 (multiple tile variants, shape-tuned) |
| Issue eff | dense (4096 v_fma_mix_f32, tight loop) |
| Latency hiding | software-pipelined (Tensile PGR/PLR prefetch) |
| Strategy | **packed-FMA (`v_fma_mix_f32`), NOT WMMA** (RDNA3 has no matrix units) |

## OUR OWN (tinygrad) — measured
| axis | tinygrad WMMA path | tinygrad FMA path |
|---|---|---|
| Throughput | **42** (best) | **<=4.8** (sweep; final TBD) |
| Roofline frac | 34% of 122 / 69% of 61 | ~8% |
| Operand reuse | L2/IC cache (PMC atlas L2 hit 54-87%), NO explicit operand-LDS-staging opt | TBD (PMC) |
| Occupancy | single-wave (local_size 32); more-waves REGRESS (no LDS to free VGPR) | TBD |
| Issue eff | v_wmma (capped) | TBD |
| Latency hiding | NONE (linearizer pin, Lever B) | NONE |
| Strategy | WMMA (bet on tensor cores -> capped at 42 on RDNA3) | FMA but poorly tiled |

## tinygrad's LDS mechanisms (what it CAN express)
- `GROUP/GROUPTOP` (expander.py:133-142) = REDUCE-staging (LDS for K-partials). NOT operand-staging.
- rangeify.py:289-298 (is_pcontig -> bufferize LOCAL) = a contiguous-access local buffer in some cases -- may or
  may not give operand-tile reuse; the search+PMC will reveal if any FMA config achieves operand reuse.
- NO dedicated operand-tile-into-LDS opt (the rocBLAS mechanism). tinygrad relies on cache for operand reuse.

## The gap to localize (what the exhaustive search + PMC will quantify per axis)
For tinygrad's BEST FMA config (from the sweep), measure axes 3-6 (PMC) to find which cap it:
- If LOW L2-hit / low AI -> memory-bound = no operand reuse (the operand-staging gap).
- If low occupancy -> VGPR/wave-bound (the LDS-frees-VGPR gap).
- If low issue eff -> address/overhead ALU or latency stalls (the tiling/pipelining gap).
"rocBLAS quality" = near-roofline on ALL axes; "our own quality" = wherever the search lands + the PMC-localized
binding axis. The DELTA per axis = the concrete reason tinygrad isn't at rocBLAS quality.

## Files
why-tensile-works-fma-not-wmma (rocBLAS disasm). Sweep: /tmp/fma_sweep.py. PMC: extra/qk_pmc_capture.py.
