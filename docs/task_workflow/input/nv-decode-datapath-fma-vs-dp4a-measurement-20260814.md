# NV decode datapath measurement: fp32 FMA vs int8 DP4A, llama MMQ vs tinygrad GEMV (2026-08-14)

Date: 2026-08-14. Target: RTX 5090, sm_120, CUDA 13.2. Settles the first-principles
question "why is llama's decode GEMV faster than ours" with measured numbers, not
assumptions.

## 1. Isolated datapath peak (same harness discipline)

`dp4a_peak_cuda.cu` (existing) and `fma_peak_cuda.cu` (added) use the identical
register-resident, runtime-trip-count, never-taken-store harness. Both print the
warp-instruction issue rate (`warps * iters * NACC`), so the two are directly
comparable. `blocks=32768, tpb=256, nacc=8`.

| instruction | warp-issue rate | MACs/instr | measured TMAC/s |
| --- | ---: | ---: | ---: |
| `fma.rn.f32` (our fused GEMV) | 954.9 G/s | 1 | 30.6 |
| `dp4a` (llama MMQ) | 951.6 G/s | 4 | 122.2 |

Same issue rate (~950 G warp-instr/s), but DP4A performs 4 int8 MACs per
instruction versus 1 fp32 MAC per FFMA. **DP4A has 4.0x the MAC throughput of
fp32 FMA on this die.** This confirms the decode-GEMV lever: llama's integer dot
buys 4x compute headroom per issued instruction over our dequant-to-fp32 FMA loop.

Accounting correction: the existing dp4a README reports "3.8 TMAC/s / 7.6 INT8
TOPS", which multiplies the warp-instruction rate by 8 but omits the 32 lanes per
warp. The corrected per-lane figure is 122 TMAC/s. The relative conclusion
(DP4A = 4x FMA) is unaffected.

## 2. Real-shape GEMV: llama MMQ vs tinygrad primitive

Same packed Q6_K weights (3.44 MB, 1024x4096 K/V role) and same Q8_1 activation
payload (tinygrad bridges it to fp16 for its ABI). Both timed with CUDA events over
200 graph replays, 3 reps.

| kernel | median us/replay |
| --- | ---: |
| llama `mul_mat_vec_q<Q6_K>` (DP4A, single fused launch) | 4.10 |
| tinygrad `q6k_gen_partial_1024_4096_4` + external sum | 43.01 |

10.5x gap at this shape. The gap is NOT purely the datapath (that factor is 4x):
the tinygrad primitive measured here is the 4-way-split-K partial form with an
external `sum(axis=1)` reduction (2 CUDA nodes), not the production cooperative
fused kernel, so the partial+reduction structure contributes most of the remaining
2.6x. Numerics clean on both arms.

## 3. Consequence

- The theoretical synthesis holds and is measurable: adopt llama's DP4A + single
  fused matvec (4x compute headroom + amortized dequant), while keeping our fusion
  of the support kernels (rope/kv/norm) that llama still pays for.
- The fused RMSNorm->Q8_1 provider (`rmsnorm_q8_1_llama_provider_4096`) and the
  DP4A consumers (`q4k_q8_dp4a_*`, `q6k_q8_dp4a_*`, `q4k_q8_mmvq_direct_*`) already
  exist in-repo; they are the correct next substrate, not the fp16-direct GEMVs.

## Evidence

- `/tmp/dp4a_peak` (recompiled) and `/tmp/fma_peak` (added harness) output.
- llama oracle: `scratchpad/llama_cuda_quantized_live_oracle.py --quant Q6_K`.
- tinygrad matched role: `scratchpad/q6k_matched_tinygrad_role_benchmark.py`
  (stale `_cpu_quantizers` 3-tuple unpack fixed in this change).
