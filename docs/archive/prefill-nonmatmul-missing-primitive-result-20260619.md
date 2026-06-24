# RESULT — chasing the prefill e2e(41) vs matmul-kernel(63) gap: NOT a missing fusion primitive; prefill is near-ceiling

Followed the "we're missing primitives" intuition into the ~35% non-matmul prefill gap. Measured the non-matmul
breakdown + kernel inventory vs llama. The intuition is half-right (the gap is real) but the cause is NOT what it
looked like.

## Measured
1. **Matmul kernel ≈ 63 TFLOPS in-model** (810µs/gateup; tinygrad WMMA ≈ Tensile ≈ llama mul_mat_q) — NOT the
   bottleneck (prior result).
2. **Non-matmul GPU (JIT-invariant, eager DEBUG=2) ≈ 21.6ms:** attention (`r_16_32_2_8_16_4_4_128_4`, 652µs/ea)
   **~12ms** = the single biggest non-matmul; lm_head 2.7ms; **183 elementwise kernels** (cast/silu/mul/residual/
   transpose) 4.5ms; norms ~0.
3. **Kernel inventory vs llama (the key data point):**
   - **tinygrad prefill = 729 launches** (183 elementwise + 546 reduce/matmul/attn).
   - **llama prefill = 2168 launches** (428 mul_mat_q + 428 quantize_mmq + 290 rms_norm + 153 copyBuffer + 144
     rope + 142 bin_bcast + 140 convert_unary + 72 k_set_rows).
   - **tinygrad already FUSES MORE than llama (729 ≪ 2168) yet is slower e2e (41 vs 49 TFLOPS).**

## Conclusion — the missing primitive is NOT in prefill
- **Fusion / kernel-count is NOT the gap** — tinygrad emits 3× fewer kernels than llama and still loses. So "fuse
  the glue" is refuted as the lever; tinygrad's scheduler already fuses aggressively (the `.contiguous()`-isolated
  matmuls are deliberate for warmstart-TC matching, not a fusion failure).
- The e2e 41 vs matmul-kernel 63 gap = **attention (~12ms, the dominant non-matmul) + lm_head + the inherent
  cost of the non-matmul work**, plus inter-kernel scheduling/overlap (llama runs 3× more kernels *faster* → its
  runtime overlaps them better than tinygrad serializes). Neither is a missing *compute* primitive.
- The only "missing primitive" candidate in prefill is a **fused flash-attention-WMMA with LDS K/V reuse** for the
  ~12ms attention — but that is the SAME wall already hit (Increment 2 refuted reuse-free flash as memory-bound;
  the LDS-tiling flash primitive is the walled codegen). And it's only ~6% of the forward → ~1.06× at most.
- **Prefill is near its practical tinygrad ceiling (~82% of llama):** matmul kernel maxed (63 TFLOPS), fusion
  already aggressive, the residual gap is attention-efficiency + runtime overlap (both hard/walled, small prize).

## Redirect — the missing-primitive opportunity is DECODE, not prefill
The campaign meta-pattern localized the BIG, clear integration gap to DECODE: tinygrad's int-dot GEMV is 76% peak
standalone → 44% in-model. THAT 32-point gap (vs prefill's modest 82%→ceiling) is the real opportunity, and its
lever is concrete and measured: **fused-mmvq integration** (amortize activation→Q8 quant across input-sharing
GEMVs + sustain llama's max-occupancy launch) and/or **spec-decode** (bandwidth-justified weight-read amortization).
Prefill's matmul is done; decode's integration is where the missing-primitive work pays off.

## Measurement caveat (banked)
JIT per-kernel attribution is UNCAPTURABLE on this stack: `time_sum_s`=0 on TinyJit replay, ProfileRangeEvent/PMC
emit nothing on replay, eager bypasses warmstart-TC, and wall is clock-noisy (192–356ms across runs). Non-matmul
absolute times are JIT-invariant (eager-valid); the e2e gap's exact split (attention vs overlap) is bounded by this.

## Files
`/tmp/pf_dbg.txt` (eager prefill DEBUG=2), llama trace `/tmp/llama_pp`. Prior:
`prefill-tensile-transpose-free-result-20260619.md`, `inference-perf-measured-map-20260619.md`.
