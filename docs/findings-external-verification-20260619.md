# External verification of the measured findings (web sources, 2026-06-19)

## CONFIRMED — exact matches
1. **RX 7900 XTX hardware specs (we used 960 GB/s, 122 TFLOPS, ~96MB IC):**
   - Memory bandwidth **960 GB/s** (384-bit GDDR6) — EXACT.
   - **122.8 TFLOPS** FP16 matrix (AI accelerators) — matches our 122 peak.
   - **96 MB** Infinity Cache; AMD's "effective BW up to 3500 GB/s" for cache-resident data.
   - Source: KitGuru / WareDB RX 7900 XTX reviews.
   - **Corollary for our decode finding:** the 5GB Q4_K weights >> 96MB IC → streamed from HBM at 960 GB/s, NOT
     IC-served (the 3500 effective only helps reused/resident data). Confirms decode weight-GEMV = HBM-BW-bound.

2. **llama.cpp kernel structure (our rocprof kernel-trace: decode mul_mat_vec_q+quantize_q8_1; prefill mul_mat_q/MMQ):**
   - "mul_mat_q (MMQ): inputs quantized to **Q8_1 on the fly**, then matrix-matrix-quantized kernels" — EXACT (our
     prefill = 74% mul_mat_q + quantize_mmq_q8_1).
   - "Weights stay quantized, **inputs quantized at runtime to Q8**, then an optimized **dot product** kernel
     (vec_dot)" — EXACT (our decode = mul_mat_vec_q + quantize_q8_1; and our "int-dot e2e integration: amortize the
     activation->Q8 quant" mechanism).
   - "dequant-then-MMA for prefill, vec_dot for decode" — matches our prefill(matmul)/decode(GEMV) split.
   - Source: DeepWiki ggml-org/llama.cpp.

3. **Regime split (our measured: decode bandwidth-bound, prefill compute-bound) — TEXTBOOK CONSENSUS:**
   - "LLM decoding at batch 1 sits firmly in the **memory bandwidth-bound** regime"; "decode has low AI and is
     bandwidth-bound"; "GEMV operations... memory-bound."
   - "**Prefill has high AI and is compute-bound**" (TDS article literally titled "Prefill Is Compute-Bound.
     Decode Is Memory-Bound."). Roofline: low-AI -> BW-capped, high-AI -> compute-capped.
   - Sources: Towards Data Science; arxiv roofline/PIM papers; apxml.

## REFINEMENT (not contradiction)
The literature says prefill is COMPUTE-bound (matmul/high-AI). Our finding that matmul-kernel speedups give ~1.00x
e2e is CONSISTENT and refines it: prefill IS compute-bound, but on tinygrad the bottleneck compute is the
**symbolic-shape attention** (non-TC, inefficient), NOT the FFN matmul (already TC-efficient). Hence concrete-KV
(makes attention TC-able) = 1.24x lever. "Compute-bound" at the regime level; the bottleneck compute is attention.

## NOT externally verified (tinygrad-specific, internal)
- The symbolic-var -> no-TC mechanism, the JIT graph-batch ramp (32/64/128/256), HCQ busy-wait — these are
  tinygrad-internal; verified by our own code-reading + measurement, not external sources (no public reference).

## Verdict
The FOUNDATIONAL findings (hardware specs, llama kernel structure, decode-BW/prefill-compute regime split, the
int-dot quant-activation mechanism) are EXACTLY confirmed by external sources. Our independent measurements landed
on the established consensus + the documented llama.cpp design. The tinygrad-specific levers (concrete-KV, symbolic
attention) are refinements within that consensus, verified internally.
