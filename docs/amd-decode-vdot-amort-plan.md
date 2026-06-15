# Phase E — amortized q8 quant + the v_dot4 GEMV: the untested e2e combination (scope)

Date: 2026-06-15. The hand-asm crack succeeded: the v_dot4 int-dot GEMV near-saturates (49.5% peak, 91% of
readraw, would exceed llama.cpp on a healthy GPU). The open question: does it translate to e2e decode tok/s?
Two prior nulls each killed ONE half; the combination was never tested:
- **D1**: optimized v_dot4 GEMV, but UNAMORTIZED quant (q8-quantize x per linear, 7x/layer) -> the quant
  overhead dominated (vdot e2e read half the bytes but at 61 GB/s) -> null (30.2 = fp 30.3).
- **A0**: AMORTIZED quant (quantize x once/token, shared across the 7 linears, 4x/layer), but used the SLOW
  SCALAR int-dot kernel (v_mad, 860 VALU, VGPR 68) -> null (28 tok/s; A0 blamed kernel occupancy).
- **NEVER TESTED**: amortized quant + the v_dot4 BUILTIN GEMV (404 VALU, near-saturating standalone).

## Why this is the genuine make-or-break (not a re-run of a known null)
A0 concluded "the int-dot KERNEL (occupancy), not the quant, is the wall" -- but A0's kernel was the SCALAR
int-dot (v_mad, 860 VALU). The v_dot4 builtin is LEANER (404 VALU) -> likely lower register pressure ->
may pipeline where the scalar didn't. And D1's null was specifically the unamortized quant, not the kernel.
So neither null tested: lean v_dot4 kernel + amortized quant. This phase does exactly that.

## E0 -- make-or-break (cheapest, first)
Wire amortized quant + the existing `q4k_q8_1_vdot_builtin_partial_kernel` into decode, measure e2e vs fp.
- **Quant cache**: a module-level dict keyed by `x_vec.uop.key`, cleared per step. q/k/v compute the SAME
  attn-norm output (CSE'd to one UOp) -> cache hit -> ONE quant feeds q,k,v; gate/up share the ffn-norm
  output -> ONE quant feeds gate,up. So 4 quants/layer, not 7 (A0's mechanism, verified ~22% hit).
- **Dispatch** (Q4K_VDOT_AMORT, in `Q4KPrimitiveLinear.__call__`): `q,scales = cache.get(key) or
  q8_1_quantize(x_vec); qbw = q8_1_bias_pack_u32_kernel; partial = custom_kernel(words, qbw, scales,
  fxn=q4k_q8_1_vdot_builtin_partial_kernel(...))`. Default-off flag.
- **Measure**: e2e decode tok/s (cli --benchmark) vs fp baseline, same session. Verify the cache HIT-rate
  (q/k/v + gate/up share). Accuracy: int8 activation is lossy (X0: 0.5-1% per-layer, fine) -- check output
  coherence.
- **Gate**: e2e > fp -> the v_dot4 kernel pipelines AND amortized quant pays -> the near-saturating kernel
  translates -> first real decode win toward llama.cpp; proceed to E1.
- e2e ~= fp -> the v_dot4 builtin kernel ALSO doesn't pipeline e2e (occupancy, like A0's scalar) -> the
  near-saturating standalone never translates -> the e2e wall is kernel-PIPELINING (occupancy), not kernel
  throughput, definitively for the best kernel. A real, decisive negative.

## E1 -- upgrade the kernel (if E0 wins)
Swap in the wide-load (uint4) + 4-accumulator v_dot4 GEMV (425 vs 302 GB/s standalone, from
`qk_prefetch_gemv`) for the existing vdot builtin -> more e2e headroom. Measure e2e tok/s vs E0 and llama.cpp.

## Touch points
- `tinygrad/llm/model.py` (`Q4KPrimitiveLinear.__call__` + the quant cache; gated, default-off).
- `extra/q4_k_gemv_primitive.py` (`q4k_q8_1_vdot_builtin_partial_kernel`, `q8_1_bias_pack_u32_kernel` -- exist).
- `extra/qk_layout.py` (`q8_1_quantize` -- exists).
- cli --benchmark for e2e; X0-style accuracy check.

## Pre-registered honesty
- The cache must actually CSE/hit across q/k/v and gate/up; verify the hit-rate (else it degrades to D1's
  per-linear quant).
- If null, it is the OCCUPANCY/pipelining wall (the lean v_dot4 still doesn't pipeline) -- the definitive
  closing of the standalone->e2e gap for the best possible kernel. Report honestly; do not re-tune.
- This is the LAST untested combination of the decode-kernel program. Either it wins (a real decode lever)
  or it closes the loop (kernel throughput is real, e2e is occupancy-bound -- the recurring thesis, now
  proven against a near-saturating kernel).
