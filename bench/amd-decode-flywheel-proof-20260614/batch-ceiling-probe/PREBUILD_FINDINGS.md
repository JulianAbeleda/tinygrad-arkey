# Pre-build search + measurement (before building batch specialization)

Date: 2026-06-15. The user asked: search prior art first (don't start from scratch) + confirm we can
actually measure what we want, BEFORE building the concrete-batch verification specialization.

## Prior art (we are NOT starting from scratch)
- **Speculative decoding is well-established**: SpecMemo (arXiv:2506.01986), MagicDec (2408.11049), Batch
  Speculative Decoding Done Right (2510.22876), Efficient Spec Decoding for Llama at Scale (2508.08192).
  Real spec-decode uses a FIXED draft length K -> the verification batch is CONCRETE, not a symbolic JIT
  variable. We reuse the framework; our novelty is the AUTOTUNED (cost-model) verification GEMMs on
  AMD/tinygrad.
- **TC needs concrete, aligned dims**; symbolic dims block it; PADTO pads to multiples of 16 (tinygrad
  pattern). Confirmed our Step-3 diagnosis.
- **Measurement tool**: `PROFILE=1` dumps `/tmp/profile.pkl.<user>` of per-kernel ProfileRangeEvents
  (name, start, end) -> a real per-kernel breakdown the JIT replay otherwise hides, incl. WMMA detection.

## Measurement WORKS -- and it reveals the build premise is shaky
Parsed the forward's profile.pkl:
- **ZERO WMMA kernels** -> the decode forward uses NO tensor cores. The plateau is a no-TC plateau, now
  MEASURED (not inferred). The matmuls run as plain reduce kernels (`r_toks_256_16_3_...`).
- **tinygrad FACTORS the dims** (12288 -> 256x16x3 in the kernel shape/name). So (a) the matmuls don't
  surface as "12288" matmuls, and (b) the warm-start shape-match (keyed on raw dims) can't find them --
  the kernel's full_shape is the factored/tiled shape, not the logical (M,K,N).
- **The heuristic never picks TC for these small-N matmuls even STANDALONE** (Step 2: 6.8 TF heuristic vs
  13.5 TF loop-TC). So making the batch concrete does NOT auto-enable TC -- consistent with the
  concrete-JIT forward being SLOWER (18 ms/tok) than symbolic (14).

## Implication: the naive "batch specialization -> TC -> plateau drops" build is NOT justified
Three compounding walls block TC in the forward, all now measured:
1. symbolic batch dim -> TC schedule errors (Step 3),
2. concrete batch alone -> heuristic still won't apply TC (this measurement),
3. forcing the loop's TC schedule -> blocked by dim-factoring (kernel shape != logical M,K,N) + symbolic.

So realizing TC in the decode forward needs MORE than batch specialization: a fixed-K verification path
that dispatches the FFN matmuls to PRE-TUNED concrete-N custom kernels (like the v_dot4 dispatch), so the
loop's concrete-shape TC schedule is used directly -- a substantial build, not a flag flip. The cheap
versions (concrete JIT, warm-start force) do not realize it.

## Honest recommendation
The measurement did its job: it shows the lever (loop TC, ~1.9x standalone, Step 2) is real but the decode
forward uses zero TC and won't accept it through any cheap path. Building the naive batch-specialization
would hit walls 2-3. The justified next build is the fixed-K verification path with pre-tuned concrete-N
matmul dispatch -- bigger, and worth a deliberate decision rather than charging in. The PROFILE-pkl
per-kernel + WMMA measurement is now established and reusable for whatever we build.
