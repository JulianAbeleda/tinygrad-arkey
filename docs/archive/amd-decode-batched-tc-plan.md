# Batched-decode machine search — realize tensor cores in the verification GEMM (scope)

Date: 2026-06-15. The reachable decode path: speculation supplies a CONCRETE batch K (fixed draft length);
the verification GEMMs become the bottleneck; the plateau is COMPUTE-bound with ZERO tensor cores
(measured via PROFILE). The loop already FINDS TC schedules (Step 2: 13.5 TF w/ TC vs 6.8 heuristic). The
only thing blocking machine-search-decode in this regime is REALIZING TC in the forward — which symbolic
batch + dim-factoring blocked (Step 3). Speculation's concrete K removes the symbolic-batch obstacle, so
this re-tests realization at CONCRETE batch.

## Why this is THE make-or-break (everything downstream depends on it)
Single-stream decode is latency/memory-bound -> no kernel vocabulary helps (v_dot4 proved it). Batched
verification is compute-bound + no-TC -> TC is the lever. The whole batched path (speculative scaffold,
matmul_decoded, loop-tuned GEMMs) is worthless if we can't put TC on the verification matmuls on RDNA3.
So prove that FIRST, cheaply, before any scaffold.

## Make-or-break (cheapest, do FIRST)
Concrete-batch (K=16) forward, A/B: baseline (heuristic, no TC) vs warm-start (force the loop's TC
schedule). The Step-3 hook already matches by shape signature; fix the key for concrete batch (out dims
{M,N}, both concrete now) and re-test.
- **Gate**: warm-start `apply > 0` (the loop's TC schedule applies WITHOUT KernelOptError at concrete
  batch) AND the forward ms/tok drops below the no-TC baseline.
- **PASS** -> TC realizes at concrete batch on RDNA3; the batched path is unblocked; proceed to the
  speculative scaffold (below). This is the moment machine-search-decode becomes realizable.
- **FAIL** (TC still errors at concrete batch) -> the wall is dim-factoring/fusion, deeper than symbolic
  batch -> tinygrad's RDNA3 vocabulary genuinely can't express it -> the realization needs a TileLang-class
  tile vocabulary (the option-1 "add primitives to tinygrad" or a tile-DSL backend). That is also a
  decisive result: it precisely locates the expressibility gap on real hardware.

## If PASS -> the batched-decode machine-search build
1. **Concrete-K verification path** in the model: a fixed-K (e.g. 16) JIT that does the Q4_K linears at
   concrete batch (so TC applies), via matmul_decoded (cheap dequant -> native fp16 matmul, the N0a
   "competitive batched path") or the W1b' fused dequant->WMMA primitive.
2. **Loop tunes the verification GEMMs** (the validated L0/L1 cost-model loop over the concrete-N matmul
   shapes -- it already finds the TC schedule cheaply, Step 2).
3. **Speculative scaffold**: verify K draft tokens/step. Draft = the fine-tuning lever (start trivial:
   n-gram / a tiny draft, just to measure realized-vs-ceiling).
4. **Measure**: realized batched-decode tok/s vs the ceiling (~2.4-3.5x mem-amortization) x the TC-plateau
   drop, vs single-stream 30 and llama.cpp 104.

## Pre-registered honesty
- The plateau drop is bounded by the matmul fraction of the forward (Amdahl) -- size it with PROFILE.
- Realized speedup also scales with draft acceptance rate (speculation) -- report the ceiling AND a
  realistic-acceptance estimate.
- This is the BATCHED/speculative path; it is NOT single-stream batch-1 parity (that's structurally stuck,
  proven). No relabeling.

## Measurement (established)
PROFILE=1 -> /tmp/profile.pkl.<user> -> per-kernel device time + WMMA detection (confirms TC actually
fired). Plus forward wall ms/tok. Both work.
