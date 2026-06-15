# W1b' RESULT -- the Marlin fused-dequant->WMMA primitive WORKS (2026-06-15)

After W1b.0 found the upstream hand-SHAPED_WMMA skeleton stale, the route reassessment chose
TC-opt-over-a-hand-LDS-staged-dequant. Built bottom-up in `extra/qk_marlin_w1b.py`; every gate green.

## Track 0 -- diagnosis (refined the target)
Rendered the W1 fused kernel: the slow kernel (23 GFLOPS) reads COMPRESSED weights and computes the
Q4_K dequant INLINE feeding each `__WMMA_16_16_16_half_float` -- recompute confirmed. Marlin fix
validated: stage the dequant in LDS once.

## Track B -- fast falsifier (not needed)
The framework facts (GROUP forbidden with TC) already showed the easy opt-staging path is blocked;
Track A's a0b gate then proved the hand-LDS route works, so the falsifier was moot.

## Track A -- the build (all gates GREEN)
- a0a: `Opt(OptOps.TC, 0, (-1,2,1))` fires WMMA on a hand-built `Ops.REDUCE` matmul (correct). The
  q4k `.set/.after/.end` manual accumulator does NOT make an `Ops.REDUCE`; `mul.reduce(k, arg=Ops.ADD,
  dtype=float32)` (per `cdna_asm_gemm.py::custom_uop_gemm`) does.
- a0b: **the make-or-break gate** -- TC fires WMMA on a MUL operand that is a load from a
  `DEFINE_LOCAL` (LDS) written earlier in the SAME kernel (copy -> barrier -> matmul). Correct.
  The Marlin structure is expressible on this fork.
- a1: full Marlin -- dequant the compressed Q4_K tile ONCE into the LDS fp16 tile -> barrier -> WMMA.
  Correct on real GGUF weights (rel_err 1e-4). Rendered source verified: ALL dequant shifts are
  BEFORE the barrier, ALL WMMA ops AFTER -- the per-MAC recompute (the W1 28x) is structurally gone.
- a2: timed marlin (reads compressed) vs the materialized-fp16 WMMA ceiling (same LDS-staged-WMMA
  kernel, pre-dequanted fp16 weight) at matched shape. Result: **fusing the dequant is ~free** --
  marlin is 1.07-1.08x FASTER than the fp16 ceiling on 4/5 shapes (reads less DRAM), 0.89x on one
  large-N shape; mean 1.04x. All correct.

## What this proves / does NOT prove
- PROVES: the competitive fused-WMMA *primitive* exists -- correct, reads compressed, uses tensor
  cores, dequant staged once, throughput-competitive with the fp16 ceiling. This was the W1b gate
  that blocked W2-W4. It is now open.
- Does NOT prove competitiveness with llama.cpp yet: absolute throughput is tiny (0.04-0.23 TFLOPS)
  because these are single-workgroup, un-tiled, whole-tile-in-LDS shapes (M<=32). Reaching the 83.6
  TFLOPS peak / 103.84 tok/s bar needs grid parallelism + K-tiling + occupancy tuning over a
  PARAMETRIZED version of this template -- exactly the W2 (parametrize) -> W3 (autotune) work, which
  now has a real template that contains a competitive point.
