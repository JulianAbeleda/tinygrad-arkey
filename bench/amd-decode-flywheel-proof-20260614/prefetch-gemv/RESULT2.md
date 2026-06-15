# Clean prefetched GEMV — make-or-break: the DEQUANT WORK is the cap, not loads (refines the root cause)

Date: 2026-06-15. Built register-efficient wide-load + MLP + reduction-ILP variants to beat tinygrad's
~42% toward readraw's ceiling. Result: the dequant work caps the GEMV well below readraw, and no load/
accumulator structure moves it -- a refinement (and partial correction) of the MLP root-cause framing.

## Result (degraded-GPU session; relative numbers valid)
| variant | % peak | valu | loads | structure |
|---|---|---|---|---|
| readraw (no dequant) | 54% | 44 | 9 | load ceiling |
| fp (naive) | 16% | 78 | 1 | load serialized behind dequant |
| fp_wide (load block first) | 23% | 1293 | 9 | within-block MLP |
| fp_prefetch (next-block) | 24% | 1298 | 18 | cross-block MLP |
| fp_vec / fp_vec_u3 (uint4 wide, lean) | 19% | 123 | 1-3 | wide loads, lean dequant |
| fp_acc8 (8 independent accumulators) | 23% | 122 | 1 | reduction ILP |

## The finding (refines the root cause)
- Load-MLP gives **1.45x over the NAIVE baseline** (16% -> 24%) -- real, but small.
- Reduction-ILP (8 accumulators) adds nothing beyond that (23%).
- EVERY dequant variant plateaus at ~16-24%, vs readraw's 54%. **The dequant WORK itself (unpack 4-bit ->
  fp -> multiply-accumulate per weight) is the cap, not the loads and not the accumulator chain.**
- So the earlier "memory-level-parallelism / prefetch is THE lever" framing was only partly right: MLP
  recovers a naive kernel's extra loss, but the dominant limiter is the per-weight dequant compute, which
  no load/ILP restructuring removes.

## Make-or-break verdict: hand restructuring does NOT beat tinygrad
My best hand kernel (~24% this session) is BELOW tinygrad's real q4k_gemv_partial (~42% historical, ~27%
scaled to this degraded session). tinygrad's codegen already vectorizes the dequant + captures the MLP/ILP
better than my hand kernels -- on the FULL affine, no less, while mine use a lean q*x. So there is no easy
hand-written load-restructuring win over tinygrad; tinygrad is already near the expressible dequant-GEMV
ceiling (~42%).

## What this means for the decode gap (42% -> llama's 54%)
The residual ~12 points (tinygrad 42% -> llama.cpp 54%) is NOT a load/MLP/ILP gap (those are captured or
ineffective) -- it is lower-level DEQUANT-kernel optimization (the exact bit-twiddling/vectorization of the
4-bit unpack, register allocation, instruction selection) that llama.cpp's hand-asm achieves and tinygrad's
codegen doesn't quite. That is the hand-asm / Writer boundary again, now localized precisely to the dequant
inner loop -- not the load pattern, not the reduction, not tensor cores, not fusion.

## Net
The decode GEMV bottleneck is the per-weight DEQUANT compute. tinygrad expresses it to ~42% of peak; the
load-MLP/ILP levers are either captured by tinygrad or cap at ~24% by hand. The last 42->54% to llama.cpp
is asm-level dequant-loop optimization, the hand-Writer boundary -- consistent with the whole program. The
machine-search-reachable ceiling on this kernel is tinygrad's ~42%; closing further needs hand asm, which
is exactly what the search philosophy is built to avoid.
