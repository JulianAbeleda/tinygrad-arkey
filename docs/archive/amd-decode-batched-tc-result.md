# Batched-decode TC realization — the approach, and why tinygrad-on-RDNA3 can't realize it

Date: 2026-06-15. The make-or-break for batched-decode machine search: can we put tensor cores on the
verification matmuls in the actual decode forward? Result: **TC is reachable for an isolated matmul (2x),
but blocked in the model forward through every path** — the precise, on-hardware proof of the
expressibility gap that a TileLang-class tile vocabulary fills.

## The approach (documented)
1. Single-stream decode is latency/memory-bound -> no kernel vocabulary helps (v_dot4 proved it).
2. Speculation supplies a CONCRETE batch K -> verification GEMMs become the bottleneck.
3. The batched plateau (~14 ms/tok, ~2% peak) is COMPUTE-bound with ZERO tensor cores (PROFILE-measured).
4. So TC is the lever. The loop already FINDS TC schedules (Step 2). Realize TC in the forward = the win.
   Realize via a warm-start hook in `apply_opts` (force the loop's TC schedule on matching matmuls, no BEAM).

## The make-or-break cascade (each ruled out a hypothesis, all measured)
| attempt | result | what it ruled out |
|---|---|---|
| isolated matmul (16,4096)@(4096,12288) + TC | **15.67 TF (2x heuristic 7.8)** | TC IS reachable on RDNA3 for this shape |
| warm-start, symbolic-batch forward | match 4, **error 4** | symbolic JIT batch -> TC errors |
| concrete-batch + standalone (12288,4096,16) opts | match 4, **error 4** | axis layout mismatch (12288 on axis0 vs 16) |
| concrete + model-layout (16,out) opts | match 4, **error 4** | same signature, still errors -> not just layout |
| **`(x@W.T).silu()` + TC** | **ERROR: "no reduce ops for TensorCore"** | **FUSION blocks TC (the silu epilogue)** |
| unfused `(x@W.T).contiguous()` + TC | **15.67 TF, TC applies** | unfusing the isolated matmul recovers TC |
| model forward + Q4K_UNFUSE + warm-start | match 3, **error 3** | the model's kernel STILL differs (3D batch-1, linear epilogue, precompiled block) |

## Root cause (measured, definitive)
- **Fusion blocks tensor cores in tinygrad**: `(x@W.T).silu()` -> "no reduce ops for TensorCore". The
  matmul's reduce is buried under the fused activation epilogue, so the TC opt can't find it. This is the
  W2 "fusion XOR tiling" wall, now measured precisely in the decode forward.
- **And even unfusing isn't enough in the model**: the model's matmul kernels carry a cascade of structural
  differences from an isolated `x@W.T` (a leading batch-1 dim, the `nn.Linear` epilogue, the
  `@function(precompile=True)` block context) -- each makes the TC opt error. The isolated matmul takes TC
  (2x); the model's never does, through symbolic OR concrete batch, standalone OR layout-matched opts,
  fused OR unfused.

## Conclusion: this is the expressibility gap, on real hardware
The TC lever is real (2x at the kernel level) and physics-justified (the plateau is no-TC compute-bound).
But **tinygrad-on-RDNA3 cannot express the TC'd decode-verification kernel** -- its automatic
fusion/lowering keeps producing matmul kernels that reject tensor cores, and the warm-start can't force
TC through the cascade of structural walls. This is exactly what a TileLang-class tile vocabulary is built
to avoid: author the kernel with TC + fusion + layout together, so TC is guaranteed to apply.

So machine-search-decode in the batched regime is blocked on RDNA3 not by the loop (it finds the TC
schedule) and not by physics (TC is 2x) but by **tinygrad's kernel expressibility**. Realizing it requires:
- (option 1) adding fused-matmul-with-TC tile primitives to tinygrad's vocabulary (so its fusion doesn't
  kill TC) -- deep renderer/scheduler work, or
- (option 2) a tile-DSL (TileLang-class) with an RDNA3/WMMA backend -- which doesn't exist yet (TileLang is
  CDNA-only).

This is the recurring shape of the whole program, now at its sharpest: the kernel-level win is real
(TC 2x), the loop finds it, and e2e realization is gated by tinygrad's RDNA3 expressibility. The honest
next move is a deliberate decision on option 1 vs option 2 (or CDNA hardware where TileLang already works),
not another tinygrad force-it attempt -- those are exhausted.

## Artifacts (default-off, normal decode unchanged)
`apply_opts` warm-start hook (postrange.py `_WARMSTART_OPTS`), `extra/qk_decode_warmstart.py`, the
`Q4K_UNFUSE` gate in model.py `_feed_forward`. PROFILE-pkl per-kernel + WMMA measurement established.
