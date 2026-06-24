# Option 1 built + measured — TC realizes on the verification matmuls, but it makes decode SLOWER

Date: 2026-06-15. The corrected option 1 (run the verification matmuls in fp16 so RDNA3 WMMA applies)
built and measured end-to-end. Result: a definitive negative.

## The full A/B (concrete batch T=16, the speculative-verification shape)
| config | ms/tok | what changed |
|---|---|---|
| fp32, no TC (baseline) | **18.1** | the no-TC plateau |
| fp16 FFN, no TC (heuristic) | 19.8 | +1.7 ms -- fp16 CAST overhead alone |
| fp16 FFN + TC (apply:4, ALL matmuls) | **26.2** | +6.4 ms more -- TC at batch-16 is net-NEGATIVE |

`warmstart_stats: match 4, apply 4, error 0` -- tensor cores fully applied to every FFN matmul. And the
forward got SLOWER by 8 ms/tok.

## Why TC realized but hurts e2e
1. **fp16 cast overhead**: casting the activation to fp16 adds kernels; +1.7 ms before TC even applies.
2. **TC at small batch is net-negative**: at N=16 the WMMA fragment setup + the PADTO blowup dominate. The
   factored dim 12288 = 256x16x**3**; TC PADTO pads the **3 -> 16** (~5x wasted compute on that axis). So
   the "2x" the isolated matmul showed becomes a loss in the forward context.
3. **Amdahl**: the FFN matmuls are a fraction of the per-token forward (attention, norms, the casts), so
   even a real 2x on them can't overcome the added overhead.

## The honest end of the TC thread
- Single-stream decode: latency/memory-bound -> no kernel lever helps (proven, v_dot4).
- Speculative/batched decode at K=16: TC IS realizable (dtype fix, apply:4) but **net-negative e2e** --
  cast + WMMA-at-small-batch + PADTO-on-factored-dims overhead exceeds the benefit.
- TC would only pay at LARGE batch (prefill, K>=64+, matmul-dominant, overhead amortized) -- which is not
  the speculative-decode regime.

So machine-search-TC does not close decode at single-stream OR speculative batch on this stack. The lever
is real at the isolated-kernel level (2x) and repeatedly fails to translate e2e -- the recurring thesis of
the whole program, now confirmed at the final lever.

## What this rules in / out
- RULED OUT (measured): realizing TC on the K=16 verification matmuls helps decode. It doesn't; it hurts.
- STILL OPEN (not chased): TC at larger speculative K (32-64) where the matmul fraction grows and PADTO
  amortizes -- might net positive, but that's a bigger draft/verify regime, and the PADTO-on-factored-3
  penalty persists unless the kernel layout avoids the factor-of-3.
- The genuine remaining lever for decode is NOT a kernel-vocabulary one (TC/fusion) -- it is fewer bytes
  (lower-bit / MoE) or a fundamentally different (megakernel/persistent) structure tinygrad can't express.

Artifacts: model.py Q4K_UNFUSE (fp16 path), warmstart hook + dump (postrange.py), qk_decode_warmstart.py.
All default-off; normal decode unchanged.
