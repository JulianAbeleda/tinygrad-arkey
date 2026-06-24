# ANSWERED (reliably) — why isolated 65 TF doesn't transfer: the matmul is ~24% of the WALL, not 74%

Diagnosed WITHOUT the unreliable per-kernel graph timestamps -- via a JITted FFN-only differential (D1).

## D1 (reliable: JITted block._feed_forward A/B, Tensile vs WMMA, interleaved)
- FFN-JIT (3 matmuls + silu + contiguous glue, 154.6 GFLOP): **WMMA 15.38ms = 10.0 TF effective | Tensile 15.30ms
  = 10.1 TF | speedup 1.005x**, rel_err 0, route confirmed {gateup:2, down:1}.
- Pure matmul at the measured 42 TF would be 154.6/42 = **3.68ms = ~24% of the 15.38ms wall**. The other ~76%
  (~11.7ms) is NON-matmul: the silu/contiguous glue + per-kernel host-dispatch (FFN = ~8 kernels x ~1.5-1.9ms
  dispatch, consistent with the campaign's ~1.3ms/kernel host overhead).
- Tensile (1.56x on the matmul) -> 1.005x e2e BECAUSE it speeds up only ~24% of the wall: 0.24/1.56 + 0.76 = 0.91 of
  the time -> ~1.06x ceiling, swamped by noise -> ~1.00x. (H1 "kernel doesn't perform via custom_kernel" is moot;
  the answer is H2: the matmul is a minority of the WALL.)

## The reconciliation (resolves the campaign's apparent contradictions)
- PMC atlas "prefill compute/WMMA-bound, matmul ~74%" = fraction of GPU-BUSY time. TRUE.
- D1 "matmul ~24% of the WALL" = the WALL also includes glue + per-kernel HOST-DISPATCH (busy-wait) that is NOT in
  GPU-busy. Both are right: matmul is 74% of GPU-busy but ~24% of the wall.
- => **the prefill WALL is glue+host-dispatch-bound, not matmul-GPU-bound.** A faster matmul kernel (Tensile 66,
  or any WMMA improvement) can't move the wall -- this is the precise, MEASURED reason the isolated 65 TF (and the
  whole 42->66 effort) doesn't transfer e2e. It VINDICATES the earlier host-dispatch-bound direction (I over-
  retracted it; SCLK-invariance only ruled out compute-bound, not the host-dispatch/glue component of the wall).

## Implication for the lever (consistent with everything shipped)
The prefill wall lever is **FEWER KERNELS / less per-kernel dispatch + less glue**, NOT a faster matmul. That is
exactly what concrete-KV does (fewer/cheaper attention kernels -> 1.24x). Faster-matmul approaches (Tensile,
WMMA-occupancy, FMA codegen) all -> ~1.0x because the matmul isn't the wall bottleneck. So the 42-TFLOPS "WMMA
ceiling" was largely a RED HERRING for e2e prefill: even rebuilding it to 66 would give ~1.06x, not 1.4x.

## Remaining (optional D2)
Split the ~76% into GPU-glue (silu/contiguous kernels) vs host-dispatch (busy-wait): time a 3-matmuls-only jit
(no silu/contiguous) vs the full FFN-JIT. If matmuls-only is still ~10ms -> host-dispatch dominates; if ~5ms ->
glue dominates. Either way the matmul is not the wall bottleneck; D2 only refines which non-matmul piece to attack.

## Files
/tmp/d1_ffn_jit.py (the reliable differential). Supersedes the speculative tensile-transfer-audit + its CORRECTION:
the real reason is matmul-is-minority-of-wall, measured cleanly.
