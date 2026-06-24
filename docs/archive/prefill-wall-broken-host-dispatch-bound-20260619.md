# BREAKTHROUGH — measurement wall BROKEN: prefill is HOST-DISPATCH-bound (~66% of wall), not GPU/matmul-bound

Broke the JIT-replay measurement wall via `ProfileGraphEvent` (graph/hcq.py `collect_timestamps`): the HCQ graph
records per-jit-item GPU timestamps on WARM replay (dur = sigs[en_id]-sigs[st_id]). Was looking for the wrong
event type before (ProfileRangeEvent/PMC emit nothing on replay; the GraphEvent was there). Tool:
`extra/qk_prefill_graph_profile.py`.

## The complete measured picture (warm, real GPU timestamps + perf_counter wall)
| start_pos | wall | GPU-span (compute) | host-dispatch | kernels | matmul %GPU | attention %GPU |
|---|---:|---:|---:|---:|---:|---:|
| concrete=0 | 339ms | **115ms** | **224ms (66%)** | 249 | 86% | 6% |
| symbolic | 419ms | 170ms | 249ms (59%) | 321 | 70% | 26.5% |

- **GPU compute = 115ms but wall = 339ms → 224ms (66%) is HOST DISPATCH.** Confirmed: `model.__call__` adds 0ms
  (direct `prefill_v2_jit` replay = 340ms = same) → the 224ms is INSIDE the HCQ graph-replay dispatch.
- **Wall ∝ kernel count: 339/249 = 1.36ms/kernel, 419/321 = 1.31ms/kernel** → per-kernel host-dispatch overhead
  (~1.3ms/kernel), NOT uncaptured GPU (would not scale with compute-kernel count). Within the GPU timeline: 0% gaps.

## THE COMPLETE ANSWER — why matmul TFLOPS don't translate to prefill e2e
**Prefill is HOST-DISPATCH-bound: the wall is ~66% per-kernel HCQ-graph-replay dispatch (~1.3ms × 249 kernels),
GPU compute is only ~34%.** The matmul (86% of the 115ms GPU = ~29% of the wall) is a minority of the wall, so
speeding it 1.2x saves ~5% — lost in the host-dominated wall + clock noise → ~1.00x. This is why ALL FOUR
matmul-kernel wins (Tensile, transpose-free, gate/up-schedule, +standalone) gave ~1.00x e2e. The kernels were never
the bottleneck — **the kernel COUNT / dispatch is.**

## What this means for the lever (re-ranked)
1. **FEWER KERNELS is the prefill lever.** concrete start_pos = 249 vs symbolic 321 kernels -> 1.24x e2e. The win is
   the **kernel-count reduction** (host dispatch), not the GPU (attention concrete also has 48 vs 99 attn kernels).
   Symbolic KV doesn't just slow attention GPU — it SPLITS attention into ~2x more kernels (more dispatch).
2. **Faster HCQ graph-replay dispatch** (~1.3ms/kernel is high) would cut the 224ms directly — a RUNTIME lever
   (graph/hcq.py replay), orthogonal to model code, and it would help EVERY model. Highest-leverage if real.
3. ~~matmul kernel speed (Tensile/schedule/transpose-free)~~ — GPU is 34% of wall, hidden; ~1.00x. Dead.
4. Explicit TC attention: helps only via GPU (6% of wall) AND fewer kernels — modest unless it cuts kernel count.

## Open reconciliation (minor)
Decode is "GPU-bound (W≈D, host~0%)" with 7 kernels; prefill (249 kernels) is dispatch-bound. Consistent if
dispatch is per-kernel (~1.3ms): decode's 7 kernels hide dispatch under GPU; prefill's 249 don't. The ~1.3ms/kernel
dispatch cost on HCQ-graph-replay is itself worth investigating (is it intrinsic or a tinygrad inefficiency?).

## Files
`extra/qk_prefill_graph_profile.py` (the wall-breaking tool — ProfileGraphEvent per-kernel + wall vs GPU-span).
Prior: `prefill-understanding-completeness-20260619.md` (now RESOLVED), `prefill-l1-l2-result-20260619.md`.
