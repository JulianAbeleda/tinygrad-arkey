# Prefill understanding — COMPLETENESS assessment + the measurement-wall boundary

## Complete (solid, e2e-validated, clock-controlled)
- **Levers:** matmul is NOT the prefill lever (4 kernel wins — Tensile 0.999x, transpose-free 0.997x, gate/up
  schedule 1.003x — ALL ~1.00x e2e). Symbolic-KV attention IS (concrete start_pos -> 1.24x byte-identical).
- **Mechanism:** `_attention` slices KV to `0:start_pos+T` (symbolic) -> reduce can't tile/TC -> slow generic
  reduce; concrete unblocks. Explicit TC attention (Option B) 2.56x standalone, blocked only by symbolic KV.

## INCOMPLETE (the genuine residual gap)
The exact WARM per-kernel split + whether the wall is **GPU-kernel-bound vs inter-kernel-overhead-bound**. The
1.00x-matmul result is consistent with TWO undistinguished explanations:
- **(a)** matmuls are a tiny fraction of the warm wall (cold-capture 42% collapses warm), OR
- **(b)** inter-kernel overhead/gaps dominate the wall (matmul time hidden).
Both give the same lever (attention), so the DECISION is unaffected — but a complete quantitative model is not.

## Why it's blocked — every measurement path hits a wall
| path | wall |
|---|---|
| JIT replay (warm, real) | `time_sum_s`=0; no PMC/ProfileRangeEvent emitted on replay |
| JIT capture (per-kernel data) | cold clock (time-split unreliable) + PMC perturbs (GRBM cycle-sum gave impossible 757ms > 332ms wall) |
| eager | untuned/different kernels -> wrong split |
| rocprofv3 | cannot trace HCQ/KFD dispatches at all |

## To close it (infrastructure)
A GPU-timeline profiler that works on HCQ **replay**: (1) make HCQGraph replay record per-kernel times
(profile hook on the graph submit), or (2) SQTT (native, `SQTT=1`) decode of the busy-vs-idle timeline (the
instruction-trace decode failed before; the TIME tokens may still be usable). This is its own infra project; the
lever-level understanding does not require it.
