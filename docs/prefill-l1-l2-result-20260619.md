# RESULT — prefill L1/L2: gate/up schedule (L2) REFUTED (1.00x e2e); concrete-KV (L1) VALIDATED 1.24x

Tested the two levers from the exact-split (`prefill-exact-split-result-20260619.md`). Clean clock-controlled
interleaved A/B via model.__call__ (warmstart applies) + correctness.

## L2 — gate/up warmstart schedule: kernel 1.4x, e2e 1.00x → REFUTED
- Standalone search (`extra/qk_gateup_schedule_search.py`): production gate/up = 40.0 TFLOPS; new schedule
  `(TC, UPCAST(0,2), UPCAST(1,4), UNROLL(0,16), LOCAL(1,4))` = 47.7 TFLOPS = **1.19x standalone**.
- In-model (`__call__` capture): the new schedule APPLIES (kernel r_16_192→r_16_64, GFLOPS 16090→22996 ≈ **1.4x
  on the gate/up kernel**) — confirmed it's not a no-op.
- **But clean e2e A/B (`extra/qk_gateup_sched_ab.py`, forced-high clock) = 1.003x.** rel_err 0. The faster gate/up
  kernel does NOT move the forward → **prefill is NOT gate/up-matmul-bound.**

## L1 — concrete-KV (concrete start_pos): 1.24x e2e → VALIDATED
- Clean interleaved A/B, concrete `start_pos=0` (KV=512 concrete) vs symbolic `vsp.bind(0)`:
  **CONCRETE 1540 tok/s (332ms) vs SYMBOLIC 1243 tok/s (412ms) = 1.239x**, rel_err 0 (byte-identical output).
- Concrete KV lets tinygrad apply concrete-shape codegen to the attention (the symbolic `KV=start_pos+T` blocks it).

## THE ANSWER — why the matmul TFLOPS don't translate to e2e
Across FOUR experiments, every matmul-KERNEL improvement gave ~1.00x e2e (Tensile 0.999x, transpose-free 0.997x,
gate/up schedule 1.003x), but the ATTENTION shape-fix gave 1.24x. **The prefill forward is NOT bottlenecked by
matmul-kernel throughput — it's bottlenecked by the symbolic-KV ATTENTION** (non-TC, the ~25% that runs at
106-4415 GFLOPS). That's why "63 TFLOPS matmul kernel" doesn't convert to e2e: the matmul isn't the critical path;
the symbolic attention is. The exact-split's L2 (gate/up efficiency) was a real kernel inefficiency but NOT the
e2e bottleneck; L1 (attention symbolic shape) IS.

## The prefill win (shippable)
**Concrete-start_pos prefill for the common single-chunk case (prompt ≤ 512, start_pos=0) → 1.24x, byte-identical.**
One cached concrete-0 jit, reused across requests. Chunked prefill (start_pos>0) keeps symbolic OR caches a
concrete jit per start_pos (recompile cost amortized over server reuse). Further: explicit TC attention (Option B,
2.56x standalone) on top of concrete KV could add more — the concrete shape unblocks it (the probe's salvage path).

## Files
`extra/qk_gateup_schedule_search.py`, `extra/qk_gateup_sched_ab.py`. Split: `prefill-exact-split-result-20260619.md`.
TC-attention probe: `amd-prefill-tc-attention-probe-20260617.md`.
