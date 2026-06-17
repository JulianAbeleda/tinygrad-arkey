# LDS / shared-memory tiling primitive — arc scope (2026-06-17)

A new, scoped research arc. **Not** flash-prefill, **not** `[nn]` model work. The goal is one missing
capability: **cooperative K/V tile reuse in LDS/shared memory**, expressed through tinygrad
`custom_kernel`/codegen, proven measurable, graphable, and killable.

## Why this arc exists

Flash-prefill Increment 2 (`amd-decode-prefill-v2-increment2-phase5-correction-20260617.md`) established:
- **Survived (proven):** `custom_kernel → Ops.PROGRAM → TinyJit` capture/replay; sliced KV; symbolic
  start_pos; multiple outputs; score-free fused causal attention *math*; correctness vs SDPA (single + GQA).
- **Failed (perf):** the custom score-free kernel is **170–760× SLOWER** than SDPA on honest DEBUG=2 GPU time.
- **Root cause:** no LDS reuse. The output dim `d` (W=129) is a GLOBAL lane, so each lane independently
  re-streams all of K/V from HBM → ~129× redundant traffic, ~0.19 TFLOP / 367 GB/s. Score-free but
  **reuse-free**, which is worse than SDPA's materialize-once-reuse.

So the missing primitive is **not "attention."** It is:

> **LDS/shared-memory tile load + barrier + cooperative reuse across lanes + register-resident accumulation.**

Per the new coding principle: *a performance primitive is an operation PLUS its required memory locality.* We
have the operation; we lack the locality. This is `[codegen]`/`[runtime]`/`[test]` territory — the boundary
between Tensor/UOp custom math and real GPU kernel engineering.

## What this arc WILL attempt (phased, kill-switched)

- **Phase 0** (this doc): scope + kill conditions + measurement rules.
- **Phase 1**: inventory how tinygrad represents local/shared memory + barriers today, and whether
  `custom_kernel` can emit them directly (vs only via schedule opts / BEAM, which hangs gfx1100).
  → `docs/amd-lds-tiling-existing-primitives-20260617.md`.
- **Phase 2**: the smallest LDS smoke kernel — cooperative global→LDS load, barrier, LDS→output. No attention.
- **Phase 3**: prove **LDS reuse beats redundant HBM reads** on a synthetic many-lanes-share-one-tile bench
  (DEBUG=2 GPU time). **This is the real primitive proof. STOP here** (re-scope Phase 4+ only if Phase 3 wins).

## What this arc will NOT do

- No `tinygrad/llm/model.py` edits; no Transformer integration; no `FLASH_PREFILL`; no prefill speedup claims.
- No broad abstraction before one tiny primitive wins. No failed probes left wired into runtime.
- No BEAM on gfx1100 if it hangs. If a tiny proof needs broad runtime/codegen surgery → write a design doc and
  STOP (don't do the surgery in this arc).

## Kill conditions (stop immediately + report)

- `custom_kernel` cannot express LDS/shared memory, or a barrier/sync cannot be expressed.
- Kernel silently falls back to CPU; or JIT can't capture/replay an LDS custom kernel.
- LDS reuse is **not faster** than redundant HBM reads (locality primitive ineffective here).
- Repeated compilation faults; or a tiny proof requires broad runtime surgery first.

## Measurement rules (non-negotiable — the Phase-5 lesson)

- **GPU time = DEBUG=2 per-kernel `tm`** (or `GlobalCounters.time_sum_s` with wait). NEVER wall-clock around
  `.realize()` for a perf claim (it measures host dispatch / cache no-ops — that's how Phase 3/4 fooled us).
- Every perf artifact records: per-kernel GPU times, kernel/program count, **compile time separate from exec**,
  largest intermediate, estimated global bytes R/W, one-process-sweep stability, commit SHA, hardware.
- Durable small artifacts under `bench/lds-tiling-primitive-20260617/`. No large raw logs.

## Status statement

**Flash-prefill is BANKED until an LDS/shared-memory tile primitive exists.** This arc tests whether tinygrad
`custom_kernel` can express that primitive at all, and whether it actually buys locality. Success here is *not*
"flash-prefill integrated" — it is "a tiny, tested, graphable LDS tile-reuse primitive that beats redundant
global reads," confirming the missing flash-prefill primitive is real and (maybe) reachable.
