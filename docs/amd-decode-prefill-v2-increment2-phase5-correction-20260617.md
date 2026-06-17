# Prefill v2 — Increment 2, Phase 5: CORRECTION — flash-prefill REFUTED on performance (2026-06-17)

**This corrects an error.** Phases 3–4 reported the custom flash-prefill kernel at ~2.7–2.8× over SDPA. Those
speedups were **measurement artifacts** — wall-clock around `.realize()` in a warm loop measured host dispatch
(and cache-hit no-ops), NOT GPU execution. The authoritative per-kernel GPU time (DEBUG=2 `tm`) tells the
opposite story. (Exactly the failure my own `amd-decode-measurement-confounds` note warns about: control
cache/launch/clock for short GPU benches — I didn't.)

## Honest numbers (DEBUG=2 GPU kernel `tm`, compute kernels, excl. one-time device-init copy)

`extra/qk_flash_prefill_phase5.py`, Hd=128, T=512, gfx1100:

| case | flash GPU ms | SDPA GPU ms | flash vs SDPA |
|---|---:|---:|---|
| single-head KV=512 | 45.2 | 0.3 | **171× SLOWER** |
| single-head KV=3584 | 332.4 | 1.0 | **338× SLOWER** |
| GQA KV=512 | 1374.3 | 1.8 | **756× SLOWER** |
| GQA KV=3584 | (too slow / faulted to measure) | 9.6 | ≫ |

The flash kernel runs at **~0.19 TFLOP / ~367 GB/s** regardless of shape.

## Root cause — score-free WITHOUT LDS reuse is memory-bound

In formulation B the output dim `d` (W=Hd+1=129) is a **GLOBAL lane**, so each of the 129 lanes independently
**re-streams all of K (and V) from HBM** to recompute the q·k dot — ~129× redundant reads (≈178 GB at KV=3584
vs SDPA's ≈1 GB). The kernel is bandwidth-bound at a fraction of peak. Being "score-free" (no `[T,KV]`
materialization) does NOT help if the alternative is re-reading K/V per lane. SDPA materializes scores but
**reuses** the data; that wins by 2–3 orders of magnitude here.

The loop-invariant-hoist I observed in Phase 3 was real but only removes *recompute within a thread* — it does
**not** share K/V *across* the `d` lanes (that needs shared memory).

## Why this is gated, not fixable cheaply

A real flash-2 stages each K/V tile in **LDS (shared memory)** and reuses it across the head/dim lanes — that's
the only way to kill the redundant HBM traffic. In tinygrad that means GROUP/LOCAL→LDS opts, which **BEAM**
would search for, but **BEAM hangs gfx1100** (documented), and hand-rolled LDS in a custom kernel is
dangerous-power surface. So flash-prefill is **the same wall-class as the decode overlap lever**: real but
gated behind codegen/runtime surface this fork won't add now.

## What the ladder DID establish (still valid)

The custom-kernel path is correctness- and integration-viable — only **performance** is refuted:
- bridge: `custom_kernel → Ops.PROGRAM → TinyJit` capture/replay ✓ (`test_custom_kernel_jit_bridge.py`)
- capabilities: sliced KV, **symbolic start_pos**, multiple outputs ✓ (`test_flash_prefill_custom_kernel_bridge.py`)
- expressibility: fused score-free causal attention is expressible (formulation B; single-kernel online softmax
  A is linearizer-rejected) ✓ (`test_flash_prefill_custom_kernel.py`)
- correctness: exact vs SDPA (err ~1e-3) single-head AND GQA multi-head ✓
These are kept as proofs. Their **speedup** claims (Phase-3/4 artifacts) are **superseded by this phase-5
measurement** and should not be trusted.

## Verdict / status

**Flash-prefill BANKED: correct + integration-viable, but ~170–760× SLOWER than SDPA without LDS tiling.** Do
NOT integrate into the model (Phase 6 cancelled). Prefill v2 rests at **Increment 1** (the ~13× FFN win, real
and quality-gated). Attention stays SDPA.

## Resume pointers (if reopened)
- The lever is LDS tiling (flash-2): stage K/V tiles in shared memory, reuse across lanes. Needs a way to
  express GROUP/LOCAL→LDS in a custom kernel without BEAM (which hangs gfx1100), or to call rocBLAS/a vendor
  flash. Re-fire gate: `extra/qk_flash_prefill_phase5.py` (require flash GPU `tm` < SDPA).
- Methodology lesson (re-banked into `amd-decode-measurement-confounds`): for GPU timing use DEBUG=2 per-kernel
  `tm` (or `GlobalCounters.time_sum_s` with wait), NEVER wall-clock around `.realize()` in a warm loop.

Anchors: `amd-decode-prefill-v2-increment1-20260617.md` (the real win), `amd-decode-prefill-v2-increment2-20260617.md`
(the earlier gated bank — this confirms it at the kernel level), `extra/qk_flash_prefill_custom.py`.
