# Runtime-KV Replay Baking — Diagnostic (2026-06-23)

## Finding: **RUNTIME_KV_BAKING_CAUSE_NOT_THE_ROPE_PRODUCER**

Following the CUDA-graph dynamic-scalar guidance (positions must be runtime params, not baked into the captured
graph), the hypothesis was: the model's JIT-replay append-position baking is caused by `start_pos`-dependent
**producers** — specifically RoPE (`freqs_cis[start_pos:start_pos+T]`) making the append receive stale K/V. I
tested this directly. **The hypothesis is refuted.**

## Isolated diagnostics — all ADVANCE correctly
TinyJit a decode step that opaque-appends into a persistent fp16 cache, replay with changing `start_pos`, check
which cache positions get written:

| diagnostic | positions written (2048,2049,2050,2051 expected) |
|---|---|
| raw (position-independent) append src | **all 4** ✓ |
| **rope-like** src: `k = rawk * freqs[start_pos:start_pos+1]` (a `start_pos`-dependent captured slice) | **all 4** ✓ |
| two-jit: prefill jit then a **fresh decode jit** (the model's pattern) | **all 4** ✓ |
| **canonical `.after(store)` prefill** then opaque-append decode (cache.uop = RESHAPE) | **all 4** ✓ |

A `start_pos`-dependent (rope-like) append src **replays correctly** — both the offset and a position-dependent
src advance. So the RoPE producer is **not** the baking cause, and an opaque `rope_kv_append` kernel would
**not** fix the model's baking.

## What this means
- The microbench (`RUNTIME_KV_MICROBENCH_PASS`) and **every** isolated reproduction of the suspected cause
  advance correctly across replays.
- The model's baking (only the capture position 2050 written; the **offset** baked, not just stale data) is
  **emergent in the full 36-layer forward** — not reproduced by the append/rope producer, the two-jit handoff,
  the canonical-prefill handoff, or the RESHAPE cache uop in isolation.

## Recommended next step (corrected direction)
**Do not build an opaque rope+append kernel** — the rope-src is proven not the cause. Instead **instrument the
model's captured decode graph** (RUNTIME_KV route re-applied) to find the actual `start_pos` concretizer:
1. Dump `dec.captured.linear` and check whether the append PROGRAM's `'start_pos'` var is **live** in
   `GraphRunner.vars` / `var_vals_replace` (`tinygrad/engine/jit.py:159-169`) or has been concretized.
2. Bisect the full forward — disable lm_head / sampling / FFN / other layers' caches — until the append
   advances, isolating the model component whose `start_pos` use concretizes the shared `'start_pos'` expr for
   the append node.
3. Only then design the targeted fix (which, given the diagnostic, is a graph-capture / var-binding issue in
   the full model, not a RoPE producer kernel).

## Status
Diagnostic-only. `model.py` clean (the RUNTIME_KV route from `runtime-managed-kv-cache-result-20260623.md`
remains reverted). Artifact: `bench/qk-runtime-managed-kv-cache/baking_diagnostic.json`. No default change.
