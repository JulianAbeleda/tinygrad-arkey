# Runtime-KV Replay-Baking Instrumentation — Scope (2026-06-23)

## Mission
Find the actual `start_pos` concretizer that bakes the opaque KV append's write offset in the **full model**
JIT decode replay (only the capture position is written), given that every isolated reproduction advances
(`docs/runtime-kv-baking-diagnostic-20260623.md` — rope-producer hypothesis REFUTED). Instrument, don't build
a kernel.

## Method
1. **Re-apply** the (reverted) default-off `RUNTIME_KV_CACHE` route to `model.py` (`_init_state` fp16 cache,
   `__call__` `@function` bypass, `_attention` opaque append + owned-tile read). Reproduce the baking.
2. **Var-liveness check**: capture the decode TinyJit graph; locate the `kv_append` PROGRAM node; check whether
   `'start_pos'` is live in the GraphRunner's replay vars (`tinygrad/engine/jit.py:159-169`
   `var_vals_replace`/`updated_vars`) or has been concretized to the capture value. The append's offset advances
   iff `'start_pos'` is patched per replay.
3. **Bisect** the full forward to isolate the component whose `start_pos` use concretizes the shared
   `'start_pos'` expr: reduce to 1 layer; toggle off lm_head/sampling, the `start_pos.unbind()` ctx-gate read,
   the owned-tile `vsp`, the rope `freqs_cis[start_pos]` slice — until the append advances.
4. Name the concretizer + the targeted fix (a graph-capture/var-binding change, NOT a rope kernel).

## Gates / verdicts
- `BAKING_CONCRETIZER_FOUND` — the op/path that concretizes `start_pos` is named + a bounded fix identified.
- `BAKING_VAR_NOT_LIVE_IN_REPLAY` — `'start_pos'` absent from GraphRunner replay vars; root cause = capture binding.
- `BAKING_NOT_ISOLATED` — reproduces only with the full model and resists bisection (escalate).

## Boundaries
Diagnostic/instrumentation only. Default-off route is re-applied for instrumentation then reverted (model.py
clean at the end unless a bounded fix is proven). No default change; no 14B/32B; no new kernel.
