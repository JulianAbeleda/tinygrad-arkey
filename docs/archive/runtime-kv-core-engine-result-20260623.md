# Runtime-KV Core Engine — Result (2026-06-23)

## 1. Verdict: `RUNTIME_KV_CORE_CAPABILITY_BLOCKED` (root cause precisely isolated to the `callify` execution model)
Owner-authorized full Lane 2 pursuit. The toy capability **passes**; the one-layer transformer proof **fails**, and
the failure is now isolated to the deepest possible cause across the entire runtime-KV saga: **tinygrad's `callify`
(pure-function compilation) execution model disallows the mutable in-graph cache realize the capability requires.**
Stopped at Phase 3 per the scope's stop rules (no full-model/W==D). No source/default changes (all diagnostics
reverted; default decode byte-identical `[279,1156,22148,…]`).

## 2. Design chosen
Design 1 — Mutable Buffer Op / State Object (`docs/runtime-kv-core-engine-design-20260623.md`). Pre-graph append
rejected (transformer interleaving); Design 4 rejected (`TOO_BROAD` alias analysis).

## 3. Engine semantics (target)
In-graph cache store + later cache load, ordered, persistent across replay, without full-MAXC materialization,
fallback-safe. The blocker prevented reaching an implementation.

## 4. Toy proof — `TOY_MUTABLE_REPLAY_PASS` (`bench/qk-runtime-kv-core-engine/toy.json`)
Persistent fp16 buffer + opaque append at runtime `start_pos` + owned-tile prefix read, **plain TinyJit** replay:
persistence rel_rmse e-7, reset clean, **no full-MAXC materialization**. The capability works when **not callified**.

## 5. One-layer proof — `ONE_LAYER_RUNTIME_KV_CORRECTNESS_FAIL` (`bench/qk-runtime-kv-core-engine/one_layer.json`)
One real transformer block, real q/k/v producer, opaque append, owned read, real prefill, eager multi-step:
- prefill cache finite; **append input k FINITE (absmax 209)**; **cache after append = NaN** → garbage token (151936)
  from step 1.
- NaN under **both** canonical-store prefill and assign-fill → not the cache fill.
- forcing k/v contiguous → no effect.
- forcing `src.realize()` → **disallowed** (`ALLOW_DEVICE_USAGE`, `tinygrad/device.py:25`).

**Root cause**: the model decode runs under **`callify` (pure-function compilation)**. Callify forbids the eager
mutable-buffer realize the opaque append needs, so inside the callified forward the append's `src` is not realized
in order → the append reads unrealized data → writes NaN. **The toy passes only because it is plain TinyJit, not
callified.** This finally explains the entire saga (toy always passed, model always baked): it is the **execution
model**, not the append kernel / persistence / args / dtype / cache / prefill (all previously refuted).

## 6. Full-model shadow route
**Not reached** (Phase 3 stop). 

## 7. W==D result
**Not reached.** (Predicted upside if ever unblocked: ~+11–13% → llama parity, per the standing MAXC-shrink.)

## 8. Materialization removal evidence
The opaque route does remove `E_49152` in the toy graph, but the one-layer correctness gate fails first, so no
in-model materialization-removal claim is made.

## 9. Correctness
Toy: correct (e-7). One-layer: **fails** (NaN/151936). The decisive, reusable correctness fact: **callify ⇒ no
eager mutable-state realize ⇒ the opaque append cannot order its src ⇒ NaN.**

## 10. Hardening / default decision
**Not reached.** Runtime-KV stays deferred as **core-engine work** requiring a change to tinygrad's pure-function
(callify/Tensor-purity) model — explicitly the scope's hard stop ("rewriting the entire Tensor purity model →
STOP"). The only non-purity-changing alternative (run decode without callify) forfeits the JIT/graph speedup that
makes decode fast.

## 11. Files changed
New: `docs/runtime-kv-core-engine-design-20260623.md`, this result doc, and
`bench/qk-runtime-kv-core-engine/{authority,toy,one_layer}.json`. **No `tinygrad/` source or default changes** —
the RUNTIME_KV diagnostic route + the `src.realize`/`contiguous` probes were re-applied then **reverted**; default
decode byte-identical. Updated `docs/README.md`, `structure/Development/session-handoff.md`.

## 12. Git status
`model.py` + `extra/qk_kv_cache_state_token.py` clean (diagnostics reverted). New design/result docs + 3 artifacts +
doc updates only. No default flip, no machine search, no 14B/32B, no attention/GEMV work.

## Bottom line
The ~+11% (→ llama parity) runtime-KV prize is **real and on the critical path**, but it is **gated by tinygrad's
core `callify`/Tensor-purity execution model** — the deepest, now-precise root cause. It is a core-engine /
Tensor-purity project (out of bounded scope), not a model/route/kernel task. With attention + GEMV at parity, **8B
decode is complete at the bounded layer (~88–89% of llama)**; the remaining lever requires tinygrad-core mutable-
state-in-pure-graph support.
