# Runtime-KV Core Engine — Design Doc (2026-06-23)

## 1. Chosen design
**Design 1 — Mutable Buffer Op / State Object** (a `KVState` owning a persistent buffer; `store_slice(start_pos,
k,v)` mutates it; `read_prefix(start_pos)` reads it, ordered, with no full-buffer materialization, persisting across
replays). This is the semantically-honest target and matches production KV-cache systems.

## 2. Rejected designs
- **Pre-graph append**: impossible — transformer layers interleave append/read (layer N k/v depend on layer N−1's
  attention read). (Superseded the earlier scope.)
- **Design 4 (bounded KV alias rule)**: prior attempts hit symbolic-alias / read-after-write walls → `TOO_BROAD`.
- **Designs 2/3**: viable as sub-mechanisms but do not resolve the actual blocker found below.

## 3. Exact semantics (target)
A replayed graph writes `cache[layer,kv,head,start_pos,dim]` and a later in-graph read sees it; the mutation
persists into the next replay; ordering is explicit; **no full-MAXC materialization (`E_49152`)**; unsupported
cases fall back to the canonical materialized path.

## 4. The actual blocker (discovered Phase 2→3) — why this is `RUNTIME_KV_ENGINE_DESIGN_TOO_BROAD_STOP`
The toy proof **passes** and the one-layer proof **fails**, and the difference precisely isolates the root cause:

- **Toy (`TOY_MUTABLE_REPLAY_PASS`)**: persistent buffer + opaque append + owned read under **plain TinyJit** —
  mutation persists, no materialization, rel_rmse e-7.
- **One-layer (`ONE_LAYER_RUNTIME_KV_CORRECTNESS_FAIL`)**: the same opaque append, with **real model k/v**, writes
  **NaN to the cache given FINITE input k** (absmax 209), from decode step 1. Independent of cache-fill method
  (canonical-store *and* assign-fill both NaN); forcing k/v contiguous has no effect; forcing `src.realize()` is
  **disallowed** — `ALLOW_DEVICE_USAGE` (`tinygrad/device.py:25`).

**Root cause: the model decode runs under `callify` (pure-function compilation).** Callify forbids the eager
mutable-buffer realize that the opaque append relies on, so inside the callified forward the append's `src` is not
realized in order → the kernel reads unrealized data → writes NaN. The toy passes *only because it is plain TinyJit,
not callified.* This is **not** the append kernel, persistence, GraphRunner args, dtype, cache, or prefill (all
previously refuted) — it is the **execution model**.

## 5. UOp/scheduler/JIT changes required (and why they're out of bounded scope)
To make Design 1 work, callify/`@function` must support **mutable in-graph state**: an in-place store + later load
within one pure-function graph that (a) orders the store before the load, (b) persists the buffer across replays,
and (c) does not require materializing the buffer as a pure value. That is a change to tinygrad's **Tensor purity /
pure-function execution model** — exactly the scope's hard stop ("solution requires rewriting the entire Tensor
purity model → STOP"). The alternative (run the decode *without* callify) forfeits the JIT/graph compilation that
makes decode fast — self-defeating.

## 6. HCQ graph changes
N/A at this layer — the blocker is upstream (callify/schedule), not HCQ. GraphRunner arg-patching and HCQ buffer
handling were previously proven correct.

## 7. Fallback
Default canonical materialized path (current default owned route) — unchanged, byte-identical.

## 8. Test ladder (result)
Phase 2 toy **PASS** → Phase 3 one-layer **FAIL (callify)** → **STOP** (do not proceed to full-model/W==D).

## 9. Risks / rollback
All diagnostic source changes reverted; default decode byte-identical. No rollback needed.

## Verdict: `RUNTIME_KV_ENGINE_DESIGN_TOO_BROAD_STOP`
The design (Design 1) is correct as a spec, but its implementation requires modifying tinygrad's pure-function
(callify) execution model to permit mutable in-graph cache state — a core Tensor-purity change beyond the bounded
project scope. Stopped per the scope's stop rules.
