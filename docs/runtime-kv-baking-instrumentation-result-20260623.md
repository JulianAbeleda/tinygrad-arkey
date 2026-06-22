# Runtime-KV Replay-Baking Instrumentation — Result (2026-06-23)

## Verdict: **RUNTIME_KV_BAKING_TRIGGER_NARROWED_NOT_FULLY_ISOLATED**

Instrumented the full-model `RUNTIME_KV_CACHE` route to find the `start_pos` concretizer that bakes the opaque
append's write offset in JIT decode replay. The **rope-producer hypothesis is refuted** (again, now in-model),
and the trigger is narrowed to **the model's canonical-store prefill before the decode jit** — but the exact
root cause is a deep tinygrad buffer-identity / TinyJit-capture interaction, not isolable without core internals
work. Route re-applied for instrumentation, then **reverted** (`model.py` clean, default decode byte-identical).

## Key fact
The captured decode graph's `kv_append` PROGRAM has `ProgramInfo.vars = {'start_pos'}` — **`start_pos` IS a
declared live runtime var on the append**. Yet replays don't advance the write. So this is not a "var was never
declared / was concretized at the kernel arg" case at the surface; something upstream prevents the per-replay
patch from taking effect (or the captured append writes a fresh buffer).

## Hypotheses tested — all refuted
| hypothesis | result |
|---|---|
| RoPE producer bakes the data (`freqs_cis[start_pos]` slice) — *user hypothesis* | **REFUTED** — isolated append with a `start_pos`-dependent (rope-like) src advances; model bakes the **offset** (only capture position written), not stale data |
| `@function` prefill | **REFUTED** — one `@function` forward then fresh decode jit advances |
| short canonical prefill jit (3 steps) | **REFUTED** — advances |
| high-`start_pos` prefill (2045–2047, <2048) | **REFUTED** — advances |
| 256-step prefill from 0 | **REFUTED** — advances |
| filled cache via `cache.assign(numpy[0:2050]).realize()` | **REFUTED** — advances |
| cache `uop` RESHAPE / materialization | **REFUTED** — `cache_kv.uop` is `Ops.RESHAPE` in **both** advancing and baking cases; `.contiguous().realize()` keeps RESHAPE and does not fix it |
| runtime route fires during prefill | **REFUTED** — batched prefill (T>1, route does NOT fire) still bakes |

## Narrowed trigger
The baking is triggered **specifically when the cache is filled via the model's canonical store before the
decode jit** — both the full token-by-token (2050-step) prefill **and** the batched (T>1) prefill **bake**. But
filling the **same** `[0:2050]` via a direct `cache.assign(numpy).realize()` **advances**. The only difference
between assign-fill (advances) and canonical-store-fill (bakes) is the buffer/scheduler state left on
`cache_kv` — and both leave `cache_kv.uop = RESHAPE`, so it is **not** the visible uop op. It is a deeper
buffer-identity / pending-store / TinyJit-capture-state difference.

Observation: with a canonical-store prefill, the decode jit's append writes only the **eager (call-0)** position;
the captured/replayed calls (call-1+) do **not** persist to `block.cache_kv` — i.e. the captured graph's append
writes a fresh buffer **only** in this post-canonical-store-prefill state.

## Recommended next step (precise)
Compare, byte-for-byte, the `cache_kv` Tensor / UOp / BUFFER identity **after canonical-store-fill vs
assign-fill** (both RESHAPE) — `cache_kv.uop.base`, the underlying `Buffer` object identity, and any lingering
`AFTER`/`STORE` in the realize graph that the decode-jit capture would re-trace. Then either:
1. force `cache_kv` back to a **pristine BUFFER identity** after the canonical-store prefill (clone into a fresh
   realized buffer) before the decode jit; or
2. fill the cache via the **opaque append during prefill** too (needs a short-ctx-correct attention read to
   avoid the `gqa` functional-reduce read-after-write hazard at ctx<2048).

This is a tinygrad buffer-identity / TinyJit-capture interaction, **not** the RoPE producer and **not** a new
kernel. The `@function` bypass + fp16 cache fixes from `runtime-managed-kv-cache-result-20260623.md` stand.

## Status
Instrumentation-only. `model.py` clean (RUNTIME_KV route reverted; default decode byte-identical
`[279, 1156, 22148, …]`). Artifacts: `bench/qk-runtime-managed-kv-cache/instrumentation.json`. No default
change; no 14B/32B; no new kernel.
