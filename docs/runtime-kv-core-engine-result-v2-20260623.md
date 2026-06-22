# Runtime-KV Core Engine — Result v2 (2026-06-23) — MAJOR CORRECTION

## Verdict: `RUNTIME_KV_CORE_ENGINE_NO_WD_TRANSFER` — but with two findings that **supersede the prior "callify hard-stop"**

Executing the full capability scope produced a **byte-identical, non-baking** runtime-KV route, which corrects the
earlier conclusion, and **re-frames the remaining +11% lever as a bounded tile/cache-layout change, not a
core-runtime / Tensor-purity project.** No source/default changes (the route is correct but yields 0% W==D as
implemented, so not shipped).

## Finding 1 (correction): runtime-KV correctness IS achievable — the bake was the *opaque append*, not callify
A new route, **`RUNTIME_KV_CORE` = native cache store (`cache[slice].store(stack(k,v))`) + owned tile reads the
cache halves as `AFTER`-nodes (`cache[0,0].after(store)`)**, behind a default-off flag with the `@function` bypass:
- **64-token byte-identical to default on two prompts** (`bench/qk-runtime-kv-core-engine/full_model_shadow.json`).
- Default (no flag) decode unchanged.

This **supersedes** `docs/runtime-kv-core-engine-result-20260623.md`'s "callify pure-function model hard-stop." That
bake was specific to the **opaque `custom_kernel` append** (which can't realize its src under callify). The
**native store + `AFTER`-read does not bake** — runtime-KV correctness is not fundamentally blocked.

## Finding 2 (re-frame): the materialization is the owned tile SLICING the cache, not a missing persistence capability
W==D (interleaved repeats, ctx1024 & 4096): **0% vs default** (−0.2/+0.2/−0.8% = noise). The `AFTER`-read of cache
**slices** (`cache[0,0]`) still materializes the K/V halves — exactly as the **default owned route** does (it also
reads `assigned_kv[0,0]`, a slice). So:

> The ~1.5ms/token materialization (the +11% MAXC-shrink lever) is **the owned tile reading the cache via slices**,
> which callify must materialize. It is **NOT** a missing runtime-managed-KV persistence capability, and **NOT** a
> Tensor-purity limitation.

The entire multi-task "runtime-KV core persistence / callify / @function persistence" framing was **over-scoped**.
Correctness was always achievable; the only real lever is removing the slice materialization.

## The real, bounded lever: buffer-identity whole-buffer read
`callify.transform_precompiled_call` (read this task): a precompiled-call input that is an **`AFTER` node is not
force-contiguous**, and `_precompiled_output_redirect` returns a **`BUFFER` with `has_buffer_identity()` directly**
(no store/after materialization). So reading the cache as a **whole buffer** (not a slice) under an `AFTER`-node
would avoid materialization. Two bounded ways:
- **(a) Separate K/V cache buffers** (`cache_k`, `cache_v`): each is a whole buffer with identity → tile reads
  `cache_k.after(store_k)` directly. Needs a one-time prefill→decode copy of the cache halves (per prompt, not per
  token). Model change, in scope.
- **(b) Kernel V-offset**: pass the one `cache_kv` buffer; the tile reads K at offset 0 and V at offset
  `Hkv·MAXC·Hd`. One-line kernel-arg change (touches the .hip).

Either is a **bounded tile/cache-layout change**, not a core-engine/Tensor-purity project. Expected upside ~+11% →
llama parity (the standing MAXC-shrink). This is now a focused next experiment, not a multi-day engine project.

## Phase ladder result
- P0 design / P1 toy: PASS (prior).
- P2/P3 **correctness**: **PASS** (byte-identical native-store+AFTER-read) — corrects the prior fail.
- P4 **W==D**: **NO TRANSFER** (slice materialization remains; `RUNTIME_KV_CORE_ENGINE_NO_WD_TRANSFER`).
- P5+: not reached (no W==D to harden).

## What changed / files
**No `tinygrad/` source or default changes** — the `RUNTIME_KV_CORE` route (native store + AFTER-read) was
implemented, validated byte-identical, measured at 0% W==D, and **reverted** (a correct-but-no-gain route isn't
worth shipping; default decode byte-identical `[279,1156,22148,…]`). New: this result doc + updated
`bench/qk-runtime-kv-core-engine/full_model_shadow.json`. README + session-handoff updated. The prior result doc
(`runtime-kv-core-engine-result-20260623.md`) is **superseded** (callify hard-stop corrected).

## Recommendation
The +11% → llama parity is reachable via the **buffer-identity whole-buffer cache read** — a bounded tile/cache
change (separate K/V buffers, option (a)), not a core-engine project. That is the precise, small next experiment.
The "runtime-managed KV / TinyJit persistence / Tensor-purity" lane is **retired as mis-scoped**.

## Git status
`model.py` clean (route reverted; default byte-identical). New/updated docs + one artifact only. No default flip, no
machine search, no 14B/32B, no attention/GEMV optimization (the buffer-identity read is scoped, not implemented).
