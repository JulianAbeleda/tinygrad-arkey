# Runtime-KV Buffer Identity / Rebase Follow-On — Result (2026-06-23)

## Verdict: **RUNTIME_KV_BUFFER_IDENTITY_DIFF_NOT_FOUND**

The buffer-identity hypothesis is **refuted**. The canonical-store-fill state and the assign-fill state have
**identical** uop structure, and a **full pristine-buffer rebase does not fix** the token-level baking. Critically,
this task found that the prior instrumentation's advance/bake proxy (block-0 cache *positions written*) is
**unreliable** — under true **token correctness**, the full-model `RUNTIME_KV_CACHE` decode bakes regardless of
prefill method or cache buffer. The cache buffer is **not** the cause. Route re-applied for instrumentation, then
**reverted** (`model.py` clean, default decode byte-identical `[279, 1156, 22148, …]`).

## 1. What the previous instrumentation found
`RUNTIME_KV_BAKING_TRIGGER_NARROWED_NOT_FULLY_ISOLATED`: the append `PROGRAM` declares `start_pos` live
(`ProgramInfo.vars = {'start_pos'}`); rope-producer refuted; baking appeared tied to the model's canonical-store
prefill before the decode jit (vs `cache.assign(numpy).realize()` which "advanced"). This task tested the
buffer-identity / rebase fix for that narrowed trigger — **and overturned the framing**.

## 2. Buffer identity diff — `BUFFER_IDENTITY_DIFF_NOT_FOUND`
`extra/qk_runtime_kv_buffer_identity_probe.py` dumps `cache_kv` identity after (a) the model's canonical-store
prefill and (b) a fresh `assign`-fill of the same `[0:2050]`. Result: `identity_diff_fields = {}` — **identical**
uop op (`RESHAPE`), dtype, shape, base op, and full bounded `op_counts`. Only the Python `id()`s and the realized
`Buffer` object ids differ (expected — different objects). There is **no structural buffer-identity difference**
to exploit.

## 3. Rebase probe — `PRISTINE_REBASE_COPY_STILL_BAKES`
At the prefill→decode handoff, every `block.cache_kv` was replaced with a **fresh** `Tensor.zeros(fp16)` buffer
holding the same data (`nf.assign(Tensor(data)).realize()`), then the decode jit was captured. Result: **still
bakes** — `tokens=[34208, 151936, 151936, …]`. A full pristine-buffer rebase (new object, new address, clean
op-chain) does **not** fix the baking. **The cache buffer is not the cause.**

## 4. The reframing — token correctness vs positions-written
The prior bisection ("no-prefill advances", "short prefill advances", "chunked prefill advances") used **block-0
cache positions-written** as the advance/bake proxy. That is **unreliable**: block-0's append can write its
positions while the full-model output is still garbage. Re-tested under **token correctness** (byte-identical to
baseline):
- **chunked prefill (T=512×4) with real tokens → BAKES** (`MODE=1 tokens=[13876, 151936, 151936, …]` vs baseline
  `[13876, 38835, 34208, …]`). The earlier "chunked advances" was a positions-written false positive.
- rebase → bakes on tokens too.

So the full-model `RUNTIME_KV` decode bakes (garbage after step 1) **regardless** of prefill method
(full/chunked/large-T/token-by-token) or cache buffer (canonical-store/assign/rebased). The only **confirmed**
advancing case remains the **isolated microbench** (`RUNTIME_KV_MICROBENCH_PASS`), not the full model.

## 5. Opaque-prefill probe (Phase 3)
**Not run.** Phase 3 (fill prefill KV via the opaque append instead of the canonical store) is a different
**cache-fill method** — but §3 proves a full cache **rebase** does not fix the baking, so the cache fill method
cannot be the cause. Running it would not change the verdict. Stopped honestly per the scope ("stop on the first
named blocker"); the named blocker is `BUFFER_IDENTITY_DIFF_NOT_FOUND` + cache-not-the-cause.

## 6–9. In-model route / graph identity / correctness / W==D
**Not reached** — no fix advanced token-correct multi-step decode, so the route gates (Phase 4–7) did not run.

## 10. Candidate / default decision
**Not registered.** No fix passed. Defaults unchanged.

## 11. Remaining blockers
The full-model `RUNTIME_KV` decode bakes on **token correctness**, and it is **not** a buffer-identity / cache /
prefill-method issue (all refuted). The append's `start_pos` is a declared live var, yet multi-step decode is
wrong. **Next investigation (must gate on token correctness, not positions-written):** instrument at the
`GraphRunner` kernel level — dump the `kv_append` CALL's `ji_args` / the `start_pos` kernarg value across
eager/capture/replay (does it actually change per replay in the full-model graph?), and verify the **owned-tile
read** correctness in-model per layer (the captured graph showed only 1 `owned_flash_tile` node where 36 layers
should each have one — a possible toposort/dedup artifact, or a real per-layer routing issue worth confirming). The
cache/buffer/prefill/rebase lane is exhausted.

## 12. Artifacts and commands
- `extra/qk_runtime_kv_buffer_identity_probe.py` → `bench/qk-runtime-managed-kv-cache/buffer_identity_diff.json`.
- Run: `DEV=AMD JIT=1 Q4K_GEMV_WARP=1 Q4K_GEMV_WARP_DOWN=1 RUNTIME_KV_CACHE=1 PYTHONPATH=. .venv/bin/python extra/qk_runtime_kv_buffer_identity_probe.py`.
- In-model repro requires re-applying the reverted `RUNTIME_KV_CACHE` route (`_init_state` fp16 + `__call__`
  `@function` bypass + `_attention` opaque-append/owned-read), then a chunked/real prefill + decode jit at ctx≥2048.

## 13. Files changed
Source: **none shipped** — the `RUNTIME_KV_CACHE` route was re-applied for instrumentation then **reverted**
(`model.py` byte-clean, default decode byte-identical). New: `extra/qk_runtime_kv_buffer_identity_probe.py`,
`docs/runtime-kv-buffer-identity-rebase-{scope,result}-20260623.md`,
`bench/qk-runtime-managed-kv-cache/buffer_identity_diff.json`.

## 14. Working tree status
`model.py` clean (reverted). New probe + docs + bench artifact only. No default change; no 14B/32B; no paged KV;
no new attention tile; no RoPE kernel; no activation/norm/GEMV work; no native core rewrite.
