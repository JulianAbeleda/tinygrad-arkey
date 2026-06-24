# Runtime-Managed KV Cache — Result (2026-06-23)

## Verdict: **RUNTIME_KV_PERSISTENCE_FAIL**

The runtime boundary is **proven in isolation** (`RUNTIME_KV_MICROBENCH_PASS`) and the integration got **much
further** than the prior single-graph attempts: bypassing `@function(precompile=True)` + an fp16 runtime cache
made **eager** multi-step decode persist, the **copy is removed**, the route fires per-layer, and the **first
decode token is byte-identical**. But under **TinyJit decode replay** the opaque append's **write position
bakes** at the capture `start_pos` (replays don't advance) → multi-step generation collapses to garbage. The
route was **reverted** (default decode byte-identical; `model.py` clean).

## 1. Verdict
`RUNTIME_KV_PERSISTENCE_FAIL` — runtime cache persists eager + in the microbench, but in-model TinyJit decode
the append's write position does not advance across replays (and the owned-tile opaque read is ctx-restricted
to ≥2048).

## 2. Online research alignment
Matches the scope's references (TensorRT-LLM block pool, vLLM PagedAttention, FlashInfer append/attention split,
SGLang reset lifecycle, vAttention contiguous-first): KV cache is **runtime-owned state** with explicit
allocate/reset/append/read lifecycle. This task built exactly that contiguous, single-request runtime object —
and the lifecycle (allocate/reset/append/read) works (microbench). The wall is tinygrad's TinyJit replay of an
opaque append whose K/V source depends on `start_pos`.

## 3. What changed architecturally
A fork-local `RuntimeKVCache` (allocate/reset/append/k_view/v_view) owns a persistent **realized fp16** buffer;
the model's per-layer forward **bypasses `@function(precompile=True)`** for the decode route so the opaque
append persists in the realized buffer instead of via `@function`'s pure-graph state (which required the
full-MAXC materialization). Attention reads the persistent cache through the existing owned AMDGCN tile.

## 4. Runtime boundary chosen and why
**Option C (layer-local append-then-owned-attention) + Option B (runtime buffer ownership), via an `@function`
bypass.** Bypassing `@function` is the minimal change that removes the precompile buffer-substitution that lost
persistence in `KV_OPAQUE_READ_CORRECTNESS_FAIL`; a full two-graph split (Option A) was unnecessary for the
single-request proof. Decode is byte-identical without `@function` (verified), so the bypass is correctness-safe.

## 5. Microbench result — `RUNTIME_KV_MICROBENCH_PASS`
`extra/qk_runtime_kv_cache_probe.py` (`bench/.../microbench.json`): isolated `RuntimeKVCache` + opaque append +
owned tile, plain TinyJit:
- **Persistence across multi-step replay**: each step appends a *distinct* token; the attention correctly sees
  **all prior appends** (rel_rmse ~3.8e-7 at sp 2048–2051) → accumulation persists.
- **Reset / no-stale**: clean across generations (3.7e-7).
- **Graph identity**: `[kv_append, owned_flash_tile_gqa, owned_flash_combine]`, **no full-MAXC copy**,
  append-before-tile.

## 6. Integration result
- **fp16 dtype fix (key):** the canonical cache is `dtypes.float` (fp32), but the opaque append writes fp16
  (b16 stores) → fp32 cache = wrong byte offsets = no persistence. Forcing the runtime cache to fp16 (matching
  the append kernel + owned tile + microbench) made **eager** multi-step persist (positions 2048/2049/2050).
- **Route fires per-layer** (36 `kv_append` calls/forward); **prefill→decode handoff** works (prefill via the
  `@function` canonical store persists into the fp16 cache; the **first decode token is byte-identical**).
- **JIT-replay position baking (blocker):** under TinyJit decode replay, only the **capture** position (2050)
  is written; replays at 2051/2052/2053 do **not** advance → garbage after the first token. The microbench's
  append advances correctly (its K/V src is an external tensor); the model's append K/V depend on `start_pos`
  via rope (`freqs_cis[start_pos:start_pos+T]`), which appears to concretize `start_pos` for the append launch.
  (The baseline canonical store advances fine, so `start_pos` itself replays — the opaque-append-with-
  start_pos-dependent-src is the differentiator.)

## 7. Graph / kernel identity result
When the route fires (decode capture-ctx ≥ 2048): **no `E_49152` full-MAXC copy**, owned tile + combine present,
reads the persistent cache (not `assigned_kv`), no `gqa_coop_vec` fallback. **Copy removal confirmed.**
(`bench/.../graph_identity.json`.)

## 8. Correctness result
**FAIL.** First decode token byte-identical; subsequent tokens garbage (151936) due to the JIT-replay position
baking. Eager and the microbench are correct; in-model JIT decode is not.

## 9. W==D result
**Not reached** — correctness failed at Phase 4. (Even had it passed, the owned-tile ctx≥2048 restriction means
the route could not fire at ctx1024 → could not clear +5%@ctx1024.)

## 10. Candidate / default decision
**Not registered.** Correctness failed; per the scope `runtime_managed_kv_owned_attention_8b` is not added to
`bench/qk-decode-eval/candidates.json`. Defaults unchanged.

## 11. Remaining 8B gap
Unchanged. The ~1.4 ms KV copy stays. **Two specific, now-isolated blockers remain** for a runtime-managed
cache: (a) **TinyJit must replay an opaque append whose write offset advances even when its K/V src depends on
`start_pos`** (the position must stay a runtime scalar, not concretize); (b) the owned tile (the opaque read)
needs a **short-ctx-correct variant** (ctx-scaled split count) to apply below ctx2048. Both are bounded,
named follow-ons — this task narrowed the wall from "@function persistence" to these two.

## 12. Follow-on limitations
Single-request/B=1/T=1 only; no paged/prefix/eviction/multi-request (scoped out). The `@function` bypass is
decode-route-only (prefill keeps `@function`); a production path would need the prefill→decode handoff hardened.

## 13. Artifacts and commands
- `extra/qk_runtime_kv_cache_probe.py` → `bench/qk-runtime-managed-kv-cache/{microbench,graph_identity}.json`.
- Microbench: `DEV=AMD JIT=1 PYTHONPATH=. .venv/bin/python extra/qk_runtime_kv_cache_probe.py`.
- In-model repro requires re-applying the reverted `RUNTIME_KV_CACHE` route (`__call__` `@function` bypass +
  `_init_state` fp16 cache + `_attention` opaque-append/owned-read), then `... RUNTIME_KV_CACHE=1 ...`.

## 14. Files changed
Source: **none shipped** — the `RUNTIME_KV_CACHE` route (`model.py` `__call__`/`_init_state`/`_attention`) was
added and **reverted**; `model.py` byte-clean, default decode byte-identical (`[279, 1156, 22148, …]`). New:
`extra/qk_runtime_kv_cache_probe.py`, `docs/runtime-managed-kv-cache-result-20260623.md`,
`bench/qk-runtime-managed-kv-cache/{microbench,graph_identity}.json`.

## 15. Working tree status
`model.py` clean (reverted). New probe + docs + bench artifacts only. No default change; no kernel built; no
14B/32B; no paged KV / prefix / eviction / server scheduler; no new attention tile; no native core rewrite.
