# Runtime-Managed KV Cache — Implementation Scope / Claude Prompt (2026-06-23)

## Mission

Implement the architecture lane that remains after bounded 8B primitive exhaustion:

**Move decode KV cache persistence out of the single pure `@function(precompile=True)` graph and into a runtime-managed
KV cache object, then use explicit append + opaque owned-attention read to avoid the full-MAXC functional copy.**

This follows:

- `KV_OPAQUE_READ_CORRECTNESS_FAIL`
- `KV_RUNTIME_MANAGED_CACHE_REQUIRED`
- `FUSED_FLASH_SINGLE_TILE_SCOPE_REDUCED_TO_OPAQUE_READ`

The local result proved:

- opaque append + owned-tile read works standalone;
- copy removal works for graph identity;
- first in-model step can be correct;
- multi-step `@function(precompile=True)` decode loses KV persistence without the canonical full-buffer
  materialization;
- therefore the full-MAXC copy is both waste **and** the current persistence linkage.

So the next step is not another single-graph trick. It is a runtime/state architecture.

## Online Check Summary

External systems align with our local result:

| System / reference | Relevant pattern | How it maps to this project |
|---|---|---|
| TensorRT-LLM KV Cache System | KV cache is a runtime block pool; runtime preallocates/distributes blocks; generation attention consumes KV cache as a first-class runtime structure. | Supports a runtime-owned KV object instead of a pure tensor returned by the model graph. |
| vLLM PagedAttention / Hybrid KV Cache Manager | CPU-side cache manager owns block tables; GPU KV data lives in blocks; attention kernels understand the cache layout. | Confirms cache lifetime, allocation, and lookup are runtime responsibilities; attention must read through the runtime interface. |
| FlashInfer | Separate append APIs for paged KV cache and decode attention APIs over KV cache; supports paged/ragged KV formats. | Directly matches our needed split: append/update cache, then attention reads cache. |
| SGLang RadixAttention | KV reuse is managed above the model graph; prefix/cache lifecycle is explicit. | Highlights reset/reuse correctness hazards; cache identity and prompt boundaries must be explicit. |
| vAttention | KV memory management can change while preserving contiguous attention-kernel compatibility. | Suggests we can start with contiguous runtime KV, not jump to paged/block-table attention. |

Sources:

- TensorRT-LLM KV cache docs: `https://nvidia.github.io/TensorRT-LLM/latest/features/kvcache.html`
- TensorRT-LLM memory/runtime KV pool: `https://nvidia.github.io/TensorRT-LLM/reference/memory.html`
- TensorRT-LLM KV reuse: `https://nvidia.github.io/TensorRT-LLM/advanced/kv-cache-reuse.html`
- vLLM hybrid KV cache manager: `https://docs.vllm.ai/en/stable/design/hybrid_kv_cache_manager/`
- vLLM paged attention design: `https://docs.vllm.ai/en/latest/design/paged_attention/`
- FlashInfer paged KV append: `https://docs.flashinfer.ai/api/page.html`
- FlashInfer attention kernels: `https://docs.flashinfer.ai/api/attention.html`
- FlashInfer repo feature list: `https://github.com/flashinfer-ai/flashinfer`
- SGLang RadixAttention blog: `https://lmsys.org/blog/2024-01-17-sglang/`
- vAttention: `https://arxiv.org/html/2405.04437v2`

### What the online check adds

Our local conclusion is aligned, but the web check adds implementation requirements that must be scoped explicitly:

1. **Cache lifecycle is a first-class API**, not an implementation detail.
   - allocate, reset, append, read, and release must be explicit.

2. **Prompt/request identity matters.**
   - stale KV across prompts is a correctness bug, not just a performance issue.

3. **Do contiguous first.**
   - Do not start with paged/block-table cache. vAttention supports preserving a contiguous attention reader while
     changing runtime memory management.

4. **Append and attention should be separate runtime phases.**
   - FlashInfer's API split mirrors our proven append + owned-tile read decomposition.

5. **Serving concerns must be fenced.**
   - Prefix reuse, eviction, block pools, multi-request batching, and KV sharing are later phases. The first gate is
     single-request Qwen3-8B/gfx1100 decode correctness and W==D.

## Required Reading Before Editing

Read these first:

1. `docs/runtime-kv-opaque-read-result-20260623.md`
2. `docs/runtime-kv-opaque-read-followon-scope-20260623.md`
3. `docs/kv-cache-stateful-jit-capability-result-20260622.md`
4. `docs/kv-cache-copy-elimination-result-20260622.md`
5. `docs/fused-flash-single-tile-result-20260622.md`
6. `docs/8b-remaining-architecture-understanding-result-20260622.md`
7. `docs/decode-gap-audit-consolidated-20260622.md`
8. `docs/decode-ffn-gemv-warp-result-20260622.md`
9. `docs/q4k-gemv-warp-promotion-hardening-result-20260622.md`
10. `structure/Development/performance-primitive-research-principles.md`
11. `structure/Development/session-handoff.md`

Inspect code:

- `tinygrad/llm/model.py`
- `extra/qk_kv_cache_state_token.py`
- `extra/qk_kv_opaque_read_probe.py`
- `extra/qk_kv_append_microprobe.py`
- `extra/qk_owned_flash_decode_graph_node.py`
- `extra/qk_owned_flash_decode.hip`
- `extra/llm_generate.py`
- `tinygrad/llm/cli.py`
- server/generate paths if used by this repo

## Scope Boundary

First implementation target:

```text
single-request, batch-1, T=1 decode, Qwen3-8B-Q4_K_M, gfx1100, contiguous KV runtime object
```

Explicit non-goals:

- no paged/block-table cache;
- no prefix sharing/reuse;
- no eviction;
- no multi-request server scheduler;
- no 14B/32B;
- no batch >1;
- no speculative decode / T>1;
- no new attention tile;
- no tinygrad core alias-analysis rewrite;
- no default change.

## Target Architecture

Introduce a fork-local runtime KV object that owns persistent cache buffers and their lifecycle.

Suggested concept:

```python
class RuntimeKVCache:
  cache_kv: Tensor          # shape [2, B, Hkv, MAXC, Hd] or current model-compatible layout
  max_context: int
  cur_pos: int | UOp-bound externally
  generation_id: int
  valid: bool

  def reset(...)
  def append(k, v, start_pos)
  def k_view(...)
  def v_view(...)
```

The object must:

- allocate/own persistent KV storage outside the pure decode step return value;
- reset on prompt boundary;
- append K/V via the proven opaque append kernel;
- feed persistent K/V directly to the existing owned AMDGCN attention tile;
- avoid `assigned_kv = cache_kv.uop.after(...)` on the runtime route;
- remain default-off and fallback-safe.

## Critical Design Choice

The last probe failed because append and attention were still inside the same precompiled pure function and
`@function` did not preserve runtime state across replay without canonical materialization.

Therefore the first funded gate must choose one runtime boundary:

### Option A — Two-Graph Decode Step

Per token/layer:

1. compute q/k/v;
2. realize opaque append into runtime cache;
3. run attention graph reading persistent cache;
4. continue model.

Risk:

- 36 layers means append graph overhead may be high;
- if append requires host sync, it can eat the saved 1.4 ms;
- this may require splitting the model forward more than desired.

### Option B — Runtime Cache Object With Persistent UOp / Buffer Ownership

The cache object owns buffers outside the pure function. The decode graph receives the buffer as an external
pre-existing input, while append is a runtime-side side effect ordered before attention.

Risk:

- still must prove TinyJit/HCQ ordering without full materialization;
- may require model/generate loop restructuring.

### Option C — Layer-Local Append-Then-Attention Wrapper

Wrap only the attention block:

- q/k/v projection remains in the model graph;
- append + owned attention are routed through an explicit runtime helper per layer;
- helper ensures ordering and persistence.

Risk:

- may break current `@function(precompile=True)` compile structure;
- must quantify overhead.

The implementation should choose the smallest option that proves:

- persistence across replay;
- no full-MAXC copy;
- byte-identical multi-step decode;
- W==D transfer.

## Phase Plan

### Phase 0 — Evidence Lock

Before edits:

- confirm clean git status;
- record HEAD;
- reproduce or inspect `KV_OPAQUE_READ_CORRECTNESS_FAIL`;
- reproduce or inspect opaque append microprobe pass;
- record post-warp baseline tok/s;
- record current copy kernel identity/cost;
- create result doc draft:
  - `docs/runtime-managed-kv-cache-result-20260623.md`

### Phase 1 — Runtime Boundary Microbench

Build a microbench before touching the model:

```text
extra/qk_runtime_kv_cache_probe.py
```

It must test:

1. allocate persistent cache;
2. reset cache;
3. append positions 0, 1, 2, 7 using opaque append;
4. run existing owned tile reading persistent cache;
5. repeat across multiple "generation" resets;
6. run with TinyJit capture/replay if applicable;
7. no full-MAXC copy;
8. no stale values after reset;
9. no host sync in the critical path if measurable.

Artifact:

- `bench/qk-runtime-managed-kv-cache/microbench.json`

Allowed verdicts:

- `RUNTIME_KV_MICROBENCH_PASS`
- `RUNTIME_KV_APPEND_ORDER_FAIL`
- `RUNTIME_KV_RESET_FAIL`
- `RUNTIME_KV_STALE_CACHE_FAIL`
- `RUNTIME_KV_HOST_SYNC_TOO_EXPENSIVE`
- `RUNTIME_KV_MICROBENCH_NOT_EXPRESSIBLE`

Stop if microbench fails.

### Phase 2 — Minimal Generate-Loop Integration

Only if Phase 1 passes.

Add a default-off flag:

```text
RUNTIME_KV_CACHE=1
```

Implement the narrowest model/generate integration that avoids relying on `@function` to carry KV persistence.

Requirements:

- route only B=1/T=1/Qwen3-8B/gfx1100;
- keep canonical path untouched;
- reset runtime cache at prompt/generation start;
- append each layer's K/V exactly once per token;
- attention reads persistent cache through existing owned tile;
- no `assigned_kv` full-copy on this route;
- fallback to canonical path on unsupported shape or failure;
- DEBUG logs route/fallback reason.

Possible files:

- `tinygrad/llm/model.py`
- `extra/llm_generate.py`
- new helper `extra/qk_runtime_kv_cache.py`
- optional probe/harness only if model changes are too invasive

Stop if integration requires broad server/runtime rewrite before a single-request proof.

### Phase 3 — Graph / Kernel Identity Gate

Write:

- `bench/qk-runtime-managed-kv-cache/graph_identity.json`

Required checks:

- no `E_49152` or equivalent full-MAXC copy;
- opaque append node present;
- owned tile and combine present;
- route reads persistent cache, not `assigned_kv`;
- cache reset visible between prompts;
- no fallback to `gqa_coop_vec`;
- no host-side synchronization in the per-token critical path beyond existing `.item()` measurement.

If full copy remains:

- classify `RUNTIME_KV_COPY_STILL_PRESENT`;
- stop or revert unsafe source changes.

### Phase 4 — Correctness Gate

Before timing:

- greedy byte-identical for at least 64 tokens;
- two prompts in one process, both byte-identical;
- same prompt twice in one process, byte-identical both times;
- start_pos replay across at least 0..63;
- compare generated token stream against baseline;
- no garbage token collapse like `151936`;
- if possible, fixed attention output vs numpy at several positions.

Allowed failure verdicts:

- `RUNTIME_KV_CORRECTNESS_FAIL`
- `RUNTIME_KV_STALE_CACHE_FAIL`
- `RUNTIME_KV_PERSISTENCE_FAIL`
- `RUNTIME_KV_PREFILL_HANDOFF_FAIL`

If correctness fails, revert unsafe source changes and preserve probe/result artifacts.

### Phase 5 — W==D Gate

Baseline:

```bash
DEV=AMD JIT=1 Q4K_GEMV_WARP=1 Q4K_GEMV_WARP_DOWN=1
```

Candidate:

```bash
DEV=AMD JIT=1 Q4K_GEMV_WARP=1 Q4K_GEMV_WARP_DOWN=1 RUNTIME_KV_CACHE=1
```

Measure:

- ctx512;
- ctx1024;
- ctx2048;
- ctx4096.

Timing discipline:

- `.item()` inside timed window;
- repeated or in-process A/B;
- report spread;
- prove route fired;
- tokens match.

Pass gate:

- `>= +5% @ctx1024`;
- no ctx512 regression beyond noise;
- no ctx2048/4096 regression beyond noise;
- byte-identical.

Expected:

- if copy removal transfers cleanly: about `74 -> 82-85 tok/s @ctx1024`;
- if runtime overhead is high: classify honestly.

Allowed W==D verdicts:

- `RUNTIME_KV_WD_PASS`
- `RUNTIME_KV_LOCAL_PASS_WD_FAIL`
- `RUNTIME_KV_OVERHEAD_EATS_COPY_WIN`

### Phase 6 — Candidate / Default Decision

Only if correctness and W==D pass:

- register candidate:
  - `runtime_managed_kv_owned_attention_8b`
- `default_on=false`;
- `default_eligible=true` only if byte-identical and fallback-safe;
- include shape/device guard metadata;
- document server/prompt lifecycle limitations.

No default flip in this task.

### Phase 7 — Result Doc

Write:

- `docs/runtime-managed-kv-cache-result-20260623.md`

Required sections:

1. Verdict.
2. Online research alignment.
3. What changed architecturally.
4. Runtime boundary chosen and why.
5. Microbench result.
6. Integration result.
7. Graph/kernel identity result.
8. Correctness result.
9. W==D result.
10. Candidate/default decision.
11. Remaining 8B gap.
12. Follow-on limitations.
13. Artifacts and commands.
14. Files changed.
15. Working tree status.

Allowed final verdicts:

- `RUNTIME_KV_WD_PASS`
- `RUNTIME_KV_LOCAL_PASS_WD_FAIL`
- `RUNTIME_KV_OVERHEAD_EATS_COPY_WIN`
- `RUNTIME_KV_CORRECTNESS_FAIL`
- `RUNTIME_KV_PERSISTENCE_FAIL`
- `RUNTIME_KV_STALE_CACHE_FAIL`
- `RUNTIME_KV_COPY_STILL_PRESENT`
- `RUNTIME_KV_HOST_SYNC_TOO_EXPENSIVE`
- `RUNTIME_KV_MICROBENCH_NOT_EXPRESSIBLE`
- `RUNTIME_KV_SCOPE_TOO_BROAD_DEFER`

## Boundaries

- No default change.
- No 14B/32B.
- No paged KV.
- No prefix reuse.
- No eviction/cache sharing.
- No server scheduler work beyond proving no stale state in single process.
- No new attention tile.
- No native tinygrad core alias-analysis rewrite.
- No activation/norm/GEMV work.
- Do not claim success from microbench only.
- Do not leave unsafe route enabled after failure.

## Claude Prompt

You are in `/home/ubuntu/tinygrad-arkey` on branch `qk-prefill-flag-leak-resolution`.

Read `docs/runtime-managed-kv-cache-implementation-scope-20260623.md` completely and execute it.

The online check aligns with the local result: production systems treat KV cache as runtime-owned state with explicit
append/read/lifecycle APIs. Our single-graph tricks are exhausted. The task is to implement the first bounded
runtime-managed KV proof for single-request Qwen3-8B/gfx1100 decode.

Start with the runtime boundary microbench. Do not touch model integration until the microbench proves persistence,
reset, append->owned-attention read ordering, no full-MAXC copy, and no stale cache. Then add a default-off
`RUNTIME_KV_CACHE=1` route only if the microbench passes.

Use post-`Q4K_GEMV_WARP` as the baseline:

```bash
DEV=AMD JIT=1 Q4K_GEMV_WARP=1 Q4K_GEMV_WARP_DOWN=1
```

Candidate:

```bash
DEV=AMD JIT=1 Q4K_GEMV_WARP=1 Q4K_GEMV_WARP_DOWN=1 RUNTIME_KV_CACHE=1
```

Hard gates:

- no full-MAXC copy;
- persistent KV across multi-step decode;
- reset/no stale cache across two prompts in one process;
- byte-identical 64-token greedy output;
- W==D `>= +5% @ctx1024`;
- no ctx regression;
- default-off and fallback-safe.

Write:

- `extra/qk_runtime_kv_cache_probe.py`
- `bench/qk-runtime-managed-kv-cache/microbench.json`
- `bench/qk-runtime-managed-kv-cache/graph_identity.json` if integrated
- `bench/qk-runtime-managed-kv-cache/wd.json` if W==D reached
- `docs/runtime-managed-kv-cache-result-20260623.md`

Do not implement paged KV, prefix reuse, eviction, 14B/32B, server scheduling, new attention tiles, native tinygrad
core alias analysis, activation/norm/GEMV work, or default flips.

Report final verdict, commands, artifacts, files changed, whether defaults changed, route registration status, and
git status.
