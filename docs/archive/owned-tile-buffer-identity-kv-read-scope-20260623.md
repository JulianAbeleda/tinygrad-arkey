# Owned Tile Buffer-Identity KV Read — Exhaustive Scope / Claude Prompt (2026-06-23)

## Mission

Execute the corrected next bounded experiment after `runtime-kv-core-engine-result-v2`.

The major correction:

```text
The +11% KV tax is NOT a missing TinyJit persistence capability.
It is the owned attention tile reading K/V through sliced cache views, which callify materializes.
```

Therefore the next lever is:

```text
make owned AMDGCN attention read buffer-identity KV inputs,
not sliced/materialized K/V views.
```

Expected value:

- remove/reduce `E_49152` full-MAXC materialization;
- recover the MAXC-shrink transfer:
  - about `+11.8%` at ctx1024 from MAXC 1536;
  - about `+12.9%` at ctx1024 from MAXC 1280;
- move Qwen3-8B decode from `~86 tok/s` to `~96-97 tok/s`, llama-parity class.

This is a bounded tile/cache-layout ABI experiment, not:

- a Runtime-KV core-engine project;
- a TinyJit purity project;
- attention algorithm work;
- GEMV work;
- machine search.

## Required Reading

Read these first, in order:

1. `docs/runtime-kv-core-engine-result-v2-20260623.md`
2. `docs/runtime-kv-core-engine-result-20260623.md`
3. `docs/post-default-runtime-kv-diagnostic-result-20260623.md`
4. `docs/owned-amdgcn-tile-real-cache-revalidation-result-20260623.md`
5. `docs/owned-amdgcn-tile-short-ctx-result-20260623.md`
6. `docs/post-owned-attention-default-audit-result-20260623.md`
7. `docs/amd-gpu-holistic-primitive-model-20260623.md`
8. `docs/cross-vendor-isa-primitive-audit-and-search-result-20260623.md`
9. `structure/Development/performance-primitive-research-principles.md`
10. `structure/Development/session-handoff.md`

Inspect code:

- `tinygrad/llm/model.py`
- `extra/qk_owned_flash_decode_graph_node.py`
- `extra/qk_owned_flash_decode.hip`
- `tinygrad/codegen/opt/callify.py`
- `tinygrad/engine/jit.py`
- `tinygrad/engine/realize.py`
- `extra/qk_decode_runtime_overhead.py`
- `extra/qk_decode_time_tax_audit.py`
- `extra/qk_isa_primitive_audit.py`

## Current Understanding

Current owned route:

```text
cache_kv: [2, layers, Hkv, MAXC, Hd]
K input to tile: cache_kv[0, layer].after(store)
V input to tile: cache_kv[1, layer].after(store)
```

Problem:

```text
cache_kv[0, layer] and cache_kv[1, layer] are sliced views, not buffer-identity inputs.
callify materializes them before passing to the precompiled owned tile.
```

Observed:

- native cache store + `AFTER` read is byte-correct;
- W==D is `0%` because materialization remains;
- `E_49152` is not removed;
- correctness/persistence is not the blocker;
- buffer identity is the blocker.

Relevant callify finding:

```text
An AFTER-node input is not automatically force-contiguous.
A BUFFER with has_buffer_identity() can be read directly.
```

So the next experiment must feed the owned tile whole-buffer identity inputs.

## Candidate Designs

### Design A — Separate K/V Cache Buffers

Change the owned route's cache layout behind an env flag:

```text
K_cache: [layers, Hkv, MAXC, Hd]
V_cache: [layers, Hkv, MAXC, Hd]
```

Then pass per-layer K/V buffers to the owned tile in a way that preserves buffer identity.

Possible variants:

1. one K buffer and one V buffer per layer;
2. one global K buffer + one global V buffer with layer offset handled by tile/kernel;
3. view-free per-layer Tensor/Buffer handles allocated directly.

Pros:

- clean ABI: tile receives K and V as whole identity buffers;
- removes the `[2, ...]` leading-axis slice;
- aligns model cache layout with tile contract;
- likely removes `E_49152`.

Risks:

- model cache layout changes behind flag;
- prefill/decode store path must write both caches correctly;
- if per-layer views still lose buffer identity, may need per-layer buffers or kernel layer offsets.

This is the preferred first path.

### Design B — Single Cache Buffer + Kernel V Offset

Keep canonical cache layout but pass a whole buffer identity input to the tile, and make the tile compute K/V offsets:

```text
cache_kv_whole: [2, layers, Hkv, MAXC, Hd]
tile reads K at offset(0, layer, ...)
tile reads V at offset(1, layer, ...)
```

Possible signatures:

```text
tile(Q, cache, part, meta, start_pos, layer)
combine(part, meta, out)
```

or:

```text
tile(Q, cache, part, meta, start_pos, base_k, base_v)
```

Pros:

- less model cache layout disruption;
- keeps prefill/decode storage closer to existing path.

Risks:

- tile ABI/kernel change;
- layer offset scalar(s);
- must avoid passing sliced K/V views anyway;
- correctness risk in offset math.

This is fallback if Design A cannot preserve buffer identity cheaply.

### Design C — callify/Buffer Identity Narrow Fix

Teach callify/tinygrad to recognize this exact cache slice pattern and pass base buffer + offset to the precompiled call.

Pros:

- potentially generalizes buffer-offset ABI.

Risks:

- touches compiler/lowering semantics;
- can become broad view/alias support;
- less bounded than A/B.

Only scope if A and B fail and the slice pattern is clearly transformable without broad alias analysis.

## Non-Goals

- Do not revive `RUNTIME_KV_CORE_ENGINE` as a core persistence task.
- Do not implement general mutable Tensor state.
- Do not do machine search.
- Do not optimize attention math/tile internals unless needed for offset ABI.
- Do not reopen Q4K GEMV.
- Do not do 14B/32B.
- Do not flip defaults.

## Required Artifact Directory

```text
bench/qk-owned-tile-buffer-identity-kv-read/
```

Required artifacts:

- `authority.json`
- `materialization_attribution.json`
- `design_a_probe.json`
- `design_b_probe.json` if needed
- `isa_audit.json`
- `wd.json`
- `decision.json`

Required result doc:

- `docs/owned-tile-buffer-identity-kv-read-result-20260623.md`

Update if result is material:

- `docs/README.md`
- `structure/Development/session-handoff.md`
- `bench/qk-decode-eval/candidates.json` only if route state changes

## Phase 0 — Authority Lock

Record:

- HEAD;
- git status;
- GPU/arch;
- default route state;
- owned attention default-on status;
- Q4K warp state;
- model path;
- current baseline tok/s;
- current `E_49152` / materialization evidence;
- current candidate registry state.

Artifact:

- `bench/qk-owned-tile-buffer-identity-kv-read/authority.json`

Verdicts:

- `AUTHORITY_LOCKED`
- `AUTHORITY_INCOMPLETE_STOP`

## Phase 1 — Reconfirm Materialization Attribution

Purpose:

Prove the target before editing.

Required checks:

1. Identify current owned-tile K/V inputs.
2. Confirm they are sliced views:
   - `cache_kv[0, layer]`;
   - `cache_kv[1, layer]`.
3. Confirm callify materializes them.
4. Confirm `E_49152` or equivalent appears.
5. Confirm native cache store + AFTER-read correctness is not enough to remove materialization.
6. Confirm a whole buffer identity input can avoid materialization in a minimal callify/custom_kernel probe.

Artifact:

- `bench/qk-owned-tile-buffer-identity-kv-read/materialization_attribution.json`

Verdicts:

- `MATERIALIZATION_ATTRIBUTED_TO_KV_SLICE_READ`
- `MATERIALIZATION_NOT_REPRODUCED_STOP`
- `WHOLE_BUFFER_IDENTITY_PROBE_PASS`
- `WHOLE_BUFFER_IDENTITY_PROBE_FAIL_STOP`

Stop unless materialization is attributed to slice read.

## Phase 2 — Design A Probe: Separate K/V Cache Buffers

Purpose:

Test the preferred bounded solution.

Implementation boundary:

- behind explicit env flag only, e.g.:

```text
OWNED_TILE_KV_IDENTITY=1
```

or:

```text
DECODE_ATTN_KV_IDENTITY=1
```

Requirements:

1. Allocate/maintain K and V caches separately for owned route.
2. Preserve default canonical cache when flag off.
3. Prefill writes K and V correctly.
4. Decode writes K and V correctly.
5. Owned tile receives buffer-identity K/V inputs.
6. No `E_49152` / full-MAXC K/V materialization.
7. Token correctness:
   - at least two prompts;
   - 64 decode tokens;
   - ctx512 and ctx1024 first.
8. Fallback:
   - unsupported shape/device returns canonical path.

Key implementation choices to evaluate:

| choice | question |
|---|---|
| per-layer K/V buffers | easiest identity? memory overhead? |
| global K/V buffers with layer offset | fewer buffers, needs tile offset |
| model cache abstraction | can cache API hide layout difference? |
| prefill migration | one-time copy vs native separate fill |

Artifact:

- `bench/qk-owned-tile-buffer-identity-kv-read/design_a_probe.json`

Verdicts:

- `DESIGN_A_BUFFER_IDENTITY_PASS`
- `DESIGN_A_CORRECTNESS_FAIL`
- `DESIGN_A_MATERIALIZATION_REMAINS`
- `DESIGN_A_TOO_INVASIVE`

If Design A passes correctness and removes materialization, go to W==D.

If Design A is blocked only by layer-offset identity, try Design B.

## Phase 3 — Design B Probe: Whole Cache + Kernel Offset

Run only if Design A fails to preserve buffer identity or is too invasive.

Purpose:

Keep canonical cache storage but pass whole-buffer identity to tile and compute K/V offsets inside the kernel.

Requirements:

1. Modify owned tile ABI behind env flag only.
2. Pass whole `cache_kv` buffer, not sliced K/V.
3. Pass scalar layer/base offsets as runtime or baked constants.
4. Tile computes:
   - K base offset;
   - V base offset;
   - Hkv/MAXC/Hd indexing.
5. Token correctness.
6. `E_49152` removed/reduced.
7. ISA audit still acceptable:
   - no spills;
   - no pathological VGPR jump;
   - v_dot2/LDS/cross-lane still present.

Artifact:

- `bench/qk-owned-tile-buffer-identity-kv-read/design_b_probe.json`

Verdicts:

- `DESIGN_B_WHOLE_CACHE_OFFSET_PASS`
- `DESIGN_B_CORRECTNESS_FAIL`
- `DESIGN_B_MATERIALIZATION_REMAINS`
- `DESIGN_B_ISA_REGRESSION`

## Phase 4 — ISA Audit

Run on any changed owned tile code object.

Required:

- use `extra/qk_isa_primitive_audit.py`;
- compare to prior owned tile:
  - `v_dot2`;
  - LDS;
  - cross-lane;
  - VGPR;
  - scratch/spill.

Artifact:

- `bench/qk-owned-tile-buffer-identity-kv-read/isa_audit.json`

Verdicts:

- `ISA_UNCHANGED_OR_ACCEPTABLE`
- `ISA_REGRESSION_STOP`
- `ISA_TOOLING_LIMITED`

## Phase 5 — W==D

Run only if correctness passes and materialization is removed/reduced.

Required contexts:

- 512;
- 1024;
- 2048;
- 4096.

Required configs:

1. current default owned route;
2. buffer-identity route.

Required checks:

- token correctness;
- route fires;
- no silent fallback;
- `E_49152` absent/reduced;
- tok/s;
- ms/token;
- repeated spread;
- no ctx512 regression.

Expected:

- target `>= +5%` at ctx1024;
- likely `~+10-13%` if materialization is fully removed;
- llama-parity class.

Artifact:

- `bench/qk-owned-tile-buffer-identity-kv-read/wd.json`

Verdicts:

- `BUFFER_IDENTITY_KV_WD_PASS`
- `BUFFER_IDENTITY_KV_NO_WD_TRANSFER`
- `BUFFER_IDENTITY_KV_REGRESSION`
- `BUFFER_IDENTITY_KV_CORRECTNESS_FAIL`

## Phase 6 — Decision / Registration

If W==D passes:

- update candidate metadata;
- keep default-on policy separate unless owner authorizes;
- decide whether this replaces current owned route or becomes an env-gated variant.

If W==D fails:

- classify why:
  - materialization not fully removed;
  - new copy introduced;
  - overlap/no transfer;
  - correctness/fallback issue.

Artifact:

- `bench/qk-owned-tile-buffer-identity-kv-read/decision.json`

Verdicts:

- `BUFFER_IDENTITY_KV_PROMOTION_READY`
- `BUFFER_IDENTITY_KV_KEEP_DEFAULT_OFF`
- `BUFFER_IDENTITY_KV_REST_NO_TRANSFER`
- `BUFFER_IDENTITY_KV_BLOCKED`

## Required Result Doc

Write:

- `docs/owned-tile-buffer-identity-kv-read-result-20260623.md`

Required sections:

1. Verdict.
2. Authority/config.
3. Correction from Runtime-KV framing.
4. Materialization attribution.
5. Design A result.
6. Design B result if run.
7. ISA audit.
8. W==D.
9. Correctness.
10. Candidate/default decision.
11. Files changed.
12. Git status.

## Stop Rules

Stop and classify if:

- materialization is not actually caused by K/V slicing;
- whole-buffer identity still materializes in minimal probe;
- token correctness fails;
- `E_49152` remains unchanged;
- ISA regresses badly;
- implementation requires broad callify/view alias support;
- W==D does not transfer.

## Claude Prompt

You are in `/home/ubuntu/tinygrad-arkey` on branch `qk-prefill-flag-leak-resolution`.

Read and execute:

```text
docs/owned-tile-buffer-identity-kv-read-scope-20260623.md
```

Also read:

```text
docs/runtime-kv-core-engine-result-v2-20260623.md
docs/post-default-runtime-kv-diagnostic-result-20260623.md
docs/owned-amdgcn-tile-real-cache-revalidation-result-20260623.md
docs/post-owned-attention-default-audit-result-20260623.md
docs/amd-gpu-holistic-primitive-model-20260623.md
```

Critical correction:

Do not pursue the retired core Runtime-KV persistence framing. The new result says correctness is achievable and the
remaining +11% tax is caused by owned tile K/V cache slicing/materialization. The task is to make the owned tile read
buffer-identity KV inputs.

Execute in order:

1. Authority lock.
2. Reconfirm materialization attribution to K/V slice reads.
3. Probe Design A: separate K/V cache buffers preserving buffer identity.
4. If A is blocked, probe Design B: whole cache buffer + kernel V/layer offset.
5. ISA-audit any changed code object.
6. Run W==D only if correctness passes and `E_49152` is removed/reduced.
7. Write result doc and artifacts.

Hard boundaries:

- no attention math optimization;
- no GEMV work;
- no machine search;
- no 14B/32B;
- no default flip unless explicitly authorized;
- no broad callify/view alias rewrite;
- token correctness is authority.

Final response must include:

- final verdict;
- Design A result;
- Design B result if run;
- whether `E_49152` was removed;
- W==D result if reached;
- ISA audit result;
- files changed;
- git status.
