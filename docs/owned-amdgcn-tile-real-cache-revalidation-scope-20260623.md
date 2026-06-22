# Owned AMDGCN Tile Real-Cache Revalidation — Scope / Claude Prompt (2026-06-23)

## Mission

Stop runtime-KV work and revalidate the owned AMDGCN attention tile against **real in-model KV cache data**.

Latest verdict:

**`RUNTIME_KV_ARG_PATCH_VALUES_CORRECT_DATA_STALE`**

The GraphRunner / CUDA-graph-style hypotheses are refuted:

- append `start_pos` kernargs patch correctly;
- owned tile `start_pos` patches correctly;
- `JIT_BATCH_SIZE=1` and plain eager `m.forward` still fail;
- this is not graph replay, scalar patching, or cache identity.

New root cause:

**The owned AMDGCN tile expects fp16 K/V, but the canonical model cache is fp32. In-model it reads fp32 cache bytes as
fp16, producing garbage/NaN.**

Therefore the top priority is:

1. quarantine the owned-tile route/candidate as not real-cache-correct;
2. fix the dtype/data contract;
3. revalidate owned tile under real multi-step decode token correctness;
4. only then resume runtime-KV.

Do not continue runtime-KV implementation until this passes.

## Required Reading

Read these first:

1. `docs/runtime-kv-graphrunner-arg-patch-result-20260623.md`
2. `bench/qk-runtime-managed-kv-cache/graphrunner_arg_probe.json`
3. `docs/runtime-kv-buffer-identity-rebase-result-20260623.md`
4. `docs/runtime-managed-kv-cache-result-20260623.md`
5. `docs/runtime-kv-opaque-read-result-20260623.md`
6. `docs/decode-attention-route-b-b4-external-graph-node-result-20260621.md`
7. `docs/decode-attention-route-b-b4-external-graph-node-scope-20260621.md`
8. `docs/decode-attention-route-b-b3-owned-amdgcn-result-20260621.md`
9. `docs/b4-cheaper-combine-result-20260622.md`
10. `docs/decode-gap-audit-consolidated-20260622.md`
11. `structure/Development/performance-primitive-research-principles.md`
12. `structure/Development/session-handoff.md`

Inspect code:

- `tinygrad/llm/model.py`
- `extra/qk_owned_flash_decode.hip`
- `extra/qk_owned_flash_decode_graph_node.py`
- `extra/qk_b4_decode_eval.py`
- `extra/qk_b4_combine_ab.py`
- `extra/qk_kv_cache_state_token.py`
- `bench/qk-decode-eval/candidates.json`
- `bench/qk-decode-eval/binding_templates.json`

## Current Facts To Preserve

| Finding | Status |
|---|---|
| Owned tile standalone fp16 correctness | PASS in synthetic/microbench. |
| Owned tile graph-node integration | PASS mechanically. |
| Owned tile W==D under previous harness | Not authoritative for real-cache correctness; likely degenerate/insufficient. |
| Owned tile with canonical in-model fp32 cache | FAIL; NaN/garbage K read. |
| GraphRunner arg patching | Correct; not root cause. |
| Runtime-KV cache identity / rebase | Not root cause of latest failure. |
| Runtime-KV route | BLOCKED until owned tile reads real cache correctly. |

## Core Questions

Answer these with direct evidence:

1. What dtype is `self.cache_kv` in canonical model decode?
2. What dtype/layout does `owned_flash_tile_gqa` actually read?
3. Where did prior B4/B5 validation fail to exercise real fp32 cache data?
4. What is the smallest safe dtype-contract fix?
5. Does the fixed route produce byte-identical multi-step decode tokens?
6. Does the fixed route preserve or improve W==D?

## Phase Plan

### Phase 0 — Quarantine / Authority Lock

Before editing:

- confirm clean git status;
- record HEAD;
- inspect candidate registry for owned AMDGCN tile status;
- ensure no default path uses owned tile;
- if candidate metadata says `default_eligible=true`, change to `false` or add explicit broken-real-cache marker;
- add a short note in the result doc draft that runtime-KV is blocked on this revalidation.

Start result doc:

- `docs/owned-amdgcn-tile-real-cache-revalidation-result-20260623.md`

### Phase 1 — Reproduce The Real-Cache Failure

Create or extend a probe:

- `extra/qk_owned_tile_real_cache_repro.py`

It must compare:

1. canonical `gqa_coop_vec` baseline;
2. `DECODE_ATTN_AMDGCN_TILE=1` owned tile on canonical model cache;
3. synthetic fp16 cache owned tile reference;
4. optionally runtime fp16 cache route if useful.

Required checks:

- token correctness, not just positions or graph identity;
- at least 4 decode steps after prefill;
- layer-0 Q/K/V finite checks;
- cache dtype and shape;
- first layer where K becomes NaN or mismatch appears;
- whether failure happens in eager `m.forward`, `JIT_BATCH_SIZE=1`, and TinyJit replay;
- whether K/V pointer addresses match expected cache object;
- whether owned tile sees fp32 bytes as half.

Artifact:

- `bench/qk-owned-amdgcn-tile-real-cache/repro.json`

Verdicts:

- `OWNED_TILE_REAL_CACHE_FAIL_REPRODUCED`
- `OWNED_TILE_REAL_CACHE_FAIL_NOT_REPRODUCED`
- `OWNED_TILE_FAILURE_NOT_DTYPE`

Stop if failure is not reproduced; write result and do not patch blindly.

### Phase 2 — Dtype Contract Fix Candidates

Try the smallest safe fix. Evaluate in order.

#### Candidate A — fp16 route cache

Make the owned-tile route use an fp16 KV cache, matching the tile contract.

Options:

- initialize `cache_kv` as fp16 only under an owned-tile/runtime-KV flag;
- cast/store K/V as fp16 for the route;
- preserve canonical fp32 cache path for default `gqa_coop_vec`.

Risks:

- may change baseline/canonical behavior if not isolated;
- prefill/decode handoff must fill fp16 cache correctly;
- model route guards must prevent accidental default use.

#### Candidate B — fp32-aware owned tile

Change the owned tile to read fp32 K/V cache and convert to fp16 or fp32 internally.

Risks:

- changes kernel memory bandwidth / performance;
- may lose the intended v_dot2 fp16 path unless converted carefully;
- more kernel work.

#### Candidate C — explicit fp32->fp16 staging before owned tile

Convert canonical fp32 prefix into fp16 staging buffer for owned tile.

Risks:

- may reintroduce a copy tax;
- likely not W==D-promotable if per-token/full-prefix;
- acceptable only as diagnostic unless bounded.

For each candidate:

- no default change;
- route must be env-gated;
- canonical default must remain byte-identical.

Artifact:

- `bench/qk-owned-amdgcn-tile-real-cache/dtype_fix.json`

Verdicts:

- `OWNED_TILE_FP16_CACHE_FIX_PASS`
- `OWNED_TILE_FP32_KERNEL_FIX_PASS`
- `OWNED_TILE_STAGING_DIAGNOSTIC_PASS`
- `OWNED_TILE_DTYPE_FIX_FAIL`
- `OWNED_TILE_DTYPE_FIX_TOO_EXPENSIVE`

### Phase 3 — Correctness Gate

Before W==D:

- greedy byte-identical vs `gqa_coop_vec` for at least 64 tokens;
- two prompts in one process;
- same prompt twice in one process;
- no NaN K/V at layer 0 or any sampled layer;
- no garbage token collapse;
- ctx where route fires must be explicitly reported;
- test eager, `JIT_BATCH_SIZE=1`, and TinyJit if applicable.

Artifact:

- `bench/qk-owned-amdgcn-tile-real-cache/correctness.json`

Failure verdicts:

- `OWNED_TILE_REAL_CACHE_CORRECTNESS_FAIL`
- `OWNED_TILE_NAN_KV_PERSISTS`
- `OWNED_TILE_ROUTE_FALLBACK_ONLY`
- `OWNED_TILE_CTX_RESTRICTION_BLOCKS_GATE`

### Phase 4 — W==D Gate

Only after correctness passes.

Baseline:

```bash
DEV=AMD JIT=1 Q4K_GEMV_WARP=1 Q4K_GEMV_WARP_DOWN=1
```

Candidate examples:

```bash
DEV=AMD JIT=1 Q4K_GEMV_WARP=1 Q4K_GEMV_WARP_DOWN=1 DECODE_ATTN_AMDGCN_TILE=1 <plus fix flag if needed>
```

Measure:

- ctx512;
- ctx1024;
- ctx2048;
- ctx4096.

If the owned tile is still ctx-restricted and cannot fire at ctx1024, report this honestly:

- `OWNED_TILE_CTX1024_BLOCKED`

Timing discipline:

- `.item()` inside timed window;
- repeated or in-process A/B;
- prove route fired;
- tokens match;
- report spread.

Pass:

- `>= +5%@ctx1024` if route supports ctx1024;
- `>= +7%@ctx4096`;
- no regression;
- byte-identical.

Artifact:

- `bench/qk-owned-amdgcn-tile-real-cache/wd.json`

### Phase 5 — Registry / Runtime-KV Decision

After W==D or correctness result:

1. Update candidate metadata:
   - broken route must be `default_eligible=false`;
   - fixed route may be `default_eligible=true` only if correctness + W==D pass;
   - otherwise keep `default_eligible=false`.

2. Decide runtime-KV status:
   - If owned tile real-cache correctness passes: runtime-KV may resume.
   - If owned tile remains broken: runtime-KV stays blocked.
   - If owned tile only works with fp16 route cache: runtime-KV must use the same fp16 cache contract.

3. Write follow-on scope only if needed:
   - `docs/runtime-kv-after-owned-tile-fix-scope-20260623.md`

## Final Result Doc

Write:

- `docs/owned-amdgcn-tile-real-cache-revalidation-result-20260623.md`

Required sections:

1. Verdict.
2. Why this revalidation was required.
3. What prior B4/B5 validation missed.
4. Reproduction of real-cache failure.
5. Dtype/layout contract.
6. Fix candidates tested.
7. Correctness result.
8. W==D result.
9. Registry/default decision.
10. Runtime-KV implication.
11. Remaining blockers.
12. Artifacts and commands.
13. Files changed.
14. Working tree status.

Allowed final verdicts:

- `OWNED_TILE_REAL_CACHE_WD_PASS`
- `OWNED_TILE_REAL_CACHE_CORRECTNESS_PASS_WD_FAIL`
- `OWNED_TILE_REAL_CACHE_CTX1024_BLOCKED`
- `OWNED_TILE_REAL_CACHE_CORRECTNESS_FAIL`
- `OWNED_TILE_REAL_CACHE_DTYPE_FIX_REQUIRED`
- `OWNED_TILE_REAL_CACHE_DTYPE_FIX_TOO_EXPENSIVE`
- `OWNED_TILE_REAL_CACHE_QUARANTINED`
- `OWNED_TILE_REAL_CACHE_FAILURE_NOT_REPRODUCED`

## Boundaries

- No default change.
- No 14B/32B.
- No runtime-KV continuation until owned tile is real-cache-correct.
- No new attention tile unless the dtype fix requires a minimal kernel contract fix.
- No native tinygrad codegen/renderer work.
- No paged KV.
- No activation/norm/GEMV work.
- Do not claim success from standalone synthetic fp16 cache.
- Do not claim success without token correctness.
- Revert unsafe source changes on failure.

## Claude Prompt

You are in `/home/ubuntu/tinygrad-arkey` on branch `qk-prefill-flag-leak-resolution`.

Read `docs/owned-amdgcn-tile-real-cache-revalidation-scope-20260623.md` completely and execute it.

The latest GraphRunner instrumentation proved args are correct and graph replay is not the root cause. The real issue
is owned AMDGCN tile data correctness: the tile expects fp16 K/V, while canonical in-model `cache_kv` is fp32. In-model
the tile reads fp32 cache bytes as fp16, causing NaN/garbage K and broken multi-step decode. Runtime-KV is blocked until
the owned tile reads real in-model cache data correctly.

Your task:

1. quarantine/mark the existing owned tile route as not real-cache-valid if metadata implies otherwise;
2. reproduce the failure with token correctness and finite K/V checks;
3. fix the dtype/data contract using the smallest safe route;
4. revalidate multi-step real-cache token correctness;
5. only then run W==D;
6. update registry/default eligibility and runtime-KV follow-on status.

Do not continue runtime-KV implementation. Do not rely on synthetic fp16 standalone correctness. Do not change defaults.
Report final verdict, commands, artifacts, source changes, registry changes, default status, and git status.
