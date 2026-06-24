# Owned AMDGCN Tile Post-Promotion — Four-Step Scope / Claude Prompt (2026-06-23)

## Mission

Execute the post-promotion follow-on plan for the now-promotable owned AMDGCN decode-attention route through four
bounded steps:

1. **Owner default decision hardening**
2. **FO2 native fp16 owned-tile cache evaluation**
3. **Runtime-KV deferral / resume criteria**
4. **Project synthesis update**

Latest authority:

- `docs/owned-amdgcn-tile-short-ctx-result-20260623.md`
- verdict: **`OWNED_TILE_SHORT_CTX_WD_PASS`**
- candidate: `decode_attention_llama_flash_tile_owned_amdgcn_b4`
- status: `default_eligible=true`, `default_on=false`, `PROMOTABLE_ALLCTX`

This is not a new attention-kernel search. The owned route is already promoted as default-eligible. This task decides
whether to flip defaults, whether to remove the remaining fp32->fp16 cast tax, how to park/runtime-KV, and how to
rewrite the project state so stale "attention exhausted" conclusions stop propagating.

## Required Reading

Read these first:

1. `docs/owned-amdgcn-tile-short-ctx-result-20260623.md`
2. `docs/owned-amdgcn-tile-short-ctx-scope-20260623.md`
3. `docs/owned-amdgcn-tile-real-cache-revalidation-result-20260623.md`
4. `docs/owned-amdgcn-tile-two-followons-scope-20260623.md`
5. `docs/decode-ffn-gemv-warp-result-20260622.md`
6. `docs/q4k-gemv-warp-promotion-hardening-result-20260622.md`
7. `docs/decode-gap-audit-consolidated-20260622.md`
8. `docs/runtime-kv-graphrunner-arg-patch-result-20260623.md`
9. `docs/runtime-managed-kv-cache-result-20260623.md`
10. `structure/Development/performance-primitive-research-principles.md`
11. `structure/Development/session-handoff.md`
12. `docs/README.md`

Inspect code/artifacts:

- `tinygrad/llm/model.py`
- `extra/qk_owned_tile_short_ctx_probe.py`
- `extra/qk_owned_tile_real_cache_repro.py`
- `extra/qk_b4_decode_eval.py`
- `bench/qk-owned-amdgcn-tile-short-ctx/{failure_matrix,correctness,wd}.json`
- `bench/qk-decode-eval/candidates.json`
- `bench/qk-decode-eval/binding_templates.json`

## Current Facts To Preserve

| Fact | Status |
|---|---|
| dtype contract bug | fixed by mandatory fp16 cast before owned tile. |
| short ctx blocker | was guard-only; `DECODE_ATTN_AMDGCN_MIN_CTX` default lowered 2048 -> 512. |
| correctness | byte-identical to gqa at ctx512/1024/2048, real cache, multi-step. |
| W==D | +6.1%@512, +8.4%@1024, +11.5%@2048, +15.5%@4096. |
| default eligibility | `default_eligible=true`, `default_on=false`. |
| runtime-KV | deferred; now incremental, not promotion-critical. |
| fp16 cache | optional FO2 to remove cast/materialization tax. |
| stale conclusion to replace | "attention exhausted / B5 sub-bar" is no longer the current project state. |

## Step 1 — Owner Default Decision Hardening

### Goal

Decide whether the route is ready for default-on, or keep it default-off with a precise owner note.

### Required checks

Run one final hardening pass against the promoted route:

Baseline:

```bash
DEV=AMD JIT=1 Q4K_GEMV_WARP=1 Q4K_GEMV_WARP_DOWN=1
```

Candidate:

```bash
DEV=AMD JIT=1 Q4K_GEMV_WARP=1 Q4K_GEMV_WARP_DOWN=1 DECODE_ATTN_AMDGCN_TILE=1
```

Required:

- in-process A/B if available, or repeated process A/B with tight spread;
- ctx512/1024/2048/4096;
- 64-token greedy correctness at ctx1024;
- at least two prompts;
- same prompt twice in one process;
- route fired at each ctx;
- fallback works on unsupported shape/device if cheap to test;
- default path unchanged with flag off;
- no regression relative to promoted artifact.

Artifact:

- `bench/qk-owned-amdgcn-tile-post-promotion/default_decision.json`

Allowed verdicts:

- `OWNER_DEFAULT_READY`
- `OWNER_DEFAULT_KEEP_OFF_PENDING_MORE_BAKE`
- `OWNER_DEFAULT_BLOCKED_REGRESSION`
- `OWNER_DEFAULT_BLOCKED_FALLBACK`

### Default decision rules

If `OWNER_DEFAULT_READY`:

- write an owner note recommending default-on;
- do **not** flip `default_on=true` unless explicitly authorized by the user in this task;
- if user has explicitly authorized default flip, update model policy and candidate metadata.

If not ready:

- keep `default_on=false`;
- preserve `default_eligible=true` only if correctness/W==D still pass;
- document exact blocker.

## Step 2 — FO2 Native fp16 Owned-Tile Cache Evaluation

### Goal

Evaluate whether a native fp16 cache for the owned-tile route removes the fp32->fp16 cast/materialization tax and stacks on
top of the promoted route.

This is optional optimization, not required for promotion.

### Current issue

The fixed route casts Q/K/V to fp16 before tile. This is correct, but the K/V cache path may pay an avoidable
materialization/cast copy. FO2 asks whether the owned route can maintain a native fp16 KV cache contract.

### Required first decision

Before editing, answer:

| question | required answer |
|---|---|
| Is cast/materialization visible in rendered kernels? | identify kernel names/time |
| How much wall/GPU time can native fp16 cache recover? | measured or bounded estimate |
| Does gqa/default path require fp32 cache? | yes/no |
| Can fp16 cache be route-local without changing default? | yes/no |
| Is prefill->decode handoff safe with fp16 cache? | proof needed |

### Candidate designs

#### A. Route-local fp16 shadow cache

- keep canonical fp32 cache for default/gqa;
- maintain fp16 shadow cache only for owned route;
- prefill populates fp16 shadow cache;
- decode appends/casts into fp16 shadow cache;
- owned tile reads fp16 shadow cache directly.

Risk:

- doubles KV memory;
- prefill population cost;
- sync/lifecycle complexity.

#### B. Native fp16 cache only under owned-tile flag

- when route flag is enabled, initialize `cache_kv` as fp16 for the model;
- require all writers/readers in that route to respect fp16;
- fallback to gqa may need conversion or be disabled.

Risk:

- fallback complexity;
- default path must remain fp32/off-route.

#### C. Keep cast path

- if cast cost is small relative to attention win, do not complicate route.

### Required probe

Create or extend:

- `extra/qk_owned_tile_fp16_cache_probe.py`

Measure:

- cast/materialization kernels and time;
- route correctness with fp16 cache or shadow cache;
- W==D at ctx512/1024/2048/4096 if correctness passes;
- memory overhead if shadow cache.

Artifacts:

- `bench/qk-owned-amdgcn-tile-post-promotion/fp16_cache_probe.json`
- `bench/qk-owned-amdgcn-tile-post-promotion/fp16_cache_wd.json` if W==D reached

Allowed verdicts:

- `FP16_CACHE_WD_PASS`
- `FP16_CACHE_CORRECTNESS_FAIL`
- `FP16_CACHE_OVERHEAD_EATS_WIN`
- `FP16_CACHE_TOO_COMPLEX_DEFER`
- `FP16_CACHE_NOT_WORTH_IT_KEEP_CAST`

### Gate

Do not ship fp16 cache route unless:

- byte-identical;
- fallback-safe or strictly guarded;
- W==D improves over promoted cast route or clearly reduces memory/copy tax;
- no default behavior change.

## Step 3 — Runtime-KV Deferral / Resume Criteria

### Goal

Update runtime-KV status now that owned attention is promotable without copy elimination.

### Required analysis

Produce a table:

| runtime-KV question | current answer |
|---|---|
| Is runtime-KV needed for promotion? | no |
| Is runtime-KV still valuable? | yes, potentially incremental copy removal |
| Is runtime-KV blocked by owned-tile correctness? | no, tile side fixed |
| What blocker remains? | opaque append NaN / persistence route from prior probes |
| Should runtime-KV resume before FO2? | decide |
| Should runtime-KV resume before default decision? | likely no |

Artifact:

- `bench/qk-owned-amdgcn-tile-post-promotion/runtime_kv_status.json`

Allowed verdicts:

- `RUNTIME_KV_DEFER_INCREMENTAL`
- `RUNTIME_KV_RESUME_AFTER_FP16_CACHE`
- `RUNTIME_KV_RESUME_NOW`
- `RUNTIME_KV_RETIRED_BY_PROMOTED_ATTENTION`

Expected recommendation:

- defer runtime-KV unless FO2 shows the cast/copy tax is still large and the route needs further gains;
- if resumed, scope from the post-fix state and do not reuse stale GraphRunner/cache theories.

## Step 4 — Project Synthesis Update

### Goal

Update project documentation so the current state is not stale.

### Required changes

Update:

- `structure/Development/session-handoff.md`
- `docs/README.md`
- any active "current state" / "frontier" doc that still says:
  - attention exhausted;
  - B4/B5 sub-bar;
  - owned tile not real-cache-correct;
  - runtime-KV is the next promotion-critical lane;
  - ctx1024 blocked;
  - default_eligible=false for B4 owned tile.

Do **not** rewrite old historical docs. Add superseding notes or README/session handoff pointers.

Required synthesis table:

| primitive/lane | latest status | default eligibility | next action |
|---|---|---|---|
| Q4K_GEMV_WARP | W==D pass | true | owner default decision / parallel |
| owned AMDGCN attention | W==D pass all ctx | true | owner default decision |
| runtime-KV | deferred incremental | false/n/a | resume only after default/FO2 decision |
| native fp16 cache | optional | n/a | FO2 |
| attention Route B older B4/B5 combine narrative | superseded | n/a | historical only |

Artifact/doc:

- `docs/post-owned-attention-promotion-synthesis-20260623.md`

Allowed verdicts:

- `PROJECT_SYNTHESIS_UPDATED`
- `PROJECT_SYNTHESIS_STALE_REFERENCES_FOUND`
- `PROJECT_SYNTHESIS_BLOCKED`

## Final Result Doc

Write:

- `docs/owned-tile-post-promotion-four-step-result-20260623.md`

Required sections:

1. Verdict.
2. Step 1 default decision hardening.
3. Step 2 fp16 cache evaluation.
4. Step 3 runtime-KV status.
5. Step 4 project synthesis update.
6. Candidate/default metadata state.
7. Remaining blockers.
8. Artifacts and commands.
9. Files changed.
10. Working tree status.

Allowed final verdicts:

- `POST_PROMOTION_DEFAULT_READY_SYNTHESIS_UPDATED`
- `POST_PROMOTION_DEFAULT_FLIPPED_SYNTHESIS_UPDATED` only if owner explicitly authorizes default flip
- `POST_PROMOTION_KEEP_DEFAULT_OFF_SYNTHESIS_UPDATED`
- `POST_PROMOTION_FP16_CACHE_WD_PASS`
- `POST_PROMOTION_FP16_CACHE_DEFERRED`
- `POST_PROMOTION_BLOCKED_REGRESSION`
- `POST_PROMOTION_SCOPE_INCOMPLETE`

## Boundaries

- Do not flip defaults unless explicitly authorized.
- No 14B/32B.
- No runtime-KV implementation unless Step 3 explicitly returns `RUNTIME_KV_RESUME_NOW` and user authorizes.
- No new attention tile.
- No native tinygrad codegen/renderer work.
- No paged KV.
- No activation/norm/GEMV implementation beyond documenting Q4K_GEMV_WARP status.
- Do not overwrite historical docs; supersede them.
- Do not claim default-ready if final A/B or correctness regresses.

## Claude Prompt

You are in `/home/ubuntu/tinygrad-arkey` on branch `qk-prefill-flag-leak-resolution`.

Read `docs/owned-tile-post-promotion-four-step-scope-20260623.md` completely and execute it.

The owned AMDGCN attention route is now promotable:

- `OWNED_TILE_SHORT_CTX_WD_PASS`
- byte-identical real-cache decode
- +6.1%@512, +8.4%@1024, +11.5%@2048, +15.5%@4096
- `default_eligible=true`, `default_on=false`

Execute the four follow-ons:

1. Final owner default-decision hardening A/B.
2. FO2 native fp16 cache evaluation, only if bounded and useful.
3. Runtime-KV deferral/resume decision in light of the promoted attention route.
4. Project synthesis update so stale "attention exhausted" and "runtime-KV next" narratives are superseded.

Do not flip defaults unless explicitly authorized. Do not move to 14B/32B. Do not implement runtime-KV in this task
unless separately authorized. Write all artifacts and the final result doc, update session handoff/README/synthesis,
and report commands, files, registry/default state, and git status.
