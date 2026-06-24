# Owned AMDGCN Tile Short-Context Correctness / Promotion Scope (2026-06-23)

## Mission

Make the now real-cache-correct owned AMDGCN decode-attention route work at short/mid context, especially
**ctx1024**, so it can be evaluated for default eligibility.

Latest state:

- `OWNED_TILE_REAL_CACHE_CTX1024_BLOCKED`
- dtype-contract bug fixed: canonical fp32 cache is cast to fp16 before the owned tile;
- real-cache multi-step decode is byte-identical where the route runs;
- W==D is strongly positive:
  - `+11.5% @ctx2048`
  - `+16.0% @ctx4096`
- ctx1024 is blocked by the route's short-context / split policy restriction.

This scope is about **short-ctx correctness and promotion gating**, not runtime-KV copy elimination.

## Required Reading

Read these first:

1. `docs/owned-amdgcn-tile-real-cache-revalidation-result-20260623.md`
2. `docs/owned-amdgcn-tile-real-cache-revalidation-scope-20260623.md`
3. `docs/runtime-kv-graphrunner-arg-patch-result-20260623.md`
4. `docs/runtime-managed-kv-cache-result-20260623.md`
5. `docs/decode-attention-route-b-b4-external-graph-node-result-20260621.md`
6. `docs/b4-cheaper-combine-result-20260622.md`
7. `docs/b4-split-kv-combine-tax-result-20260621.md`
8. `docs/split-kv-economics-audit-result-20260621.md`
9. `docs/decode-gap-audit-consolidated-20260622.md`
10. `docs/decode-ffn-gemv-warp-result-20260622.md`
11. `docs/q4k-gemv-warp-promotion-hardening-result-20260622.md`
12. `structure/Development/performance-primitive-research-principles.md`
13. `structure/Development/session-handoff.md`

Inspect code:

- `tinygrad/llm/model.py`
- `extra/qk_owned_flash_decode.hip`
- `extra/qk_owned_flash_decode_graph_node.py`
- `extra/qk_owned_tile_real_cache_repro.py`
- `extra/qk_b4_decode_eval.py`
- `extra/qk_b4_policy_sweep.py`
- `extra/qk_b4_combine_ab.py`
- `bench/qk-decode-eval/candidates.json`
- `bench/qk-decode-eval/binding_templates.json`

## Current Facts To Preserve

| Fact | Status |
|---|---|
| Owned tile dtype bug | Fixed by mandatory fp16 cast before tile. |
| Default path | unchanged. |
| Owned tile at ctx2048/4096 | byte-identical and W==D positive. |
| Owned tile at ctx1024 | blocked / short-ctx incorrect or disabled. |
| Runtime-KV | unblocked on tile correctness but deferred; copy elimination is no longer the immediate promotion blocker. |
| Prior B3/B4/B5 degenerate-cache claims | not authority for real-cache correctness. |

## Core Questions

Answer with direct evidence:

1. Why exactly is ctx1024 blocked?
   - route guard?
   - split count too high?
   - empty/invalid split?
   - softmax meta NaN?
   - combine reading uninitialized `part`/`meta`?
   - owned tile overread?
   - numeric underflow/overflow?

2. What is the smallest split/ctx policy that is correct at ctx512/1024/2048/4096?

3. Does the fixed policy keep the strong ctx2048/4096 W==D gains?

4. Does ctx1024 clear `>= +5%` W==D?

5. If ctx1024 cannot clear, is the route still an owner long-context knob or default-ineligible?

## Phase Plan

### Phase 0 — Baseline / Authority Lock

Before edits:

- confirm clean git status;
- record HEAD;
- confirm real-cache dtype fix is present;
- run/inspect default gqa baseline;
- run/inspect current owned route:
  - ctx512;
  - ctx1024;
  - ctx2048;
  - ctx4096;
- confirm tokens match at ctx2048/4096 and fail/block at ctx1024;
- confirm candidate metadata says fixed-real-cache but default-ineligible.

Start result doc:

- `docs/owned-amdgcn-tile-short-ctx-result-20260623.md`

### Phase 1 — Short-Context Failure Reproduction

Create or extend:

- `extra/qk_owned_tile_short_ctx_probe.py`

Probe matrix:

| ctx | S | combine | expected |
|---:|---:|---|---|
| 512 | current default | base/hd/hw if relevant | identify fail/block |
| 1024 | current default | base/hd/hw if relevant | identify fail/block |
| 2048 | current default | current route | must pass |
| 4096 | current default | current route | must pass |

For each ctx/S:

- compare owned tile output vs `gqa_coop_vec` or numpy/reference;
- report `rel_max`, `rel_rmse`;
- detect NaN/Inf in:
  - Q;
  - K/V cache slice;
  - `part`;
  - `meta` (`m`, `l`);
  - combine output;
- count valid/nonempty splits;
- report `n_valid`, `S`, split span, empty split count;
- report whether any split has `n_start >= n_valid`;
- verify combine ignores empty splits correctly;
- dump first bad head/split/dim if failure.

Artifact:

- `bench/qk-owned-amdgcn-tile-short-ctx/failure_matrix.json`

Allowed verdicts:

- `SHORT_CTX_EMPTY_SPLIT_NAN`
- `SHORT_CTX_OVERSPLIT_UNINITIALIZED_META`
- `SHORT_CTX_COMBINE_EMPTY_SPLIT_BUG`
- `SHORT_CTX_TILE_OVERREAD`
- `SHORT_CTX_NUMERIC_BUG`
- `SHORT_CTX_ROUTE_GUARD_ONLY`
- `SHORT_CTX_FAILURE_NOT_REPRODUCED`

Stop if failure is not reproduced.

### Phase 2 — Split Policy / Empty-Split Fix

Try policy/fix candidates in order.

#### Candidate A — ctx-aware S clamp

Choose S such that every split has work:

```text
S_eff <= n_valid
```

and ideally a minimum tokens-per-split:

```text
tokens_per_split >= 16 or another empirically justified floor
```

Implement as route policy first, not kernel rewrite.

#### Candidate B — kernel empty-split guard

If oversplitting is desired for occupancy, make tile/combine empty-split-safe:

- tile writes neutral meta for empty split:
  - `m = -inf`
  - `l = 0`
  - `part = 0`
- combine ignores `l == 0` / `m == -inf` safely;
- no NaN from `exp(-inf - -inf)`.

#### Candidate C — hybrid policy

Use ctx-aware S clamp plus empty-split-safe kernel for robustness.

#### Candidate D — fallback below threshold

If short ctx cannot be made correct and fast:

- owned tile route only fires at ctx >= threshold;
- default_eligible remains false unless promotion policy allows ctx-gated route.

Artifact:

- `bench/qk-owned-amdgcn-tile-short-ctx/split_policy.json`

Allowed verdicts:

- `SHORT_CTX_SPLIT_POLICY_PASS`
- `SHORT_CTX_EMPTY_SPLIT_GUARD_PASS`
- `SHORT_CTX_HYBRID_PASS`
- `SHORT_CTX_FALLBACK_REQUIRED`
- `SHORT_CTX_FIX_REGRESSES_LONG_CTX`

### Phase 3 — Token Correctness Gate

Before W==D:

- greedy byte-identical vs default gqa for at least 64 tokens;
- two prompts in one process;
- same prompt twice in one process;
- ctx512/1024/2048/4096 as supported;
- no NaN K/V or output;
- no garbage token collapse;
- route fires where expected;
- fallback fires where expected.

Artifact:

- `bench/qk-owned-amdgcn-tile-short-ctx/correctness.json`

Failure verdicts:

- `SHORT_CTX_TOKEN_CORRECTNESS_FAIL`
- `SHORT_CTX_STALE_CACHE_FAIL`
- `SHORT_CTX_ROUTE_FALLBACK_ONLY`
- `SHORT_CTX_LONG_CTX_REGRESSION`

### Phase 4 — W==D Gate

Only after correctness passes.

Baseline:

```bash
DEV=AMD JIT=1 Q4K_GEMV_WARP=1 Q4K_GEMV_WARP_DOWN=1
```

Candidate:

```bash
DEV=AMD JIT=1 Q4K_GEMV_WARP=1 Q4K_GEMV_WARP_DOWN=1 DECODE_ATTN_AMDGCN_TILE=1
```

Measure:

- ctx512;
- ctx1024;
- ctx2048;
- ctx4096.

Timing discipline:

- `.item()` inside timed window;
- repeated or in-process A/B;
- prove route/fallback per ctx;
- tokens match;
- report spread.

Promotion gate:

- `>= +5% @ctx1024`;
- `>= +7% @ctx4096`;
- no ctx512 regression beyond noise;
- byte-identical.

If route intentionally falls back at ctx512, report ctx512 as fallback/control.

Artifact:

- `bench/qk-owned-amdgcn-tile-short-ctx/wd.json`

Verdicts:

- `OWNED_TILE_SHORT_CTX_WD_PASS`
- `OWNED_TILE_SHORT_CTX_CORRECTNESS_PASS_WD_FAIL`
- `OWNED_TILE_SHORT_CTX_CTX1024_STILL_BLOCKED`
- `OWNED_TILE_SHORT_CTX_LONG_CTX_ONLY_KNOB`

### Phase 5 — Registry / Default Decision

After W==D:

- update `bench/qk-decode-eval/candidates.json`;
- if W==D gates pass:
  - `default_eligible=true`;
  - `default_on=false` unless owner explicitly asks to flip;
  - include ctx policy and dtype contract.
- if ctx1024 still blocked:
  - keep `default_eligible=false`;
  - mark as long-context owner knob;
  - include positive ctx2048/4096 results.

Runtime-KV status:

- If short-ctx correctness/W==D passes, runtime-KV can resume as incremental copy-elimination.
- If route remains ctx-gated, runtime-KV must inherit that restriction or wait for a short-ctx attention read.

### Phase 6 — Result Doc

Write:

- `docs/owned-amdgcn-tile-short-ctx-result-20260623.md`

Required sections:

1. Verdict.
2. Why short-ctx was blocked.
3. Failure matrix.
4. Split policy / kernel fix.
5. Correctness result.
6. W==D result.
7. Registry/default decision.
8. Runtime-KV implication.
9. Remaining blockers.
10. Artifacts and commands.
11. Files changed.
12. Working tree status.

Allowed final verdicts:

- `OWNED_TILE_SHORT_CTX_WD_PASS`
- `OWNED_TILE_SHORT_CTX_CORRECTNESS_PASS_WD_FAIL`
- `OWNED_TILE_SHORT_CTX_CTX1024_STILL_BLOCKED`
- `OWNED_TILE_SHORT_CTX_LONG_CTX_ONLY_KNOB`
- `OWNED_TILE_SHORT_CTX_FIX_REGRESSES_LONG_CTX`
- `OWNED_TILE_SHORT_CTX_FAILURE_NOT_REPRODUCED`

## Boundaries

- No default change.
- No 14B/32B.
- No runtime-KV implementation in this task.
- No native tinygrad codegen/renderer work.
- No paged KV.
- No activation/norm/GEMV work.
- Do not use degenerate/zero cache as correctness authority.
- Do not claim success without token correctness.
- Revert unsafe source changes on failure.

## Claude Prompt

You are in `/home/ubuntu/tinygrad-arkey` on branch `qk-prefill-flag-leak-resolution`.

Read `docs/owned-amdgcn-tile-short-ctx-scope-20260623.md` completely and execute it.

The owned AMDGCN tile real-cache dtype bug is fixed and the route is now byte-identical with strong W==D gains at
ctx2048/4096, but default eligibility is blocked because ctx1024 is not supported/correct. Your task is to make the
owned tile short-context-correct, especially ctx1024, or decisively classify why it cannot be.

Start by reproducing the short-context failure with real cache and token correctness. Then diagnose split policy,
empty splits, combine meta, overread, and NaN sources. Try ctx-aware split policy first, then empty-split-safe kernel
guards if needed. Revalidate token correctness before any W==D claim.

Do not continue runtime-KV. Do not change defaults. Do not use synthetic/zero cache as authority.

Report final verdict, commands, artifacts, source changes, registry/default changes, runtime-KV implication, and git
status.
