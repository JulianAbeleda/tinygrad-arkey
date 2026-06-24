# Runtime-KV Buffer Identity / Rebase Follow-On — Scope / Claude Prompt (2026-06-23)

## Mission

Continue the runtime-managed KV cache lane from the latest narrowed result:

**`RUNTIME_KV_BAKING_TRIGGER_NARROWED_NOT_FULLY_ISOLATED`**

The prior probes refuted the RoPE-producer hypothesis. The append `PROGRAM` has `ProgramInfo.vars = {'start_pos'}`,
so `start_pos` is declared live at the append surface. The failure now appears to be a **cache buffer identity /
TinyJit capture-state interaction** left by the model's canonical-store prefill.

Goal:

1. Compare `cache_kv` identity after canonical-store prefill vs direct `cache.assign(numpy).realize()` fill.
2. Name the hidden identity/capture-state difference that causes append replay to write a fresh buffer or baked
   position after canonical prefill.
3. Test the smallest fix:
   - **A. rebase/clone `cache_kv` into a pristine realized buffer before decode JIT**, or
   - **B. fill prefill KV using the opaque append path too**.
4. If one fix passes, continue to graph identity, correctness, and W==D gates. If neither passes, classify and stop.

Do not build a RoPE kernel. Do not reopen attention/norm/GEMV. Do not change defaults.

## Required Reading

Read these first:

1. `docs/runtime-kv-baking-instrumentation-result-20260623.md`
2. `bench/qk-runtime-managed-kv-cache/instrumentation.json`
3. `docs/runtime-managed-kv-cache-result-20260623.md`
4. `docs/runtime-managed-kv-cache-implementation-scope-20260623.md`
5. `docs/runtime-kv-baking-diagnostic-20260623.md`
6. `bench/qk-runtime-managed-kv-cache/baking_diagnostic.json`
7. `docs/runtime-kv-opaque-read-result-20260623.md`
8. `docs/kv-cache-stateful-jit-capability-result-20260622.md`
9. `structure/Development/performance-primitive-research-principles.md`
10. `structure/Development/session-handoff.md`

Inspect code:

- `tinygrad/llm/model.py`
- `extra/qk_runtime_kv_cache_probe.py`
- `extra/qk_kv_cache_state_token.py`
- `extra/qk_kv_opaque_read_probe.py`
- `tinygrad/tensor.py` around `assign`, `contiguous`, `realize`
- `tinygrad/ops.py` for UOp identity/base/buffer semantics
- `tinygrad/engine/jit.py`
- `tinygrad/engine/realize.py`
- `tinygrad/engine/schedule.py`
- `tinygrad/codegen/linearize.py` / relevant rangeify if needed only for identity tracing

## Current Facts To Preserve

Do not retest settled hypotheses unless needed to validate instrumentation.

| Hypothesis / fact | Status |
|---|---|
| Append lacks live runtime var | **Refuted** — append `ProgramInfo.vars = {'start_pos'}`. |
| RoPE producer bakes stale data | **Refuted** — isolated rope-like source advances; model bakes offset / fresh-buffer behavior. |
| `@function` prefill alone causes baking | **Refuted**. |
| Short/high/256-step prefill causes baking | **Refuted**. |
| Direct `cache.assign(numpy).realize()` fill causes baking | **Refuted** — direct assign-fill advances. |
| Visible `cache_kv.uop` op explains it | **Refuted** — both cases show `Ops.RESHAPE`. |
| Model canonical-store prefill before decode JIT | **Confirmed trigger** — token-by-token and batched canonical prefill both bake. |
| Failure observation | captured append writes the eager/capture position only; replay writes do not persist to `block.cache_kv`, likely writing a fresh buffer. |

## Central Question

After the two fill paths produce the same visible shape and `cache_kv.uop = RESHAPE`, what differs?

Compare at least:

- Python object identity:
  - `id(block.cache_kv)`
  - `id(block.cache_kv.uop)`
  - `id(block.cache_kv.uop.base)` if available
  - `id(block.cache_kv.uop.buffer)` / realized buffer if available
- UOp structure:
  - op;
  - dtype;
  - shape;
  - base chain;
  - `src` chain up to a bounded depth;
  - any `AFTER`, `STORE`, `ASSIGN`, `BUFFER`, `COPY`, `RESHAPE`, `VIEW`, `DEFINE_VAR`, `BIND`.
- Buffer identity:
  - realized buffer object;
  - device;
  - size;
  - base buffer;
  - allocation address if available;
  - any replacement after realize / contiguous / assign.
- JIT capture state:
  - graph input buffers for append;
  - append `ji_args` buffer address across eager/capture/replay;
  - var_vals for `start_pos` across calls;
  - whether GraphRunner uses the persistent cache buffer or a substituted/fresh buffer.

## Phase Plan

### Phase 0 — Evidence Lock

Before edits:

- confirm clean git status;
- record HEAD;
- inspect latest instrumentation artifact;
- reproduce or inspect:
  - canonical-store-fill -> decode append bakes;
  - direct assign-fill -> decode append advances;
  - append `ProgramInfo.vars = {'start_pos'}`.

Start result doc:

- `docs/runtime-kv-buffer-identity-rebase-result-20260623.md`

### Phase 1 — Identity Diff Tool

Create:

- `extra/qk_runtime_kv_buffer_identity_probe.py`

It must build two controlled states:

1. **canonical-store-fill state**
   - fill cache through the model's canonical store path, token-by-token and batched if practical;
   - this should reproduce baking.

2. **assign-fill state**
   - fill same `[0:N]` positions with `cache.assign(numpy).realize()`;
   - this should advance.

For each state, dump:

- tensor/uop/buffer identity fields;
- bounded UOp graph fingerprints;
- append graph captured buffer addresses/ids;
- append runtime var patch info;
- replay write-position check.

Artifact:

- `bench/qk-runtime-managed-kv-cache/buffer_identity_diff.json`

Required verdicts:

- `BUFFER_IDENTITY_DIFF_FOUND`
- `BUFFER_IDENTITY_DIFF_NOT_FOUND`
- `BUFFER_IDENTITY_PROBE_INCONCLUSIVE`

### Phase 2 — Fix Candidate A: Pristine Buffer Rebase

If the diff suggests canonical-store fill leaves a bad identity / lingering store state, test a rebase before decode JIT:

Possible rebase forms to try:

1. `cache_kv = cache_kv.contiguous().realize()` if it truly changes buffer identity and advances.
   - Note: prior instrumentation says simple re-realize did **not** fix it, so do not stop here.

2. Allocate a new fresh runtime cache tensor and copy the data into it:
   - `fresh = Tensor.empty_like(cache_kv, dtype=dtypes.float16).realize()`
   - copy only valid prefix or full buffer;
   - replace runtime cache object with `fresh`;
   - ensure buffer address/object differs from canonical-store buffer.

3. Use a custom copy/rebase kernel if tensor copy preserves bad identity.

4. Reinitialize `block.cache_kv` / runtime cache handles so decode JIT captures the fresh buffer, not the old canonical-store identity.

The rebase must be performed once at prefill->decode handoff, not per token.

Test:

- after canonical-store prefill;
- run rebase;
- capture decode JIT;
- replay start_pos 2050..2053;
- verify all positions advance and persist.

Artifact:

- `bench/qk-runtime-managed-kv-cache/rebase_probe.json`

Allowed verdicts:

- `PRISTINE_REBASE_ADVANCES`
- `PRISTINE_REBASE_COPY_STILL_BAKES`
- `PRISTINE_REBASE_TOO_EXPENSIVE`
- `PRISTINE_REBASE_CORRECTNESS_FAIL`

Cost gate:

- one-time rebase cost must be reported;
- if full-MAXC rebase cost is paid once per prompt, compare it to per-token full-MAXC copy saved;
- if cost is per-token, reject.

### Phase 3 — Fix Candidate B: Opaque-Prefill Fill

If rebase fails or is too expensive, test filling the prefill cache through the opaque append path so canonical store never taints cache identity.

Do not require the owned attention read during prefill. This phase only needs to fill K/V correctly.

Options:

1. Token-by-token opaque append for a short prefill range.
   - simplest correctness proof;
   - likely too slow for production but useful to isolate identity.

2. Batched opaque append kernel for T>1 prefill positions.
   - only if token-by-token proof passes;
   - not required for first gate unless cheap.

3. Hybrid:
   - canonical prefill computes K/V;
   - opaque append writes K/V into runtime cache;
   - attention/prefill output can still use canonical path if needed.

Test:

- fill [0:N] via opaque append;
- capture decode JIT;
- replay start_pos N..N+3;
- verify positions advance and persist.

Artifact:

- `bench/qk-runtime-managed-kv-cache/opaque_prefill_probe.json`

Allowed verdicts:

- `OPAQUE_PREFILL_ADVANCES`
- `OPAQUE_PREFILL_TOO_SLOW`
- `OPAQUE_PREFILL_CORRECTNESS_FAIL`
- `OPAQUE_PREFILL_NOT_EXPRESSIBLE`

### Phase 4 — Minimal In-Model Route If A Or B Passes

Only if Phase 2 or 3 proves advancing replay.

Add default-off route:

```text
RUNTIME_KV_CACHE=1
```

Route requirements:

- strict Qwen3-8B/gfx1100/B=1/T=1 guard;
- canonical path untouched;
- prefill->decode handoff uses the chosen fix:
  - rebase, or
  - opaque prefill fill;
- decode appends via opaque append;
- attention reads persistent cache through existing owned tile;
- no assigned_kv full-MAXC copy;
- no unsafe source route left enabled if correctness fails.

### Phase 5 — Graph Identity Gate

Write:

- `bench/qk-runtime-managed-kv-cache/rebased_graph_identity.json`

Required:

- no `E_49152` or equivalent full-MAXC copy per decode token;
- append node present;
- owned tile/owned combine present;
- replay positions advance after prefill;
- graph uses persistent fresh/rebased cache buffer;
- no fallback;
- no per-token host sync beyond measurement.

### Phase 6 — Correctness Gate

Before W==D:

- 64-token greedy byte-identical;
- two different prompts in one process;
- same prompt twice in one process;
- no stale cache;
- no 151936 collapse;
- first token and later tokens match;
- start_pos sequence around prefill boundary verified.

Allowed failure verdicts:

- `REBASED_RUNTIME_KV_CORRECTNESS_FAIL`
- `REBASED_RUNTIME_KV_STALE_CACHE_FAIL`
- `REBASED_RUNTIME_KV_PERSISTENCE_FAIL`

### Phase 7 — W==D Gate

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

Known issue:

- existing owned tile is ctx-restricted/short-ctx fragile. If the route cannot fire below ctx2048, report this
  honestly and do not claim `+5%@ctx1024` promotion. You may still measure ctx2048/4096 as research-only.

Pass gate:

- `>= +5%@ctx1024` if route supports ctx1024;
- no regression;
- byte-identical;
- tight spread.

Verdicts:

- `REBASED_RUNTIME_KV_WD_PASS`
- `REBASED_RUNTIME_KV_CTX1024_BLOCKED`
- `REBASED_RUNTIME_KV_LOCAL_PASS_WD_FAIL`
- `REBASED_RUNTIME_KV_OVERHEAD_EATS_WIN`

## Final Result Doc

Write:

- `docs/runtime-kv-buffer-identity-rebase-result-20260623.md`

Required sections:

1. Verdict.
2. What the previous instrumentation found.
3. Buffer identity diff.
4. Rebase probe.
5. Opaque-prefill probe if reached.
6. In-model route result if reached.
7. Graph identity.
8. Correctness.
9. W==D.
10. Candidate/default decision.
11. Remaining blockers.
12. Artifacts and commands.
13. Files changed.
14. Working tree status.

Allowed final verdicts:

- `RUNTIME_KV_BUFFER_IDENTITY_FIXED_WD_PASS`
- `RUNTIME_KV_BUFFER_IDENTITY_FIXED_CTX1024_BLOCKED`
- `RUNTIME_KV_BUFFER_IDENTITY_FIXED_CORRECTNESS_FAIL`
- `RUNTIME_KV_BUFFER_IDENTITY_REBASE_ADVANCES_SCOPE_READY`
- `RUNTIME_KV_BUFFER_IDENTITY_OPAQUE_PREFILL_SCOPE_READY`
- `RUNTIME_KV_BUFFER_IDENTITY_DIFF_FOUND_NO_FIX`
- `RUNTIME_KV_BUFFER_IDENTITY_DIFF_NOT_FOUND`
- `RUNTIME_KV_BUFFER_IDENTITY_INCONCLUSIVE`

## Boundaries

- No default change.
- No 14B/32B.
- No paged KV.
- No new attention tile.
- No RoPE kernel.
- No native tinygrad core rewrite unless the result is explicitly "scope ready", not implemented.
- No activation/norm/GEMV work.
- Do not retest closed hypotheses unless validating instrumentation.
- Do not leave unsafe source changes enabled after failure.

## Claude Prompt

You are in `/home/ubuntu/tinygrad-arkey` on branch `qk-prefill-flag-leak-resolution`.

Read `docs/runtime-kv-buffer-identity-rebase-scope-20260623.md` completely and execute it.

The latest instrumentation refuted the RoPE-producer hypothesis. The append `PROGRAM` declares `start_pos` live,
but after the model's canonical-store prefill, decode JIT replay writes only the capture/eager position or a fresh
buffer. Direct `cache.assign(numpy).realize()` fill of the same prefix advances correctly. Both states show
`cache_kv.uop = RESHAPE`, so the difference is hidden buffer identity / lingering store / TinyJit capture state.

Your task:

1. Build a buffer identity diff tool comparing canonical-store-fill vs assign-fill.
2. Name the hidden difference if possible.
3. Test pristine-buffer rebase after canonical prefill.
4. If needed, test opaque-prefill fill.
5. Only if a fix advances replay, reintroduce the default-off runtime KV route and run graph identity,
   correctness, and W==D gates.

Do not build a RoPE kernel. Do not reopen attention/norm/GEMV. Do not change defaults. Preserve and report all
artifacts and stop honestly on the first named blocker.
