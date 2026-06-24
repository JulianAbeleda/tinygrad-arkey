# Prefill Long-Context Root-Cause Audit Scope (2026-06-24)

## Objective

Resolve the remaining long-context prefill issue with a deterministic causal audit. This scope targets a final pass to determine whether degradation is:

- harness protocol drift,
- real role growth in multi-chunk whole-prefill,
- route mismatch,
- or memory-profile artifact.

No route flips, no kernel search, and no decode changes in this scope.

## Why this scope is needed

The latest session confirmed:

- synced whole-prefill still shows a graph/Tensile gap at short-to-mid context,
- single concrete chunks are optimistic at longer prompt lengths,
- whole multi-chunk per-role attribution did not finish due `MemoryError` during PROFILE.

The issue is therefore narrowed but not fully proven.

## Read list (required)

- `docs/prefill-long-context-harness-authority-and-role-tax-scope-20260624.md`
- `docs/prefill-long-context-harness-authority-and-role-tax-result-20260624.md`
- `docs/prefill-post-decode-parity-frontier-result-20260623.md`
- `bench/qk-prefill-post-decode-parity-frontier/baseline_prefill.json`
- `bench/qk-prefill-post-decode-parity-frontier/time_tax.json`
- `bench/qk-prefill-long-context-no-regression-audit/time_tax_by_context.json`
- `extra/qk_prefill_whole_synced.py`
- `extra/qk_prefill_per_role_time_tax.py`
- `bench/qk-decode-eval/HARNESS_GUIDE.md`

## Authority lock required before this run

Create `authority.json` with:

- git hash and dirty state,
- branch and machine metadata,
- ROCm/runtime version if available,
- model path and max context,
- full env snapshot:
  - `DEV`, `JIT`, `PREFILL_V2`, `PREFILL_GRAPH_GEMM`, `PREFILL_TENSILE_GEMM`,
    `PREFILL_GEMM_8WAVE`, `PREFILL_GEMM_DBUF`, `PREFILL_GEMM_PLRA`,
    `PREFILL_GEMM_PLRAB`, `PREFILL_TENSILE_GEMM`, `PREFILL_CONCRETE_KV`, `PREFILL_SERVER_PROFILE`.

Unset variables must be explicit `null` or `"unset"` fields.

## Artifact set

Use:

`bench/qk-prefill-root-cause-long-context-20260624/`

Required artifacts:

- `authority.json`
- `measurement_plan.json`
- `whole_prefill_by_ctx_raw.json`
- `whole_prefill_chunk_series.json`
- `whole_prefill_8192_growth.json`
- `single_chunk_vs_whole_prefill.json`
- `route_coverage_by_ctx_and_role.json`
- `per_role_tax_timeseries_by_ctx.json`
- `kv_attention_split_timeseries.json`
- `memory_pressure_watch.json`
- `decision.json`

Final result doc:

- `docs/prefill-long-context-root-cause-audit-result-20260624.md`

## Hypothesis order and acceptance tests

### H0: Harness protocol issue

Test: compare one-shot chunk-by-chunk same-lane output against a direct whole run contract for all contexts.
Failure => `PREFILL_ROOTCAUSE_HARNESS_TRAP_ONLY`.

### H1: Route identity issue

Test: validate expected route IDs for graph-GEMM default and Tensile for each context.
Failure => `PREFILL_ROOTCAUSE_ROUTE_MISMATCH`.

### H2: Real whole-context role-growth

Test: role tax grows with `start_pos` and context; confirm whether growth is concentrated in:
- `kv_proj`,
- `ffn_down`,
- attention buckets (`attention_qk`, `attention_pv`),
- and/or copies/materialization.
Failure to find growth while gap remains => `PREFILL_ROOTCAUSE_INTEGRATION_BOUND_UNRESOLVED`.

### H3: Memory-profile distortion

Test: PROFILE runs complete without OOM and produce stable per-role aggregates for repeated trials.
Failure => `PREFILL_ROOTCAUSE_MEMORY_PROFILE_BLOCKER`.

## Execution lanes

### Lane A: synced whole-prefill raw chunk telemetry

Command:

`DEV=AMD JIT=1 PREFILL_V2=1 PYTHONPATH=. .venv/bin/python extra/qk_prefill_whole_synced.py`

Required:

- contexts: `512,1024,2048,4096,8192`,
- explicit `start_pos` timing output for each chunk,
- per-context raw chunk series and direct sum-based whole-prefill,
- repeat counts and spread.

Record to `whole_prefill_by_ctx_raw.json` and `whole_prefill_chunk_series.json`.

### Lane B: direct comparator

Command:

`DEV=AMD JIT=1 PREFILL_V2=1 PREFILL_TENSILE_GEMM=1 PREFILL_GRAPH_GEMM=0 PYTHONPATH=. .venv/bin/python extra/qk_prefill_whole_synced.py`

Use the exact same chunk schedule as Lane A.
Record route tags and bytes-identity checks.

### Lane C: whole-prefill growth table

Use outputs from Lane A/B to build:

- `whole_prefill_8192_growth.json`:
  - `ctx`
  - `slot_start_pos`
  - `slot_ms`
  - `slot_tok_s`
  - `slot_share_vs_whole`
- `single_chunk_vs_whole_prefill.json`:
  - `ctx`
  - `start_pos`
  - `single_chunk_tok_s`
  - `whole_prefill_tok_s`
  - `ratio`

### Lane D: memory-safe per-role tax over multi-chunk

Update `extra/qk_prefill_per_role_time_tax.py` into a looped one-chunk collector:

- keep a single model instance,
- capture PROFILE per start_pos/chunk,
- immediately aggregate role totals by `Context(PROFILE=1)`,
- clear `Compiled.profile_events` between chunks,
- write per-role rows with `ctx`, `start_pos`, `role`, `shape`, `calls`, `ms`, `share`.

If OOM still triggers, reduce:

- warmup count,
- context count in profile loop,
- number of repeated runs,
and rerun.

Record to `per_role_tax_timeseries_by_ctx.json` and `route_coverage_by_ctx_and_role.json`.

### Lane E: kv/attention split decomposition

From lane C + D derive:

- `kv_attention_split_timeseries.json`:
  - `ctx`
  - `start_pos`
  - `kv_proj_ms`
  - `attention_qk_ms`
  - `attention_pv_ms`
  - `copy_materialization_ms`
  - `other_ms`

This is the final decomposition to identify a concrete integration bound.

### Lane F: memory pressure telemetry

Before and after each run capture:

- `rocm-smi --showmeminfo vram --json`,
- active process memory usage,
- profile buffer size and number of profile events retained.

Write to `memory_pressure_watch.json`.

## Stop rules

- `PREFILL_ROOTCAUSE_ROUTE_MISMATCH` if expected route flags are not observed.
- `PREFILL_ROOTCAUSE_OOM_PROFILE` if profile lane cannot complete at 8192 with documented mitigation attempts.
- `PREFILL_ROOTCAUSE_8192_MISSING` if no `8192` context measurement exists in whole-prefill lane.
- `PREFILL_ROOTCAUSE_NO_GROWTH_SIGNAL` if role growth is not attributable and gap remains.

No route changes are made until these stop rules are cleared.

## Final decision

`decision.json` must select exactly one primary label:

- `PREFILL_ROOTCAUSE_HARNESS_TRAP_ONLY`
- `PREFILL_ROOTCAUSE_LONG_CTX_INTEGRATION_BOUND`
- `PREFILL_ROOTCAUSE_ROLE_GROWTH_BOUND`
- `PREFILL_ROOTCAUSE_ATTENTION_OR_KV_BOUND`
- `PREFILL_ROOTCAUSE_LAYOUT_COPY_BOUND`
- `PREFILL_ROOTCAUSE_MEMORY_PROFILE_BLOCKER`

Also include `next_step` as one of:

- `FULL_ROLE_TAX_REPEAT_WITH_LOW_MEM_PROFILE`
- `NONSEARCH_INTEGRATION_FIX_SCOPE`
- `DECODE_SAFE_CROSS_CHECK_FIRST`

