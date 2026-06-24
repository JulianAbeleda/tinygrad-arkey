# Prefill Long-Context Integration Hardening Scope (2026-06-24)

## Objective

Convert the root-cause result (`PREFILL_ROOTCAUSE_LONG_CTX_INTEGRATION_BOUND`) into a hardening execution plan that removes long-context prefill integration slope without changing kernels or defaults.

This scope is explicitly **non-search** and **non-default-flip**. It must determine where the integration tax sits in the whole-path and whether it scales linearly across chunks.

## Why this scope exists

- `docs/prefill-long-context-root-cause-audit-result-20260624.md` shows the long-context drop is real, not a pure harness trap.
- `single_chunk_vs_whole_prefill` indicates a near-1.4x drop by `start_pos=0` optimism at 8192.
- `kv_attention_split_timeseries` had zero buckets; attention and copy decomposition is incomplete.
- `route_coverage_by_ctx_and_role` is `actionable:false` at 8192 and is likely a post-processing limitation, not evidence of no problem.

## Scope boundaries

- Keep `PREFILL_GEMM_*` defaults as they are for this run (`eightwave` default path should stay default-on). 
- Do not flip any prefill defaults during this scope.
- Do not change decode (`decode` and KV/attention stack untouched).
- Do not run new search experiments during this scope.

## Source context to read first

- `structure/Development/session-handoff.md`
- `docs/prefill-long-context-root-cause-audit-result-20260624.md`
- `docs/prefill-long-context-root-cause-audit-scope-20260624.md`
- `docs/prefill-long-context-harness-authority-and-role-tax-result-20260624.md`
- `bench/qk-prefill-root-cause-long-context-20260624/`

## Required tools

- `extra/qk_prefill_whole_synced.py`
- `extra/qk_prefill_per_role_time_tax.py`
- `extra/qk_decode_time_tax_audit.py` (if shared split helper is reused)
- Optional low-level profilers used by those scripts as needed (`rocm-smi`, `radeontop`, `nvidia-smi` style replacements for host/device trace)

## Required artifact folder

`bench/qk-prefill-long-context-integration-hardening-20260624/`

## Required outputs

- `authority.json`
- `whole_prefill_by_ctx_raw.json`
- `whole_prefill_chunk_series.json`
- `single_chunk_vs_whole_prefill.json`
- `runtime_overlap_by_ctx.json`
- `per_role_time_tax_timeseries_by_ctx.json`
- `route_coverage_by_ctx_and_role.json`
- `kv_attention_split_timeseries.json`
- `memory_pressure_watch.json`
- `decision.json`

## Required contexts

- 512, 1024, 2048, 4096, 8192

## Hypothesis ladder (stopping at first confirmed)

1. `H0`: the remaining mismatch is still a measurement artifact (extrapolation or profile-processing bug)
2. `H1`: host/runtime sync dominates and grows with chunk index/context
3. `H2`: in-model dispatch/copy layout tax grows across the multi-chunk path
4. `H3`: attention-copy boundary attribution is real and recoverable via existing integration patching

## Execution lanes

### Lane A: exact whole-prefill ladder, no extrapolation

Run synced whole-prefill timing and capture every chunk sample actually used by each context. For each context, collect `start_pos ∈ [0, chunk_size, ..., ctx-chunk_size]` exactly.

Command contract:

```
DEV=AMD JIT=1 PREFILL_V2=1 PYTHONPATH=. .venv/bin/python extra/qk_prefill_whole_synced.py
```

Then capture a comparator command in parallel:

```
DEV=AMD JIT=1 PREFILL_V2=1 PREFILL_TENSILE_GEMM=1 PREFILL_GRAPH_GEMM=0 PYTHONPATH=. .venv/bin/python extra/qk_prefill_whole_synced.py

```

Acceptance criteria:

- every `8192` `start_pos` must be measured (16 rows, no extrapolated fallback)
- no hardcoded chunk assumptions outside `chunk_size` and explicit context ladder
- artifact keys include repeats, sample count, spread, and exact command line

### Lane B: host/runtime overlap decomposition

Introduce (or temporarily instrument) wall/queue overlap counters in the prefill benchmark path to isolate:

- host-side prep (`token_ids`/input setup, graph warmup churn)
- `dev.synchronize()`/`Tensor.item()` boundary waits
- GPU-only launch span

Persist per-ctx metrics in `runtime_overlap_by_ctx.json` with columns:

- `wall_ms`
- `gpu_only_ms`
- `host_sync_ms`
- `launch_overhead_ms`
- `sync_calls`

### Lane C: full-lattice per-role attribution (memory-safe)

Update/execute per-role tax collection on the same full context ladder as Lane A. The output must be multi-chunk-aware (not a single `start_pos=0` chunk).

Requirements:

- one compiled model instance per run
- clear `Compiled.profile_events` between chunk runs
- persist after each context (`per_role_time_tax_timeseries_by_ctx.json`)
- mark rows with `authority`, `actionable`, and `ctx`, `start_pos`

### Lane D: kv/attention split decomposition

Extend role parsing to derive concrete buckets:

- `attention_qk_ms`
- `attention_pv_ms`
- `kv_proj_ms`
- `copy_materialization_ms`
- `other_ms`

Persist both per-chunk and aggregated by ctx files.

### Lane E: decision lock

Build decision JSON from all lanes in this order:

- `PREFILL_LONGCTX_INTEGRATION_HARDENING_HOSTSYNC_BOUND`
- `PREFILL_LONGCTX_INTEGRATION_HARDENING_DISPATCH_BOUND`
- `PREFILL_LONGCTX_INTEGRATION_HARDENING_ATTENTION_COPY_BOUND`
- `PREFILL_LONGCTX_INTEGRATION_HARDENING_NO_GROWTH_CONFIRMED`

If one of the first three labels is selected, add exact patch hypotheses (file + function + line-region) and hand the project to a follow-up hardening doc.

## Stop/abort rules

- incorrect route lock against explicit lane flags
- harness inconsistency between graph and tensile at same ctx for `single_chunk` comparisons
- OOM unmitigated after 2 memory mitigation passes
- any new evidence that requires decode changes

## Non-goals

- no default-flip or emit-search in this scope
- no decode route experiments
- no model retraining or architecture changes

## Success criteria

Hardening scope is complete when one of the above `PREFILL_LONGCTX_INTEGRATION_HARDENING_*` labels is fully evidence-backed and next steps are explicit: either code-bound patch plan or final assertion that this is still non-hardenable with current primitives.
