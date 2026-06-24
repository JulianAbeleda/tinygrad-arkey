# Prefill Aggressive-Theoretical Closure Scope (2026-06-24)

## Objective

Prove whether prefill can reach the documented aggressive projection and decide if that projection is reachable with current tooling.

- Baseline: `3485.17 / 3404.00 / 3176.80 / 2720.58 / 2177.03` tok/s
- Confirmed (`eightwave`): `3597.23 / 3505.13 / 3263.13 / 2784.15 / 2217.39` tok/s
- Aggressive/theoretical: `4593.45 / 4486.47 / 4187.02 / 3585.72 / 2869.33` tok/s

Source: `docs/prefill-baseline-confirmed-aggressive-bound-handoff-20260624.md`, `bench/qk-prefill-long-context-no-regression-audit/` artifacts, `bench/qk-prefill-post-decode-parity-frontier/`.

## Why this has a different proof profile than decode

Prefill aggressive values are a corridor projection from unresolved integration residual closure, not a direct measured candidate baseline. So this scope has two proof modes:

- **Closed-theoretical mode**: aggressive upper-bound envelope is achievable under a non-search gap-closure hypothesis.
- **Direct-measure mode**: measured runs show equivalent or near-equivalent gains without opening broad search.

If direct mode fails, capture whether the shortfall is a tooling/integration proof limit.

## Success criteria

- For direct mode: measured prefill tok/s at required contexts equals or exceeds aggressive targets within tolerance, with no correctness or integration regressions.
- For theoretical mode: prove the dominant blockers are integration-in-model and not missing kernel/route candidates, so aggressive is not attainable under current scope.

## Scope

- Non-search first, then bounded follow-up only.
- No default flips except explicitly gated candidate comparisons.
- No free-form emit search unless closed by this scope with proof that the gap is in a bounded transferable candidate space.

## Required tooling

- `extra/qk_prefill_whole_synced.py`
- `extra/qk_prefill_per_role_time_tax.py`
- route-check helpers already used by prefill scripts (no new helper required)

Required output directory:

- `bench/qk-prefill-aggressive-target-proof-20260624/`

Required artifacts:

- `authority.json`
- `whole_prefill_baseline.json`
- `whole_prefill_candidates.json`
- `whole_prefill_chunk_series.json`
- `per_role_time_tax_timeseries_by_ctx.json`
- `route_coverage_by_ctx_and_role.json`
- `kv_attention_split_timeseries.json`
- `single_chunk_vs_whole_prefill.json`
- `memory_pressure_watch.json`
- `decision.json`

## Commands to run

```bash
DEV=AMD JIT=1 PREFILL_V2=1 PYTHONPATH=. .venv/bin/python extra/qk_prefill_whole_synced.py
DEV=AMD JIT=1 PREFILL_V2=1 PREFILL_GEMM_DBUF=0 PREFILL_GEMM_PLRA=1 PYTHONPATH=. .venv/bin/python extra/qk_prefill_whole_synced.py
DEV=AMD JIT=1 PREFILL_V2=1 PREFILL_GEMM_8WAVE=1 PYTHONPATH=. .venv/bin/python extra/qk_prefill_whole_synced.py
DEV=AMD JIT=1 PREFILL_V2=1 PREFILL_GEMM_PIPELINE=1 PREFILL_GEMM_PIPELINE_TM=2 PREFILL_GEMM_PIPELINE_TN=2 PYTHONPATH=. .venv/bin/python extra/qk_prefill_whole_synced.py
DEV=AMD JIT=1 PREFILL_V2=1 PREFILL_GEMM_PIPELINE=1 PREFILL_GEMM_PIPELINE_TM=4 PREFILL_GEMM_PIPELINE_TN=2 PYTHONPATH=. .venv/bin/python extra/qk_prefill_whole_synced.py
```

## Candidate matrix

1. **Baseline control**: shipped prefill path (eightwave default-on if promoted)
2. **Existing confirmed comparator**: `PREFILL_GEMM_8WAVE=1` explicit
3. **Needs-review safety branch** (bounded):
   - `PREFILL_GEMM_PIPELINE=1 PREFILL_GEMM_PIPELINE_TM=2 PREFILL_GEMM_PIPELINE_TN=2`
   - `PREFILL_GEMM_PIPELINE=1 PREFILL_GEMM_PIPELINE_TM=4 PREFILL_GEMM_PIPELINE_TN=2`
4. **Secondary fallback**: `PREFILL_GEMM_DBUF=0 PREFILL_GEMM_PLRA=1` (isolated follow-up only)

Primary baseline command:

```bash
DEV=AMD JIT=1 PREFILL_V2=1 PYTHONPATH=. .venv/bin/python extra/qk_prefill_whole_synced.py
```

Direct compare command (copy baseline with `PREFILL_GEMM_*` override):

```bash
DEV=AMD JIT=1 PREFILL_V2=1 PREFILL_GEMM_8WAVE=1 PYTHONPATH=. .venv/bin/python extra/qk_prefill_whole_synced.py
```

## Phase 0 — Authority lock

Record commit, dirty state, hardware, model path, max context, and all `PREFILL_*` flags. Include memory snapshot before/after.

## Phase 1 — Whole-prefill authority

Measure whole-prefill synced at:

- `512, 1024, 2048, 4096, 8192`

For each candidate row, capture `tok/s`, repeats, spread, and `start_pos` chunk series.

Requirements:

- Use same timing protocol as no-regression baseline artifacts.
- Include comparator with `PREFILL_GEMM_DBUF=0 PREFILL_GEMM_PLRA=1` only if feasible in the same protocol.

## Phase 2 — Long-context growth and slot-series evidence

- For each context collect chunk-wise series via `whole_prefill_chunk_series.json`.
- Create `single_chunk_vs_whole_prefill.json` with chunk ratio by context.
- This is the direct proof that integration slope exists (or does not).

## Phase 3 — Role/integration attribution with route coverage

- Run `extra/qk_prefill_per_role_time_tax.py` in a multi-chunk-safe way.
- Require single model instance reuse per run, profile-events clear per chunk.
- Capture `route_coverage_by_ctx_and_role.json` and confirm role route stability.
- Capture `kv_attention_split_timeseries.json` if parser support exists.

## Phase 4 — Memory and profile safety

Capture memory pressure before/after each 8192 run and after profile loops.

- guardrail: OOM must be remediated within two passes (profile loop shrinkage + run decomposition)
- if unresolved, emit fail label and stop broad comparison.

## Falsification map (proof this aggressive target cannot be reached in current scope)

- **No transfer despite candidates**: candidate-only gains stay below corridor and follow same ctx trend as current `PREFILL_LONGCTX_NO_TRANSFER` style outcome.
- **Integration slope confirmed**: multi-chunk growth is > token-linear even when candidate fixed, and chunk-wise whole/ single ratio keeps worsening with ctx.
- **Role stability with no core transfer**: `ffn_gate_up`, `ffn_down`, `kv_proj`, `qo_proj` shares stable across start_pos and context, indicating fixed-role transfer failure.
- **Metadata mismatch**: route-lock mismatch or `route_coverage_by_ctx_and_role` missing 8192 coverage.
- **Safe-stop by harness**: OOM or profiling truncation prevents full role attribution, meaning only theoretical proof can be claimed until rerun support is added.

## Failure-label outputs (single primary)

- `PREFILL_AGGRESSIVE_TARGET_UNPROVEN__IN_MODEL_INTEGRATION`
- `PREFILL_AGGRESSIVE_TARGET_UNPROVEN__CHUNK_SLOPE`
- `PREFILL_AGGRESSIVE_TARGET_UNPROVEN__UNATTRIBUTED_ROLES`
- `PREFILL_AGGRESSIVE_TARGET_UNPROVEN__PIPELINE_GATE_FAIL`

Include a root-cause section in `decision.json` referencing:

- `docs/prefill-long-context-root-cause-audit-result-20260624.md`
- `docs/prefill-long-context-integration-hardening-scope-20260624.md`
- `bench/qk-prefill-post-decode-parity-frontier/time_tax.json`
- `bench/qk-prefill-post-decode-parity-frontier/tensile_gap_attribution.json`

## Exit outcome

- PASS => aggressive target appears reachable from measured evidence and this scope can be turned into a follow-up promotion package.
- FAIL => classify as **non-search integration-bound** and propose a tooling-instrumentation follow-up (not broad prefill emit search).
