# Prefill + Decode Next Workstreams Codex Scope (2026-06-24)

## Objective

Give a fresh Codex session one executable map for the next performance cycle:

1. harden prefill long-context scaling,
2. refresh decode-vs-llama authority,
3. expand decode search only if the refreshed decode evidence leaves a material, bounded gap.

This is an umbrella scope. It does not replace the detailed lane docs; it tells Codex which lane to run first, what evidence is authoritative, and when to stop.

## Current Big-Picture State

### Prefill

- `eightwave` is promoted as the current prefill graph-GEMM emit default.
- The open issue is not "find another emit knob first"; it is long-context integration scaling.
- Current root-cause label: `PREFILL_ROOTCAUSE_LONG_CTX_INTEGRATION_BOUND`.
- Priority: highest, because this is the remaining risk for long prompts and long chats.

### Decode

- Current default decode path is healthy:
  - lifecycle bundle: `DECODE_LIFECYCLE_RECHECK_BUNDLE_PASS`
- latest baseline: `bench/qk-decode-lifecycle-recheck-bundle/decode-lifecycle-recheck-20260624-172026`
- current A/B vs old internal route: +13.02% to +18.41% across ctx 512..4096
  - pre/post unknown-lockstep: `DECODE_UNKNOWN_BUCKET_LOCKSTEP_PROVEN`
- The baseline is now recorded in the run above; run periodic diff again before any broad decode search to measure drift from the prior snapshot.
- Priority: refresh clean decode-vs-llama first, then decide whether decode search is worth funding.

## Execution Order

### Phase 1: Prefill Long-Context Hardening

Run this first.

Detailed scope:

- `docs/prefill-long-context-integration-hardening-scope-20260624.md`

Primary artifact directory:

- `bench/qk-prefill-long-context-integration-hardening-20260624/`

Read before executing:

- `docs/prefill-long-context-root-cause-audit-result-20260624.md`
- `docs/prefill-long-context-harness-authority-and-role-tax-result-20260624.md`
- `docs/prefill-eightwave-promotion-result-20260624.md`
- `bench/qk-prefill-root-cause-long-context-20260624/`

Required outputs:

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
- `summary.md`

Required contexts:

- 512, 1024, 2048, 4096, 8192

Command starting point:

```bash
cd /home/ubuntu/tinygrad-arkey
DEV=AMD JIT=1 PREFILL_V2=1 PYTHONPATH=. .venv/bin/python extra/qk_prefill_whole_synced.py
```

Comparator starting point:

```bash
cd /home/ubuntu/tinygrad-arkey
DEV=AMD JIT=1 PREFILL_V2=1 PREFILL_TENSILE_GEMM=1 PREFILL_GRAPH_GEMM=0 PYTHONPATH=. .venv/bin/python extra/qk_prefill_whole_synced.py
```

Authority rules:

- Whole-prefill synced timing is authoritative.
- Single-chunk `start_pos=0` timing is diagnostic only.
- For ctx 8192, the full chunk lattice must be measured; do not extrapolate from one chunk.
- Keep `eightwave` on the default path.
- Do not flip defaults during this phase.
- Do not start a new prefill emit search until the integration tax is localized.

Phase 1 decision labels:

- `PREFILL_LONGCTX_INTEGRATION_HARDENING_HOSTSYNC_BOUND`
- `PREFILL_LONGCTX_INTEGRATION_HARDENING_DISPATCH_BOUND`
- `PREFILL_LONGCTX_INTEGRATION_HARDENING_ATTENTION_COPY_BOUND`
- `PREFILL_LONGCTX_INTEGRATION_HARDENING_NO_GROWTH_CONFIRMED`

Stop criteria:

- Stop with a result doc once one label is evidence-backed.
- If the result is host/runtime/dispatch bound, scope a code patch in a new follow-up doc before editing defaults.
- If the result is no-growth confirmed, mark prefill long-context as stable and move to Phase 2.

### Phase 2: Decode-Vs-Llama Authority Refresh

Run this after Phase 1, unless decode has regressed or changed defaults.

Detailed scopes:

- `docs/decode-lifecycle-recheck-bundle-periodic-scope-20260624.md`
- `docs/decode-ctx-slope-lifecycle-primitive-audit-scope-20260624.md`

Primary artifact directories:

- `bench/qk-decode-lifecycle-recheck-bundle/`
- `bench/qk-decode-ctx-slope-lifecycle-primitive-audit-20260624/`

Required outputs:

- `llama_vs_tinygrad_decode_by_ctx.json`
- `decode_role_time_by_ctx.json`
- `attention_qk_pv_softmax_split_by_ctx.json`
- `kv_identity_materialization_by_ctx.json`
- `q4k_route_coverage_by_role.json`
- `programs_and_syncs_by_ctx.json`
- `smallop_residual_census.json`
- `unknown_primitive_candidates.json`
- `coverage_score_update.json`
- `decision.json`
- `summary.md`

Required contexts:

- 512, 1024, 2048, 4096

Lifecycle recheck command:

```bash
cd /home/ubuntu/tinygrad-arkey
DEV=AMD JIT=1 PYTHONPATH=. .venv/bin/python extra/qk_decode_lifecycle_recheck_periodic.py --out-root bench/qk-decode-lifecycle-recheck-bundle
```

Kernel capture command:

```bash
cd /home/ubuntu/tinygrad-arkey
DEV=AMD JIT=1 PYTHONPATH=. .venv/bin/python extra/qk_decode_audit_common.py --contexts 512,1024,2048,4096
```

Ctx-slope audit command:

```bash
cd /home/ubuntu/tinygrad-arkey
DEV=AMD JIT=1 PYTHONPATH=. .venv/bin/python extra/qk_decode_ctx_slope_lifecycle_audit.py
```

Authority rules:

- Use W==D, synced `.item()` timing for tinygrad.
- Use explicit llama source artifact for every llama tok/s value.
- Mark stale llama sources as stale; do not silently mix current tinygrad with old llama numbers.
- Keep `DECODE_ATTN_KV_IDENTITY=1` and `DECODE_ATTN_AMDGCN_TILE=1` as the default route under test.
- Confirm no `E_49152` materialization regression.
- Confirm unknown-bucket closure pre/post if a new lifecycle bundle is run.

Phase 2 decision labels:

- `DECODE_VS_LLAMA_REFRESH_PASS_ABOVE_PARITY`
- `DECODE_VS_LLAMA_REFRESH_PASS_PARITY`
- `DECODE_VS_LLAMA_CTX_SLOPE_REVIEW_REQUIRED`
- `DECODE_VS_LLAMA_SOURCE_STALE_REVIEW_REQUIRED`
- `DECODE_ROUTE_OR_MATERIALIZATION_REGRESSION`

Stop criteria:

- If tinygrad is at/above llama at all measured ctx and no unexplained bucket is above 2% wall, do not start broad decode search.
- If ctx slope falls below parity at long ctx, move to Phase 3 with the named role/bucket as the search target.
- If llama source is stale, refresh or explicitly label the comparison stale before drawing a performance conclusion.

### Phase 3: Decode Search Expansion

Run this only after Phase 2 identifies a material bounded target.

Detailed scopes:

- `docs/decode-ctx-slope-lifecycle-primitive-audit-scope-20260624.md`
- `docs/decode-unknown-bucket-full-visibility-scope-20260624.md`
- `docs/exhaustive-gpu-lifecycle-primitive-audit-scope-20260624.md`
- `docs/gpu-lifecycle-primitive-coverage-tracker-20260624.md`

Candidate search targets, ranked by likely value:

1. ctx-growing attention subrole gap if Phase 2 shows QK/PV/combine slope.
2. small-op residual if Phase 2 shows a measurable, fusable wall share.
3. route coverage gap if any promoted Q4K/GEMV path is expected but inactive.
4. materialization/copy gap if `E_49152` or another hidden copy returns.
5. unknown lifecycle primitive if an unclassified bucket is above 2% wall share.

Search boundaries:

- Do not search weight-GEMV just because it is a large role; current evidence says it is at or above llama parity.
- Do not run a broad grid without a named primitive, knob set, correctness gate, and stop rule.
- Do not promote any decode route without token correctness, route fire, materialization check, ISA/resource check, and W==D transfer.

Required search row schema:

- `candidate_id`
- `target_phase`
- `primitive_class`
- `hypothesis`
- `knobs`
- `expected_effect`
- `correctness_gate`
- `route_gate`
- `materialization_gate`
- `isa_gate`
- `authority_benchmark`
- `stop_rule`

Phase 3 decision labels:

- `DECODE_SEARCH_TARGET_READY`
- `DECODE_SEARCH_EXECUTED_WIN`
- `DECODE_SEARCH_EXECUTED_NO_TRANSFER`
- `DECODE_SEARCH_DEFERRED_NO_BOUNDED_TARGET`
- `DECODE_CORE_RUNTIME_REQUIRED`

## Final Handoff Requirements

Every phase must end with:

- a result doc in `docs/`
- a `decision.json` in the phase artifact directory
- exact commands and environment variables
- source paths for all llama/tinygrad comparisons
- a short table with tok/s, delta tok/s, and percent vs baseline where applicable
- explicit next step: patch, search, rest, or defer

## Recommended Next Codex Prompt

```text
Read docs/prefill-decode-next-workstreams-codex-scope-20260624.md and execute Phase 1 only. Produce the required artifacts and result doc, then stop with the decision label and next patch/search recommendation.
```
