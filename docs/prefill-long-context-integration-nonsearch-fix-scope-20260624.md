# Prefill Long-Context Non-Search Integration Fix Scope (2026-06-24)

## Objective

Close the `NONSEARCH_INTEGRATION_FIX_SCOPE` gap from the long-context root-cause lane with a final authority check that confirms whether the integration drop remains after full-lattice timing and attribution.

## Current signal for this session

- `docs/prefill-long-context-root-cause-audit-result-20260624.md` ended on `PREFILL_ROOTCAUSE_LONG_CTX_INTEGRATION_BOUND` and required a follow-up.
- `docs/archive/prefill-long-context-integration-hardening-result-20260624.md` confirmed `PREFILL_LONGCTX_INTEGRATION_HARDENING_NO_GROWTH_CONFIRMED` for current graph route.
- The missing step is a fresh, explicit non-search follow-up artifact bundle with completion verdict.

## Required read-first

- [qk-prefill-integration-hardened authority artifacts](/home/ubuntu/tinygrad-arkey/bench/qk-prefill-long-context-integration-hardening-20260624)
- [qk-prefill-root-cause artifacts](/home/ubuntu/tinygrad-arkey/bench/qk-prefill-root-cause-long-context-20260624)
- [Scope from last session](/home/ubuntu/tinygrad-arkey/docs/prefill-long-context-harness-authority-and-role-tax-scope-20260624.md)

## Scope and boundaries

- Non-search attribution and authority confirmation only.
- No kernel edits.
- No defaults changes.
- No machine-search.
- Keep `eightwave`/graph-GEMM default behavior as-is for the authority lane.

## Required outputs

Create one run at:

- `bench/qk-prefill-long-context-integration-fix-20260624/`

Required artifacts:

- `authority.json`
- `whole_prefill_by_ctx_raw.json`
- `whole_prefill_chunk_series.json`
- `runtime_overlap_by_ctx.json`
- `single_chunk_vs_whole_prefill.json`
- `per_role_time_tax_timeseries_by_ctx.json`
- `route_coverage_by_ctx_and_role.json`
- `kv_attention_split_timeseries.json`
- `memory_pressure_watch.json`
- `decision.json`

## Execution command

```bash
cd /home/ubuntu/tinygrad-arkey
DEV=AMD JIT=1 PREFILL_V2=1 PYTHONPATH=. .venv/bin/python \
  extra/qk_prefill_integration_hardening.py \
  bench/qk-prefill-long-context-integration-fix-20260624 \
  --contexts 512,1024,2048,4096,8192 \
  --repeats 3 --inner 4 --profile-repeats 1
```

## Acceptance criteria

Read the folder artifacts and apply this decision ladder:

1. `single_chunk_vs_whole` stays ~1.0 at all contexts (no single-chunk optimism).
2. `route_coverage_by_ctx_and_role` shows `ffn_gate_up`, `ffn_down`, `kv_proj`, `qo_proj`, and `other` as expected.
3. `runtime_overlap_by_ctx` has bounded launch overhead; if growth returns, classify as host/runtime bound and exit with a patch hypothesis.
4. `kv_attention_split_timeseries` is present with non-negative buckets for all rows.

Completion rule:

- If all gates above pass and growth is not observed over full-lattice contexts, mark
  `PREFILL_LONGCTX_INTEGRATION_HARDENING_NO_GROWTH_CONFIRMED` + explicit closeout.
- If growth returns and can be attributed to launch/boundary overhead, classify that growth bound and queue a deterministic follow-up fix lane.

## Evidence target

Final result doc:
- `docs/prefill-long-context-integration-nonsearch-fix-result-20260624.md`

Decision artifact to publish:
- [bench/qk-prefill-long-context-integration-fix-20260624/decision.json](/home/ubuntu/tinygrad-arkey/bench/qk-prefill-long-context-integration-fix-20260624/decision.json)
