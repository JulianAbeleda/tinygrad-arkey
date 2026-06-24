# Prefill Long-Context Non-Search Integration Fix Result (2026-06-24)

## Verdict

**`PREFILL_LONGCTX_INTEGRATION_HARDENING_NO_GROWTH_CONFIRMED` (follow-through pass complete)**

The follow-up lane reproduced stable, full-lattice, multi-context prefill behavior with the current graph-GEMM path and did not re-open a long-context integration slope signal.

## Scope and lock

Result uses:

- Command: [scope execution command in scope doc](/home/ubuntu/tinygrad-arkey/docs/prefill-long-context-integration-nonsearch-fix-scope-20260624.md#execution-command)
- Artifact directory: [bench/qk-prefill-long-context-integration-fix-20260624](/home/ubuntu/tinygrad-arkey/bench/qk-prefill-long-context-integration-fix-20260624)

Authority contract:

- Full-lattice whole-prefill at contexts `512,1024,2048,4096,8192`.
- Graph route: `PREFILL_GRAPH_GEMM` on, `PREFILL_TENSILE_GEMM` off by default.
- `PREFILL_V2=1`, `DEV=AMD`, `JIT=1`.
- Full chunk lists for `8192` (`start_pos`: 0..7680 step 512), zero extrapolation.
- Authority and output files include command/env/git branch/dirty + memory and coverage rows.

## Key artifacts

- [authority.json](/home/ubuntu/tinygrad-arkey/bench/qk-prefill-long-context-integration-fix-20260624/authority.json)
- [whole_prefill_by_ctx_raw.json](/home/ubuntu/tinygrad-arkey/bench/qk-prefill-long-context-integration-fix-20260624/whole_prefill_by_ctx_raw.json)
- [runtime_overlap_by_ctx.json](/home/ubuntu/tinygrad-arkey/bench/qk-prefill-long-context-integration-fix-20260624/runtime_overlap_by_ctx.json)
- [single_chunk_vs_whole_prefill.json](/home/ubuntu/tinygrad-arkey/bench/qk-prefill-long-context-integration-fix-20260624/single_chunk_vs_whole_prefill.json)
- [per_role_time_tax_timeseries_by_ctx.json](/home/ubuntu/tinygrad-arkey/bench/qk-prefill-long-context-integration-fix-20260624/per_role_time_tax_timeseries_by_ctx.json)
- [route_coverage_by_ctx_and_role.json](/home/ubuntu/tinygrad-arkey/bench/qk-prefill-long-context-integration-fix-20260624/route_coverage_by_ctx_and_role.json)
- [kv_attention_split_timeseries.json](/home/ubuntu/tinygrad-arkey/bench/qk-prefill-long-context-integration-fix-20260624/kv_attention_split_timeseries.json)
- [decision.json](/home/ubuntu/tinygrad-arkey/bench/qk-prefill-long-context-integration-fix-20260624/decision.json)

## Harness summary by context

| ctx | whole tok/s | wall_ms | gpu_only_ms | host_sync_ms | single/whole |
|---:|---:|---:|---:|---:|---:|
| 512 | 3574 | 143.252 | 129.520 | 35.749 | 1.000000 |
| 1024 | 3573 | 286.605 | 242.162 | 71.600 | 1.000010 |
| 2048 | 3572 | 573.362 | 485.079 | 143.181 | 0.999877 |
| 4096 | 3571 | 1147.044 | 969.836 | 286.530 | 0.999458 |
| 8192 | 3569 | 2295.024 | 1940.997 | 573.244 | 0.999787 |

Notes:

- No single-chunk optimism at these settings (`ratio` remains ~1.0).
- No extrapolated chunk rows in any context.
- Route coverage remains complete for the expected graph GEMM roles plus `other`.
- `launch_overhead_ms` is effectively 0 in this measurement configuration.

## Attributed role and split closure

- `route_coverage_by_ctx_and_role.json` shows actionable coverage across
  `ffn_gate_up`, `ffn_down`, `kv_proj`, `qo_proj`, and `other` at all contexts.
- `kv_attention_split_timeseries.json` remains present with non-negative values; `kv_proj` + `other` dominate.
- Attention/copy buckets are still parsed as 0 for this current split naming path; this indicates naming capture shape in the existing schema, not a growth regression.

## Decision

- `bench/qk-prefill-long-context-integration-fix-20260624/decision.json` contains:
  - `label`: `PREFILL_LONGCTX_INTEGRATION_HARDENING_NO_GROWTH_CONFIRMED`
  - `next_step`: `NONSEARCH_INTEGRATION_FIX_SCOPE`
  - `requires`: `confirm_integration_fix`

## Closure

- This non-search follow-up run produced a fresh closure-grade authority sample confirming that the long-context prefill growth signal is not present under full-lattice wall measurement in this lane.
- In practice: **no immediate non-search code fix is authorized from this pass**; treat this as *integration posture closed for this cycle* and move back to the session owner workflow (decode/other frontier lanes).
- If a new long-context slope regression reappears, the next step is to restart this scope with:
  - a comparator command that toggles one explicit runtime-lane control (`PREFILL_TENSILE_GEMM=1`) in the same full-lattice mode, and
  - a per-profile name normalization pass for QK/PV/copy classification so those split buckets are attributable if material activity is present.
