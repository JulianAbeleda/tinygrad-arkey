# Prefill Long-Context Integration Hardening Result (2026-06-24)

## Verdict

The hardening lane confirms the long-context growth signal is now mainly a path-comparison/instrumentation issue in tensile, while graph default remains stable in integrated long-context timing.

- Graph lane (`route_graph_gemm_on_tensile_off`): `PREFILL_LONGCTX_INTEGRATION_HARDENING_NO_GROWTH_CONFIRMED`
- Tensile lane (`route_graph_gemm_off_tensile_on`): `PREFILL_LONGCTX_INTEGRATION_HARDENING_HOSTSYNC_BOUND`

## Inputs

- `bench/qk-prefill-long-context-integration-hardening-20260624/authority.json`
- `bench/qk-prefill-long-context-integration-hardening-20260624-tensile/authority.json`
- `bench/qk-prefill-long-context-integration-hardening-20260624/whole_prefill_by_ctx_raw.json`
- `bench/qk-prefill-long-context-integration-hardening-20260624-tensile/whole_prefill_by_ctx_raw.json`
- `bench/qk-prefill-long-context-integration-hardening-20260624/whole_prefill_chunk_series.json`
- `bench/qk-prefill-long-context-integration-hardening-20260624-tensile/whole_prefill_chunk_series.json`
- `bench/qk-prefill-long-context-integration-hardening-20260624/runtime_overlap_by_ctx.json`
- `bench/qk-prefill-long-context-integration-hardening-20260624-tensile/runtime_overlap_by_ctx.json`
- `bench/qk-prefill-long-context-integration-hardening-20260624/single_chunk_vs_whole_prefill.json`
- `bench/qk-prefill-long-context-integration-hardening-20260624/route_coverage_by_ctx_and_role.json`
- `bench/qk-prefill-long-context-integration-hardening-20260624-tensile/route_coverage_by_ctx_and_role.json`
- `bench/qk-prefill-long-context-integration-hardening-20260624/kv_attention_split_timeseries.json`
- `bench/qk-prefill-long-context-integration-hardening-20260624-tensile/kv_attention_split_timeseries.json`
- `bench/qk-prefill-long-context-integration-hardening-20260624/memory_pressure_watch.json`
- `bench/qk-prefill-long-context-integration-hardening-20260624-tensile/memory_pressure_watch.json`
- `bench/qk-prefill-long-context-integration-hardening-20260624/decision.json`
- `bench/qk-prefill-long-context-integration-hardening-20260624-tensile/decision.json`

## Execution

- Graph: `DEV=AMD JIT=1 PREFILL_V2=1 PYTHONPATH=. .venv/bin/python extra/qk_prefill_integration_hardening.py --out bench/qk-prefill-long-context-integration-hardening-20260624 --contexts 512,1024,2048,4096,8192`
- Tensile comparator: `DEV=AMD JIT=1 PREFILL_V2=1 PREFILL_TENSILE_GEMM=1 PREFILL_GRAPH_GEMM=0 PYTHONPATH=. .venv/bin/python extra/qk_prefill_integration_hardening.py --out bench/qk-prefill-long-context-integration-hardening-20260624-tensile --contexts 512,1024,2048,4096,8192`

## Whole-prefill throughput (synced)

| ctx | Graph tok/s | Tensile tok/s | Graph/Tensile | Decision evidence |
|---:|---:|---:|---:|---|
| 512 | 3578 | 3306 | 108.23% | graph still ahead, no long-context drop |
| 1024 | 3578 | 3303 | 108.33% | graph stable vs 8192 ladder |
| 2048 | 3575 | 3303 | 108.23% | linear growth held in graph lane |
| 4096 | 3575 | 3303 | 108.23% | graph unchanged across chunks |
| 8192 | 3576 | 3304 | 108.23% | graph no-growth confirmed |

## Chunk coverage and extrapolation

- 8192 was measured with the full 16-chunk lattice for both lanes (0..7680, step 512).
- `extrapolated_chunks` is 0 for all contexts, in both graph and tensile outputs.
- `memory_pressure_watch.json` indicates no additional memory growth after context startup:
  - graph max used 21.776 GB, total 25.753 GB
  - tensile max used 24.401 GB, total 25.753 GB

## Single-chunk versus whole-prefill

| ctx | single tok/s | whole tok/s | ratio |
|---:|---:|---:|---:|
| 512 | 3578.0 | 3578 | 1.000 |
| 1024 | 3578.4 | 3578 | 1.000 |
| 2048 | 3576.4 | 3575 | 1.000 |
| 4096 | 3574.3 | 3575 | 0.9998 |
| 8192 | 3574.3 | 3576 | 0.9995 |

- At these points, single-chunk start-pos diagnostics are no longer showing the previous multi-chunk optimism gap (ratio is ~1.0).

## Runtime overlap decomposition

- Graph lane:
  - host sync is ~12.49% of wall at all contexts.
  - launch overhead is effectively 0% at 512 and grows to ~2.9% by 8192.
  - `launch_overhead_ms` is bounded and scales with chunk count, not exploding with context.
- Tensile lane:
  - host sync is ~12.49% of wall at all contexts.
  - launch overhead dominates (~86.9% wall), even as context scales.
  - GPU-only work remains ~0.6% of wall, confirming route/dispatch inefficiency in this comparator.

## Role/tax and coverage observations

- `route_coverage_by_ctx_and_role.json`:
  - graph lane: graph/GEMM roles are active and marked actionable across all contexts.
  - tensile lane: only `other / non_gemm` role is present in this run.
- `kv_attention_split_timeseries.json`:
  - graph lane: `kv_proj` is non-zero but `attention_qk_ms`, `attention_pv_ms`, and `copy_materialization_ms` remain 0.
  - tensile lane: `other` captures almost all time, with `kv_proj` and both attention/copy fields also 0 in this schema.
- This means attention/copy decomposition is still a blind spot for attributing the residual split; it is present in schema but currently unresolved.

## Closure and next step

- Decision files:
  - `bench/qk-prefill-long-context-integration-hardening-20260624/decision.json`
    - `PREFILL_LONGCTX_INTEGRATION_HARDENING_NO_GROWTH_CONFIRMED`
  - `bench/qk-prefill-long-context-integration-hardening-20260624-tensile/decision.json`
    - `PREFILL_LONGCTX_INTEGRATION_HARDENING_HOSTSYNC_BOUND`
- Practical next step:
  - keep graph+eightwave as the current long-context authority lane;
  - scope a runtime-latency / launch-path follow-up specifically for tensile parity if comparator benchmarking is still required;
  - do not open kernel search from this evidence alone.
