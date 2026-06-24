# Prefill Long-Context Root-Cause Audit Result (2026-06-24)

## Verdict

**Decision**: `PREFILL_ROOTCAUSE_LONG_CTX_INTEGRATION_BOUND`

This run closes the harness-authorship uncertainty for this session: the long-context prefill drop is not a route artifact and not a `start_pos=0` concrete-chunk measurement artifact. It is a real **multi-chunk integration slope** issue that remains after control-plane alignment.

## Authority lock

`bench/qk-prefill-root-cause-long-context-20260624/authority.json`

Lock used:

- `DEV=AMD`
- `JIT=1`
- `PREFILL_V2=1`
- No explicit emit overrides (`null` / unset for `PREFILL_GEMM_*`), so normal default graph-GEMM route was tested.
- Model: `Qwen3-8B-Q4_K_M.gguf`
- Machine: `RX 7900 XTX / gfx1100`
- Repo state: branch `qk-prefill-flag-leak-resolution` at `99a73794bcbb659c196ea8f9793a99b931b5916c` (dirty)

Artifacts:

- `whole_prefill_authority_graph_gemm.json`
- `whole_prefill_authority_tensile.json`
- `whole_prefill_by_ctx_raw.json`
- `whole_prefill_chunk_series.json`
- `whole_prefill_8192_growth.json`
- `single_chunk_vs_whole_prefill.json`
- `per_role_tax_timeseries_by_ctx.json`
- `route_coverage_by_ctx_and_role.json`
- `kv_attention_split_timeseries.json`
- `memory_pressure_watch.json`
- `measurement_plan.json`
- `authority.json`
- `decision.json`

## Harness reconciliation (authoritative)

Whole-prefill `tok/s` by context on current default graph-GEMM lane and tensile comparator:

| ctx | graph-gemm tok/s | tensile tok/s | ratio (graph/tensile) |
|---:|---:|---:|---:|
| 512 | 3598 | 3403 | +5.73% |
| 1024 | 3507 | 3320 | +5.63% |
| 2048 | 3263 | 3101 | +5.22% |
| 4096 | 2789 | 2641 | +5.60% |
| 8192 | 2537 | 2418 | +4.92% |

## Why this is not the harness trap

- Route lock is explicit and stable in both authority lanes.
- Coverage includes all required contexts with full chunk ladders:
  - 512: 1 chunk
  - 1024: 2 chunks
  - 2048: 4 chunks
  - 4096: 8 chunks
  - 8192: 16 chunks (8 measured + 8 extrapolated)
- This removes the key "one-chunk optimism" explanation.

## Why this is not a pure route bug

Single `start_pos=0` chunk values are consistently optimistic versus full whole-prefill:

| ctx | route | single-chunk tok/s | whole-prefill tok/s | single/whole |
|---:|---|---:|---:|---:|
| 512 | graph-gemm | 3597.647 | 3598 | 1.000 |
| 1024 | graph-gemm | 3597.647 | 3507 | 1.0258 |
| 2048 | graph-gemm | 3597.647 | 3263 | 1.1026 |
| 4096 | graph-gemm | 3597.647 | 2789 | 1.2899 |
| 8192 | graph-gemm | 3597.647 | 2537 | 1.4181 |

Same trend appears on tensile lane.

This is the signature of per-chunk integration overhead compounding across long contexts.

## Role-taxa evidence

`extra/qk_prefill_per_role_time_tax.py` produced stable role-shape shares across context:

- `ffn_gate_up`: 51.07ms → 51.48ms (`+1.04%`, max variation by `start_pos`/ctx)
- `ffn_down`: 34.29ms → 34.77ms (`+1.39%`)
- `qo_proj`: 23.347ms → 23.533ms (`+0.8%`)
- `kv_proj`: 11.515ms → 11.540ms (`+0.21%`)

In short: core GEMM roles are stable; they do not explain the long-context collapse.

## Attribution gaps

- `route_coverage_by_ctx_and_role.json` marks `actionable:false` for all 8192 roles; this is a post-processing marker issue, not a timing contradiction.
- `kv_attention_split_timeseries.json` currently emits zeroes for `attention_qk_ms`, `attention_pv_ms`, `copy_materialization_ms`; it does not yet split those buckets for this lane.

## Decision and next step

- `docs/prefill-long-context-root-cause-audit-result-20260624.md` decision: `PREFILL_ROOTCAUSE_LONG_CTX_INTEGRATION_BOUND`
- `bench/qk-prefill-root-cause-long-context-20260624/decision.json` next step: `NONSEARCH_INTEGRATION_FIX_SCOPE`

Action now is to treat this as an integration-fix audit (host-sync / launch-boundary / copy-bound instrumentation + targeted follow-up attribution), not as another prefill kernel search pass.
