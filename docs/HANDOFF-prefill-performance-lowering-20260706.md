# Handoff: Prefill Performance Lowering

Date: 2026-07-06

This is the pickup doc for the 8B/14B prefill performance-lowering work. It is intentionally a routing document:
read the listed files before building anything, and do not duplicate existing gates/probes.

## Goal

Recover prefill performance without relying on handwritten/raw instruction-list kernels.

Promotion is out of scope for this handoff. The current goal is pre-promotion completion: every non-promotion row should
either be `done` with gate-backed evidence or explicitly `blocked` with the exact missing compiler/scheduler/policy work.

Authoritative report:

```bash
PYTHONPATH=. python3 -m extra.qk.prefill_performance_lowering_report --pre-promotion --compact
PYTHONPATH=. python3 -m extra.qk.prefill_performance_lowering_report --pre-promotion --orchestration --compact
```

Source of truth:

- `extra/qk/prefill_performance_lowering_registry.py`
- `extra/qk/prefill_performance_lowering_report.py`
- `docs/prefill-performance-lowering-scope-20260706.md`
- `docs/prefill-lowering-spark-orchestration-100pct-20260706.md`

## Current Percent Table

These percentages are estimates for planning only. The status fields and gate outputs remain the hard authority.

| Target | Row | Owner | Status | Completion | Why |
| --- | --- | --- | --- | ---: | --- |
| 8B | fp16 recovery scope | policy | `pending` | 20% | Scope exists, but generated fast fp16 prefill has not replaced raw graph-GEMM. |
| 8B | baseline | policy | `done` | 100% | Current warmstart and bounded whole-prefill route-attribution evidence are in place. |
| 8B | single-operand LDS staging | codegen | `in_progress` | 55% | Small and route-bound A/B staging evidence exists; medium performance and coop B ownership remain blocked. |
| 8B | both-operands staging | codegen | `pending` | 35% | Tiny generated both-operand probe passes; route-bound medium proof is missing. |
| 8B | cooperative partition | scheduler | `pending` | 45% | Tiny coop B probe and route-bound case exist; medium source-B shape is not owned by the rewrite. |
| 8B | optional double buffer | scheduler | `not_started` | 0% | Sidecar only; defer until phases 1-4 leave a measured residual. |
| 14B | packed/MMQ recovery scope | policy | `pending` | 25% | Scope and substrates are tracked; generated MMQ does not dominate direct-packed VALU. |
| 14B | baseline | policy | `pending` | 45% | Direct-packed memory-safe baseline is documented but still performance-limited. |
| 14B | tile contract | scheduler | `done` | 100% | Bounded full-role tile contract is centralized and gated. |
| 14B | WMMA surface decision | vocab | `done` | 100% | Shaped WMMA surface is selected and backed by evidence. |
| 14B | small lifecycle | codegen | `done` | 100% | Small bounded multi-tile lifecycle is gated. |
| 14B | synthetic role shape | scheduler | `pending` | 60% | Role-shape contract and gate exist; full execution needs scheduler-owned tile loops. |
| 14B | model authority | policy | `blocked` | 30% | Blocker gate exists; QK route policy cannot select 14B MMQ prefill route ids. |
| 14B | Q6 residual decision | policy | `blocked` | 20% | Q6 direct-generated route exists; no generated Q6_K prefill MMQ route or residual attribution artifact exists. |

Current strict row-count completion is 4/14 pre-promotion rows (`done`). Current estimated average completion emitted by
the report is 52.5%. Treat the estimate as directional only: the remaining work is compiler/scheduler/policy work, not
bookkeeping.

## Current Real Blockers

### 8B

1. `PREFILL_GRAPH_GEMM=1` is still the old fast raw instruction-list path.
2. Generated single-operand staging has evidence, but the medium warmstart path is performance-flat.
3. Cooperative B partition is proven in a tiny generated probe, but the medium source-B shape carries a `GLOBAL` tile
   context. The postrange rewrite now fails closed instead of generating invalid UOps.
4. Both-operand staging is only proven in a tiny custom probe, not route-bound medium execution.
5. Optional double buffering is not a blocker yet. Do not start it unless the route-bound cooperative path works and
   still leaves a large measured residual.

### 14B

1. Tile contract, surface decision, and small lifecycle are done.
2. Full role-shape execution is blocked by `scheduler_owned_tile_loop_missing`.
3. MMQ model authority is blocked because `tinygrad/llm/route_policy.py` does not support the 14B prefill MMQ route ids.
4. Q6 residual policy is blocked because there is no generated Q6_K prefill MMQ route and no residual attribution
   artifact.

## Do Not Duplicate

These are already built and should be reused:

- 8B baseline authority:
  - `extra/qk/prefill_v2_schedule_table_gate.py`
  - `extra/qk/prefill_whole_synced.py`
  - `bench/prefill-v2-schedule-table/latest.json`
  - `bench/prefill-whole-synced/latest.json`
- 8B fp16 staging probes:
  - `extra/qk/prefill_graph_gemm_single_operand_stage_gate.py`
  - `extra/qk/prefill_graph_gemm_fp16_stage_gate.py`
  - `extra/qk/prefill_graph_gemm_route_bound_stage_gate.py`
  - `extra/qk/prefill_graph_gemm_tile_loop_stage_gate.py`
  - `extra/qk/prefill_graph_gemm_medium_stage_gate.py`
- 8B cooperative B probes:
  - `extra/qk/cooperative_stage_lanemap.py`
  - `extra/qk/prefill_graph_gemm_coop_partition_gate.py`
  - `extra/qk/prefill_graph_gemm_coop_route_contract_gate.py`
- 14B Q4_K/Q8_1 tile contract and lifecycle:
  - `extra/qk/q4k_wmma_tile_lowering.py`
  - `extra/qk/q4k_wmma_full_role_contract_gate.py`
  - `extra/qk/q4k_wmma_tiled_lifecycle_gate.py`
  - `extra/qk/q4k_wmma_tiled_role_shape_exec_gate.py`
  - `extra/qk/q4k_wmma_tiled_no_hand_kernel_gate.py`
- 14B policy blockers:
  - `extra/qk/prefill_14b_model_authority_gate.py`
  - `extra/qk/prefill_14b_q6_decision_gate.py`

## Reading List

Read these before coding:

1. `docs/prefill-performance-lowering-scope-20260706.md`
2. `docs/prefill-lowering-spark-orchestration-100pct-20260706.md`
3. `docs/codegen-wmma-lds-staging-design-20260705.md`
4. `docs/handwritten-kernel-exhaustive-lowering-scope-20260706.md`
5. `docs/q4k-wmma-full-role-lowering-solution-scope-20260705.md`
6. `docs/route-b-iu8-wmma-mmq-design-20260705.md`
7. `extra/qk/prefill_performance_lowering_registry.py`
8. `extra/qk/prefill_performance_lowering_report.py`
9. `tinygrad/codegen/opt/postrange.py`
10. `tinygrad/schedule/rangeify.py`
11. `tinygrad/schedule/wmma.py`
12. `tinygrad/llm/prefill_routes.py`
13. `tinygrad/llm/route_policy.py`

Read these for evidence/artifacts:

1. `bench/prefill-graph-gemm-medium-stage/latest.json`
2. `bench/prefill-graph-gemm-coop-route-contract/latest.json`
3. `bench/q4k-wmma-full-role-contract/latest.json`
4. `bench/q4k-wmma-tiled-role-shape-exec/latest.json`
5. `bench/q4k-wmma-tiled-lifecycle/latest.json`
6. `bench/q4k-wmma-scheduler-surface/latest.json`

## Recommended Next Work

### First pick: 8B source-B tile ownership

Target files:

- `tinygrad/codegen/opt/postrange.py`
- `tinygrad/schedule/rangeify.py`
- `extra/qk/prefill_graph_gemm_medium_stage_gate.py`
- `extra/qk/prefill_graph_gemm_coop_route_contract_gate.py`

Done means:

- The route-bound cooperative B case no longer skips due to source-B `GLOBAL` tile context.
- The generated route compiles without invalid `PTRCAT`/late-vectorization failures.
- The medium gate beats the current warmstart LOCAL baseline by the configured margin.
- No raw `Ops.INS` or `extra/qk/prefill/wmma.py` markers enter the strict path.

### Second pick: 14B scheduler-owned tiled loop

Target files:

- `extra/qk/q4k_wmma_tile_lowering.py`
- `extra/qk/q4k_wmma_tiled_role_shape_exec_gate.py`
- `extra/qk/prefill_int8_wmma_spec.py`
- scheduler/codegen files only after proving no existing helper already covers the loop boundary.

Done means:

- `scheduler_owned_tile_loop_missing` is replaced by a generated scheduler-owned loop over `m_tile`, `n_tile`, and
  `group_tile`.
- Full role-shape gate remains bounded and does not materialize `[groups, M, N]` RAW.
- The gate still does not claim model promotion.

### Third pick: policy after scheduler path exists

Do not add route-policy support before the scheduler/codegen route is real. When it is real:

- Add supported route ids in `tinygrad/llm/route_policy.py`.
- Add representative 14B policy rows.
- Re-run `extra/qk/prefill_14b_model_authority_gate.py`.
- Add Q6 residual attribution before changing Q6 policy.

## Validation Commands

Fast focused validation:

```bash
PYTHONPATH=. python3 -m pytest \
  test/unit/test_prefill_performance_lowering.py \
  test/unit/test_prefill_14b_policy_gates.py \
  test/unit/test_q4k_wmma_tile_lowering.py \
  test/unit/test_q4k_wmma_tiled_gates.py \
  test/unit/test_prefill_graph_gemm_medium_stage_gate.py \
  test/unit/test_prefill_graph_gemm_route_bound_stage_gate.py -q
```

Current report:

```bash
PYTHONPATH=. python3 -m extra.qk.prefill_performance_lowering_report --pre-promotion --orchestration --compact
```

AMD medium 8B gate:

```bash
timeout 720 bash -lc 'PYTHONPATH=. python3 -m extra.qk.prefill_graph_gemm_medium_stage_gate --run-amd --pin-clock --compact'
```

14B tiled role-shape gate:

```bash
timeout 300 bash -lc 'PYTHONPATH=. python3 -m extra.qk.q4k_wmma_tiled_role_shape_exec_gate --compact'
PYTHONPATH=. python3 -m extra.qk.q4k_wmma_full_role_contract_gate --compact
```

Policy blockers:

```bash
PYTHONPATH=. python3 -m extra.qk.prefill_14b_model_authority_gate
PYTHONPATH=. python3 -m extra.qk.prefill_14b_q6_decision_gate
```

Both policy commands are expected to exit nonzero until their blockers are solved.
