# Prefill Lowering Spark Orchestration to 100% Minus Promotion - 2026-07-06

## Objective

This is the execution plan for getting the prefill lowering work to **100% complete before route promotion**.

Promotion rows stay explicitly out of scope:

- `prefill_performance_target_1_promotion`
- `prefill_performance_target_2_promotion`

The goal is to clear every non-promotion active blocker in:

```sh
PYTHONPATH=. python3 -m extra.qk.prefill_performance_lowering_report --orchestration --compact
```

and leave only promotion/authority-review decisions for humans.

## Done Definition

The pre-promotion work is done when all of the following are true:

| Target | Done condition before promotion |
| --- | --- |
| 8B fp16 graph-GEMM recovery | Generated route-bound prefill owns WMMA+LDS/cooperative staging at medium and whole-prefill shapes without raw `Ops.INS`. |
| 8B performance | Generated pp512 prefill materially recovers toward the historical graph-GEMM class and beats current strict-pure baseline. |
| 8B provenance | Gates prove no default-selected path executes `extra/qk/prefill/wmma.py` or raw instruction-list WMMA. |
| 14B packed/MMQ | Q4_K/Q8_1 tiled role-shape execution no longer reports `scheduler_owned_tile_loop_missing`; synthetic roles execute bounded generated loops. |
| 14B model authority evidence | The generated packed/MMQ path is selectable and measurable on representative model roles without falling back to direct-packed VALU. |
| Q6 residual policy | Q6_K residual cost is measured and classified: keep direct-packed, add generated MMQ, or defer with explicit budget. |
| Harness | Central registry/report/artifacts explain every remaining row; no scattered one-off benchmark truth. |

Promotion is not part of this done definition. Promotion starts only after the above state is reached.

## Current Blockers

Current non-promotion active blockers from the registry:

| Row | Owner | Current blocker |
| --- | --- | --- |
| `prefill_performance_target_1_fp16_recovery` | policy | Raw graph-GEMM is still the old fast substrate; generated replacement has not recovered performance. |
| `prefill_performance_target_1_baseline` | policy | Representative AMD baselines need current authority evidence. |
| `prefill_performance_target_1_single_operand_stage` | codegen | Medium B tile-only staging is flat; coop diagnostic skips the real rewrite because source B has non-lane ranges outside warp+reduce. |
| `prefill_performance_target_1_both_operands_stage` | codegen | Both-source staging is only a custom probe; route-bound medium proof is missing. |
| `prefill_performance_target_1_coop_partition` | scheduler | Route-bound coop executes but does not beat baseline; medium source-B shape is not owned by the coop rewrite. |
| `prefill_performance_target_2_packed_mmq_recovery` | policy | Generated packed/MMQ substrate does not dominate direct-packed VALU yet. |
| `prefill_performance_target_2_baseline` | policy | Direct-packed baseline remains the authority path. |
| `prefill_performance_target_2_synthetic_shape` | scheduler | `scheduler_owned_tile_loop_missing`. |
| `prefill_performance_target_2_model_authority` | policy | No model-authority route policy proves MMQ-first packed prefill. |
| `prefill_performance_target_2_q6_residual_decision` | policy | No explicit Q6 residual policy. |

Sidecar:

| Row | Owner | Condition |
| --- | --- | --- |
| `prefill_performance_target_1_optional_double_buffer` | scheduler | Only run if phases 1-4 leave more than 20% residual vs historical graph-GEMM trajectory. |

## Orchestration Rules

Use Codex Spark agents for implementation/scaffolding, then one stronger review pass for integration quality.

Every Spark worker must:

- Work from `master`.
- Own disjoint files or a clearly stated responsibility boundary.
- Reuse existing gates/harnesses before adding new ones.
- Avoid raw `Ops.INS`, `DeviceKernel`, or string-injected ISA as a solution path.
- Update artifacts only for gates it owns.
- Return changed paths, commands run, pass/fail status, and next blocker if still blocked.
- Stop if it is only trying random rewrites without improving a gate or producing a sharper blocker.

The main integrator must:

- Keep `extra/qk/prefill_performance_lowering_registry.py` and `docs/prefill-performance-lowering-scope-20260706.md` synced.
- Run focused tests after each worker merge.
- Commit clean checkpoints with bracketed messages.
- Push each checkpoint.
- Reject patches that create duplicate harnesses or hide fallback routes.

## Parallel Workstreams

### Worker A - 8B Codegen: Medium Source-B Ownership

Owner area: `codegen`

Rows:

- `prefill_performance_target_1_single_operand_stage`
- `prefill_performance_target_1_both_operands_stage`

Primary files:

- `tinygrad/codegen/opt/postrange.py`
- `tinygrad/schedule/rangeify.py`
- `tinygrad/schedule/wmma.py`
- `extra/qk/prefill_graph_gemm_medium_stage_gate.py`
- `extra/qk/prefill_graph_gemm_route_bound_stage_gate.py`
- `extra/qk/prefill_graph_gemm_fp16_stage_gate.py`
- relevant tests under `test/unit/`

Task:

Make the cooperative/source-B stage own the medium source-B operand shape instead of skipping it when the source carries non-lane `GLOBAL`/`UPCAST`/`UNROLL` ranges outside warp+reduce.

Required implementation direction:

- Start from the current skip payload in `bench/prefill-graph-gemm-medium-stage/latest.json`.
- Identify which ranges are tile-loop carriers and which ranges are fragment identity.
- Preserve tile loops as enclosing loops; stage only the cooperative 16x16 fragment identity.
- Do not reintroduce the previous invalid late vector local-store shape.
- If a full fix is too large, add a centered diagnostic that proves the exact legal transformed UOp shape needed.

Gates:

```sh
PYTHONPATH=. python3 -m pytest test/unit/test_prefill_graph_gemm_medium_stage_gate.py test/unit/test_prefill_graph_gemm_route_bound_stage_gate.py -q
timeout 720 bash -lc 'PYTHONPATH=. python3 -m extra.qk.prefill_graph_gemm_medium_stage_gate --run-amd --pin-clock --compact'
timeout 300 bash -lc 'PYTHONPATH=. python3 -m extra.qk.prefill_graph_gemm_route_bound_stage_gate --run-amd --local-stage a --compact'
```

Done:

- Medium gate shows the cooperative case rewrites at least one source-B candidate.
- Emitted source has fp16 WMMA, shared local, and barrier for the cooperative path.
- Numeric status is `ok`.
- No raw `Ops.INS`.
- Remaining blocker, if any, is performance only, not legality/skipped rewrite.

Spark prompt:

```text
You are Worker A CODEGEN in /home/ubuntu/tinygrad-arkey. Own the 8B medium source-B codegen legality path. Do not edit policy docs except to report evidence. Reuse existing gates. Make the route-bound cooperative/source-B stage own the medium B source shape instead of skipping `GLOBAL/UPCAST/UNROLL` ranges outside warp+reduce. Keep tile loops outside the staged fragment. Run focused tests and bounded AMD gates. Return changed paths, commands, and whether the medium coop rewrite emits WMMA/shared/barrier without raw Ops.INS.
```

### Worker B - 8B Scheduler: Cooperative Partition Performance

Owner area: `scheduler`

Rows:

- `prefill_performance_target_1_coop_partition`
- conditional `prefill_performance_target_1_optional_double_buffer`

Primary files:

- `extra/qk/cooperative_stage_lanemap.py`
- `extra/qk/prefill_graph_gemm_coop_partition_gate.py`
- `extra/qk/prefill_graph_gemm_coop_route_contract_gate.py`
- `extra/qk/prefill_graph_gemm_medium_stage_gate.py`
- `tinygrad/codegen/opt/postrange.py` only if coordination with Worker A is required

Task:

Turn the custom cooperative B-tile lane map into a route-bound medium performance win.

Required implementation direction:

- Use the existing lane map: lanes `0..15` stage the unique 16x16 B tile into 256 halfs; all lanes read through `lane&15`.
- Keep the route-bound path generated.
- Once Worker A clears source-B ownership, tune tile movement and fragment readback for speed.
- Only explore double buffering if route-bound cooperative staging works but leaves more than 20% residual.

Gates:

```sh
PYTHONPATH=. python3 -m pytest test/unit/test_prefill_graph_gemm_coop_route_contract_gate.py test/unit/test_prefill_graph_gemm_medium_stage_gate.py -q
timeout 720 bash -lc 'PYTHONPATH=. python3 -m extra.qk.prefill_graph_gemm_medium_stage_gate --run-amd --pin-clock --compact'
PYTHONPATH=. python3 -m extra.qk.prefill_graph_gemm_coop_route_contract_gate --compact
```

Done:

- `medium_gate_route_bound_coop_partition_executes == true`.
- `route_bound_coop_partition_beats_baseline == true`.
- `PREFILL_GRAPH_GEMM_COOP_ROUTE_CONTRACT_PASS`.
- No raw `Ops.INS`.

Spark prompt:

```text
You are Worker B SCHEDULER in /home/ubuntu/tinygrad-arkey. Own route-bound cooperative B-tile movement and performance. Reuse the existing coop partition probe and medium/contract gates. Do not build a separate benchmark. Make the route-bound cooperative case beat baseline by at least the contract threshold, or produce a precise centered blocker after Worker A's source-B ownership state. Return changed paths, commands, and artifact deltas.
```

### Worker C - 8B Whole-Prefill Authority

Owner area: `policy`

Rows:

- `prefill_performance_target_1_fp16_recovery`
- `prefill_performance_target_1_baseline`

Primary files:

- `extra/qk/prefill_v2_schedule_table_gate.py`
- `extra/qk/prefill_whole_synced.py`
- `docs/8b-prefill-generated-route-closed-20260705.md`
- `docs/prefill-performance-lowering-scope-20260706.md`
- registry/report files only for current evidence

Task:

Keep 8B baseline and whole-prefill authority evidence current while Workers A/B change the generated substrate.

Required implementation direction:

- Refresh current strict-pure baseline with bounded runs.
- Refresh generated route run after A/B changes.
- Keep raw graph-GEMM numbers as historical comparison only.
- Make route attribution explicit in artifacts.

Gates/commands:

```sh
PYTHONPATH=. python3 -m extra.qk.prefill_v2_schedule_table_gate --compact
timeout 720 bash -lc 'PYTHONPATH=. python3 -m extra.qk.prefill_v2_schedule_table_gate --run-amd --pin-clock --compact'
timeout 1200 bash -lc 'DEVICE_IN_FUNCTION_BUG=1 ALLOW_DEVICE_USAGE=1 PYTHONPATH=. python3 extra/qk/prefill_whole_synced.py --mode smoke --whole-lengths 512'
```

Done:

- Current 8B baseline artifact is reproducible.
- Whole-prefill smoke shows whether A/B changes improve, regress, or do not move pp512.
- Registry/doc state reflects measured status.

Spark prompt:

```text
You are Worker C 8B POLICY/AUTHORITY in /home/ubuntu/tinygrad-arkey. Own baseline and whole-prefill evidence only. Do not modify core codegen. Run bounded authority/smoke gates, update artifacts/docs/registry if evidence changes, and keep route attribution explicit. Return exact commands, numbers, and changed files.
```

### Worker D - 14B Scheduler: Generated Full-Role Tile Loop

Owner area: `scheduler`

Rows:

- `prefill_performance_target_2_synthetic_shape`

Primary files:

- `extra/qk/q4k_wmma_tile_lowering.py`
- `extra/qk/q4k_wmma_tiled_role_shape_exec_gate.py`
- `extra/qk/q4k_wmma_tiled_lifecycle_gate.py`
- `extra/qk/prefill_int8_wmma_spec.py`
- `tinygrad/llm/prefill_routes.py`
- `tinygrad/llm/generated_candidates.py`
- `extra/qk/route_manifest.py`
- relevant tests under `test/unit/`

Task:

Replace `scheduler_owned_tile_loop_missing` with a bounded generated tile loop for Q4_K/Q8_1 role-shape execution.

Required implementation direction:

- Reuse `describe_int8_wmma_tile_lowering` and the existing lifecycle emitter.
- Own loops over `tile_m`, `tile_n`, and Q4_K groups without materializing full `[groups, M, N]` RAW tensors.
- Keep live RAW at tile scope, currently 256 elems.
- Preserve role-shape bounds and artifact fields: `raw_tile_steps`, `live_raw_elems`, `forbidden_full_raw_elems`.
- Do not fallback to direct-packed/VALU.
- Do not use raw hand kernels.

Gates:

```sh
PYTHONPATH=. python3 -m pytest test/unit/test_q4k_wmma_tiled_gates.py test/unit/test_q4k_wmma_tile_lowering.py -q
timeout 300 bash -lc 'PYTHONPATH=. python3 -m extra.qk.q4k_wmma_tiled_role_shape_exec_gate --compact'
PYTHONPATH=. python3 -m extra.qk.q4k_wmma_full_role_contract_gate --compact
```

Done:

- Role-shape exec gate attempts and executes generated tiled loop work without `scheduler_owned_tile_loop_missing`.
- Artifact shows bounded graph/kernel counts.
- WMMA surface remains selected.
- No direct-packed fallback.

Spark prompt:

```text
You are Worker D 14B SCHEDULER in /home/ubuntu/tinygrad-arkey. Own Q4_K/Q8_1 tiled role-shape execution. Reuse existing tile contract/lifecycle helpers. Replace `scheduler_owned_tile_loop_missing` with a bounded generated tile loop over tile_m/tile_n/group that keeps RAW tile-local and avoids graph explosion. Do not fallback to direct-packed. Run q4k focused tests and role-shape gate. Return changed paths, artifacts, and remaining blocker if any.
```

### Worker E - 14B Model Authority and Baseline

Owner area: `policy`

Rows:

- `prefill_performance_target_2_packed_mmq_recovery`
- `prefill_performance_target_2_baseline`
- `prefill_performance_target_2_model_authority`

Primary files:

- `tinygrad/llm/prefill_routes.py`
- `tinygrad/llm/generated_candidates.py`
- `extra/qk/route_manifest.py`
- `extra/qk/q4k_prefill_route_spec.py`
- `extra/qk/q6k_prefill_route_spec.py`
- `docs/prefill-14b-llama-parity-trace-20260704.md`
- `docs/prefill-performance-lowering-scope-20260706.md`

Task:

Create model-authority evidence that the generated MMQ path is selectable and measurable for relevant 14B roles.

Required implementation direction:

- Do not promote default routes.
- Add/extend gates that prove route selection uses the generated MMQ candidate for target role shapes.
- Keep direct-packed fallback explicit and classified.
- Record baseline vs generated path timing once Worker D can execute synthetic/full role paths.

Gates:

```sh
PYTHONPATH=. python3 -m extra.qk.prefill_performance_lowering_report --target target_2 --compact
PYTHONPATH=. python3 -m extra.qk.q4k_wmma_full_role_contract_gate --compact
timeout 300 bash -lc 'PYTHONPATH=. python3 -m extra.qk.q4k_wmma_tiled_role_shape_exec_gate --compact'
```

Done:

- Model-authority artifact names which roles select generated MMQ.
- Direct-packed fallback is absent for covered Q4_K/Q8_1 roles or explicitly classified for uncovered roles.
- Baseline gap is measured enough to decide whether performance work remains before promotion.

Spark prompt:

```text
You are Worker E 14B POLICY/AUTHORITY in /home/ubuntu/tinygrad-arkey. Own target_2 route selection, model-authority evidence, and baseline classification. Do not promote defaults. Reuse route manifest and generated candidate surfaces. Build or update central gates that prove generated MMQ route selection for representative 14B Q4_K/Q8_1 roles once Worker D execution exists. Return changed paths, commands, and route-attribution evidence.
```

### Worker F - Q6 Residual Decision

Owner area: `policy`

Rows:

- `prefill_performance_target_2_q6_residual_decision`

Primary files:

- `extra/qk/q6k_prefill_route_spec.py`
- `tinygrad/llm/prefill_routes.py`
- `tinygrad/llm/generated_candidates.py`
- `docs/prefill-14b-llama-parity-trace-20260704.md`
- `docs/prefill-performance-lowering-scope-20260706.md`

Task:

Determine whether Q6_K needs a generated MMQ path before promotion or can remain direct-packed with an explicit residual budget.

Required implementation direction:

- Measure or classify Q6_K wall-share after Q4_K/Q8_1 generated path exists.
- If Q6 is small residual, document keep-direct-packed policy and route attribution.
- If Q6 remains material, scope a Q6 generated MMQ worker as a follow-up.

Gates:

```sh
PYTHONPATH=. python3 -m extra.qk.prefill_performance_lowering_report --target target_2 --compact
```

Done:

- Q6 residual row has an evidence artifact.
- Policy says one of: keep direct-packed, implement Q6 MMQ now, or defer with threshold and reason.

Spark prompt:

```text
You are Worker F Q6 POLICY in /home/ubuntu/tinygrad-arkey. Own only the Q6 residual decision. Do not implement Q4_K codegen. Measure/classify Q6_K residual after generated Q4_K/Q8_1 route evidence exists. Update central docs/registry with a clear keep/implement/defer policy. Return changed paths and evidence.
```

### Worker G - Harness Consistency and Review Packet

Owner area: `policy`

Rows:

- all non-promotion rows, read-only except docs/report/tests

Primary files:

- `extra/qk/prefill_performance_lowering_registry.py`
- `extra/qk/prefill_performance_lowering_report.py`
- `test/unit/test_prefill_performance_lowering.py`
- `docs/prefill-performance-lowering-scope-20260706.md`
- this doc

Task:

Keep orchestration centralized and produce review packets from worker outputs.

Required implementation direction:

- No core codegen/scheduler changes.
- Add consistency checks only when they prevent real drift.
- Keep rows derived from registry; do not hard-code row order in tests.
- After every worker integration, print the orchestration JSON and update docs/registry if blockers changed.

Gates:

```sh
PYTHONPATH=. python3 -m pytest test/unit/test_prefill_performance_lowering.py -q
PYTHONPATH=. python3 -m extra.qk.prefill_performance_lowering_report --orchestration --compact
```

Done:

- One review packet lists:
  - cleared blockers,
  - remaining non-promotion blockers,
  - evidence artifacts,
  - exact commands to reproduce,
  - files changed per worker.

Spark prompt:

```text
You are Worker G HARNESS/REVIEW in /home/ubuntu/tinygrad-arkey. Own only registry/report/docs/tests for orchestration. Keep the central report accurate after worker outputs. Do not modify implementation code. Add minimal consistency tests if needed. Produce a review packet for humans: cleared blockers, remaining non-promotion blockers, artifacts, commands, and changed files.
```

## Integration Order

Parallel launch:

1. Worker A: 8B source-B codegen ownership.
2. Worker D: 14B scheduler-owned tile loop.
3. Worker C: 8B baseline/authority evidence.
4. Worker G: harness/review stays live as a sidecar.

After A reaches a legal medium coop rewrite:

5. Worker B: 8B coop performance.

After D reaches executable synthetic role-shape loop:

6. Worker E: 14B model authority.
7. Worker F: Q6 residual decision.

Final review:

8. Strong review pass over all integrated work.
9. Run focused gate suite.
10. Update scope/registry/report.
11. Stop before promotion rows.

## Focused Final Gate Suite

Run this before declaring 100% minus promotion:

```sh
PYTHONPATH=. python3 -m pytest \
  test/unit/test_prefill_performance_lowering.py \
  test/unit/test_prefill_graph_gemm_medium_stage_gate.py \
  test/unit/test_prefill_graph_gemm_route_bound_stage_gate.py \
  test/unit/test_prefill_graph_gemm_coop_route_contract_gate.py \
  test/unit/test_q4k_wmma_tiled_gates.py \
  test/unit/test_q4k_wmma_tile_lowering.py \
  -q

PYTHONPATH=. python3 -m extra.qk.prefill_performance_lowering_report --orchestration --compact
PYTHONPATH=. python3 -m extra.qk.prefill_graph_gemm_coop_route_contract_gate --compact
PYTHONPATH=. python3 -m extra.qk.q4k_wmma_full_role_contract_gate --compact
timeout 300 bash -lc 'PYTHONPATH=. python3 -m extra.qk.q4k_wmma_tiled_role_shape_exec_gate --compact'
```

GPU authority runs:

```sh
timeout 720 bash -lc 'PYTHONPATH=. python3 -m extra.qk.prefill_graph_gemm_medium_stage_gate --run-amd --pin-clock --compact'
timeout 720 bash -lc 'PYTHONPATH=. python3 -m extra.qk.prefill_v2_schedule_table_gate --run-amd --pin-clock --compact'
timeout 1200 bash -lc 'DEVICE_IN_FUNCTION_BUG=1 ALLOW_DEVICE_USAGE=1 PYTHONPATH=. python3 extra/qk/prefill_whole_synced.py --mode smoke --whole-lengths 512'
```

## Stop Conditions

Stop and ask for review if:

- A worker needs to change the same core file as another active worker in incompatible ways.
- A patch introduces raw `Ops.INS`, handwritten ISA, or direct-packed fallback as the claimed generated solution.
- A gate runs longer than its timeout twice with no narrower diagnostic.
- A worker changes route authority/promotion state.
- The next step is only broad search without a measurable gate target.

## Human Review Surface

Humans should only need to review:

- Whether the generated route is clean enough for future promotion.
- Whether the measured performance improvement is sufficient.
- Whether any remaining Q6 residual is acceptable.
- Whether to start promotion rows.

Everything else should be Spark-owned implementation, gate refresh, and registry/report synchronization.
