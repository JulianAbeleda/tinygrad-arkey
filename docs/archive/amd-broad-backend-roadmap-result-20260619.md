# AMD Broad Backend Roadmap Result

Date: 2026-06-19

Scope:

- `docs/amd-broad-backend-roadmap-scope-20260619.md`

Artifact generator:

- `extra/qk_amd_broad_backend_roadmap.py`

Generated artifacts:

- `bench/amd-broad-backend-roadmap/authority.json`
- `bench/amd-broad-backend-roadmap/oracle_suite.json`
- `bench/amd-broad-backend-roadmap/schedule_metadata_ir_result.json`
- `bench/amd-broad-backend-roadmap/wait_scheduler_result.json`
- `bench/amd-broad-backend-roadmap/register_resource_result.json`
- `bench/amd-broad-backend-roadmap/software_pipeline_result.json`
- `bench/amd-broad-backend-roadmap/bb5a_renderer_allocator_scope.json`
- `bench/amd-broad-backend-roadmap/bb5a1_pipeline_ir_scope.json`
- `bench/amd-broad-backend-roadmap/bb5a1_pipeline_ir_result.json`
- `bench/amd-broad-backend-roadmap/bb5a_full_plan.json`
- `bench/amd-broad-backend-roadmap/bb5a_execution_result.json`
- `bench/amd-broad-backend-roadmap/bb5a2_double_buffer_lds_result.json`
- `bench/amd-broad-backend-roadmap/bb5a2_solution_scope.json`
- `bench/amd-broad-backend-roadmap/bb5a2_lds_stage_plan_result.json`
- `bench/amd-broad-backend-roadmap/bb5a2_lowering_hook_result.json`
- `bench/amd-broad-backend-roadmap/bb5a2_render_isa_evidence_result.json`
- `bench/amd-broad-backend-roadmap/bb5a2_real_lowering_integration_result.json`
- `bench/amd-broad-backend-roadmap/bb5a2_pipelined_dataflow_result.json`
- `bench/amd-broad-backend-roadmap/bb5a3_wait_scheduler_integration_result.json`
- `bench/amd-broad-backend-roadmap/bb5a4_allocator_resource_result.json`
- `bench/amd-broad-backend-roadmap/bb5a5_resource_policy_result.json`
- `bench/amd-broad-backend-roadmap/bb5a6_correctness_result.json`
- `bench/amd-broad-backend-roadmap/bb5a7_performance_gate_result.json`
- `bench/amd-broad-backend-roadmap/bb5a8_tensile_mapping_result.json`
- `bench/amd-broad-backend-roadmap/bb5a8_authority_kernel_capture_result.json`
- `bench/amd-broad-backend-roadmap/bb5a9_causal_delta_package_result.json`
- `bench/amd-broad-backend-roadmap/bb5a10_tensile_layout_audit_result.json`
- `bench/amd-broad-backend-roadmap/bb5a10_implementation_plan_result.json`
- `bench/amd-broad-backend-roadmap/bb5a10_p1_layout_spec_result.json`
- `bench/amd-broad-backend-roadmap/bb5a10_p2_rendered_lds_result.json`
- `bench/amd-broad-backend-roadmap/bb5a10_p3_kloop_stage_result.json`
- `bench/amd-broad-backend-roadmap/bb5a10_p4_wait_barrier_result.json`
- `bench/amd-broad-backend-roadmap/bb5a10_p5_resource_policy_result.json`
- `bench/amd-broad-backend-roadmap/bb5a10_p6_structural_candidate_result.json`
- `bench/amd-broad-backend-roadmap/bb5a10_p7_correctness_scope_result.json`
- `bench/amd-broad-backend-roadmap/bb5a10_p7a_p7b_correctness_result.json`
- `bench/amd-broad-backend-roadmap/bb5a10_p7c_numeric_correctness_result.json`
- `bench/amd-broad-backend-roadmap/q8_transfer_result.json`
- `bench/amd-broad-backend-roadmap/model_gate_result.json`
- `bench/amd-broad-backend-roadmap/result.json`

## Verdict

`BROAD_BACKEND_ACCEPTED_BB5A10_P7C_SMALL_NUMERIC_DONE_P7D_NEXT_Q8_BLOCKED`.

BB-0 is accepted as a broad AMD backend/compiler project. BB-1 oracle suite is materialized from existing q8 decode,
prefill Tensile, clock-authority, and decode-native tooling artifacts. BB-2 schedule metadata IR, BB-3 semantic
wait-scheduler probe, and BB-4 resource accounting probe pass. BB-5 is formally blocked on real renderer/allocator
integration. BB-5a is scoped, BB-5a.1 passes as a read-only pipeline IR surface, the full remaining BB-5a
implementation plan is materialized and executed through the roadblock sequence. BB-5a.2 double-buffered LDS lowering
now passes through gated source/ELF evidence, BB-5a.3 wait scheduler integration passes, BB-5a.4 resource control
passes, BB-5a.5 policy passes, and BB-5a.6 correctness passes. BB-5a.7 blocks: current pure tinygrad authority
prefill is `42.0 TFLOPS`, below the `60.0 TFLOPS` gate. BB-5a.8 completes the static Tensile-to-tinygrad mapping and
then captures the timing-equivalent pure-tinygrad authority kernel as source/ELF/disassembly/resource evidence. The
captured kernel reaches `43.026 TFLOPS`, emits `64` `v_wmma` instructions, uses `0` LDS bytes, and emits `0`
`ds_load_b128`. BB-5a.9 completes the causal-delta package and converts the rest into parallel implementation tracks:
LDS layout, K-loop scheduling, and resource policy may proceed together. BB-5a.10 completes the focused Tensile layout
audit against the selected rocBLAS authority function. The selected function is isolated from `/tmp/td_all.txt`, has
`25088` LDS bytes, `256` VGPRs, scratch `0`, `ds_store_b64` LDS writes, `ds_load_b128` WMMA operand reads, `80`
`v_wmma`, `545` waits, and `6` barriers. The audit proves enough to implement a non-bitexact staged-LDS candidate,
but not enough to claim a bit-identical Tensile LDS layout clone. BB-5a.10 implementation planning is now complete:
P0 is done; P1-P5 should run as one coordinated batch over layout, renderer lowering, K-loop staging, waits/barriers,
and resource policy; P6/P7/P8 are structural/correctness/performance gates; P9 keeps q8 transfer blocked until P8
passes. P1 is now complete: the selected-layout spec defines two logical LDS regions, accepts `ds_store_b64` as the
selected authority store path, requires `ds_load_b128` feeding WMMA, and keeps bitexact Tensile layout out of scope.
P2-P6 now pass as a structural ISA/ELF candidate: nonzero LDS, selected-kernel-compatible LDS stores, `ds_load_b128`
feeding WMMA, staged order, semantic waits/barriers, and resource policy. This is not correctness or performance yet.
P7 executable correctness is scoped into P7a-P7e. P7a validates known-good LDS-WMMA hardware correctness; P7b wraps the
structural candidate with real kernargs, LDS allocation, lidx/gidx, and an output store; P7c/P7d prove small and
authority-shape numeric correctness; P7e packages the P8 handoff. P7a-P7c now pass: known-good LDS-WMMA RMSE is
`0.000209`, the structural candidate has an executable wrapper with output, and a selected-compatible
`ds_store_b64 -> ds_load_b128 -> WMMA` small tile is numerically correct with RMSE `0.00020901396055705845`. Q8
transfer remains blocked.

This does **not** reopen q8-specific N2 scheduler work. Do not start BB-6 q8 transfer until BB-5 has a real
software-pipelined prefill pass.

## BB-0 - Acceptance

Verdict: `BROAD_BACKEND_ACCEPTED`.

Boundary:

- this is reusable AMD backend/compiler work;
- q8 native transfer is a downstream consumer;
- default decode and prefill behavior remain unchanged;
- q8 artifact and Tensile artifact routes remain research/policy evidence unless separately accepted.

## BB-1 - Oracle Suite

Verdict: `PASS`.

Authority rows:

| row | baseline | oracle | gate |
|---|---:|---:|---|
| q8 decode gate/up consumer | `166.649us` native AMD DSL | `93.54us` hipcc/LLD | q8 transfer continues at `<=75us`, strong pass `<=60us` |
| prefill ffn_gate/up | `42.0 TFLOPS` tinygrad WMMA | `~65.6 TFLOPS` Tensile | pure tinygrad `>=60 TFLOPS` |
| prefill ffn_down | `42.0 TFLOPS` tinygrad WMMA | `~69.8 TFLOPS` Tensile | pure tinygrad `>=60 TFLOPS` |
| tooling smoke | visibility only | HCQ/PMU artifacts | no timing promotion from packet counts |

The oracle suite records correctness, timing, launch contract, resource metadata, authority level, source artifacts, and
fallback policy for each row.

## BB-2 - Schedule Metadata IR

Verdict: `PASS_SCHEDULE_METADATA_IR`.

Added:

- `tinygrad/renderer/amd/schedule.py`
- `extra/qk_amd_schedule_metadata_probe.py`

Result artifact:

- `bench/amd-broad-backend-roadmap/schedule_metadata_ir_result.json`

BB-2 is a read-only metadata layer. It does not change emitted code. It classifies existing AMD UOps and instruction
objects into schedule fields needed by later phases:

- latency class;
- memory space;
- vector width;
- wait group;
- barrier scope;
- live-range boundary;
- issue cluster;
- prefetch/LDS stage;
- register-pressure budget placeholder.

Gate result:

- q8-shaped instruction probe: `PASS`;
- WMMA-shaped lowered-UOp probe: `PASS`;
- metadata survives lowering/dumping: `PASS`;
- semantic change: `False`.

BB-3 wait scheduling and BB-4 register/resource control may now start. BB-5 software-pipelined prefill remains blocked
on BB-3/BB-4. BB-6 q8 transfer remains blocked until shared backend capability exists.

## BB-3 - Semantic Wait/Scheduler Probe

Verdict: `PASS_SEMANTIC_WAIT_SCHEDULER_PROBE`.

Added:

- scheduler action planning in `tinygrad/renderer/amd/schedule.py`;
- `extra/qk_amd_wait_resource_probe.py`.

Result artifact:

- `bench/amd-broad-backend-roadmap/wait_scheduler_result.json`

The probe uses BB-2 metadata to plan and emit scheduler-only instruction changes on q8-shaped and WMMA-shaped
instruction streams:

- `s_clause` before global-memory clauses;
- `s_waitcnt` before dependent consumers;
- `s_delay_alu` after VALU/WMMA issue points.

Gate result:

- q8-shaped instruction stream: `PASS`;
- WMMA-shaped instruction stream: `PASS`;
- instruction bytes changed: `PASS`;
- correctness preservation class: `scheduler_hints_only_no_dataflow_change`;
- default behavior changed: `False`.

This is not yet renderer-wide scheduling. It proves the semantic planner/emitter can produce intended AMD scheduler
instructions from metadata on bounded probes.

## BB-4 - Register/Resource Accounting Probe

Verdict: `PASS_RESOURCE_ACCOUNTING_PROBE`.

Added:

- instruction register-span accounting in `tinygrad/renderer/amd/schedule.py`;
- BB-4 output in `extra/qk_amd_wait_resource_probe.py`.

Result artifact:

- `bench/amd-broad-backend-roadmap/register_resource_result.json`

The probe reports VGPR/SGPR spans, register operand counts, instruction counts, and accounting stability before/after
BB-3 scheduler hints for q8-shaped and WMMA-shaped instruction streams.

Gate result:

- q8-shaped resource accounting: `PASS`;
- WMMA-shaped resource accounting: `PASS`;
- scheduler hints do not change register accounting: `PASS`;
- default behavior changed: `False`.

This is accounting only. It is not a spill-free allocator or occupancy optimizer yet.

## BB-5 - Software-Pipelined Prefill Probe

Verdict: `BLOCKED_REAL_RENDERER_ALLOCATOR_INTEGRATION`.

Added:

- `extra/qk_amd_software_pipeline_probe.py`

Result artifact:

- `bench/amd-broad-backend-roadmap/software_pipeline_result.json`

BB-5 consumes the Tensile codegen oracle, BB-3 scheduler probe, BB-4 resource accounting, and the prior pure-tinygrad
software-pipeline attempt.

Gate result:

| item | result |
|---|---|
| current tinygrad WMMA baseline | `42.0 TFLOPS` |
| prior hand-UOp double-buffer attempt | `47.2 TFLOPS`, no improvement over `48.5 TFLOPS` base |
| required pure tinygrad gate | `>=60 TFLOPS` |
| Tensile oracle | `~65.6 TFLOPS` ffn_gate/up |
| BB-3 scheduler hints | probe-level only |
| BB-4 resource accounting | accounting only, no allocator control |
| `OptOps.PREFETCH/PIPELINE/DOUBLE_BUFFER` | missing |

Decision:

- BB-5 does not pass the TFLOPS gate.
- The formal BB-5 block is accepted because the probe identifies the exact missing layer: true K-loop software-pipeline
  lowering plus spill-free allocator/resource control.
- Stop before BB-6. Q8 transfer remains blocked until BB-5 is reopened as real renderer/allocator integration and then
  reaches the prefill gate.

## BB-5a - Renderer/Allocator Scope

Verdict: `BB5A_SCOPE_COMPLETE_IMPLEMENTATION_NOT_READY`.

Added:

- `docs/amd-broad-backend-bb5a-renderer-allocator-scope-20260619.md`
- `extra/qk_amd_bb5a_renderer_allocator_scope.py`

Result artifact:

- `bench/amd-broad-backend-roadmap/bb5a_renderer_allocator_scope.json`

BB-5a converts the BB-5 block into implementation-ready scope. It checks the current repo surfaces and records the
missing capability rows needed before BB-5 can be reopened:

| row | current state | minimum pass |
|---|---|---|
| pipeline IR surface | `missing` | stage IDs survive lowering and metadata dumping |
| double-buffered LDS lowering | `missing_renderer_integration` | two LDS stages and non-byte-identical ISA |
| semantic wait scheduler integration | `probe_level_only` | wait movement integrated into AMD rendering |
| allocator/live-range control | `accounting_only` | spill-free candidate or deterministic resource rejection |
| resource policy | `missing_control_policy` | shape/resource select-reject reasons before render |
| correctness harness | `missing_for_real_renderer_path` | small WMMA and authority prefill correctness pass |
| performance gate | `blocked_below_gate` | pure tinygrad authority prefill `>=60 TFLOPS` |

The new next step is BB-5a.1: implement a pipeline IR surface. Its first minimum pass is not TFLOPS; it is proving that
stage-aware pipeline metadata survives tinygrad lowering and dumping for a WMMA prefill-shaped kernel. Q8 transfer
remains blocked.

## BB-5a.1 - Pipeline IR

Verdict: `PASS_PIPELINE_IR_SURFACE`.

Added:

- `docs/amd-broad-backend-bb5a1-pipeline-ir-scope-20260619.md`
- `extra/qk_amd_bb5a1_pipeline_ir_scope.py`
- `extra/qk_amd_bb5a1_pipeline_ir_probe.py`

Result artifacts:

- `bench/amd-broad-backend-roadmap/bb5a1_pipeline_ir_scope.json`
- `bench/amd-broad-backend-roadmap/bb5a1_pipeline_ir_result.json`

BB-5a.1 defines the durable software-pipeline metadata contract needed before double-buffered LDS lowering can start.
It explicitly does not claim renderer movement or TFLOPS.

Implemented in `tinygrad/renderer/amd/schedule.py`:

- `AMDPipelineStageMeta`
- `pipeline_stage_metadata_from_records`
- `pipeline_stage_summary`
- `pipeline_stage_dump`

The schema fields are:

- `pipeline_id`
- `phase`
- `stage_id`
- `stage_count`
- `producer_distance`
- `k_axis`
- `buffer_role`
- `lds_slot`
- `dependency_group`
- `semantic_order`
- `resource_budget`

The scoped work packages were:

| row | target | minimum pass |
|---|---|---|
| BB-5a.1a | stage schema | serializable `AMDPipelineStageMeta` plus summary/dump helpers |
| BB-5a.1b | stage extraction | read-only extraction produces prologue/steady rows with `producer_distance=1` |
| BB-5a.1c | probe | proves roles, phases, LDS slots, dependency groups, and unchanged defaults |
| BB-5a.1d | roadmap integration | aggregate names BB-5a.1 implementation/result state explicitly |

Gate result:

| check | result |
|---|---|
| stage count is `2` | pass |
| phases include `prologue` and `steady` | pass |
| roles include `global_load`, `lds_store`, `lds_load`, `wmma_consume` | pass |
| LDS slots include `0` and `1` | pass |
| dependency groups present | pass |
| steady producer distance is `1` | pass |
| semantic order monotonic | pass |
| default behavior changed | `False` |
| performance claim | `False` |

The current next step is BB-5a.2: lower two LDS stages from this metadata into non-byte-identical AMD ISA without
changing defaults.

## BB-5a Full Implementation Plan

Verdict: `BB5A_FULL_PLAN_READY_BB5A2_NEXT`.

Added:

- `docs/amd-broad-backend-bb5a-full-implementation-plan-20260619.md`
- `extra/qk_amd_bb5a_full_plan.py`

Result artifact:

- `bench/amd-broad-backend-roadmap/bb5a_full_plan.json`

The remaining work is now planned as one dependency chain instead of one-off scope steps:

| phase | status | minimum pass |
|---|---|---|
| BB-5a.2 double-buffered LDS lowering | `READY` | two LDS stages lower into non-byte-identical AMD ISA without changing defaults |
| BB-5a.3 semantic wait scheduler integration | `BLOCKED_ON_BB5A2` | dependency-aware waits attach to the lowered stream with reasons |
| BB-5a.4 allocator/live-range control | `BLOCKED_ON_BB5A2` | spill-free or deterministic resource rejection with VGPR/SGPR/LDS/occupancy metadata |
| BB-5a.5 resource policy | `BLOCKED_ON_BB5A3_BB5A4` | select/reject pipelined candidate with resource reasons |
| BB-5a.6 correctness harness | `BLOCKED_ON_BB5A5` | small WMMA and authority prefill correctness pass |
| BB-5a.7 performance gate / BB-5 reopen | `BLOCKED_ON_BB5A6` | pure tinygrad authority prefill `>=60 TFLOPS` with real pipelined ISA |
| BB-6 q8 transfer handoff | `BLOCKED_ON_BB5A7` | native q8 `<=75us` continue, `<=60us` strong pass |

This plan keeps BB-6 blocked until BB-5a.7 passes or formally blocks with an implemented reusable backend capability.

## BB-5a Execution

Verdict: `BB5A_EXECUTION_BLOCKED_BB5A7_PERFORMANCE_GATE`.

Added:

- `extra/qk_amd_bb5a_execute_plan.py`

Result artifacts:

- `bench/amd-broad-backend-roadmap/bb5a_execution_result.json`
- `bench/amd-broad-backend-roadmap/bb5a2_double_buffer_lds_result.json`
- `bench/amd-broad-backend-roadmap/bb5a3_wait_scheduler_integration_result.json`
- `bench/amd-broad-backend-roadmap/bb5a4_allocator_resource_result.json`
- `bench/amd-broad-backend-roadmap/bb5a5_resource_policy_result.json`
- `bench/amd-broad-backend-roadmap/bb5a6_correctness_result.json`
- `bench/amd-broad-backend-roadmap/bb5a7_performance_gate_result.json`

BB-5a execution ran the full phase chain. BB-5a.2 blocks:

| check | result |
|---|---|
| pipeline IR pass | pass |
| metadata has LDS slots `0` and `1` | pass |
| LDS stage plan pass | pass |
| `DEFINE_LOCAL` lowering hook pass | pass |
| AMD ELF LDS descriptor evidence | pass |
| render source integration | pass |
| pipelined LDS/WMMA source skeleton | pass |
| non-byte-identical source evidence | pass |
| default behavior changed | `False` |
| performance claim | `False` |

Downstream phases now execute until the performance gate:

| phase | status |
|---|---|
| BB-5a.3 | `PASS_BB5A3_WAIT_SCHEDULER_INTEGRATION` |
| BB-5a.4 | `PASS_BB5A4_ALLOCATOR_RESOURCE_CONTROL` |
| BB-5a.5 | `PASS_BB5A5_RESOURCE_POLICY` |
| BB-5a.6 | `PASS_BB5A6_CORRECTNESS` |
| BB-5a.7 | `BLOCKED_BB5A7_PERFORMANCE_GATE_NOT_MET` |
| BB-6 | `BLOCKED_ON_BB5A7_PERFORMANCE_GATE` |

The remaining block is not BB-5a.2. The active block is BB-5a.7 performance: pure tinygrad authority prefill must reach
`>=60 TFLOPS`; the current authority row is `42.0 TFLOPS`.

## BB-5a.2 Solution Scope

Verdict: `BB5A2_SOLUTION_SCOPED_REAL_LOWERING_REQUIRED`.

Added:

- `docs/amd-broad-backend-bb5a2-real-lds-lowering-solution-scope-20260619.md`
- `extra/qk_amd_bb5a2_solution_scope.py`

Result artifact:

- `bench/amd-broad-backend-roadmap/bb5a2_solution_scope.json`

The solution has three layers:

| layer | target | minimum pass |
|---|---|---|
| Layer 1 stage-to-LDS plan | `tinygrad/renderer/amd/schedule.py` | `AMDLDSStagePlan` maps slots `0/1` to deterministic alias-safe LDS slots with required bytes |
| Layer 2 postrange/rangeify lowering | `tinygrad/codegen/opt/postrange.py`, `tinygrad/schedule/rangeify.py` | lowered UOps preserve `lds_slot=0/1` through local-buffer cleanup |
| Layer 3 render/ISA evidence | AMD render/ELF/probe path | AMD render/assembly sees two-slot LDS structure and differs from serialized baseline |

Required probe:

- `extra/qk_amd_bb5a2_real_lds_lowering_probe.py`
- `bench/amd-broad-backend-roadmap/bb5a2_real_lds_lowering_result.json`

Layer 1, Layer 2, Layer 3, render-source integration, and the pipelined LDS/WMMA source skeleton now pass. BB-5a.2 is
complete for this gated roadmap sequence.

## BB-5a.2 Layer 1 - LDS Stage Plan

Verdict: `PASS_LDS_STAGE_PLAN`.

Added:

- `extra/qk_amd_bb5a2_lds_stage_plan_probe.py`

Implemented in `tinygrad/renderer/amd/schedule.py`:

- `AMDLDSStagePlan`
- `lds_stage_plan_from_pipeline`
- `lds_stage_plan_dump`

Result artifact:

- `bench/amd-broad-backend-roadmap/bb5a2_lds_stage_plan_result.json`

Gate result:

| check | result |
|---|---|
| input pipeline IR pass | pass |
| slot count is `2` | pass |
| slots `0` and `1` present | pass |
| alias-safe offsets | pass |
| required local bytes recorded | `8192` |
| dependency groups present | pass |
| lowering status | `planned` |
| default behavior changed | `False` |
| performance claim | `False` |

Layer 1 is complete. Layer 2 lowers this plan into durable local definitions.

## BB-5a.2 Layer 2 - DEFINE_LOCAL Lowering Hook

Verdict: `PASS_DEFINE_LOCAL_LOWERING_HOOK`.

Added:

- `extra/qk_amd_bb5a2_lowering_hook_probe.py`

Implemented in `tinygrad/renderer/amd/schedule.py`:

- `AMDLDSLoweredSlot`
- `lower_lds_stage_plan_to_define_locals`
- `lds_lowering_dump`

Result artifact:

- `bench/amd-broad-backend-roadmap/bb5a2_lowering_hook_result.json`

Gate result:

| check | result |
|---|---|
| input LDS stage plan pass | pass |
| lowered slot count is `2` | pass |
| `DEFINE_LOCAL` UOps emitted | pass |
| address space is LDS/local | pass |
| `DEFINE_LOCAL` slots distinct | `9000`, `9001` |
| planned slots preserved | `0`, `1` |
| planned offsets preserved | `0`, `4096` |
| lowered bytes match plan | `8192` |
| default behavior changed | `False` |
| performance claim | `False` |

This is still gated helper evidence only. It does not prove AMD renderer consumption, assembly/ELF LDS layout, or
non-byte-identical ISA.

## BB-5a.2 Layer 3 - Render/ELF Evidence

Verdict: `PASS_RENDER_ELF_LDS_EVIDENCE`.

Added:

- `extra/qk_amd_bb5a2_render_isa_evidence_probe.py`

Implemented in `tinygrad/renderer/amd/elf.py`:

- `kernel_descriptor_from_elf`
- `group_segment_fixed_size_from_elf`

Result artifact:

- `bench/amd-broad-backend-roadmap/bb5a2_render_isa_evidence_result.json`

Gate result:

| check | result |
|---|---|
| input lowering hook pass | pass |
| candidate has two `DEFINE_LOCAL` slots | pass |
| candidate ELF LDS bytes match plan | `8192` |
| serialized baseline ELF LDS bytes | `4096` |
| ELF hash non-byte-identical | pass |
| instruction stream identical | `True` |
| default behavior changed | `False` |
| performance claim | `False` |

This proves the AMD ELF packer sees the two-slot LDS allocation through `group_segment_fixed_size`. It does not yet
prove real pipelined source/ISA movement because the probe intentionally uses the same one-instruction stream for the
candidate and baseline.

## BB-5a.2 Completion, BB-5a.3-BB-5a.6, BB-5a.7 Block, And BB-5a.8 Mapping

Additional completed artifacts:

- `bench/amd-broad-backend-roadmap/bb5a2_real_lowering_integration_result.json`: `PASS_RENDER_SOURCE_LDS_INTEGRATION`
- `bench/amd-broad-backend-roadmap/bb5a2_pipelined_dataflow_result.json`: `PASS_PIPELINED_LDS_WMMA_SOURCE_SKELETON`
- `bench/amd-broad-backend-roadmap/bb5a3_wait_scheduler_integration_result.json`: `PASS_BB5A3_WAIT_SCHEDULER_INTEGRATION`
- `bench/amd-broad-backend-roadmap/bb5a4_allocator_resource_result.json`: `PASS_BB5A4_ALLOCATOR_RESOURCE_CONTROL`
- `bench/amd-broad-backend-roadmap/bb5a5_resource_policy_result.json`: `PASS_BB5A5_RESOURCE_POLICY`
- `bench/amd-broad-backend-roadmap/bb5a6_correctness_result.json`: `PASS_BB5A6_CORRECTNESS`
- `bench/amd-broad-backend-roadmap/bb5a7_performance_gate_result.json`: `BLOCKED_BB5A7_PERFORMANCE_GATE_NOT_MET`
- `bench/amd-broad-backend-roadmap/bb5a8_tensile_mapping_result.json`: `PASS_STATIC_TENSILE_TINYGRAD_MAPPING_CAUSAL_PROOF_BLOCKED`
- `bench/amd-broad-backend-roadmap/bb5a8_authority_kernel_capture_result.json`: `PASS_AUTHORITY_KERNEL_CAPTURE_CAUSAL_INPUTS_READY`
- `bench/amd-broad-backend-roadmap/bb5a9_causal_delta_package_result.json`: `PASS_BB5A9_CAUSAL_DELTA_PACKAGE_IMPLEMENTATION_TRACKS_READY`

Gate summary:

| phase | result |
|---|---|
| BB-5a.2 dataflow | two LDS stores, two LDS loads, one AMD WMMA source intrinsic |
| BB-5a.3 scheduler | `insert_s_clause`, `ensure_s_waitcnt`, `insert_s_delay_alu` attach to lowered LDS/WMMA stream |
| BB-5a.4 resources | VGPR span `32`, LDS bytes `8192`, candidate accepted by probe policy |
| BB-5a.5 policy | pipelined candidate selected with shape/resource reasons |
| BB-5a.6 correctness | small AMD WMMA rel_err `0.0002596`; authority prefill rel_err `0.000348` |
| BB-5a.7 performance | blocked: pure tinygrad `42.0 TFLOPS` vs `60.0 TFLOPS` gate |
| BB-5a.8 mapping | static mapping complete; causal proof blocked on actual timed tinygrad kernel source/ISA/resource capture |
| BB-5a.8 capture | timing-equivalent tinygrad authority kernel captured at `43.026 TFLOPS` with source/ELF/disassembly/resource evidence |
| BB-5a.9 package | causal deltas proven; parallel implementation tracks ready |

BB-5a.8 answers the tooling question directly:

| question | answer |
|---|---|
| Can we map Tensile against current tinygrad/BB-5a? | yes, at static source/ISA-skeleton level |
| Can we prove the causal claim yet? | no |
| Proven matches | macro tile `128x128x16`; WMMA fragment `16x16x16x1` |
| Proven gaps | Tensile timing `65.6 TFLOPS` vs tinygrad `42.0 TFLOPS`; Tensile `LRVW16/ds_load_b128` vs BB-5a skeleton `DS_LOAD_B32` |
| Structural-but-not-timed evidence | BB-5a two LDS slots, waits/clauses/delay, VGPR/LDS accounting |
| Missing proof | full source/ISA/resource artifact for the same pure-tinygrad authority kernel measured at `42.0 TFLOPS` |

This completes the mapping and capture scope. BB-5a.8 authority capture wrote:

- `bench/amd-broad-backend-roadmap/bb5a8_authority_kernel_capture/tinygrad_ffn_gate_up_authority.hip`
- `bench/amd-broad-backend-roadmap/bb5a8_authority_kernel_capture/tinygrad_ffn_gate_up_authority.hsaco`
- `bench/amd-broad-backend-roadmap/bb5a8_authority_kernel_capture/tinygrad_ffn_gate_up_authority.disasm`

Captured tinygrad authority facts:

| fact | value |
|---|---:|
| best timing | `43.026 TFLOPS` |
| reference tinygrad row | `42.0 TFLOPS` |
| reference Tensile row | `65.6 TFLOPS` |
| `v_wmma` | `64` |
| LDS bytes | `0` |
| `ds_load_b128` | `0` |
| `scratch_*` | `0` |
| launch global/local | `(192, 16, 1)` / `(32, 1, 1)` |

BB-5a.9 classifies the causal deltas and removes the false serial dependency between the remaining implementation
surfaces:

| track | status | next |
|---|---|---|
| A causal delta | complete | use P0 deltas as acceptance criteria |
| B LDS layout | ready | nonzero LDS in ELF and DS traffic in disasm |
| C K-loop scheduler | ready | prologue plus steady-state two-slot K-loop with semantic waits |
| D resource policy | ready | scratch/private/VGPR rejection before timing |
| E candidate gate | blocked | correctness plus pure tinygrad `>=60 TFLOPS` |
| F q8 transfer | blocked | only after BB-5 performance passes |

BB-5a.10 then audits the selected Tensile layout evidence and clears candidate-spec readiness. The next valid work is
BB-5a.10 implementation: build one non-bitexact staged authority-shape candidate with selected-kernel-compatible LDS
stores, `ds_load_b128` feeding WMMA, semantic waits/barriers, and scratch-free resource policy. BB-6 q8 transfer
remains disallowed until a real pure-tinygrad prefill candidate passes the `>=60 TFLOPS` gate.

## Phase Status

| phase | status |
|---|---|
| BB-0 | `BROAD_BACKEND_ACCEPTED` |
| BB-1 | `PASS` |
| BB-2 | `PASS_SCHEDULE_METADATA_IR` |
| BB-3 | `PASS_SEMANTIC_WAIT_SCHEDULER_PROBE` |
| BB-4 | `PASS_RESOURCE_ACCOUNTING_PROBE` |
| BB-5 | `BLOCKED_REAL_RENDERER_ALLOCATOR_INTEGRATION` |
| BB-5a | `BB5A_SCOPE_COMPLETE_IMPLEMENTATION_NOT_READY` |
| BB-5a.1 | `PASS_PIPELINE_IR_SURFACE` |
| BB-5a-plan | `BB5A_FULL_PLAN_READY_BB5A2_NEXT` |
| BB-5a-execution | `BB5A_EXECUTION_BLOCKED_BB5A7_PERFORMANCE_GATE` |
| BB-5a.2-solution | `BB5A2_SOLUTION_SCOPED_REAL_LOWERING_REQUIRED` |
| BB-5a.2-layer-1 | `PASS_LDS_STAGE_PLAN` |
| BB-5a.2-layer-2 | `PASS_DEFINE_LOCAL_LOWERING_HOOK` |
| BB-5a.2-layer-3 | `PASS_RENDER_ELF_LDS_EVIDENCE` |
| BB-5a.2-integration | `PASS_RENDER_SOURCE_LDS_INTEGRATION` |
| BB-5a.2-dataflow | `PASS_PIPELINED_LDS_WMMA_SOURCE_SKELETON` |
| BB-5a.3 | `PASS_BB5A3_WAIT_SCHEDULER_INTEGRATION` |
| BB-5a.4 | `PASS_BB5A4_ALLOCATOR_RESOURCE_CONTROL` |
| BB-5a.5 | `PASS_BB5A5_RESOURCE_POLICY` |
| BB-5a.6 | `PASS_BB5A6_CORRECTNESS` |
| BB-5a.7 | `BLOCKED_BB5A7_PERFORMANCE_GATE_NOT_MET` |
| BB-5a.8-mapping | `PASS_STATIC_TENSILE_TINYGRAD_MAPPING_CAUSAL_PROOF_BLOCKED` |
| BB-5a.8-capture | `PASS_AUTHORITY_KERNEL_CAPTURE_CAUSAL_INPUTS_READY` |
| BB-5a.9-causal-delta | `PASS_BB5A9_CAUSAL_DELTA_PACKAGE_IMPLEMENTATION_TRACKS_READY` |
| BB-5a.10-layout-audit | `PASS_TENSILE_LAYOUT_AUDIT_CANDIDATE_SPEC_READY_NOT_BITEXACT` |
| BB-5a.10-plan | `PASS_BB5A10_IMPLEMENTATION_PLAN_READY` |
| BB-5a.10-P1-layout-spec | `PASS_BB5A10_P1_LAYOUT_SPEC_READY` |
| BB-5a.10-P2-rendered-LDS | `PASS_BB5A10_P2_RENDERED_LDS_STORE_READ` |
| BB-5a.10-P3-kloop-stage | `PASS_BB5A10_P3_KLOOP_STAGE_SCHEDULER` |
| BB-5a.10-P4-wait-barrier | `PASS_BB5A10_P4_WAIT_BARRIER_SCHEDULE` |
| BB-5a.10-P5-resource-policy | `PASS_BB5A10_P5_RESOURCE_POLICY` |
| BB-5a.10-P6-structural-candidate | `PASS_BB5A10_P6_STRUCTURAL_CANDIDATE` |
| BB-5a.10-P7-correctness-scope | `PASS_BB5A10_P7_CORRECTNESS_SCOPE_READY` |
| BB-5a.10-P7a-P7b-correctness | `PASS_BB5A10_P7A_P7B_EXECUTABLE_WRAPPER` |
| BB-5a.10-P7c-numeric-correctness | `PASS_BB5A10_P7C_SMALL_NUMERIC_CORRECTNESS` |
| BB-6 | `BLOCKED_ON_BB5A_IMPLEMENTATION` |
| BB-7 | `BLOCKED_ON_PRIMITIVE_MOVEMENT` |

## Decision

Stop before BB-6. The next valid work is BB-5a.10 P7d: authority-shape correctness smoke. Performance work must
produce a measured pure tinygrad authority prefill candidate at `>=60 TFLOPS` before q8 transfer can start.

Disallowed still:

- q8-only native scheduler patch;
- manual static-diff `s_waitcnt` / `s_clause` / `s_delay_alu` edits;
- reopening load-shape, waitcnt grouping, reduction topology, or WMMA knob sweeps as standalone work;
- default behavior changes.
