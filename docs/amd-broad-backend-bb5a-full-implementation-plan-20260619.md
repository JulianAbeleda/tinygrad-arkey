# AMD Broad Backend BB-5a Full Implementation Plan

Date: 2026-06-19

Parent:

- `docs/amd-broad-backend-roadmap-result-20260619.md`
- `docs/amd-broad-backend-bb5a-renderer-allocator-scope-20260619.md`
- `docs/amd-broad-backend-bb5a1-pipeline-ir-scope-20260619.md`

Artifact generator:

- `extra/qk_amd_bb5a_full_plan.py`

Generated artifact:

- `bench/amd-broad-backend-roadmap/bb5a_full_plan.json`

## Verdict

`BB5A_FULL_PLAN_READY_BB5A2_NEXT`.

BB-5a.1 now passes as a read-only pipeline IR surface. The remaining work should be executed from this full plan rather
than re-scoped one phase at a time. The plan keeps q8 transfer blocked until the shared backend passes prefill gates.

## Current State

Completed:

- BB-0 broad backend accepted.
- BB-1 oracle suite exists.
- BB-2 schedule metadata IR passes.
- BB-3 semantic wait/scheduler probe passes as probe-only.
- BB-4 resource accounting probe passes as accounting-only.
- BB-5 formally blocks on renderer/allocator integration.
- BB-5a scopes missing renderer/allocator implementation.
- BB-5a.1 passes a read-only two-stage pipeline IR surface.

Current aggregate:

- `BROAD_BACKEND_ACCEPTED_BB5A1_PASS_BB5A2_NEXT`

Next implementation target:

- BB-5a.2 double-buffered LDS lowering.

## Non-Negotiable Rules

- No q8 transfer until BB-5a and reopened BB-5 pass.
- No default behavior change until correctness, performance, graph, and policy gates pass.
- No static ISA patch counted as backend progress.
- No hand assembly hidden behind a normal tinygrad kernel.
- No TFLOPS claim from metadata-only work.
- No model gate before primitive movement exists.

## Phase Plan

### BB-5a.2 Double-Buffered LDS Lowering

Goal: lower the BB-5a.1 two-stage metadata into real distinct LDS slots.

Inputs:

- `AMDPipelineStageMeta`
- `bb5a1_pipeline_ir_result.json`
- existing AMD UOp/metadata classification
- prior failed hand-UOp double-buffer evidence

Likely files:

- `tinygrad/renderer/amd/schedule.py`
- `tinygrad/codegen/opt/postrange.py`
- `tinygrad/codegen/late/linearizer.py`
- AMD renderer/autogen integration points used for final instruction emission
- `extra/qk_amd_bb5a2_double_buffer_lds_probe.py`

Required implementation:

- map `lds_slot=0/1` metadata to two logical LDS regions or offsets;
- preserve producer/consumer stage identity through lowering;
- prevent local buffer cleanup from collapsing the two slots;
- dump stage-to-LDS mapping before render and after lowering.

Pass artifact:

- `bench/amd-broad-backend-roadmap/bb5a2_double_buffer_lds_result.json`

Minimum pass:

- two LDS stages visible in metadata/lowering;
- generated structure is non-byte-identical to serialized baseline where ISA is available;
- default behavior unchanged;
- no TFLOPS claim required.

Kill:

- if two slots cannot survive lowering;
- if output is byte-identical to the serialized single-buffer path;
- if the only working path is hand assembly.

### BB-5a.3 Semantic Wait Scheduler Integration

Goal: integrate dependency-aware waits into the AMD render/lowering path.

Inputs:

- BB-3 scheduler action planner;
- BB-5a.1 dependency groups;
- BB-5a.2 LDS stage mapping.

Likely files:

- `tinygrad/renderer/amd/schedule.py`
- AMD renderer instruction emission path
- `extra/qk_amd_bb5a3_wait_scheduler_integration_probe.py`

Required implementation:

- place `s_waitcnt vmcnt` at consuming points;
- distinguish global, LDS, scalar, barrier, and WMMA dependencies;
- emit `s_clause` and `s_delay_alu` from semantic reasons;
- preserve barriers needed for correctness.

Pass artifact:

- `bench/amd-broad-backend-roadmap/bb5a3_wait_scheduler_integration_result.json`

Minimum pass:

- wait/scheduler actions attach to lowered WMMA prefill-shaped instruction stream;
- action dump records instruction index and reason;
- default behavior unchanged unless explicitly enabled by a probe flag;
- correctness class remains no dataflow change.

Kill:

- if waits remain probe-only;
- if placement is static text insertion;
- if correctness requires conservative wait-after-every-load serialization.

### BB-5a.4 Allocator And Live-Range Control

Goal: move from resource accounting to real candidate acceptance/rejection.

Inputs:

- BB-4 resource accounting;
- BB-5a.1 resource budgets;
- BB-5a.2/5a.3 staged lowering and wait plans.

Likely files:

- `tinygrad/renderer/amd/schedule.py`
- `tinygrad/renderer/isa/__init__.py`
- AMD register allocation/pre-regalloc hook points
- `extra/qk_amd_bb5a4_allocator_resource_probe.py`

Required implementation:

- track accumulator, prefetch, pointer, and LDS-index live ranges;
- estimate VGPR/SGPR/LDS pressure before final render;
- reject or transform high-pressure candidates deterministically;
- record spill risk and occupancy estimate.

Pass artifact:

- `bench/amd-broad-backend-roadmap/bb5a4_allocator_resource_result.json`

Minimum pass:

- candidate metadata includes VGPR/SGPR/LDS/spill-risk/occupancy;
- a known unsafe high-pressure candidate is rejected with a reason, or a safe candidate remains within budget;
- default behavior unchanged.

Kill:

- if allocation remains accounting-only;
- if candidates silently spill;
- if occupancy/resource policy cannot explain selection or rejection.

### BB-5a.5 Resource Policy

Goal: choose when the pipelined path is enabled.

Inputs:

- BB-5a.2 lowering result;
- BB-5a.3 scheduler result;
- BB-5a.4 resource result;
- authority prefill shapes.

Likely files:

- `tinygrad/renderer/amd/schedule.py`
- a small policy helper under AMD renderer or codegen opt
- `extra/qk_amd_bb5a5_resource_policy_probe.py`

Required implementation:

- select/reject by shape, stage count, LDS bytes, VGPR/SGPR pressure, estimated occupancy, and correctness coverage;
- keep unsupported cases on existing tinygrad path;
- emit deterministic policy explanation.

Pass artifact:

- `bench/amd-broad-backend-roadmap/bb5a5_resource_policy_result.json`

Minimum pass:

- policy selects a supported WMMA prefill-shaped candidate or rejects it with concrete resource reasons;
- unsupported shapes fall back;
- default behavior unchanged.

Kill:

- if policy is a hardcoded shape-only switch;
- if rejection reasons are not tied to measured or computed resources;
- if default behavior changes before correctness/performance gates.

### BB-5a.6 Correctness Harness

Goal: prove the pipelined candidate is numerically valid.

Inputs:

- BB-5a.2 lowering;
- BB-5a.3 wait scheduling;
- BB-5a.4 resource control;
- BB-5a.5 policy.

Likely files:

- `extra/qk_amd_bb5a6_correctness_probe.py`
- possible focused unit tests if the repo has matching local test patterns

Required implementation:

- small deterministic WMMA matmul comparison;
- authority prefill matmul comparison;
- graph/TinyJit replay smoke after primitive correctness passes;
- capture code hash, launch contract, metadata, and tolerance.

Pass artifact:

- `bench/amd-broad-backend-roadmap/bb5a6_correctness_result.json`

Minimum pass:

- small WMMA correctness passes;
- one authority prefill matmul correctness passes;
- no graph replay instability on the smoke row if graph route is attempted;
- default behavior unchanged.

Kill:

- if correctness only passes by falling back to serialized lowering;
- if graph replay invalidates stage buffers or metadata;
- if tolerances exceed existing WMMA/prefill policy.

### BB-5a.7 Performance Gate / BB-5 Reopen

Goal: rerun BB-5 with real implementation evidence.

Inputs:

- BB-5a.2 through BB-5a.6 pass artifacts;
- authority prefill oracle suite;
- controlled clock/DPM methodology.

Likely files:

- `extra/qk_amd_software_pipeline_probe.py`
- `extra/qk_amd_bb5a7_performance_gate.py` if a separate runner is cleaner

Required implementation:

- measure pure tinygrad authority prefill matmul;
- prove generated ISA is real pipelined structure, not byte-identical serialized path;
- record correctness, launch, code hash, resource metadata, and timing authority.

Pass artifact:

- `bench/amd-broad-backend-roadmap/bb5a7_performance_gate_result.json`
- updated `bench/amd-broad-backend-roadmap/software_pipeline_result.json`

Minimum pass:

- pure tinygrad `>=60 TFLOPS` on `ffn_gate/up` or `ffn_down`;
- no Tensile or handwritten fallback;
- correctness passes;
- default behavior remains unchanged until model/policy acceptance.

Formal block:

- acceptable only if a real renderer/allocator implementation exists and the gate still fails for measured reasons.

Kill:

- if performance relies on external code objects;
- if ISA structure is still serialized;
- if the measured win disappears under controlled clock methodology.

### BB-6 Q8 Transfer Handoff

Goal: apply the shared backend capability to q8 only after prefill proves the shared machinery exists.

Start condition:

- BB-5a.7 passes or formally blocks with a reusable implemented scheduler/resource capability.

Required q8 gates:

- native q8 consumer `<=75us` to continue;
- `<=60us` strong pass;
- W==D decode improves `>=3%` before any policy promotion;
- dNLL `<=0.01` for any lossy path;
- default remains unchanged until accepted.

Kill:

- if shared backend improves prefill but q8 remains above `75us`;
- if q8 requires a q8-only native scheduler patch;
- if model-level W==D does not move.

## Artifact Dependency Graph

Required order:

1. `bb5a2_double_buffer_lds_result.json`
2. `bb5a3_wait_scheduler_integration_result.json`
3. `bb5a4_allocator_resource_result.json`
4. `bb5a5_resource_policy_result.json`
5. `bb5a6_correctness_result.json`
6. `bb5a7_performance_gate_result.json`
7. reopened `software_pipeline_result.json`
8. only then BB-6 q8 transfer artifacts

Parallel-safe work:

- BB-5a.3 planner integration can be prototyped in parallel with BB-5a.2 only as a probe, but it cannot pass before
  real LDS stage mapping exists.
- BB-5a.4 resource accounting extensions can be prototyped in parallel, but allocator/resource pass cannot pass before
  staged lowering produces a candidate.
- BB-5a.6 harness scaffolding can be created early, but correctness pass cannot pass before the candidate exists.

## Roadmap Completion Gate

BB-5a is complete when:

- BB-5a.2 through BB-5a.6 pass;
- BB-5a.7 either reaches `>=60 TFLOPS` pure tinygrad or records a formal implemented-backend block;
- all artifacts include code hash/source pointer, correctness class, resource metadata, default-behavior status, and
  next action.

Until then:

- BB-6 remains blocked;
- q8 native transfer remains disallowed;
- current default tinygrad behavior remains unchanged.
