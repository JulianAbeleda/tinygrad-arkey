# AMD Broad Backend Roadmap Scope

Date: 2026-06-19

Purpose: define the roadmap-sized AMD backend project implied by the decode and prefill closeouts. This is the scope
for a reusable backend/compiler investment, not the start of another bounded q8 decode patch.

## Current Decision

Verdict: `ROADMAP_SCOPE_ONLY`.

Do not start a q8-specific native scheduler/renderer implementation. The decode-native pass list closed that path:

- q8 `ffn_gate/up` role body evidence exists;
- N2 candidate count is `0`;
- max isolated timing-grade movement is `14.087us`, below the `>=30us` q8 feature gate;
- PMC/SQTT are not currently counter/timeline authority from saved q8 artifacts;
- no native W==D projection is justified.

The only valid native path left is a broad AMD backend project that teaches tinygrad to emit schedule/resource behavior
currently supplied by hipcc/LLD and Tensile. That project must be accepted as backend work up front.

## Authority Inputs

Decode authority:

- q8 handwritten hipcc/LLD artifact route is research-only/default-off and improves W==D by about `1.05-1.06x`;
- q8 artifact dNLL is `+0.002887`, acceptable for a research flag but not default policy;
- native AMD DSL q8 consumer is correct but slow: `166.649us`;
- hipcc/LLD q8 oracle is about `93.54us`;
- no bounded native feature explains the `~73us` gap.

Prefill authority:

- pure tinygrad WMMA prefill rests around `42-50 TFLOPS` under controlled clock;
- Tensile/oracle prefill reaches about `66-69 TFLOPS`;
- manual UOp double-buffer prefetch compiles and is correct but produces byte-identical ISA;
- software pipelining is therefore not expressible at the current UOp/renderer level;
- closing the gap requires renderer scheduling, double-buffered LDS lowering, deferred waits, and spill-free resource
  control.

Cross-primitive exhaustion:

- bounded q8 primitive work is closed;
- bounded prefill WMMA knob search is closed;
- remaining rows are project-level: wait scheduling, `s_clause` / `s_delay_alu`, register/live-range control,
  occupancy/resource policy, LDS staging, and software-pipelined K-loops.

## Roadmap Acceptance Gate

Start this project only if one of these is true:

1. `BROAD_BACKEND_ACCEPTED`: the project explicitly accepts a multi-week reusable AMD backend/compiler investment
   without pretending it is a q8 patch.
2. `TOOLING_READY_FOR_FEATURE`: future attribution names a feature clearing `>=30us` on q8 decode or `>=15 TFLOPS` on
   prefill.

If neither condition is true, the correct state is:

- shipped pure-tinygrad decode and prefill baselines;
- q8 artifact retained as default-off research;
- Tensile/prefill artifact retained as research/policy evidence;
- native scheduler/resource transfer documented as roadmap only.

## Non-Goals

- No q8-only scheduler patch.
- No default-on external artifact route.
- No manual `s_clause`, `s_delay_alu`, or `s_waitcnt` insertion from static diff alone.
- No reopened load-shape, waitcnt-grouping, reduction-topology, BEAM, or WMMA knob sweep as standalone work.
- No hand-maintained assembly hidden behind the backend.
- No success claim from a standalone microbench unless the owning primitive/model gate moves.
- No default behavior change before model, quality, and fallback gates pass.

## Required Backend Capabilities

The broad project must produce reusable mechanisms, not per-kernel text edits.

Minimum capabilities:

- schedule metadata IR for latency class, memory space, vector width, dependency group, barrier scope, live-range
  boundary, preferred issue order, prefetch stage, LDS stage, and register-pressure budget;
- latency-aware instruction scheduling for VALU/SALU/global/LDS/WMMA instructions;
- semantic placement for `s_waitcnt`, `s_clause`, and `s_delay_alu`;
- register allocation and live-range control sufficient to avoid known accumulator/prefetch spill cliffs;
- software-pipelined global -> LDS -> register K-loop lowering;
- double-buffered LDS allocation and alternating stage semantics;
- occupancy/resource policy tied to VGPR/SGPR/LDS pressure and launch shape;
- graph-safe integration so scheduled kernels survive TinyJit replay and buffer rebinding;
- attribution/diagnostic output that preserves program name, code hash, launch contract, role, disassembly, resource
  metadata, and timing authority.

## Tracks

### Track A - Governance and Oracle Suite

Goal: keep the project grounded in real authority cases.

Deliverables:

- q8 decode consumer oracle: native tinygrad, hipcc/LLD artifact, W==D, dNLL, disassembly, resource metadata;
- prefill oracle: pure tinygrad WMMA, Tensile schedule, controlled-clock throughput, correctness, disassembly;
- smoke oracle: small deterministic AMD kernel for scheduler/regression checks;
- one artifact ledger with commands, env, hashes, and pass/fail thresholds.

Gate:

- all oracle rows have correctness, timing, disassembly, code hash, launch contract, resource metadata, and fallback
  policy.

### Track T - Attribution Tooling

Goal: preserve hardware visibility for future decisions while accepting that decode N2 is not currently gated on it.

Deliverables:

- PMC decode status and counter schema;
- SQTT/ATT timeline status and body attribution status;
- primitive role timing joins;
- feature attribution matrix with explicit authority level;
- blocker taxonomy for counter/timeline gaps.

Gate:

- tooling never upgrades visibility-only packet counts into timing authority;
- any future `TOOLING_READY_FOR_FEATURE` claim names the primitive, role, code hash, feature, movement, and authority.

### Track IR - Schedule Metadata IR

Goal: represent the schedule facts the renderer currently discards.

Deliverables:

- schedule metadata attached to UOps or lowered AMD instructions;
- dependency-group model for waits/barriers;
- live-range and stage boundaries;
- resource budgets visible before final rendering;
- dumps that compare metadata against emitted ISA.

Gate:

- the same metadata path can describe one q8-shaped probe and one WMMA GEMM probe without semantic changes.

Kill:

- if the IR devolves into per-kernel handwritten annotations, stop the project or reclassify it as hand assembly.

### Track S - Wait and Instruction Scheduler

Goal: emit intended dependency schedules rather than conservative renderer order.

Deliverables:

- semantic `s_waitcnt` placement;
- controlled `s_clause` / `s_delay_alu` insertion;
- latency-aware instruction ordering;
- regression tests proving ISA movement where expected.

Gate:

- intended ISA changes appear in both a q8-shaped probe and a WMMA-shaped probe;
- correctness is unchanged;
- either q8 moves by `>=15us` as a component of a larger path or prefill contributes measurable TFLOPS movement.

Kill:

- if this repeats the known `0.837us` waitcnt-only result and does not enable later software pipelining.

### Track R - Register and Resource Control

Goal: keep scheduled kernels from collapsing under register pressure, spills, or occupancy loss.

Deliverables:

- live-range splitting/extension controls;
- accumulator and prefetch register budgeting;
- VGPR/SGPR/LDS pressure reports;
- occupancy policy that is connected to launch shape and measured movement.

Gate:

- a WMMA more-accumulator or prefetch probe avoids the known spill cliff;
- a q8 probe changes VGPR/occupancy/resource metadata in a controlled way;
- no broad AMD regression is introduced.

### Track P - Software-Pipelined Prefill WMMA

Goal: generate the Tensile-class double-buffered K-loop in pure tinygrad.

Required behavior:

- prologue loads tile `k+1`;
- steady state overlaps global load/LDS store for the next tile with WMMA on the current tile;
- two LDS buffers prevent aliasing between current reads and next writes;
- `vmcnt` waits are deferred until the prefetched data is consumed;
- barriers are placed for correctness, not after every global load.

Gate:

- one prefill `ffn_gate/up` or `ffn_down` kernel reaches `>=60 TFLOPS` without external artifacts;
- disassembly proves non-byte-identical schedule versus the current serialized kernel;
- correctness passes and graph/model integration remains available.

### Track Q - Q8/MMVQ Scheduler Transfer

Goal: apply the reusable scheduler/resource capability to the q8 decode consumer only after the shared backend pieces
exist.

Deliverables:

- q8 native consumer with controlled instruction order, waits, resource metadata, and role-local timing;
- W==D decode run;
- quality/dNLL run if any lossy route changes;
- fallback to existing default path.

Gate:

- continue if native q8 consumer reaches `<=75us`;
- strong pass if native q8 consumer reaches `<=60us`;
- W==D decode must improve by `>=3%` with dNLL `<=0.01` for any lossy path;
- default remains unchanged until policy accepts the route.

Kill:

- if the shared backend improves prefill but q8 remains above `75us`, close q8 native transfer and keep q8 artifact as
  research only.

### Track M - Model and Graph Integration

Goal: ensure backend wins survive the real tinygrad execution path.

Deliverables:

- TinyJit replay support;
- graph-safe buffer rebinding;
- fallback path for unsupported shapes/devices;
- W==D decode and pp prefill harness rows;
- dNLL/quality rows for lossy decode paths;
- default-off flags until policy gates pass.

Gate:

- model-level timing moves in the expected direction;
- graph replay is stable;
- correctness and quality gates pass;
- default behavior remains unchanged until explicitly accepted.

## Phases

### BB-0 - Project Acceptance

Output:

- explicit decision: `BROAD_BACKEND_ACCEPTED` or `ROADMAP_ONLY`.

Pass:

- project owners accept the scope as backend/compiler work and accept that early phases may not move q8 immediately.

### BB-1 - Oracle Suite

Output:

- `bench/amd-broad-backend-roadmap/oracle_suite.json`.

Pass:

- q8 decode, prefill WMMA/Tensile, and smoke rows are reproducible with correctness, timing, disassembly, code hashes,
  launch contracts, and resource metadata.

### BB-2 - Schedule Metadata IR

Output:

- `bench/amd-broad-backend-roadmap/schedule_metadata_ir_result.json`.

Pass:

- q8-shaped and WMMA-shaped probes carry schedule metadata through lowering.

### BB-3 - Wait/Scheduler Emitter

Output:

- `bench/amd-broad-backend-roadmap/wait_scheduler_result.json`.

Pass:

- emitted ISA changes in intended locations with correctness preserved.

### BB-4 - Register/Resource Control

Output:

- `bench/amd-broad-backend-roadmap/register_resource_result.json`.

Pass:

- resource controls change VGPR/SGPR/LDS/occupancy predictably and avoid a known spill cliff.

### BB-5 - Software-Pipelined Prefill

Output:

- `bench/amd-broad-backend-roadmap/software_pipeline_result.json`.

Pass:

- pure tinygrad prefill reaches `>=60 TFLOPS` on one authority matmul and disassembly shows the double-buffered
  software pipeline.

### BB-6 - Q8 Transfer

Output:

- `bench/amd-broad-backend-roadmap/q8_transfer_result.json`.

Pass:

- q8 native consumer reaches `<=75us` to continue or `<=60us` as a strong pass;
- W==D moves `>=3%` if promoted beyond a local probe.

### BB-7 - Model Gate

Output:

- `bench/amd-broad-backend-roadmap/model_gate_result.json`.

Pass:

- at least two authority cases move, or one authority case moves strongly and the other has a documented technical
  reason for not transferring.

### BB-8 - Maintainability Decision

Output:

- final result doc and upstreamability assessment.

Pass:

- implementation is a reusable backend capability with tests and no default regressions.

Kill:

- implementation is only maintainable as one-off q8 or prefill assembly.

## Artifact Schema

Expected artifact directory:

- `bench/amd-broad-backend-roadmap/authority.json`
- `bench/amd-broad-backend-roadmap/oracle_suite.json`
- `bench/amd-broad-backend-roadmap/schedule_metadata_ir_result.json`
- `bench/amd-broad-backend-roadmap/wait_scheduler_result.json`
- `bench/amd-broad-backend-roadmap/register_resource_result.json`
- `bench/amd-broad-backend-roadmap/software_pipeline_result.json`
- `bench/amd-broad-backend-roadmap/q8_transfer_result.json`
- `bench/amd-broad-backend-roadmap/model_gate_result.json`

Minimum common fields:

- `verdict`
- `date`
- `command`
- `env`
- `device`
- `role`
- `program_name`
- `code_hash`
- `shape`
- `launch_global`
- `launch_local`
- `correctness`
- `timing`
- `disassembly_path`
- `resource_metadata`
- `authority_level`
- `fallback_policy`

## Stop Conditions

Stop or return to roadmap-only if any of these hold:

- schedule metadata is only useful as per-kernel annotations;
- wait/scheduler emitter produces ISA churn but no primitive movement;
- register/resource control requires a full allocator rewrite before any probe movement is visible;
- software-pipelined prefill cannot produce non-byte-identical ISA;
- q8 remains above `75us` after shared scheduler/resource capability exists;
- wins do not survive TinyJit/model integration;
- the implementation changes default behavior before policy and fallback gates pass.

## Roadmap Summary

The broad AMD backend project is justified only as reusable infrastructure for at least the decode q8/MMVQ and prefill
WMMA/Tensile gaps. It should start with explicit `BROAD_BACKEND_ACCEPTED`, not by smuggling the work through an N2 q8
implementation. The first real build target is schedule/resource capability; q8 native transfer is a downstream
consumer, not the project definition.
