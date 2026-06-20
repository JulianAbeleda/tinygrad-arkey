# AMD Broad Backend BB-5a.1 Pipeline IR Scope

Date: 2026-06-19

Parent:

- `docs/amd-broad-backend-bb5a-renderer-allocator-scope-20260619.md`

Artifact generator:

- `extra/qk_amd_bb5a1_pipeline_ir_scope.py`

Generated artifact:

- `bench/amd-broad-backend-roadmap/bb5a1_pipeline_ir_scope.json`

## Verdict

`BB5A1_SCOPE_COMPLETE_IMPLEMENTATION_NOT_READY`.

BB-5a.1 is the first implementation slice under BB-5a. Its job is not to make prefill fast yet. Its job is to create a
tinygrad-native software-pipeline IR contract that survives the normal lowering path and can later drive double-buffered
LDS lowering, wait scheduling, allocator policy, and performance gates.

## Problem

Current tinygrad has `Ops.STAGE`, `BufferizeOpts`, local bufferization, `AFTER`, barriers, and tensor-core WMMA lowering.
Those are not enough to express a K-loop software pipeline. The missing contract is stage semantics:

- which operation belongs to prologue, steady state, or epilogue;
- which K iteration is produced and which K iteration is consumed;
- which global load, LDS store, LDS load, and WMMA consume belong to the same dependency group;
- which LDS buffer slot is current versus next;
- what wait distance is intended;
- whether the metadata survived rangeify, postrange opt, linearize, and AMD schedule metadata dumping.

Without this contract, the renderer is free to serialize the graph or collapse the double-buffer attempt back into
byte-identical ISA, which is exactly what BB-5 observed.

## Required IR Contract

The BB-5a.1 IR surface must represent these fields:

| field | purpose |
|---|---|
| `pipeline_id` | groups all operations belonging to one software-pipelined K loop |
| `phase` | one of `prologue`, `steady`, `epilogue` |
| `stage_id` | logical stage number within the pipeline |
| `stage_count` | number of active software-pipeline stages, initially `2` |
| `producer_distance` | distance between the consumed tile and produced tile, initially `1` |
| `k_axis` | range/axis identity for the pipelined K loop |
| `buffer_role` | one of `global_load`, `lds_store`, `lds_load`, `wmma_consume`, `barrier`, `wait` |
| `lds_slot` | logical LDS buffer slot, initially `0` or `1` |
| `dependency_group` | group used by future wait/barrier scheduling |
| `semantic_order` | coarse order within a stage without becoming handwritten ISA |
| `resource_budget` | optional pre-allocator budget for prefetch registers, LDS bytes, accumulators |

The contract can be carried in one of three ways:

- a new dataclass in `tinygrad/renderer/amd/schedule.py` attached through UOp tags/metadata;
- a structured `Ops.STAGE.arg` extension if it remains compatible with existing `BufferizeOpts`;
- new opt vocabulary plus a lowering pass that materializes the metadata.

The first implementation should prefer a sidecar metadata extractor over changing core UOp semantics. Core semantics
should change only once the probe proves the contract is useful and stable.

## Acceptable First Pass

BB-5a.1 passes when all of these are true:

- a WMMA prefill-shaped synthetic pipeline can be described with two stages;
- metadata dump contains at least one `global_load`, one `lds_store`, one `lds_load`, and one `wmma_consume` row;
- the dump contains both `lds_slot=0` and `lds_slot=1`;
- at least one `steady` row has `producer_distance=1`;
- the metadata can be derived from UOps or structured stage records without changing emitted code;
- the artifact declares `default_behavior_changed=false`;
- the pass does not claim any TFLOPS movement.

This is an IR/readiness pass only. It does not satisfy BB-5a.2 or reopen BB-5.

## Non-Goals

- No q8 transfer.
- No AMD renderer instruction reordering.
- No allocator change.
- No default-on scheduling policy.
- No handwritten assembly or static disassembly patch.
- No performance claim from labels alone.

## Implementation Plan

### BB-5a.1a Stage Schema

Add a small schema for pipeline stage metadata.

Candidate location:

- `tinygrad/renderer/amd/schedule.py`

Required objects:

- `AMDPipelineStageMeta`
- `pipeline_stage_dump`
- `pipeline_stage_summary`

The schema should be serializable to JSON and independent of AMD instruction objects.

### BB-5a.1b UOp/Stage Extraction

Add a read-only extractor that can derive pipeline rows from a structured synthetic prefill-shaped UOp or stage record.

Candidate location:

- `tinygrad/renderer/amd/schedule.py`

The extractor must not alter UOps. It should only produce metadata.

### BB-5a.1c Probe

Add a probe that builds the minimum two-stage K-loop description and emits:

- `bench/amd-broad-backend-roadmap/bb5a1_pipeline_ir_result.json`

Required checks:

- stage count is `2`;
- phases include `prologue` and `steady`;
- roles include global load, LDS store, LDS load, and WMMA consume;
- LDS slots include `0` and `1`;
- dependency groups are present;
- default behavior is unchanged.

### BB-5a.1d Roadmap Integration

Update the roadmap aggregator so the next step moves from BB-5a.1 scope to BB-5a.1 implementation only after this
scope exists, and later to BB-5a.2 only after the IR result passes.

## Kill Conditions

Kill or redesign BB-5a.1 if:

- the metadata can only be attached by per-kernel handwritten annotations;
- stage rows cannot be connected to UOps/ranges at all;
- the contract cannot represent both producer and consumer sides of a two-stage K loop;
- the contract mutates default codegen before correctness/performance gates exist;
- the probe passes without proving two distinct LDS slots and a producer distance.

## Downstream Rule

BB-5a.2 may start only after BB-5a.1 has a passing IR result. BB-5a.2 must then lower the two-stage contract into real
double-buffered LDS structure and prove non-byte-identical ISA. Until then, BB-6 q8 transfer remains blocked.
