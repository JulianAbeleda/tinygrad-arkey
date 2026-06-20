# AMD Broad Backend BB-5a.2 Real LDS Lowering Solution Scope

Date: 2026-06-19

Parent:

- `docs/amd-broad-backend-roadmap-result-20260619.md`
- `docs/amd-broad-backend-bb5a-full-implementation-plan-20260619.md`

Artifact generator:

- `extra/qk_amd_bb5a2_solution_scope.py`

Generated artifact:

- `bench/amd-broad-backend-roadmap/bb5a2_solution_scope.json`

## Verdict

`BB5A2_SOLUTION_SCOPED_REAL_LOWERING_REQUIRED`.

The solution is a real lowering path, not more labels. BB-5a.1 proved a two-stage pipeline metadata contract. BB-5a.2
must now make that contract survive tinygrad lowering as two distinct LDS regions and, eventually, non-byte-identical
AMD ISA.

## Root Cause

Current tinygrad has three relevant mechanisms:

- `Ops.STAGE` / `BufferizeOpts` creates local buffering during rangeify;
- `Ops.DEFINE_LOCAL` represents local memory before final codegen;
- AMD ELF scans `Ops.DEFINE_LOCAL` to set `group_segment_fixed_size`.

But there is no path that consumes:

- `AMDPipelineStageMeta`
- `lds_slot=0/1`
- `producer_distance=1`
- pipeline role/dependency-group metadata

and turns it into:

- two durable local slots;
- stable local-memory indexes;
- two visible LDS address regions in lowered UOps/IR/ISA;
- non-byte-identical generated code versus the serialized single-buffer path.

The prior hand-UOp double-buffer attempt already allocated `(2, BLOCK, K)` shaped locals, but current linearization and
rendering still collapsed the intended overlap into byte-identical ISA. Therefore BB-5a.2 must add a compiler-visible
stage contract and a lowering hook, not only larger local allocation.

## Solution Shape

Implement a gated AMD pipeline lowering path with three layers.

### Layer 1 - Stage-to-LDS Plan

Add a read-only planner that converts `AMDPipelineStageMeta` rows into an LDS allocation plan.

Candidate location:

- `tinygrad/renderer/amd/schedule.py`

Required objects:

- `AMDLDSStagePlan`
- `lds_stage_plan_from_pipeline`
- `lds_stage_plan_dump`

Required fields:

| field | purpose |
|---|---|
| `pipeline_id` | joins plan to pipeline metadata |
| `stage_count` | initially `2` |
| `slots` | logical slots `[0, 1]` |
| `slot_roles` | producer/consumer role per slot |
| `slot_offsets` | deterministic byte or element offsets |
| `dependency_groups` | groups that future wait/barrier scheduler consumes |
| `required_local_bytes` | LDS allocation required by the plan |
| `alias_safe` | current and next tiles do not alias |
| `lowering_status` | `planned`, `lowered`, or `blocked` |

Layer 1 pass is still metadata-only, but it must be precise enough for Layer 2 to lower.

### Layer 2 - Postrange/Rangeify Lowering Hook

Add a gated lowering hook that materializes two local slots in UOps.

Candidate locations:

- `tinygrad/codegen/opt/postrange.py` if the lowering is opt-driven;
- `tinygrad/schedule/rangeify.py` if it must attach to `Ops.STAGE` / `BufferizeOpts`;
- a new small helper imported by those modules if keeping AMD-specific logic isolated is cleaner.

Required behavior:

- only runs under an explicit probe flag or candidate metadata;
- maps `lds_slot=0/1` to distinct `Ops.DEFINE_LOCAL` regions or distinct non-foldable offsets in one local region;
- preserves the slot identity through `pm_add_buffers_local`, `rangeify_codegen`, `pm_remove_bufferize`, and final
  `DEFINE_LOCAL` generation;
- emits a dump mapping stage rows to local buffer IDs/offsets.

Non-negotiable:

- no default behavior change;
- no q8 route;
- no handwritten AMD assembly.

### Layer 3 - Render/ISA Evidence

Prove that the lowered candidate reaches AMD render/assembly as distinct LDS usage.

Candidate locations:

- `tinygrad/renderer/amd/elf.py` for LDS-size/resource confirmation;
- AMD LLVM/source/ISA dump path for text/byte comparison;
- probe script for generated code hash and diff.

Required evidence:

- generated source or ISA contains two distinct LDS regions or offsets;
- the code hash differs from serialized single-buffer baseline;
- if machine code is available, instruction bytes are non-byte-identical;
- `group_segment_fixed_size` / LDS metadata matches the two-slot plan.

## Required Probe

Add:

- `extra/qk_amd_bb5a2_real_lds_lowering_probe.py`

Emit:

- `bench/amd-broad-backend-roadmap/bb5a2_real_lds_lowering_result.json`

The probe must build or inspect a WMMA prefill-shaped candidate and report:

| check | pass condition |
|---|---|
| input pipeline IR | `PASS_PIPELINE_IR_SURFACE` |
| LDS stage plan | two slots, alias-safe, required bytes recorded |
| UOp lowering | two durable local slots or offsets visible after lowering |
| renderer consumption | AMD render path sees the two-slot structure |
| code diff | source/hash/ISA differs from serialized baseline |
| defaults | unchanged |
| performance claim | false |

## Implementation Order

1. Add `AMDLDSStagePlan` and plan dump helpers.
2. Add the BB-5a.2 probe in metadata-only mode; it should fail on missing lowering.
3. Add the gated lowering hook.
4. Extend the probe to inspect lowered UOps/source/ISA.
5. Update `extra/qk_amd_bb5a_execute_plan.py` to consume the new pass artifact.
6. Only if BB-5a.2 passes, unblock BB-5a.3/BB-5a.4.

## Pass

BB-5a.2 passes only when:

- the two-stage pipeline IR maps to a concrete LDS stage plan;
- lowered UOps preserve two local slots or offsets;
- AMD render/assembly evidence is non-byte-identical to the serialized baseline;
- no default behavior changes.

## Kill

Kill or redesign the path if:

- slot identity can only be preserved by handwritten assembly;
- rangeify/local-buffer cleanup always collapses the slots;
- non-byte-identical code appears only from irrelevant naming/order churn;
- correctness would require serializing with a barrier/wait after every global load.

## Downstream

BB-5a.3 and BB-5a.4 remain blocked until BB-5a.2 passes. Q8 transfer remains disallowed.
