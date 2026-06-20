# AMD Broad Backend BB-5a Renderer/Allocator Scope

Date: 2026-06-19

Parent:

- `docs/amd-broad-backend-roadmap-scope-20260619.md`
- `docs/amd-broad-backend-roadmap-result-20260619.md`

Artifact generator:

- `extra/qk_amd_bb5a_renderer_allocator_scope.py`

Generated artifact:

- `bench/amd-broad-backend-roadmap/bb5a_renderer_allocator_scope.json`

## Verdict

`BB5A_SCOPE_COMPLETE_IMPLEMENTATION_NOT_READY`.

BB-5 proved that the existing tinygrad AMD path cannot honestly claim software-pipelined prefill. BB-5a is the missing
implementation project: real K-loop pipeline lowering, real AMD renderer scheduling, and allocator/resource control
that can survive the normal tinygrad codegen path.

This is still not q8 transfer. Q8 remains blocked until this shared capability passes a prefill authority gate.

## Current Evidence

The current repo has useful planning tools but not enough execution tooling:

- `OptOps` exposes tensor-core, axis, local, group, padding, and swap transforms, but no pipeline-stage transform.
- `tinygrad/renderer/amd/schedule.py` can classify metadata and emit scheduler hints on bounded instruction streams.
- BB-3 scheduler insertion is probe-local; it is not wired into AMD codegen/rendering.
- BB-4 resource work is accounting-only; it does not allocate registers, split live ranges, or reject schedules.
- The prior pure-tinygrad double-buffer attempt compiled and was correct, but rendered byte-identical ISA and did not
  move the prefill gate.
- The authority target remains pure tinygrad prefill `>=60 TFLOPS` versus the current controlled `~42 TFLOPS` row and
  the Tensile `~65-70 TFLOPS` oracle.

## Required Work Packages

### BB-5a.1 Pipeline IR Surface

Define a tinygrad-native representation for software pipeline stages.

Required capability:

- express prologue, steady state, and epilogue stages;
- mark stage distance, for example load tile `k+1` while consuming tile `k`;
- represent global -> register -> LDS -> WMMA dependencies without hand assembly;
- carry stage IDs through lowering and metadata dumps.

Acceptable implementation surfaces:

- new `OptOps` entries such as `PREFETCH`, `PIPELINE`, and `DOUBLE_BUFFER`;
- a new internal UOp/stage annotation pass;
- a renderer-recognized structured loop pattern, if it is mechanically produced and testable.

Not acceptable:

- per-kernel handwritten annotations;
- a static ISA text patch;
- hidden hand assembly behind a normal tinygrad kernel.

### BB-5a.2 Double-Buffered LDS Lowering

Lower a two-stage LDS buffer in the AMD renderer without collapsing it back into a serialized single-buffer sequence.

Required capability:

- allocate two logically distinct LDS regions or stage offsets;
- alternate producer and consumer stages;
- prove current-stage LDS reads do not alias next-stage LDS writes;
- preserve required barriers while avoiding a barrier after every global load when dependency semantics allow overlap.

Pass evidence:

- generated ISA is not byte-identical to the current serialized kernel;
- metadata shows at least two LDS stages;
- disassembly shows alternating global-load/LDS-store/WMMA groups across the K loop.

### BB-5a.3 Semantic Wait Scheduler Integration

Move BB-3 from a bounded probe into the AMD rendering path.

Required capability:

- place `s_waitcnt vmcnt` at real consuming points rather than immediately after every global load;
- distinguish global, LDS, scalar, and WMMA dependency groups;
- emit `s_clause` and `s_delay_alu` only from semantic scheduling decisions;
- preserve correctness across barriers, stores, and WMMA accumulation.

Pass evidence:

- wait placement changes for a lowered WMMA prefill kernel;
- the scheduler can dump its planned actions with instruction indices and reasons;
- correctness is unchanged on a deterministic small WMMA matmul.

### BB-5a.4 Allocator And Live-Range Control

Turn BB-4 accounting into a control surface.

Required capability:

- track accumulator, prefetch, pointer, and LDS-index live ranges;
- keep large accumulator kernels spill-free;
- cap or reject schedules that exceed VGPR/SGPR/LDS budgets;
- expose occupancy estimates tied to launch shape and register/LDS pressure;
- preserve graph-safe ABI and kernarg layout.

Pass evidence:

- generated metadata records VGPR, SGPR, LDS, spill count, and estimated occupancy;
- at least one known high-pressure WMMA/prefetch candidate is rejected or transformed with a reason;
- no VGPR/SGPR spill appears in the authority prefill candidate.

### BB-5a.5 Resource Policy

Choose when to enable the software pipeline.

Required capability:

- compare serialized and pipelined candidates;
- reject unprofitable shapes;
- keep default behavior unchanged until model/policy gates pass;
- expose a deterministic explanation for every selected or rejected schedule.

Pass evidence:

- policy artifact includes target shape, stage count, LDS bytes, VGPR/SGPR budget, estimated occupancy, and rejection
  reason when blocked;
- unsupported shapes fall back to the existing tinygrad path.

### BB-5a.6 Correctness Harness

Prove the rendered pipeline is semantically valid.

Required coverage:

- small deterministic WMMA matmul;
- authority prefill `ffn_gate/up` or `ffn_down` matmul shape;
- graph/TinyJit replay smoke once the primitive gate passes.

Pass evidence:

- numerical correctness matches the existing tinygrad result under the same dtype tolerance;
- graph replay does not invalidate buffers, LDS stage selection, or schedule metadata.

### BB-5a.7 Performance Gate

Reopen BB-5 only after BB-5a.1 through BB-5a.6 exist.

Required pass:

- pure tinygrad;
- no Tensile or handwritten code-object fallback;
- authority prefill matmul reaches `>=60 TFLOPS`;
- generated ISA proves real software-pipelined structure;
- correctness passes;
- default behavior remains unchanged.

## File-Level Scope

Likely implementation files:

- `tinygrad/codegen/opt/__init__.py` for any new opt vocabulary;
- `tinygrad/codegen/opt/postrange.py` or a new lowering pass for staged K-loop construction;
- `tinygrad/codegen/late/linearizer.py` if priority/toposort must preserve staged prefetch semantics;
- `tinygrad/renderer/isa/__init__.py` if pre/post-regalloc hooks need resource-policy feedback;
- `tinygrad/renderer/amd/schedule.py` for semantic scheduling, stage metadata, and resource summaries;
- AMD renderer/autogen instruction integration where final instructions are emitted;
- focused `extra/qk_amd_*` probes for correctness, ISA structure, and performance gates.

Do not modify default model routing or q8 decode routing in BB-5a.

## Completion Checklist

BB-5a is complete only when all rows below are true:

| row | required state |
|---|---|
| pipeline IR | stages survive lowering and dumping |
| double-buffer LDS | two LDS stages visible in metadata and ISA |
| wait scheduler | integrated into AMD render path, not probe-only |
| allocator/resource control | spill-free budgeted candidate or deterministic rejection |
| policy | can select/reject pipelined schedules with resource reasons |
| correctness | small WMMA and authority prefill correctness pass |
| performance | pure tinygrad authority prefill `>=60 TFLOPS` |
| defaults | unchanged until separate policy/model acceptance |

## Stop Conditions

Stop and keep BB-6 blocked if any of these remain true:

- pipeline stages do not survive lowering;
- the double-buffered kernel renders byte-identical to the serialized kernel;
- waits are still inserted only by a standalone probe;
- register work remains accounting-only;
- the authority prefill gate remains below `60 TFLOPS`;
- the implementation requires handwritten assembly or an external artifact to pass.

## Downstream Rule

Only after BB-5a passes may BB-6 scope q8 transfer. The q8 transfer must consume the same shared scheduler/resource
capability and must not introduce a q8-only native scheduler patch.
