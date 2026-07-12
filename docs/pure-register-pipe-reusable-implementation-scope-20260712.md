# Pure register pipe reusable implementation scope

Date: 2026-07-12

## Objective

Make the existing compiler-owned lifecycle reusable for a register-resident
WMMA pipe, without creating a second compiler, copying the handwritten AMD
kernel, or changing the established LDS candidate behavior.

Target execution shape:

```text
typed candidate
  -> PipelinePolicy(storage=global_register_resident)
  -> shared descriptor/range/accumulator validation
  -> shared epoch/body/drain lifecycle proof
  -> global b128 A/B loads into register fragment carriers
  -> proven typed wait dependency
  -> WMMA accumulation and global stores
  -> AMDLLVM/AMD binary + ABI + resource proof
```

The first target is `attn_qo`, shape `M=512,N=4096,K=4096`, wave32,
RDNA3 WMMA `f16 -> f32`. `attn_kv` and `ffn_down` are expansion targets only
after the first role passes all gates. `ffn_gate_up` remains on the generated
LDS policy during this work.

## Decision

This is an incremental extension, not a rewrite. Approximately 70-80 percent
of the required machinery already exists and is tested:

- `KernelStage1` epoch/slot ownership, prologue/body/drain, and lifecycle proof;
- typed `PipelinePolicy`, `RegisterPipePlan`, `WaitDependency`, and `WaitCount`;
- `Stage1StorageAdapter`, with the existing LDS path already routed through it;
- WMMA descriptor dimensions, lane remaps, CONTRACT axes, and accumulator ABI;
- AMD global b128 and WMMA instruction selection;
- native AMD physical wait analysis in `_insert_waitcnt`;
- AMDLLVM `Ops.WAIT -> llvm.amdgcn.s.waitcnt` backend seam;
- cache identity, resource-plan, and candidate admission boundaries;
- existing correctness, route, and pinned benchmark authority tooling.

The missing work is the register storage adapter, lifecycle wait provenance,
final graph/resource integration, and promotion gates.

## Current evidence and baselines

The measured pure-vs-hybrid gap is approximately 31 ms at the pinned K=8
whole-prefill protocol. The all-role LDS candidate is about 147-156 ms while
the role-selective hybrid teacher is about 116-125 ms. The in-model mechanism
is the all-role 40 KiB LDS buffer2 kernel versus non-LDS kernels for
`attn_qo`, `attn_kv`, and `ffn_down`.

The generic `PREFILL_WMMA_PIPE_PRIMITIVE=1` experiment is not the target
primitive: it is an ordinary graph WMMA path and measured about 1.9k tok/s.
Do not use it as evidence that a register pipe is impossible or complete.

The typed wait seam is now real: `WaitCount` validates AMD counter bounds,
AMDLLVM emits `llvm.amdgcn.s.waitcnt`, and gfx1100 object compilation produces
the corresponding `s_waitcnt`. An arbitrary `Ops.WAIT(WaitCount)` is not yet a
valid register route because it is not tied to a producer/load-group proof.

## 100 percent completion definition

The reusable register path is complete only when all of these are true:

1. The register candidate has explicit storage, logical-stage, wait, resource,
   role, shape, target, dtype, and canonical identity fields.
2. Shared validation proves the exact RDNA3 descriptor, scalar fp16 row/K
   ownership, four binary CONTRACT axes for A and B, descriptor remaps,
   `half.vec(16)` carriers, and `float.vec(8)` accumulator ownership.
3. The register storage adapter emits no `DEFINE_LOCAL`, no LDS windows, and
   no route-owned `Ops.INS` or native instruction payloads.
4. The lifecycle proves K=1, K=2, full K, and adversarial tail cases with two
   logical register stages and no overwrite-before-consume hazard.
5. Every WMMA consumer has a typed dependency on the correct producer/load
   group. A wait cannot be inserted without lifecycle provenance.
6. The backend lowers that dependency to a proven `WaitCount` or fails closed.
   Duplicate waits and branch/barrier/drain behavior are tested.
7. Final source, binary, ABI, launch geometry, LDS, VGPR, SGPR, scratch, spill,
   workgroup, and candidate identity facts are joined.
8. Full-output nonconstant correctness passes for `attn_qo`.
9. Isolated pinned timing excludes compilation and setup and beats the current
   generated pure LDS candidate for `attn_qo`.
10. `attn_kv` and `ffn_down` pass independent shape/tail/resource/correctness
    gates before combined promotion.
11. Whole-prefill attribution records every role, including fallback roles,
    and strict pure promotion proves there is no hybrid fallback.
12. The machine-search schema exposes only policy fields represented by the
    compiler contract and treats correctness/resource failures as hard rejects.

Anything less is a diagnostic or compile-only milestone, not pure promotion.

## Architecture to preserve

### Policy boundary

Canonical owner: `tinygrad/codegen/opt/compiler_policies.py`.

Use `PipelinePolicy` as the composition of:

```text
storage: StoragePolicy
wait: WaitPolicy
resources: ResourcePlan
logical stages: int
```

Physical LDS slots and logical register stages must remain separate. Do not map
`RegisterPipePlan.stages=2` to `KernelStage1PipelinePlan.stage_count=2`.
The existing LDS plan is a proved stage1 lifecycle with one or two physical
LDS slots; changing that meaning would invalidate current proofs.

### Lifecycle boundary

Canonical owner: `tinygrad/codegen/opt/kernel_pipeline.py`.

Reuse `stage1_lifecycle_events`, `prove_stage1_lifecycle`, accumulator graph
construction, and `prove_stage1_uop_graph`. The register path may add an
explicit logical-stage mapping or a storage-neutral plan, but must preserve the
existing LDS plan and legacy `build_stage1_uop_graph` entry point.

### Storage boundary

Canonical adapter: `Stage1StorageAdapter`.

The LDS implementation remains responsible for `DEFINE_LOCAL`, padded A/B
windows, cooperative stores, barriers, and LDS fragment loads. The register
implementation must return typed producer/fragment stages backed by ordinary
global loads and register carriers. It must not return instruction lists or
call `extra/qk/prefill/wmma.py::build_gemm_pipe`.

### Wait boundary

Graph ownership: `WaitDependency` and lifecycle proof.

Backend representation: `WaitCount` and `Ops.WAIT`.

The native AMD renderer may reuse its physical register-span wait resolver, but
that algorithm must stay behind a backend interface. Route code must not import
`AMDOps`, raw RDNA3 instruction constructors, or `Ops.INS`.

AMDLLVM now has the intrinsic seam. It must receive only a dependency-derived,
validated `WaitCount`; an arbitrary marker must not bypass load-group proof.
The existing `amdllvm_wait_dependency` fail-closed behavior remains until the
graph-to-wait lowering is implemented.

### Resource and identity boundary

Candidate identity remains in `runtime_specs.py` and `KernelCandidateContext`.
Final resource facts must be extracted from the generated program/binary, not
from host estimates. A register policy reports LDS=0 only after proving the
lowered graph contains no local allocation. Unknown VGPR/SGPR counts are not
acceptable for promotion.

## Dependency-ordered implementation packets

### R0 - Contract and negative-control freeze [complete]

Owners: compiler policy modules and focused unit tests.

Already complete:

- immutable storage, wait, resource, pipeline, register-plan contracts;
- fail-closed two-stage/b128/targeted-wait register policy;
- typed `WaitCount` bounds and AMD simm16 packing;
- policy route/cache separation and LDS backward compatibility;
- negative controls for ordinary graph WMMA and unsupported waits.

Exit: policy tests pass and no incomplete register candidate is admitted.

### R1 - First-class wait node admission [complete, backend-only]

Owners: `tinygrad/uop/spec.py`, `tinygrad/renderer/llvmir.py`, late regalloc.

Already complete:

- `Ops.WAIT` is preserved as a pseudo operation;
- spec admission rejects missing wait payloads;
- AMDLLVM lowers typed `WaitCount` to `llvm.amdgcn.s.waitcnt`;
- gfx1100 object compilation is covered.

Not complete in this packet: lifecycle emission and dependency provenance.

### R2 - Shared descriptor and operand validation

Owner: `tinygrad/codegen/opt/kernel_lds.py`.

Extract shared helpers for:

- exact RDNA3 WMMA descriptor and lane-map remaps;
- geometry/tile divisibility and per-wave factors;
- scalar fp16 source plus row/K RANGE ownership;
- A/B four binary CONTRACT axes and folded element identity;
- `half.vec(16)` fragment carrier and `float.vec(8)` accumulator ABI.

Keep these checks independent of `DEFINE_LOCAL`, `AddrSpace.LOCAL`, LDS window
size, and active LDS bytes. Leave all LDS allocation/window checks in the LDS
policy. Add regression tests proving existing LDS output and failures remain
unchanged.

Exit: the register template can call shared validation without constructing an
LDS allocation.

### R3 - Register storage template for one tile

Owner: new compiler-opt storage implementation beside `kernel_lds.py`.

Implement a typed `RegisterStorageAdapter` or equivalent with:

- global b128 A/B loads;
- explicit logical stage identity;
- register fragment carriers retaining row/K ownership;
- no `DEFINE_LOCAL`, `KernelLDSWindow`, or LDS resource claims;
- producer and fragment callbacks returning existing typed stage values;
- fail-closed shape, dtype, descriptor, and carrier checks.

Use existing precontract fixture construction as the ABI oracle. Do not copy
the native `build_gemm_pipe` instruction sequence.

Exit: host graph proves no LDS nodes, exact A/B fragment metadata, dominance of
fragment consumers, and output-store ownership for `attn_qo`.

### R4 - Logical register-stage mapping

Owner: lifecycle adapter in `kernel_pipeline.py` plus policy tests.

Define an explicit mapping between:

- logical register stages = 2;
- physical LDS slots = 0;
- current stage1 event/proof semantics.

Prove prologue, body, drain, slot/stage identity, K=1, K=2, full K, and tail
cases. Reject accidental use of LDS slot arithmetic for register storage.

Exit: lifecycle proof passes with register storage while all existing LDS
tests remain byte/behavior compatible.

### R5 - Dependency-derived wait emission

Owner: lifecycle graph and backend policy bridge.

Add typed dependency edges from each producer/load group to its WMMA consumer.
Lower only proven dependencies to `WaitCount` and `Ops.WAIT`.

Required checks:

- producer dominates consumer;
- load group and logical stage match;
- no wait without a producer;
- no duplicate wait for the same counter state;
- branch, barrier, loop-end, and kernel-drain handling;
- unsupported backend capability fails before launch.

The native physical `_insert_waitcnt` algorithm may be refactored behind a
callback interface, but its current route behavior must remain unchanged.

Exit: one register graph emits a dependency-derived wait and the proof joins
the wait metadata to the candidate identity.

### R6 - Compile-only vertical slice

Owner: normal codegen path, target `AMD:LLVM:gfx1100`.

Compile `attn_qo` through the standard graph path and require:

- `Ops.WAIT` survives spec, rangeify, devectorization, linearization, and
  regalloc;
- generated LLVM contains the typed wait intrinsic;
- generated object contains the expected wait instruction;
- no route-owned raw ISA or `Ops.INS` construction;
- exact ABI, launch dimensions, WMMA descriptor, and output stores;
- candidate identity/cache separation.

Exit: compile-only artifact is reproducible and all source/binary/resource
facts are joined. No timing promotion yet.

### R7 - Final resource and spill gate

Owner: backend resource capture and candidate admission.

Measure and join:

- LDS bytes (must be zero for register policy);
- VGPR/SGPR counts;
- scratch and spill status;
- workgroup size, wave count, occupancy limits;
- source hash and binary hash;
- ABI and candidate canonical identity.

Host estimates remain provisional. Unknown or overflowing facts reject the
candidate. Any spill rejects the fast register policy unless separately proven.

Exit: final-program `ResourcePlan` exists and is tied to the exact binary.

### R8 - Correctness and isolated timing

Owner: pure route harness and pinned benchmark authority.

Correctness first:

- nonconstant full-output parity;
- adversarial K/tail cases;
- A/B stride and output ABI checks;
- repeated runs and cache identity checks.

Timing second:

- pinned clocks;
- compile/input/output setup excluded;
- isolated kernel timing before whole-prefill;
- compare against current generated pure LDS candidate and hybrid teacher;
- record wait/resource/source/binary provenance.

Exit: `attn_qo` is correct and materially improves the pure baseline, or a
measured backend ceiling is documented without promotion.

### R9 - Role expansion

Expand only after R2-R8 pass for `attn_qo`:

- `ffn_down`: independent tile/resource/correctness gate;
- `attn_kv`: small-N occupancy and LDS/local-stage overflow audit;
- `ffn_gate_up`: remains generated LDS unless independently justified.

Each role gets its own candidate identity, binary, resource facts, and pinned
timing. No shape-only aliasing is permitted.

### R10 - Combined pure authority and machine search

Fix route attribution so every role, including fallback roles, is recorded.
Require pure authority to prove:

- all selected roles use compiler-owned policy paths;
- no hybrid/raw-ISA fallback;
- candidate identity and binary joins are complete;
- correctness/resource gates passed before timing ranking.

Only then expose storage/stage/wait/tile fields to machine search. Search must
operate over typed policy fields, not environment flag combinations that can
select an unproven route.

## Parallel execution plan

The smallest non-duplicating Spark assignments are:

### Spark A - Shared validation extraction

Work only in `kernel_lds.py` and its focused tests. Extract shared descriptor,
range, CONTRACT, and carrier validation. Do not edit lifecycle, route policy,
or backend wait code.

### Spark B - Register storage host graph

Work only in a new register storage module plus focused structural tests. Use
the shared validation interface from Spark A; if it is not available, stop at
an interface fixture rather than duplicating checks. Do not edit native ISA or
route selection.

### Spark C - Wait dependency bridge

Work only in lifecycle wait metadata, `WaitDependency` validation, and backend
bridge tests. Do not emit arbitrary `Ops.WAIT`; require producer/load-group
provenance. Preserve native `_insert_waitcnt` behavior.

These three packets can start in parallel only after agreeing on immutable
interfaces. R4 depends on A+B. R5 depends on C plus the R3 stage shape. R6-R8
are sequential gates. R9 is parallel by role after R8. R10 is final and must
not be started early.

## Explicit non-goals

- No rewrite of Tinygrad's compiler or AMD renderer.
- No copy of `build_gemm_pipe` instruction lists.
- No route-owned `Ops.INS`, `AMDOps`, or raw ISA payloads.
- No change to the current generated LDS candidate default.
- No promotion based on full barriers, compile time, or synthetic vec-only
  fixtures.
- No fabricated VGPR/SGPR/occupancy values.
- No combined pure benchmark before per-role attribution is fixed.

## Stop conditions

Stop a packet and report blocked when:

1. it would duplicate an existing policy, lifecycle, descriptor, or wait
   implementation;
2. it requires changing the established LDS output or stage1 semantics;
3. the backend cannot prove the claimed wait/resource/ABI fact;
4. a structural fixture cannot be connected to the normal graph path;
5. correctness or identity evidence is missing.

The project is genuinely blocked only if the existing UOp graph cannot retain
the required range/CONTRACT/WMMA ABI, or if the backend cannot consume a typed,
proven wait dependency. Current evidence shows neither condition.

## Existing artifacts to reuse

- `docs/pure-register-pipe-reuse-vs-rewrite-20260712.md`
- `docs/pure-pipe-modular-lifecycle-storage-scope-20260712.md`
- `docs/pure-register-pipe-primitive-exhaustive-scope-20260712.md`
- `tinygrad/codegen/opt/compiler_policies.py`
- `tinygrad/codegen/opt/kernel_pipeline.py`
- `tinygrad/codegen/opt/kernel_lds.py`
- `tinygrad/codegen/opt/postrange.py`
- `tinygrad/renderer/llvmir.py`
- `tinygrad/renderer/isa/amd.py`
- `extra/qk/wmma_pipe_spec.py`
- `extra/qk/runtime_specs.py`
- `extra/qk/prefill_whole_synced.py`
- `test/unit/test_precontract_lds_stage.py`
- `test/unit/test_kernel_pipeline.py`
- `test/unit/test_amdllvm_waitcnt.py`

