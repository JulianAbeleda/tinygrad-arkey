# Pure WMMA pipe first-class UOp: exhaustive implementation scope

## Purpose

This document scopes the remaining work required to make the generated
non-LDS WMMA pipe a real **pure** route. It extends
`docs/pure-pipe-s4b-compiler-implementation-scope-20260712.md` with the
compiler contract, ownership boundaries, dependency order, tests, and stop
conditions needed for implementation.

The target is the existing Qwen3-8B prefill workload on AMD gfx1100 (7900 XTX):

- generated `ffn_gate_up` buffer2 candidate remains the known working generated
  baseline;
- `attn_qo` is the first non-LDS vertical slice at `M=512,N=4096,K=4096`;
- `ffn_down` is `M=512,N=4096,K=12288`;
- `attn_kv` is `M=512,N=1024,K=4096`;
- hybrid remains a teacher/reference only and is never wrapped or copied into
  the pure implementation.

No route default or machine-search population changes until the first pure
vertical slice passes every gate in this document.

## Current boundary

Already implemented and host-tested:

- `WMMAPipeSpec`, immutable `WMMAPipeIR`, and typed candidate context in
  `extra/qk/wmma_pipe_spec.py`;
- sink attachment through `KernelInfo.candidate_context`;
- postrange/cache identity and ABI tests;
- typed lifecycle/resource contract with fail-closed validation;
- generated diagnostic WMMA structure and the ordinary generated-matmul
  negative result (it is not equivalent to the lean pipe primitive);
- compiler insertion-point inventory in the S4b scope.

The remaining blocker is architectural, not a missing route flag. The AMD
renderer has generated helpers for b128 loads, WMMA, stores, and waits, but the
helpers currently materialize backend `Ops.INS` operations. Existing graph
`Ops.LOAD` and aggregate `Ops.WMMA` do not encode staged ownership or targeted
`vmcnt` waits. A typed metadata object alone cannot change scheduling or
register lifetimes.

The repository also has a pure-capable `AMDLLVMRenderer` path. Generic tensor
core graphs can already lower as `LOAD -> Ops.WMMA -> STORE` through LLVM AMDGPU
intrinsics. That path does not consume `WMMAPipeIR` stage fields, does not bind
the two-stage producer/ready/consume lifecycle, and has no typed targeted-wait
dependency. Therefore attaching the pipe context to a generic matmul would be
false provenance: it would prove ordinary WMMA, not the pure pipe primitive.

## Purity contract

Pure means compiler-owned generation from graph inputs through the executable
program:

1. The route supplies declarative shapes, dtypes, buffers, and tuning fields;
   it does not supply ISA strings, instruction tuples, or a precompiled binary.
2. The graph-facing representation is a first-class typed operation. It is
   present before backend instruction selection and is part of the graph/cache
   identity.
3. The compiler owns stage assignment, load grouping, wait placement, fragment
   liveness, register allocation, launch dimensions, and resource metadata.
4. The backend may use its existing internal instruction representation after
   typed lowering, but those instructions must be produced by the compiler
   lowering pass. Route code must not construct `Ops.INS` or `AMDOps.*` payloads.
5. The executed binary, source hash, candidate identity, and route census must
   join to the same candidate. Any fallback, handwritten atom, or ambiguous
   provenance is a pure-route failure.

This distinction prevents a false negative (rejecting all backend instruction
selection) and a false positive (calling a handwritten emitter “generated”).

## Required data model

### Declarative specification

`WMMAPipeSpec` remains the user/search-facing schema. It contains only stable,
serializable fields:

- role and `(M,N,K)` shape;
- tile and `k_step` divisibility;
- operand layouts and input/output dtypes;
- stage count and pipeline tile factors;
- target and wait policy;
- optional compiler-owned tuning knobs once validated.

It must reject invalid shapes, dtypes, targets, stage counts, and wait policies
before graph construction. It must not contain register numbers, LDS offsets,
instruction text, or cache-local object references.

### Typed compiler IR

Add the smallest first-class typed operation that represents one pipe epoch or
one complete pipe loop. The exact class/enum name is an implementation choice,
but the contract must include:

- three graph buffer references: output, A, B;
- compile-time shape/layout/dtype/target fields;
- tile geometry and K-step count;
- stage count and explicit stage-slot ownership;
- load-group cardinality for A and B;
- wait policy and the dependency it protects;
- accumulator dtype and output conversion;
- tail/divisibility behavior;
- launch geometry and resource-plan handle;
- candidate identity/provenance.

The operation must be immutable and hashable. Runtime tensor pointers and
scalar dimensions remain graph arguments, not embedded addresses. Two distinct
candidate identities must never share a lowered-program cache entry.

### Lifecycle semantics

The operation defines a deterministic state machine for each stage slot:

`free -> producing -> ready -> consuming -> free`.

The verifier must prove, for every K step:

- a slot is not overwritten before its prior consumer completes;
- every WMMA operand is dominated by the corresponding load and targeted wait;
- the wait covers exactly the required load group, not an unrelated full wait;
- the next stage can overlap the current compute stage;
- loop-carried accumulators dominate all stores;
- tail handling is either statically rejected or explicitly lowered.

The lifecycle is a compiler invariant, not a comment or a route-level flag.

## Compiler ownership map

### 1. Front-end construction

Files: `extra/qk/wmma_pipe_spec.py`, prefill route selection, candidate-context
helpers.

Responsibilities:

- build the typed IR from an eligible schedule;
- validate role/shape/target/dtype constraints;
- attach canonical identity to `KernelInfo.candidate_context`;
- leave ordinary matmul graphs unchanged when the candidate is absent;
- preserve the existing generated gate/up candidate.

Forbidden: importing the hybrid emitter, constructing native instructions,
mutating global compiler state, or selecting a candidate based on timing before
correctness.

### 2. UOp verification and graph rewrite

Files: `tinygrad/uop/ops.py`, verifier/rangeify, and the relevant graph rewrite
pass.

Responsibilities:

- add the typed operation and its child/output typing;
- verify buffer dtypes, strides, shape divisibility, and target;
- verify lifecycle dominance and stage-slot uniqueness;
- lower the typed node to ordinary graph structure plus a compiler-owned pipe
  scheduling node before backend rendering;
- preserve graph key determinism.

The operation must not be silently represented as an ordinary matmul: doing so
loses stage ownership and produces the measured slow ordinary-generated path.

### 3. Scheduling and lowering

Files: `tinygrad/codegen/opt/postrange.py`, scheduling/lowering modules.

Responsibilities:

- assign workgroups, waves, and lane ownership;
- derive cooperative b128 global-load addresses;
- allocate A/B fragment lifetimes and fp32 accumulators;
- emit the two-stage K loop;
- assign produce/ready/consume/release events;
- place targeted waits after the required load group;
- expose schedule facts to resource accounting and diagnostics.

The scheduler must have a typed wait dependency. A boolean “targeted wait” flag
is insufficient because it cannot prove which loads the wait covers.

### 4. AMD instruction selection

File: `tinygrad/renderer/isa/amd.py` and associated renderer/isel code.

Responsibilities:

- map typed b128 loads to existing AMD load encodings;
- map typed WMMA operations to `v_wmma_f32_16x16x16_f16`;
- map stores and output conversion;
- lower the typed wait dependency to the required `s_waitcnt` form;
- preserve source/binary provenance and launch metadata;
- reject unsupported target features before COMGR/program launch.

Existing `AMDOps.GLOBAL_LOAD_B128`, `AMDOps.V_WMMA`, and store helpers may be
reused only behind this compiler-owned lowering boundary. They must not be
constructed by route code. If wait scheduling cannot be represented with the
current renderer API, add the smallest typed scheduler output or backend op;
do not add a route-local instruction emitter.

### 5. Register allocation and resource accounting

Files: AMD register allocator/late passes, `KernelInfo`/program metadata, and
the typed resource hook.

Responsibilities:

- account for A/B fragment and accumulator VGPR lifetimes;
- detect spills and fail closed;
- compute SGPR usage, LDS bytes, scratch bytes, waves, and launch dimensions;
- report non-LDS pipe LDS as zero only when the generated plan proves no LDS
  allocation;
- retain unknown register counts until backend lowering rather than inventing
  estimates;
- reject occupancy/resource overflow before binary execution.

Resource output must identify whether each value is measured from the final
program or is only a pre-lowering estimate.

### 6. Program, cache, and runtime ABI

Files: `tinygrad/codegen/__init__.py`, `tinygrad/engine/realize.py`,
`tinygrad/runtime/graph/hcq.py`.

Responsibilities:

- include typed candidate identity/schema in lowering and program cache keys;
- preserve output/A/B argument order and pointer dtypes;
- preserve contiguous strides and exact runtime dimensions;
- replay the same program through HCQ graph capture;
- prove two candidate identities cannot alias a cache or binary;
- restore warmstart/environment state on success and exceptions.

The runtime must receive ordinary graph buffers. It must not receive a hidden
precompiled kernel handle or route-specific pointer convention.

## Dependency-ordered work packets

### P0 — Contract freeze (host-only)

Deliver the typed operation schema, field table, invalid-contract tests, and
canonical serialization. Review against existing `WMMAPipeIR` and candidate
context so no duplicate schema is introduced.

Exit: all invalid contracts fail before lowering; ordinary graphs are unchanged.

### P1 — UOp and verifier

Add the smallest first-class operation, graph child/output typing, verifier,
rangeify behavior, and graph-key support.

Exit: a host graph contains the typed node; no backend instruction payloads are
present; ordinary matmul tests remain byte/behavior compatible.

### P2 — Lifecycle scheduler

Implement stage-slot ownership, K-loop progression, fragment liveness, load
groups, and typed wait dependencies.

Exit: a structural schedule trace proves every load/ready/WMMA edge and rejects
under-wait, over-consume, overwrite, and unsupported tail cases.

Current feasibility result: **blocked at this packet**. Existing postrange
already builds generic `Ops.WMMA` graphs, and the typed context can be attached,
but postrange ignores pipe stage/wait fields. The reusable fixtures
`build_stage1_uop_graph`, `KernelStage1ProducerStage`,
`KernelStage1FragmentStage`, `prove_stage1_uop_graph`,
`wmma_fragment_loads`, and `wmma_output_owners` validate pieces of ownership and
fragment mapping; none binds a route-level 512x4096x4096 model kernel or proves
contiguous fp16 A/B strides. The next implementation must integrate those
semantics into lowering before any LLVM compile or benchmark can be called
pure.

A bounded compiler-consumed slice now exists as
`build_wmma_pipe_barrier_chain()` in `extra/qk/wmma_pipe_spec.py`. It proves a
real `Ops.BARRIER -> input pointer/load -> Ops.WMMA -> STORE` dependency with no
`Ops.INS`, but it is restricted to `attn_qo` and represents a full workgroup
barrier, not targeted `vmcnt`. It is structural only until a valid WMMA
CONTRACT/range-axis accumulator is supplied to the normal devectorizer. The
existing `_expanded_pipeline_accumulator` fixture in
`test/unit/test_kernel_pipeline_expansion.py` is the safe pattern to extract for
the next compile-only gate.

### P3 — AMD lowering slice

Lower only `attn_qo` (`512x4096x4096`) through the normal renderer. Reuse backend
encodings behind the typed boundary; do not alter route defaults.

Exit: compile-only source/ISA contains the expected generated operations and no
route-owned native instruction construction; cache identity and launch metadata
are present.

### P4 — Resource and binary gate

Join final resource extraction, source hash, binary hash, candidate identity,
and launch dimensions. Add no-spill and occupancy-limit failures.

Exit: final-program resource facts are measured or explicitly marked unknown;
unsupported plans fail before COMGR.

### P5 — ABI/runtime gate

Run graph replay, paired tensors, two candidate identities, warmstart restore,
and HCQ capture/replay.

Exit: exact output/A/B ABI and cache separation pass in one process.

### P6 — `attn_qo` correctness and timing

Run nonconstant full-output parity, executed-binary identity, pinned kernel
timing, then gate/up-only plus `attn_qo` whole-model A/B.

Exit: no fallback, strict pure provenance, and a measured result at ctx512.

### P7 — Role parameterization

Parameterize `ffn_down` and `attn_kv` only after P6. Re-run independent shape,
tail, resource, correctness, and binary gates; do not inherit attn_qo resource
assumptions.

Exit: all three non-LDS roles have separate passing candidate identities.

### P8 — Combined pure authority

Combine generated gate/up with the three generated non-LDS roles. Run pinned
K8/warmup4/round3 at ctx512, then 1024/2048/4096.

Exit: parity, route census, no hybrid fallback, executed-binary joins, clean
artifacts, and a measured pure ceiling or 4.4k result.

### P9 — Machine-search promotion

Search only compiler-owned parameters after P8. Correctness is a hard filter;
timing is measured with clocks pinned and compile excluded. Promote only
whole-model winners with retained provenance.

## Test and artifact matrix

Every packet must add or run the smallest relevant test in each category:

| Category | Required proof |
|---|---|
| Schema | invalid shape/dtype/target/stage/wait/tail contracts fail closed |
| Graph | typed node appears only for eligible candidates; ordinary matmul unchanged |
| Lifecycle | produce/ready/consume/release ordering and targeted wait dominance |
| Purity | no route-owned `Ops.INS`, ISA text, instruction tuples, or precompiled binary |
| ABI | output/A/B order, fp16/fp32 contract, strides, dimensions, paired tensors |
| Cache | two identities produce separate lowered programs/binaries |
| Runtime | warmstart restoration, graph capture/replay, exception cleanup |
| Resources | final VGPR/SGPR/LDS/scratch/waves, no spill, overflow rejection |
| Correctness | nonconstant full-output parity for each role and combined route |
| Binary | source/binary hash joins to the same candidate identity |
| Timing | pinned clocks, compile excluded, kernel and whole-model measurements |
| Authority | route census, no fallback, pure provenance, reproducible artifact |

Required artifacts are JSON or text files containing schema version, candidate
identity, role/shape, source hash, binary hash, launch dimensions, resource
facts, route census, clock-pin status, benchmark protocol, and verdict. A
diagnostic kernel or isolated TFLOPS number is never a pure authority artifact.

## Review and rollback rules

- Review each packet before starting its dependents.
- Do not land a new `Ops` member without verifier, rangeify, renderer, cache,
  and serialization coverage in the same implementation slice.
- Do not change route defaults while P3-P5 are incomplete.
- If a test requires a handwritten instruction emitter, stop and redesign the
  compiler boundary.
- If a role fails resource or correctness gates, keep the role out of the pure
  candidate set; do not silently fall back while claiming pure execution.
- Preserve hybrid as a comparison route only; never use its timing as proof that
  the pure implementation executed.

## Completion definition

This scope is complete only when:

1. the first-class typed operation lowers through the normal graph/compiler
   pipeline without route-owned native instruction construction;
2. `attn_qo`, `ffn_down`, and `attn_kv` each pass independent ABI, lifecycle,
   resource, correctness, binary, and timing gates;
3. generated gate/up plus those three roles execute as one strict pure route;
4. pinned whole-model artifacts prove the route and identify either a 4.4k
   result or a measured pure ceiling with the remaining limiter named;
5. machine search is restricted to compiler-owned parameters and promotes only
   whole-model winners.

If the UOp/renderer redesign cannot express the lifecycle safely, the correct
completion state is **blocked at the named interface**, with the exact missing
semantics documented. A host schema, diagnostic kernel, or hybrid measurement
does not satisfy completion.
