# Pure register-resident WMMA pipe primitive: exhaustive scope

## Goal

Build the missing pure primitive for `attn_qo`, `attn_kv`, and `ffn_down`:

```text
typed candidate
  -> register-resident storage policy
  -> shared epoch/slot lifecycle
  -> global b128 A/B loads into VGPR fragments
  -> compiler-owned targeted wait dependency
  -> WMMA accumulation and stores
  -> normal graph/cache/runtime ABI
```

`ffn_gate_up` remains on the existing generated LDS primitive until the
register path is independently proven. The hybrid handwritten atom is a
behavioral reference only.

This document specializes
`docs/pure-pipe-modular-lifecycle-storage-scope-20260712.md`; it does not create
a second lifecycle framework.

## Definition of done

Path 3 is complete only when:

1. a register candidate is represented by an immutable compiler policy and exact
   candidate identity;
2. the shared lifecycle proves two-stage ownership, K progression, tails, and
   accumulator coverage;
3. the generated graph contains no local allocation for register operands;
4. A/B fragments retain scalar fp16 row/K ownership, exact CONTRACT axes and
   descriptor remaps, and `half.vec(16)` carrier types;
5. the backend consumes a typed wait dependency and emits a proven targeted wait
   or fails closed before promotion;
6. final source/binary/resource/ABI facts join to the candidate identity;
7. `attn_qo`, `ffn_down`, and `attn_kv` pass independent correctness and timing
   gates;
8. the combined pure route has no hybrid fallback and closes the measured
   roughly 31 ms gap or establishes a measured backend ceiling.

## Existing pieces to reuse

- `KernelStage1PipelinePlan`, lifecycle events, and proofs in
  `tinygrad/codegen/opt/kernel_pipeline.py`;
- WMMA descriptors, lane remaps, accumulator ownership, and CONTRACT validation
  in `tinygrad/codegen/opt/kernel_lds.py`;
- candidate context/cache identity in `tinygrad/uop/ops.py` and
  `extra/qk/runtime_specs.py`;
- postrange candidate insertion in `tinygrad/codegen/opt/postrange.py`;
- `WMMAPipeIR` and `RegisterPipePlan` policy contracts;
- existing HIP/LLVM WMMA fixtures in `test/unit/test_precontract_lds_stage.py`;
- hybrid `build_gemm_pipe` only as a schedule/resource/correctness teacher.

## Explicit non-goals

- Do not copy `extra/qk/prefill/wmma.py::build_gemm_pipe` instruction lists.
- Do not construct route-owned `Ops.INS` or `AMDOps` payloads.
- Do not change the existing LDS candidate default while the register path is
  incomplete.
- Do not call generic ordinary matmul a register pipe; the prior 1,906 tok/s
  experiment falsified that shortcut.
- Do not promote a full-barrier-only prototype as the fast primitive.

## Required contracts

### Candidate and policy

Register candidates must carry:

- `storage_kind=global_register_resident`;
- exactly two producer stages/buffers;
- 16-byte cooperative global-load width;
- `wait_policy=targeted_vmcnt` and per-stage scope;
- `resource_plan=unproven` until final compilation;
- exact role, shape, target, dtype, and canonical identity.

Incomplete register payloads must fail admission. LDS payloads remain backward
compatible and must not inherit register defaults.

### Storage policy

Implement `RegisterStoragePolicy` beside the existing LDS policy:

```text
validate(spec, geometry, descriptor)
producer(epoch, slot, reuse)
fragments(epoch, slot, ready, k_step)
resource_plan()
```

The policy must return typed producer/fragment instances understood by the
shared lifecycle core. It must not return instruction lists.

### Fragment contract

For each A/B operand:

- source remains scalar fp16 with row and K RANGE ownership;
- four binary CONTRACT axes are retained;
- CONTRACT args exactly match the range IDs and descriptor remaps;
- fragment dtype is `half.vec(16)`;
- WMMA args carry exact dimensions, dtype in/out, device, thread count, and
  upcast-axis tuple;
- accumulator is the exact `float.vec(8)` ownership layout expected by the
  descriptor.

The existing precontract validation must be generalized, not bypassed.

### Lifecycle contract

For every K epoch:

```text
produce(stage)
  -> issue A/B global loads
  -> ready dependency
  -> fragment consume
  -> WMMA
  -> release/reuse
```

The proof must reject overwrite-before-consume, consume-before-ready, missing
producer/consumer, wrong stage/slot, missing tail drain, and accumulator gaps.

Register storage has no LDS slot window. Do not map `WMMAPipeIR.stages=2`
directly to `KernelStage1PipelinePlan.stage_count=2`; the current stage-1 plan
uses `stage_count=1` with two rotating slots. Introduce an explicit semantic
mapping instead.

### Wait contract

Represent waits as typed dependencies:

```text
WaitDependency(load_group, producer_stage, consumer_stage, scope="per_stage")
```

The verifier must prove that each WMMA consumes only after the corresponding
load group is ready. The backend must either lower this dependency to the
targeted wait required by the plan or fail closed. A full `Ops.BARRIER` is a
correctness diagnostic, not a substitute for the performance primitive.

## Implementation packets

### R0 — Policy/schema freeze

Complete the register policy contract, canonical identity fields, LDS backward
compatibility, and fail-closed invalid cases. No lowering changes.

Exit: policy tests pass and register candidates cannot be mistaken for executed
programs.

### R1 — Generic lifecycle/descriptor extraction

Separate descriptor/CONTRACT/accumulator validation from LDS allocation checks.
Keep `PrecontractPipelineTemplate` behavior unchanged by moving only shared
validation helpers. Add regression tests for every existing LDS invariant.

Exit: existing LDS suite is unchanged and shared validation has no LDS-only
assumptions.

### R2 — Register operand template

Implement global-load-backed producer and fragment callbacks for one tile. Use
real row/K RANGE ownership, descriptor remaps, and carrier types. No local
allocation, no native ISA.

Exit: host graph proves A/B fragment dominance, zero LDS nodes, exact WMMA
descriptor, and accumulator/store ownership.

### R3 — Two-stage K lifecycle

Connect the register template to the shared lifecycle for `attn_qo`:

- prologue/body/drain;
- two alternating register stages;
- load-group identity;
- K tail handling;
- accumulator loop-carried state.

Exit: lifecycle proof passes for K=1,2,3, full K=4096, and adversarial tails.

### R4 — Wait dependency/backend hook

Thread `WaitDependency` through postrange and backend lowering. First prove the
dependency in source/IR; then implement targeted wait lowering for the AMD path.
No route promotion while the backend only supports full barriers.

Exit: final source/binary contains the expected wait semantics and joins to the
typed candidate; unsupported wait lowering fails before launch.

### R5 — Compile-only `attn_qo`

Compile through the normal HIP/LLVM renderer with the exact output/A/B ABI.
Require no `Ops.INS` in route-owned construction, valid devectorization, source
hash, cache separation, launch dimensions, and zero LDS allocation.

Exit: two candidate identities compile separately and the generated source has
the expected load/WMMA/store/wait structure.

### R6 — Resource and binary gate

Join final VGPR/SGPR, LDS, scratch, spill, waves, source hash, binary hash, and
candidate identity. Host estimates remain marked provisional until final
program extraction.

Exit: register candidate has measured resources and no spill/overflow.

### R7 — Correctness and timing

Run full-output nonconstant parity, then isolated pinned timing with compile and
input/output setup excluded. Compare against ordinary generated WMMA and the
hybrid teacher only as a reference.

Exit: register `attn_qo` is correct and materially faster than the current pure
generated candidate path.

### R8 — Role expansion

Parameterize `ffn_down` and `attn_kv` only after R7. KV receives independent
small-N occupancy, tail, and resource proofs.

Exit: all three roles pass independently with exact identities.

### R9 — Combined pure route

Use register storage for the three non-LDS roles and existing generated LDS for
`ffn_gate_up`. Fix per-role route attribution so fallback roles are explicit.
Run pinned ctx512, then 1024/2048/4096. Require no hybrid fallback.

Exit: whole-model wall time closes the measured gap or establishes a measured
compiler ceiling with the remaining limiter named.

### R10 — Machine search

Search only typed register-policy fields: tile, stage cadence, vector width,
register residency, and wait policy. Correctness is a hard filter; timing is
secondary; promotion requires whole-model evidence and exact binary joins.

## Required tests and artifacts

- policy/schema/cache separation;
- LDS regression suite unchanged;
- descriptor/remap/CONTRACT validation;
- register graph has no local allocation;
- lifecycle dominance, slot reuse, prologue/body/drain, and tail proofs;
- backend wait/source purity;
- exact A/B/output ABI and stride checks;
- final resource and binary joins;
- full-output parity for all roles;
- pinned isolated and whole-model timing;
- per-role route census including non-selected fallback roles;
- strict pure provenance with no hybrid execution.

Every authority artifact must record commit, candidate identity, storage policy,
wait policy, source/binary hashes, resource facts, route census, clock-pin
status, and benchmark protocol.

## Stop conditions

Stop and document the missing interface if:

- register fragments cannot be expressed without route-owned ISA;
- the compiler cannot preserve row/K ownership through devectorization;
- targeted waits cannot be represented by the backend;
- resource extraction cannot prove zero LDS/no spill;
- the only passing implementation is generic ordinary WMMA;
- a mixed route cannot prove fallback roles explicitly.

The hybrid atom may remain the teacher/reference, but it must never be relabeled
as the pure implementation.
