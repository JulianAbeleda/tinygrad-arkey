# Direct-L2/LDS execution bridge scope

> Historical implementation scope. Its completion definition is superseded by
> `pure-register-direct-l2-completion-scope-20260712.md` as of 2026-07-13.
> Preserve this as design history; do not use it to claim D100 or R100.

## Objective

Make the direct-register transport and the existing LDS transport executable
through tinygrad's normal AMD runtime, then produce a trustworthy measured
decision for `attn_qo` `(512,4096,4096)` on `AMD:gfx1100`.

The implementation must remain transport-agnostic at the execution-authority
boundary. The authority selects and validates a candidate; it does not know
whether the candidate uses LDS, register-resident staging, or a future
transport. The current decision policy may compare `direct_l2` and `lds`, but
the execution interfaces must use a generic transport name.

## Reuse boundary

The bridge must reuse these existing tinygrad primitives:

| Concern | Existing seam | Required use |
|---|---|---|
| Graph lowering | `Tensor.schedule_linear()` | Produce the semantic workload and buffer ABI. |
| Compilation | `tinygrad.engine.realize.compile_linear` | Compile the selected transport through the normal compiler path. |
| Runtime binding | `tinygrad.engine.realize.get_runtime` | Resolve a compiled `Ops.PROGRAM` to the device runtime. |
| AMD code loading | `tinygrad.runtime.ops_amd.AMDProgram` | Let the runtime load ELF, apply relocations, install descriptors, and own code lifetime. |
| Launch | `AMDProgram.__call__` / `run_linear` | Use the existing argument ABI, queue, synchronization, and timeout support. |
| Memory | `tinygrad.device.Buffer`, `Device[device].allocator`, `Tensor` | Allocate, upload, guard, and read buffers through the standard allocator. |
| Timing | existing `time_call`, profile events, `Device.synchronize` | Measure with the runtime's synchronization boundary. |
| Counters | existing AMD PMC/SQTT hooks | Capture counters only when explicitly enabled and live. |
| Cache identity | compiler/runtime cache keys | Namespace by candidate identity and environment; never reuse a stale binary. |

Do not add a second HIP/ROCm launcher, ELF loader, allocator, queue, argument
marshaller, or binary cache.

## Central modules and responsibilities

### 1. Semantic schedule authority

Create one transport-independent schedule builder, owned by the existing
prefill route layer. Its input is a typed workload and schedule policy; its
output is a semantic schedule plus explicit transport requirements.

It owns:

- role, shape, dtypes, layout, tile, wave, and WMMA/MFMA consumer identity;
- logical A/B/C buffer roles and ABI order;
- correctness/reference semantics;
- the transport-independent pair key and schedule digest.

It does not own:

- binary hashes;
- GPU allocation or dispatch;
- benchmark decisions;
- promotion policy.

The existing pair generator should call this builder instead of duplicating
semantic payload construction.

### 2. Transport policy authority

Add a centralized transport descriptor/registry. Each transport provides a
policy and a compiler transformation, not an execution implementation.

Minimum descriptors:

- `lds`: legal LDS windows, barriers, and expected LDS resource facts;
- `global_register_resident`: zero-LDS register staging, typed wait policy,
  static register mapping, and no-spill requirements.

Each descriptor must declare:

- canonical storage name;
- candidate fields exposed to machine search;
- required structural predicates;
- resource expectations;
- route binding identity;
- fallback eligibility.

The transport registry must be the only place that maps a generic transport
name to transport-specific validation. `single_buffer_execution_authority`
and `transport_execution_authority` should consume this registry rather than
hard-code direct-L2 assumptions.

### 3. Compiler/artifact authority

Split compile evidence from executable lifetime.

`CompileArtifact` is the serializable evidence record. It contains:

- candidate identity and source identity;
- target/profile/ABI;
- final source/disassembly and binary SHA-256;
- final AMD descriptor/resource artifact;
- allocator and wait proofs;
- launch ABI: argument roles, dtypes, buffer count, global/local sizes;
- capture mode and dispatch permission;
- transport descriptor identity.

`ExecutableArtifact` is an in-process object and is not serialized. It contains:

- the validated `Ops.PROGRAM` UOp;
- the resolved tinygrad runtime object from `get_runtime`;
- the bound device and cache identity;
- the compile artifact it was derived from;
- an explicit close/lifetime method.

The executable object must be created only after the serializable artifact
passes all compile/resource gates. The binary hash must be computed from the
actual compiled library used by the runtime, not merely copied from a report.

### 4. Generic execution authority

Define a generic callback/adapter contract, for example:

```text
prepare(workload, transport, environment) -> ExecutableArtifact
allocate(executable, inputs, guards) -> ExecutionBuffers
launch(executable, buffers, launch_config, timeout) -> LaunchEvidence
capture(buffers) -> OutputEvidence
release(executable, buffers) -> None
```

This layer owns no direct-L2/LDS decision. It validates the artifact/route
join, delegates to tinygrad for allocation and launch, and returns structured
evidence.

The existing hardware executor becomes a policy wrapper around this generic
authority. Its callbacks should receive an executable transport contract,
not raw implementation-specific details.

### 5. Safety authority

Safety must be process-scoped, not thread-timeout-scoped.

The parent runner owns:

- explicit opt-in;
- clean-device preflight;
- child-process creation;
- hard wall-clock timeout;
- child termination on timeout;
- post-run device health check;
- artifact/result collection;
- permanent revocation after timeout, fault, guard corruption, or numerical
  failure.

The child owns one bounded launch at a time and exits after releasing buffers.
No subsequent stage may run in the same child after a failed launch. A GPU
reset or unhealthy probe blocks the entire experiment and requires a fresh
process and explicit reauthorization.

### 6. Correctness authority

Use one shared correctness harness for both transports:

- deterministic but nonconstant FP16 inputs;
- CPU FP32 reference;
- full output comparison;
- finite-value checks;
- unchanged-input checks;
- prefix/suffix guard regions;
- declared `rtol=2e-2`, `atol=2e-2`;
- output shape/dtype/ABI checks.

The correctness harness must not know how the kernel stages data. It consumes
the generic execution contract and returns evidence.

### 7. Benchmark authority

Benchmarking is separate from correctness and promotion:

- prepare both candidates from one semantic schedule;
- compile and validate both independently;
- run correctness before timing;
- use identical device/environment/clock policy;
- randomize transport order;
- warm up before samples;
- collect at least nine samples per candidate;
- capture live L2, memory, and compute counters where supported;
- report median, spread/CV, outliers, and environment;
- pass a decision record to the existing decision policy.

The policy remains CPU-only and decides promote-direct, retain-LDS, or
blocked/inconclusive. It must never infer L2 residency from zero LDS usage.

## Dependency graph

```text
semantic schedule
        |
        +--> transport policy --> candidate payload
        |                              |
        |                              v
        +------------------------> compiler
                                       |
                         compile artifact + executable artifact
                                       |
                         generic execution/safety authority
                              /                    \
                         correctness            timing/counters
                              \                    /
                               paired decision policy
                                       |
                              retain LDS or promote
```

The critical path is semantic schedule → real executable artifact → guarded
generic launch. Benchmark policy and reporting are downstream and should not
be coupled into compilation or dispatch.

## Implementation phases

### Phase A — inventory and contracts

- Define typed workload, transport, launch ABI, compile artifact, executable
  artifact, and evidence schemas.
- Add negative tests for mismatched candidate, transport, target, shape, ABI,
  and binary identity.
- Keep all code CPU-only.

### Phase B — executable artifact from normal tinygrad compilation

- Build the selected route as a normal `Ops.PROGRAM`.
- Call `compile_linear` and `get_runtime` instead of introducing a custom
  launcher.
- Retain the runtime object and exact `ProgramInfo` launch metadata.
- Verify the runtime's loaded library hash against the compile artifact.
- Add a CPU/mock runtime test proving artifact creation and lifetime without
  touching AMD hardware.

Current implementation status: the exact `attn_qo` direct-L2 candidate now
produces a real tinygrad `Ops.PROGRAM` and passing compiler-capture evidence
through `extra/qk/prefill/attn_qo_executable_preparation.py`. The exact LDS
candidate now also produces a real `Ops.PROGRAM`, using the existing
`_emit_schedule -> build_gemm_lds2` route and a separate generic transport
compile-evidence schema. Both artifacts join to `tinygrad.runtime.bridge` with
the actual final binary hash; construction remains non-dispatching. The pair
helper also joins both artifacts through the existing semantic pair authority,
preserving one `pair_key`/schedule digest while requiring distinct binaries.

The generated LDS *precontract* lowering is still blocked by its
multidimensional `WITH_LOCAL` / register-allocation failures. That is no
longer a blocker for the paired execution milestone because the existing LDS2
route is a compatible, reusable fallback. The remaining blockers are shared
buffer/guard correctness, isolated hardware dispatch, and paired timing and
counter evidence. The shared guarded lifecycle now lives in
`extra/qk/prefill/guarded_execution.py`: its Buffer-backed adapter places
payloads inside prefix/suffix guards, performs full readback comparison and
input immutability checks, and requires health before/after dispatch. The
transport-specific dispatch callback only binds logical buffer roles to the
compiled ABI. No GPU dispatch has been performed by this milestone.

### Phase C — generic buffer/launch adapter

- Implement allocation and buffer-role binding with `Tensor`/`Buffer`.
- Add guard allocation and before/after validation.
- Call `AMDProgram.__call__` through the existing runtime path.
- Return structured launch/output evidence.
- Keep the direct-L2/LDS policy out of this layer.

### Phase D — isolated hardware canary

- Compile a separate artifact for every canary shape.
- Run one stage per child process.
- Start with the smallest shape and verify output fully.
- Require health before and after each child.
- Revoke on any timeout, reset, guard failure, input mutation, or numerical
  failure.
- Only the exact final stage may produce a benchmark authorization.

### Phase E — paired benchmark

- Build exact direct and LDS executable artifacts from one semantic schedule.
- Correctness-gate both.
- Run randomized paired timing and counters.
- Feed only complete evidence into the existing decision policy.
- Record the decision and preserve LDS as fallback unless promotion criteria
  pass.

### Phase F — generalization

Repeat the same generic path for `attn_kv`, `ffn_down`, and `ffn_gate_up`.
Only role-specific schedule builders, ABI/reference shapes, and transport
requirements should vary. Safety, execution, artifact, correctness, and
benchmark authorities must remain shared.

## Explicit non-goals

- No custom HIP/ROCm launch runtime.
- No manual ELF loader or allocator.
- No global route switch before measured evidence.
- No claim that zero LDS equals guaranteed L2 residency.
- No serialized executable handles; only serializable evidence is persisted.
- No automatic recovery-and-continue after a GPU fault.

## Definition of done

The bridge is complete when:

1. Both transports compile as real tinygrad `Ops.PROGRAM` objects.
2. The executable artifact launches through the existing AMD runtime.
3. A subprocess-isolated canary proves correctness and health at every stage.
4. Exact direct/LDS artifacts are identity-joined and independently correct.
5. At least nine paired samples and live counters are captured.
6. The decision is reproducible from persisted evidence.
7. LDS remains the safe fallback unless the declared speed/stability gate passes.
