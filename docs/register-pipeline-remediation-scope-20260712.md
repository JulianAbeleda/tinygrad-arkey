# Register-Resident Pipeline Remediation Scope

Date: 2026-07-12

## Decision

The register-resident runtime route is not eligible for hardware dispatch in
its current form.  The logical storage-adapter direction remains viable, but
the implementation does not yet make synchronization, physical register
ownership, final-resource evidence, and runtime eligibility authoritative over
the emitted program.

This remediation is complete only when the compiler has one end-to-end chain:

```
typed register schedule
  -> global loads
  -> architecturally derived wait that dominates every consumer
  -> deterministic packed stage values
  -> allocator-issued physical VGPR leases
  -> WMMA consumer
  -> final binary/resource artifact from the same allocation
  -> identity-joined runtime eligibility gate
```

No structural proof, payload marker, route name, or diagnostic report may
substitute for a fact from the final emitted program.

## Confirmed Defects

### Wait-counter semantics

The current register template passes `loads_per_stage` as `vmcnt`.  AMD
`s_waitcnt vmcnt(N)` waits until at most `N` vector-memory operations remain;
it does not wait for `N` operations to complete.  With eight issued loads,
`vmcnt(8)` may perform no wait at all.

The sequential single-stage implementation must use a full vector-memory
drain (`vmcnt(0)`) unless instruction scheduling proves the exact number of
younger outstanding operations.  The graph-level policy must describe the
dependency; the backend must own the architectural counter derivation.

### Wait dominance

The current producer places stage writes and the wait as siblings beneath a
barrier.  That proves both precede the barrier, but does not prove that the wait
precedes the stage write/pack consuming each load result.

Required graph order:

```
LOAD(A/B) -> WAIT -> PACK/STAGE_WRITE -> READY -> STAGE_READ -> WMMA
```

Every path from a staged global load to a VALU, register-stage write, or WMMA
consumer must cross the corresponding wait.  Tests must check graph dominance
and final instruction order.

### Split physical-allocation authority

The typed AMD allocator records leases, while native instruction selection
independently assigns stage registers from `FRAG_BASE + used`.  Evidence and
emission can therefore disagree.

One allocator must issue every physical interval used by stage A, stage B,
WMMA A/B fragments, accumulators, ABI values, address temporaries, virtual
registers, and reserved scratch windows.  Instruction selection may consume a
lease but may not invent a physical base.  Final artifacts must serialize the
same leases.

### Stateful half-pair reconstruction

Instruction selection currently combines adjacent half stores through mutable
`ctx._stage_pending` state.  Which node emits the packed write depends on
rewrite visitation order.  Pairing is a semantic graph transformation and
must be deterministic before physical instruction selection, or represented
by an explicit typed packed-stage operation.

### LDS/register proof conflation

The register route currently rewrites itself as `route_family="lds"`, derives
postrange options from an LDS schedule, and validates an LDS-specific fragment
layout identity.  Tile geometry and WMMA descriptors may be shared, but storage
transport and proof identity may not.

Register-resident candidates require an explicit register schedule identity
that names static slot addressing, packed VGPR stage layout, wait policy,
consumer identity, and zero LDS ownership.  LDS defaults and local-stage keys
must be unreachable from this path.

### Disconnected runtime gate

The strict compile/resource/correctness gate exists as a reporting API, but
runtime route installation does not require it.  Payload admission currently
can install global warmstart state before final binary/resource evidence is
available.

Register runtime installation must default closed.  It requires an eligibility
token or immutable result produced by the authoritative gate and joined to:

- canonical candidate identity;
- exact target and ABI;
- exact role and shape;
- final binary hash;
- final VGPR/SGPR/LDS/scratch facts;
- physical interval mapping;
- typed wait and required-edge coverage;
- correctness authority when dispatch promotion is requested.

Compile-only eligibility and dispatch eligibility must be distinct states.

## Required Ownership Boundaries

### Research and route layer

Owns candidate identity, exact applicability, search fields, and promotion
state.  It may request a typed register schedule but cannot select physical
registers, create wait immediates, or claim final resources.

### Schedule and graph layer

Owns logical A/B tiles, lifecycle, producer/consumer dependencies, static slot
semantics, and typed waits.  It cannot emit AMD instruction payloads or assign
physical VGPR numbers.

### AMD backend

Owns instruction selection, architectural wait immediates, physical leases,
register allocation, final instruction ordering, ABI metadata, and resource
facts.  It must fail closed when a logical contract cannot be represented.

### Evidence layer

Serializes facts from the final program.  It cannot reconstruct physical
mapping from the candidate payload or host estimate.  Its result controls
runtime eligibility rather than merely documenting it.

## Implementation Workstreams

### A. Synchronization

1. Replace the `loads_per_stage -> vmcnt` rule with a backend-facing completion
   requirement.
2. Use `vmcnt(0)` for the initial sequential implementation.
3. Reshape producer dependencies so WAIT dominates every staged-value use.
4. Verify final native and LLVM wait encoding from the same typed contract.
5. Reject duplicate, missing, misplaced, or non-dominating waits.
6. Add tests with unrelated younger loads before permitting nonzero waits.

Exit criteria:

- no stage write is reachable from a staged load without crossing WAIT;
- final instruction stream places the wait after all protected loads and
  before their first consumer;
- tests assert architectural completion semantics, not only bit packing.

### B. Deterministic stage representation

1. Introduce an explicit packed-stage value/write or perform pair formation in
   a deterministic pre-isel graph pass.
2. Remove correctness dependence on mutable visitation state.
3. Preserve both half-value dependencies and the producer readiness token.
4. Verify every logical half element is represented exactly once.

Exit criteria:

- graph output is invariant under legal source/topological ordering;
- missing or duplicate halves fail closed;
- no unmatched pending pair can survive lowering.

### C. Physical register authority

1. Extend the canonical allocator to reserve fixed ABI, accumulators, stage A,
   stage B, fragments, scratch, and virtual pools in one namespace.
2. Have instruction selection request/consume leases rather than calculate
   bases.
3. Derive `_vpool` exclusions from leases.
4. Emit resource intervals from those leases.
5. Validate final encoded register operands against intervals.

Exit criteria:

- allocator leases, emitted pins, and artifact intervals are identical;
- overlap and exhaustion tests fail before assembly;
- no physical base is independently manufactured by route or isel code.

### D. Register-specific scheduling

1. Add a register schedule identity separate from LDS.
2. Share only storage-independent tile and WMMA consumer contracts.
3. Remove LDS route-family rewriting, LDS fragment-layout identity, LDS
   environment defaults, and LDS local-stage ownership from register routing.
4. Keep the initial capability exact to `attn_qo 512x4096x4096`, gfx1100,
   wave32, one static physical slot, and two logical lifecycle stages.

Exit criteria:

- a register candidate can be described and compiled without constructing an
  LDS schedule spec;
- register and LDS cache/warmstart keys cannot alias;
- storage identity remains register-resident through final artifact capture.

### E. Runtime eligibility

1. Introduce immutable compile-only and dispatch eligibility results.
2. Require exact identity joins before warmstart/context installation.
3. Default missing, stale, malformed, host-estimated, or mismatched evidence to
   rejection.
4. Scope warmstart mutation and restore prior state on failure.
5. Keep correctness promotion separate from compilation.

Exit criteria:

- no register route installs from payload admission alone;
- a different binary, candidate, role, shape, target, or ABI cannot reuse an
  eligibility result;
- legacy LDS behavior remains unchanged.

## Remaining Path to Completion (2026-07-12 Audit)

The fixed-stage-use representation is complete as of `84cece736`: packed A/B
stage leases are consumed as existing physical registers rather than virtual
definitions.  The exact compile-only kernel now reaches final register
allocation with stage-read pressure removed.  The following work remains and
must be completed in order; solving a later item does not waive an earlier
gate.

### F. Epilogue lifetime closure (current blocker)

The exact `attn_qo 512x4096x4096` kernel currently keeps 64 output-address
values and the accumulator/output carriers live together at the reduction
boundary.  Linear allocation requests spills even though the arithmetic is
individually legal.  AMD:ISA intentionally has no spill fallback.

1. Serialize output stores with explicit graph dependencies.
2. Materialize or rematerialize each output address adjacent to its store.
3. Ensure store scheduling does not extend accumulator or loop-address
   lifetimes backward across the reduction.
4. Retain fixed accumulator ownership and exact output lane mapping.

Exit criteria:

- the exact kernel completes final register allocation with zero spills;
- no scratch allocation or hidden spill/fill instructions are emitted;
- output addresses have bounded, local live ranges in the final linear stream;
- epilogue ordering tests preserve every output exactly once.

### G. Final resource and binary authority

Passing allocation is necessary but not sufficient.  Host estimates and
pre-isel lease plans cannot authorize execution.

1. Capture final VGPR, SGPR, LDS, scratch, wave, and workgroup metadata from
   the assembled program.
2. Capture allocator intervals and encoded register operands from the same
   compilation.
3. Join those facts to candidate identity, target/ABI, and binary hash.
4. Reject occupancy below the explicitly selected floor; do not infer that
   zero spills implies acceptable occupancy.

Exit criteria:

- final artifact reports zero LDS and zero scratch for the register route;
- all encoded A/B/C operands lie in their authoritative intervals;
- resource metadata and binary hash are immutable and identity-joined;
- a stale or host-estimated resource record cannot pass compile eligibility.

### H. Instruction-order and memory-safety proof

Assembler success can still hide an incorrect wait, overwrite, or address.

1. Prove `global_load -> vmcnt(0) -> stage_write -> fixed_stage_use -> WMMA`
   for every staged fragment and loop epoch.
2. Prove no stage lease is overwritten before its final consumer.
3. Prove all global loads/stores are in bounds for exact and edge tiles.
4. Verify ABI parameter order, workgroup mapping, vector alignment, and tail
   behavior against the candidate identity.

Exit criteria:

- final disassembly, not only graph IR, satisfies every ordering edge;
- stage overwrite, missing wait, duplicate/missing output, and out-of-bounds
  mutations fail deterministic compile-only tests;
- no LDS/barrier instruction is present unless the route identity is LDS.

### I. Numerical promotion and fault containment

The first hardware execution remains a separate promotion event.  Compilation
must never automatically enable dispatch.

1. Start with bounded canaries and guarded allocations under an external
   timeout/reset-aware harness.
2. Compare against a trusted implementation across deterministic and adversarial
   inputs, including nonfinite and accumulation-sensitive cases.
3. Expand from one workgroup to representative tiles, then the exact shape.
4. Revoke eligibility on timeout, reset, guard corruption, nondeterminism, or
   numerical failure.

Exit criteria:

- canary and exact-shape tolerances are predefined and pass repeatedly;
- guard regions and device health checks remain clean;
- correctness evidence is joined to the exact binary/resource artifact;
- dispatch remains default-off without that evidence.

### J. L2-to-VGPR versus LDS decision

Direct L2/global-to-VGPR is a candidate schedule, not the predetermined winner.
Likely post-correctness blockers include lower occupancy from VGPR use, memory
latency not hidden by a one-slot sequential schedule, instruction overhead,
and loss of cross-wave reuse that LDS provides.

1. Benchmark the exact register and LDS binaries under identical inputs and
   measurement controls.
2. Record latency, throughput, occupancy, cache behavior, and variance by
   role/shape.
3. Select per exact identity; preserve LDS fallback and never generalize from
   one favorable shape.

Exit criteria:

- direct staging is promoted only where it is correct, stable, and materially
  better than LDS;
- regressions or inconclusive measurements retain the LDS route;
- selection evidence is versioned with compiler and binary identity.

### Anticipated failure classes

These are explicit audit targets rather than reasons to preemptively weaken
the design:

- fixed-register aliasing between ABI, accumulators, stages, and temporaries;
- occupancy collapse despite zero spills;
- instruction-encoding or descriptor limits exposed only after assembly;
- wait-counter assumptions invalidated by load regrouping;
- loop-carried dependencies that serialize too much and erase latency hiding;
- edge/tail tiles diverging from the exact aligned shape;
- numerical differences from fp16 packing or accumulator lane mapping;
- compile cache or warmstart evidence reused across a different binary;
- hardware hangs or resets despite graph-level correctness.

## Verification Matrix

### CPU-only unit tests

- wait immediate meaning and bounds;
- wait dominance over every stage consumer;
- lifecycle ownership and one-slot overwrite ordering;
- deterministic half pairing;
- lease overlap, alignment, pressure, and exhaustion;
- emitted pin-to-lease equality;
- register/LDS schedule identity separation;
- runtime rejection for absent and mismatched evidence;
- exact acceptance for identity-joined compile-only evidence.

### Compile-only integration tests

- final stream contains protected loads, `s_waitcnt vmcnt(0)`, packed stage
  writes, stage reads, and WMMA in the required order;
- no `DEFINE_LOCAL`, LDS instructions, LDS bytes, raw route-owned `Ops.INS`, or
  host-estimated final register counts;
- final binary hash and resource intervals are joined.

### Hardware promotion tests

Hardware execution is outside this remediation until all CPU and compile-only
gates pass.  Promotion then proceeds through bounded single-workgroup
correctness, guarded buffers, repeated deterministic outputs, and finally the
exact full shape.  A GPU fault, reset, timeout, guard corruption, nonfinite
output, or identity mismatch revokes dispatch eligibility.

## Commit and Review Discipline

Each workstream lands as a separately reviewable commit with its own invariant
tests.  Representation repairs must precede runtime activation.  No commit may
both weaken a gate and expand dispatch eligibility.  Documentation and tests
must describe architectural semantics independently of implementation names.

## Completion Definition

The route is directionally sound only when typed contracts are not parallel
descriptions but the actual authorities used by scheduling, instruction
selection, allocation, artifact generation, and runtime admission.  Until all
five workstreams meet their exit criteria, `global_register_resident` remains
compile-only and hardware dispatch remains disabled by default.
