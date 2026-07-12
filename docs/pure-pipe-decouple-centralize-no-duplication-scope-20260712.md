# Pure pipe decouple, centralize, and reuse scope

Date: 2026-07-12

## Objective

Finish the pure register-pipe architecture by decoupling the existing modules
and making them interchangeable, without creating a second lifecycle,
descriptor system, policy schema, allocator, wait system, or resource artifact.

The rule for this scope is strict:

```text
new behavior may be added;
new ownership types are not added when an existing owner can be extended.
```

The target path is:

```text
candidate policy
  -> existing PipelinePolicy
  -> existing shared WMMA/precontract validation
  -> existing lifecycle proof
  -> existing Stage1StorageAdapter boundary
       LDS implementation | register implementation
  -> existing WaitDependency/WaitCount seam
  -> existing AMD register allocator/resource artifact
  -> existing correctness, timing, and machine-search gates
```

## Current owners and duplication audit

| Concern | Canonical owner to keep | Existing duplication or coupling | Consolidation action |
|---|---|---|---|
| Storage/wait/resource policy | `tinygrad/codegen/opt/compiler_policies.py` | route/spec strings and compatibility re-exports | resolve route strings once into `PipelinePolicy`; keep `extra.qk` as re-export only |
| LDS lifecycle | `tinygrad/codegen/opt/kernel_pipeline.py` | `KernelStage1PipelinePlan` assumes positive LDS slots | keep legacy plan/API; expose a storage-neutral callback/protocol view rather than a second lifecycle |
| Register lifecycle mapping | `tinygrad/codegen/opt/register_pipeline.py` | `RegisterLogicalStagePlan` is separate from stage1 semantics | make it an adapter/mapping consumed by existing lifecycle functions; do not add another proof engine |
| WMMA geometry/CONTRACT/remaps | shared helpers in `kernel_lds.py` | register template and pipe fixtures can reconstruct metadata | require every storage policy to call the shared validators |
| Storage callbacks | `Stage1StorageAdapter` in `kernel_pipeline.py` | register adapter is structural and not postrange-wired | extend the adapter contract; route storage through it |
| Register roles/leases | `register_contracts.py`, `amd_register_contracts.py`, `amd_register_allocator.py` | old numeric constants and helper-local ranges in `amd.py`/research code | make descriptor the source of truth; migrate helpers incrementally |
| Wait provenance | `WaitDependency`, `WaitCount`, `WaitDependencyCoverage` | pipe structural fixtures construct wait metadata independently | derive waits from lifecycle edges; no fixture-only wait claims |
| Physical waits | `AMDISARenderer._insert_waitcnt`, AMDLLVM `Ops.WAIT` seam | physical and graph wait logic are separate | keep physical resolver backend-owned; feed it typed graph provenance |
| Final resource identity | `AMDResourceArtifact` | evaluation gates and route artifacts can carry partial facts | require one final artifact before promotion/search |
| Route selection | `prefill_graph_gemm_route.py` and `PrefillGEMMScheduleSpec` | candidate path forces LDS and installs LDS warmstart directly | dispatch from admitted `PipelinePolicy`; preserve legacy candidate defaults |
| Evidence gates | `pure_register_evaluation_gate.py` and whole-prefill authority | per-role fallback census is incomplete | make missing role/fallback evidence a hard reject |

## Non-goals

- Do not create `RegisterLifecycle`, `RegisterWaitGraph`, or another policy
  hierarchy if the existing contracts can be extended.
- Do not copy `build_gemm_pipe` or expose `AMDOps` to route/compiler graph code.
- Do not move AMD physical register numbers into generic codegen modules.
- Do not change the default LDS route while register integration is unproven.
- Do not make structural fixtures count as executable kernels.
- Do not merge the native post-regalloc wait resolver into route code.

## Centralization rules

### Rule C1: one logical policy

`PipelinePolicy` is the only composition of storage, wait, resource, and
logical stage data. `RegisterPipePlan` remains its strict register-pipe
constructor. `WMMAPipeSpec.pipeline_policy` and candidate admission must use
these objects rather than re-deriving strings.

### Rule C2: one descriptor/validation path

`kernel_lds.py` shared helpers are the sole authority for:

- RDNA3 descriptor identity and lane remaps;
- tile/wave/K divisibility;
- scalar fp16 row/K ownership;
- A/B four-binary-axis CONTRACTs;
- `half.vec(16)` and `float.vec(8)` carriers.

LDS-specific checks remain in the LDS template. Register storage must call the
shared helpers and add only zero-LDS checks.

### Rule C3: one lifecycle proof

`stage1_lifecycle_events` and `prove_stage1_lifecycle` remain the proof core.
`KernelStage1PipelinePlan` remains compatible with existing LDS callers.
`RegisterLogicalStagePlan` is only a mapping object that supplies the methods
the existing proof consumes; it must not grow a second event/proof algorithm.

### Rule C4: one storage callback boundary

`Stage1StorageAdapter` is the only callback boundary. The register adapter
must return the existing `KernelStage1ProducerStage` and
`KernelStage1FragmentStage` values. Postrange must select the adapter from
`PipelinePolicy` rather than branch on ad-hoc environment flags.

### Rule C5: one wait contract

`WaitDependency` describes logical producer/consumer/load-group coverage.
`WaitCount` describes the AMD immediate. `WaitDependencyCoverage` joins them.
No path may create a `WaitCount` without a validated dependency. Native AMD
physical queue tracking and AMDLLVM intrinsic lowering remain backend adapters.

### Rule C6: one register lease/artifact path

`RegisterRole`/`RegisterBank`/`Lease` describe logical reservations;
`AMDRegisterDescriptor` describes gfx1100 physical facts;
`AMDRegisterLeaseAllocator` realizes leases;
`AMDResourceArtifact` records final intervals and resources. Do not create a
second allocator in `amd.py`, `wmma.py`, or the graph layer.

## Dependency-ordered phases

### D0 - Ownership freeze

Create an import/ownership test or review checklist proving the table above:

- generic code imports compiler contracts, not research-plane compatibility;
- route code does not import `AMDOps` or physical register constants;
- `wmma.py` only consumes the backend allocator through its public interface;
- artifact validation consumes the shared `RegisterBank` type.

Exit: duplicate definitions are listed and assigned to removal/delegation.

### D1 - Policy normalization

Update `PrefillGEMMScheduleSpec`, `WMMAPipeSpec`, candidate admission, and
route attribution to resolve one `PipelinePolicy`. Preserve JSON and legacy
LDS behavior. Add identity tests proving LDS/register policies cannot alias.

Exit: no lowering branch compares raw storage/wait strings after policy
resolution.

### D2 - Shared validation consumption

Make `RegisterPipeTemplate` consume the extracted `kernel_lds.py` helpers with
the real `tc` descriptor. Remove any local CONTRACT/remap/carrier validation.
Keep the existing structural tests, adding negative tests for wrong descriptor,
wrong remap, wrong carrier, and lost row/K range ownership.

Exit: one validation path serves both LDS and register storage.

### D3 - Lifecycle adapter, not lifecycle rewrite

Change only the interface needed for `stage1_lifecycle_events` and
`prove_stage1_lifecycle` to consume the register mapping. Do not change their
event semantics. Explicitly prove:

- logical register stages are not physical LDS slots;
- prologue/body/drain order;
- K=1, 2, 3, full K, and tail cases;
- no overwrite before consume;
- body fragment readiness refers to the correct producer stage.

Exit: register structural tests use the existing proof and LDS tests remain
unchanged.

### D4 - Storage-policy postrange dispatch

Refactor the candidate branch in `postrange.py` and route installation so:

- LDS policy constructs the existing allocation/template path;
- register policy constructs `RegisterStorageAdapter`;
- no branch unconditionally creates `DEFINE_LOCAL` for a register policy;
- no route sets LDS warmstart state after policy selection;
- unsupported policy/backend combinations reject before launch.

Exit: compile-only register admission reaches normal graph construction rather
than a fixture-only builder, while default LDS output is preserved.

### D5 - Wait dependency emission

Derive dependencies from D3 producer/consumer events. Use existing coverage
validation, then create one typed `Ops.WAIT`/`WaitCount` per required state.
Handle duplicate state, branch, barrier, loop-end, and kernel drain. Keep
`_insert_waitcnt` unchanged until a backend bridge test proves equivalence.

Exit: a normal graph wait is provenance-tagged and cannot be manually injected
without coverage.

### D6 - Register artifact join

Feed the allocator's leases into `AMDResourceArtifact`. Require:

- role-to-physical intervals;
- VGPR/SGPR/LDS/scratch/spill facts;
- target and ABI;
- source/binary/candidate hashes;
- no overlap, overrun, unknown, or identity mismatch.

Exit: register compile admission requires a final artifact, not a host guess.

### D7 - Compile-only vertical slice

Compile `attn_qo` through normal `full_rewrite_to_sink` and AMDLLVM/AMD
backend paths. Require the real WMMA CONTRACT ABI, typed waits, no raw route
ISA, zero LDS, and artifact identity. Known synthetic WMMA fixtures are not
allowed as substitutes.

Exit: reproducible source/object artifact; no timing claim.

### D8 - Correctness, timing, roles, search

Run nonconstant full-output parity, pinned isolated timing, then whole-prefill
authority. Expand to `ffn_down` and `attn_kv` only after independent gates.
Fix per-role fallback attribution before pure machine-search promotion.

Exit: all selected roles are pure, fallback-free, correct, resource-proven,
and searchable through logical policy fields.

## Parallel agent packets

### Packet A - D0/D1 ownership and policy audit

Only touch import/ownership tests, policy normalization, and route identity
tests. Do not edit allocator or lifecycle code.

### Packet B - D2/D3 storage/lifecycle adapter

Only touch `register_pipeline.py`, shared validation call sites, and lifecycle
adapter tests. Do not create a second proof or modify physical AMD allocation.

### Packet C - D4/D5 postrange/wait bridge

Only touch policy-driven postrange dispatch and dependency-derived wait emission.
Do not alter the native wait resolver or invent a new wait schema.

### Packet D - D6-D8 artifact/authority gates

Only touch artifact joins, resource/evidence gates, route census, and benchmark
admission. Do not hide missing register binaries behind defaults.

Packets A, B, and D can run in parallel. C depends on B's adapter contract.
D can build fail-closed gates in parallel but cannot promote a route until C
produces a real artifact.

## Completion definition

Decoupling is complete when:

1. every concern has one canonical owner from the table;
2. LDS and register storage share policy, validation, lifecycle, wait, and
   artifact interfaces;
3. physical AMD details are confined to the AMD descriptor/allocator/backend;
4. normal postrange dispatch selects storage policy without hard-coded LDS;
5. no duplicate lifecycle, descriptor, allocator, wait, or resource types exist;
6. the pure register route passes compile, resource, correctness, timing, and
   fallback-free authority gates.

## Spark execution status (2026-07-12)

- **D0-D1 complete:** `ecd5d2527` and `4700a7f99` prove policy ownership and
  candidate/schedule resolution through the existing `PipelinePolicy`.
- **D2-D3 complete as a fail-closed mapping:** `93a517eec` exposes the
  zero-LDS logical two-stage plan through the existing register adapter and
  lifecycle proof. The LDS stage builder rejects the register adapter before
  lowering because true two-stage readiness is not yet proven.
- **D4-D5 complete as policy-aware admission:** `f57ec4cf4` resolves policy
  before candidate storage allocation and rejects register execution unless
  wait coverage and an executable storage adapter exist. Native waits and the
  legacy LDS route are unchanged.

The cross-phase policy, register, lifecycle, wait, artifact, route, and
evaluation suite passes with **163 tests**, 3 known pytest configuration
warnings, and 26 subtests. D6-D8 remain gated on the executable register
storage/lifecycle and final WMMA ABI; no partial execution path is being
promoted.

## Final Spark execution status (2026-07-12)

- **D6 artifact join complete:** `7365f297c` requires a nested final
  `AMDResourceArtifact`, strict source/binary/candidate joins, final-program
  resources, and rejects host estimates, spills, and missing intervals.
- **D7 one-epoch compile complete:** `dec34992d` compiles the real RDNA3
  register carriers and WMMA CONTRACT ABI through normal AMD rewriting with no
  `DEFINE_LOCAL` or raw `Ops.INS`. This is a single-epoch structural compile,
  not a full K-loop kernel.
- **D8 authority gates complete:** `a1db05b4a` strengthens resource,
  correctness/timing, role-attribution, and machine-search admission. Missing
  binary/resource/output evidence fails closed.

The remaining execution blocker is precise: the shared stage1 builder supplies
the prologue readiness marker to body fragments while the register template
requires the body epoch/slot producer marker. Exact carrier matching therefore
rejects the full K loop. This must be fixed in the shared lifecycle mapping;
falling back to the prologue marker would make the two-stage proof unsound.
