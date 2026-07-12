# Pure pipe modular lifecycle/storage/wait scope

## Objective

Reuse the existing compiler-owned WMMA pipeline for both LDS-staged and
register-resident operands. The current LDS candidate path remains the behavior
and regression oracle. The new register path must be added as a storage policy,
not as a second compiler or a copied handwritten atom.

Target roles:

- register-resident/non-LDS: `attn_qo`, `attn_kv`, `ffn_down`;
- existing LDS policy initially retained: `ffn_gate_up`.

The hybrid handwritten route remains a behavioral teacher only. It must not be
called from the pure path and its `Ops.INS` emitter must not be wrapped as a
compiler implementation.

## Current code and ownership

| Concern | Current owner | Required modular owner |
|---|---|---|
| Epoch/slot lifecycle | `tinygrad/codegen/opt/kernel_pipeline.py` | storage-independent core |
| LDS producer/fragments | `tinygrad/codegen/opt/kernel_lds.py` | `LDSStoragePolicy` |
| WMMA/accumulator construction | `postrange.py` callbacks | shared WMMA/accumulator builder |
| Candidate selection | `extra/qk/prefill_graph_gemm_route.py` | declarative storage/wait policy |
| Candidate identity | `runtime_specs.py`, `KernelInfo` | unchanged schema boundary plus policy fields |
| Wait lowering | hand AMD renderer / barriers | typed `WaitDependency` plus backend policy |
| Resource accounting | LDS candidate context and backend metadata | storage-specific plan joined to final program |
| Timing/provenance | whole-prefill authority | unchanged, with per-role route census fixed |

## Design principles

1. The lifecycle core owns ordering and proof; storage policies own where bytes
   and fragments live.
2. Wait policy is separate from storage policy. LDS and register storage must be
   able to choose different synchronization strategies.
3. Candidate route code supplies declarative policy data. It does not construct
   UOps, ISA, instruction tuples, or resource guesses.
4. Existing LDS behavior must remain byte/behavior compatible during extraction.
5. A policy is not executable until its final source, binary, resource, ABI,
   and route identity are joined.
6. Full barriers are a correctness fallback, not evidence of targeted wait
   performance.

## Modular interfaces

### Lifecycle core

Keep the existing `KernelStage1PipelinePlan`, lifecycle events, and
`build_stage1_uop_graph`/`prove_stage1_uop_graph` semantics. Generalize names and
type annotations only where they currently imply LDS. The core receives:

- stage count and slot count;
- K-tile count and tail policy;
- `StoragePolicy` callbacks;
- `WaitPolicy` callbacks;
- accumulator/WMMA callback;
- resource plan.

The core must continue to prove:

- producer dominance of every consume;
- no slot overwrite before all consumers finish;
- correct prologue/body/drain order;
- exact K-step and tail behavior;
- accumulator ownership and store coverage.

### StoragePolicy

Define a small immutable policy contract, preferably in a compiler-opt module:

```text
validate(spec, geometry, descriptor) -> None
producer(epoch, slot, reuse) -> ProducerStage
fragments(epoch, slot, ready, k_step) -> FragmentStage
resource_plan() -> ResourcePlan
storage_kind -> "lds" | "register"
```

`ProducerStage` and `FragmentStage` remain typed values already understood by
the lifecycle core. They must carry stage/slot identity and dependency edges;
they must not expose raw instruction lists.

### LDSStoragePolicy

Extract the current `PrecontractPipelineTemplate` behavior without changing it:

- global cooperative b128 loads;
- LDS stores into typed A/B windows;
- ready barrier;
- LDS b128 fragment loads;
- existing descriptor remaps and accumulator contract;
- `active_lds_bytes` and LDS legality checks.

The first extraction packet must prove existing LDS tests remain unchanged.

### RegisterStoragePolicy

Implement the non-LDS policy against the same lifecycle callbacks:

- global b128 A/B loads directly into register-resident fragment values;
- two stage buffers with explicit stage identity;
- no `DEFINE_LOCAL` or LDS windows;
- fragment values dominate WMMA consumers;
- register resource plan and no-spill gate;
- K-step progression matching the hybrid teacher.

The policy must not copy `build_gemm_pipe` instruction construction. It may use
the atom's schedule facts as a reference for tile shape, stage cadence, and
correctness expectations.

### WaitPolicy

Define a typed dependency rather than a free string or integer:

```text
WaitDependency(load_group, producer_stage, consumer_stage, scope)
```

Required implementations:

- `FullBarrierWait`: compiler-owned full workgroup barrier, correctness probe;
- `TargetedVmcountWait`: backend-owned lowering of the required global-load
  dependency, initially fail-closed until the backend can prove it.

The lifecycle verifier must reject a consumer that has no dependency covering
its producer. It must also reject a wait policy that claims targeted behavior
without a final backend artifact proving the wait.

### ResourcePlan

Separate pre-lowering estimates from final program facts:

- storage kind and LDS bytes;
- VGPR/SGPR known/unknown status;
- scratch bytes and spill status;
- workgroup size, waves, occupancy limits;
- provenance (`host_estimate` vs `final_program`).

Register policy must report LDS=0 only after validating that no local allocation
enters the lowered graph. Unknown register counts remain unknown until backend
lowering; they must never be fabricated.

## Candidate schema changes

Extend the typed candidate pipeline payload with explicit policy fields:

- `storage_kind`: `lds` or `register`;
- `stage_count` and `buffer_count`;
- `wait_policy`: `full_barrier` or `targeted_vmcnt`;
- `load_group` and fragment residency;
- resource-plan schema/version;
- backend capability identifier.

Backward compatibility rules:

- existing LDS candidate payloads deserialize unchanged;
- missing new fields default only to the established LDS behavior;
- register candidates must be rejected unless all fields are explicit;
- canonical identity includes the policy fields;
- cache keys cannot alias LDS and register candidates with the same shape.

## Postrange integration

The candidate branch in `postrange.py` currently constructs an LDS allocation and
`PrecontractPipelineTemplate`. Refactor it to:

1. validate the candidate policy;
2. create the appropriate `StoragePolicy`;
3. pass policy callbacks into the existing lifecycle core;
4. construct shared WMMA/accumulator ownership;
5. run the lifecycle proof;
6. attach resource facts and candidate identity to the sink.

The existing LDS branch must remain the default for all current candidates until
the register policy passes its gates. No route code should set LDS warmstart
keys directly after this extraction; policy installation belongs at one scoped
compiler boundary.

## Backend boundary

The pure-capable LLVM path already lowers ordinary loads, WMMA, stores, and full
barriers. It does not currently lower targeted VM waits. Therefore:

- the full-barrier register policy may be used for compile/correctness tests;
- it must not be promoted for performance from barrier-only results;
- targeted wait lowering is a separate backend packet;
- native `AMDOps`/raw ISA may not be imported into route code;
- backend-generated instruction representation is acceptable only after the
  typed compiler boundary and with generated provenance.

## Dependency-ordered implementation packets

### M0 — Schema freeze

Add policy fields, canonical identity tests, backward-compatible LDS fixtures,
and explicit rejection of incomplete register candidates.

### M1 — Extract lifecycle/storage interface

Introduce the minimal `StoragePolicy`, `WaitPolicy`, and `ResourcePlan` types.
Adapt the current LDS template without behavior change. Run all existing LDS
pipeline, precontract, and candidate ABI tests.

### M2 — Shared WMMA/accumulator adapter

Move descriptor validation, CONTRACT axes, accumulator ownership, and WMMA
construction behind a shared callback/adapter. Preserve existing postrange
output and devectorizer expectations.

### M3 — Register producer/fragments (host structural)

Implement register-resident producer and fragment callbacks for `attn_qo`
512x4096x4096. Prove stage identity, load grouping, fragment dominance, no LDS
allocation in the graph, and full-output store ownership. No route promotion.

### M4 — Wait-policy integration

Thread typed `WaitDependency` through lifecycle proof and postrange. Add the
full-barrier implementation first. Add targeted-vmcnt only after a backend hook
and final-source proof exist.

### M5 — Compile-only vertical slice

Compile `attn_qo` through the normal AMD LLVM/HIP path. Require valid CONTRACT/
range axes, WMMA descriptor, barrier/wait source, ABI, cache identity, and
resource metadata. Reject synthetic vec-only accumulators.

### M6 — Register resource/binary gate

Join final source/binary hash, LDS=0, VGPR/SGPR, scratch/spill, workgroup, and
candidate identity. Fail closed on unknown or overflowing plans.

### M7 — Correctness and isolated timing

Run nonconstant full-output parity, then pinned isolated kernel timing against
ordinary generated WMMA. Compile time is excluded. A barrier-only result is
diagnostic and cannot advance to whole-model promotion.

### M8 — Role expansion

Parameterize `ffn_down` and `attn_kv` only after `attn_qo` passes M3-M7. KV gets
independent tail/occupancy/resource proof.

### M9 — Combined pure authority

Combine register policies for the three roles with existing generated LDS
`ffn_gate_up`. Fix per-role route attribution before measuring. Run pinned ctx512
then larger contexts, requiring no hybrid fallback and exact binary joins.

### M10 — Machine-search promotion

Search only policy fields represented in the typed compiler contract. Correctness
is a hard filter; timing precedes promotion; whole-model results are mandatory.

## Test matrix

- schema identity and LDS/register cache separation;
- ordinary graph compatibility with policy absent;
- lifecycle dominance and slot reuse for both storage policies;
- CONTRACT/range-axis and WMMA descriptor validity;
- no `Ops.INS` or raw ISA in route-owned graph construction;
- register policy emits no local allocation;
- wait dependency coverage and fail-closed unsupported policies;
- exact A/B/output ABI, dtypes, strides, and shape;
- source/binary/resource/candidate identity joins;
- full-output nonconstant correctness and adversarial tails;
- pinned isolated and whole-model timing;
- per-role route census including fallback roles;
- no hybrid fallback for pure promotion.

## Completion definition

Modularization is complete when:

1. the existing LDS route runs through the extracted interfaces unchanged;
2. the register policy builds a valid compiler graph for `attn_qo`;
3. a backend-consumed wait dependency is proven in final source/binary;
4. resources and ABI are measured from the final program;
5. all three non-LDS roles pass independent correctness/timing gates;
6. the combined pure route closes the measured gap or establishes a measured
   backend ceiling;
7. route census proves no hybrid fallback;
8. machine search varies only the modular policy fields.

If M3-M5 require a second renderer or route-owned emitter, stop and record the
missing backend interface instead of claiming modular pure execution.
