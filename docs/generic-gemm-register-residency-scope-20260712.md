# Generic GEMM Register Residency Scope

Date: 2026-07-12

## Goal

Extract the logical register-resident tile contract from the current AMD/WMMA
prototype so the same storage and lifecycle machinery can serve GEMM consumers
such as WMMA, MFMA, dot2, and vector ALU. Keep physical register allocation and
instruction lowering backend-owned. Preserve the existing LDS path and the
current pure-8B execution gates.

This is an abstraction scope, not permission to generalize an unproven kernel
across shapes. Every consumer and shape still requires its own descriptor,
correctness, resource, and timing evidence.

## 100% definition

The work is complete when:

1. A compiler-neutral register-tile descriptor represents role, dtype, vector
   width, tile shape, slot policy, static-addressing requirement, layout, and
   lifetime without importing AMD or WMMA types.
2. The shared lifecycle builder consumes that descriptor through the existing
   `Stage1StorageAdapter`; LDS and register storage remain interchangeable at
   one boundary.
3. A backend adapter converts logical tiles into physical register leases,
   alignment, packing, wait dependencies, and resource facts. No route code
   contains physical register numbers or raw ISA.
4. Consumer adapters validate and lower WMMA, MFMA, dot2, and vector-ALU tile
   layouts independently. A consumer cannot claim compatibility from dtype or
   shape alone.
5. Static VGPR addressing is explicit. Dynamic modulo indices are rejected
   unless the backend proves an equivalent lowering; no indirect-VGPR fiction
   or LDS fallback is accepted as register residency.
6. Existing AMD/WMMA behavior is unchanged and all current tests pass.
7. At least one non-WMMA GEMM consumer reaches normal lowering through the same
   generic descriptor and backend adapter with no duplicate lifecycle code.
8. Resource, correctness, timing, and machine-search gates consume the generic
   descriptor plus consumer/backend evidence and reject incomplete joins.

## Current assets and boundaries

Reuse these owners; do not recreate them:

| Layer | Existing owner | Genericization boundary |
|---|---|---|
| Logical lifecycle | `tinygrad/codegen/opt/kernel_pipeline.py` | Keep epoch/slot/produce/consume/release proof; parameterize storage descriptor |
| Storage callbacks | `Stage1StorageAdapter`, `RegisterStorageAdapter` | Keep callback ABI; move logical tile metadata below callbacks |
| Descriptor/range checks | `tinygrad/codegen/opt/kernel_lds.py` | Keep shared shape/range helpers; remove LDS naming from neutral helpers only |
| Policy/waits/resources | `tinygrad/codegen/opt/compiler_policies.py` | Add typed residency/consumer fields only if existing policy cannot represent them |
| Register leases | `tinygrad/renderer/isa/amd_register_allocator.py` | AMD owns physical packing, spans, and capacity |
| AMD WMMA lowering | `tinygrad/renderer/isa/amd.py` | Consumer/backend adapter; never move physical pins into codegen |
| Evidence | `AMDResourceArtifact`, pure-register gates | Join generic descriptor, consumer, backend, source, and binary identities |

The logical descriptor must not import `dtypes`-specific AMD enums, `AMDOps`,
or WMMA lane-remap implementation details. Consumer adapters may depend on the
validated tensor-core descriptor and backend capabilities.

### G0 field inventory

The current prototype has one logical source of truth and two adapters.  This
table records ownership before further extraction; names in the backend and
consumer columns must not leak into the compiler-neutral contract.

| Existing field | Current owner | Contract owner | Notes |
|---|---|---|---|
| `RegisterPipeTemplate.pipe_tm`, `pipe_tn` | `register_pipeline.py` | logical tile | A/B fragment counts |
| `RegisterPipeTemplate.schedule`, `logical_buffer_count` | `register_pipeline.py` | logical tile/lifecycle | slot count and addressing mode |
| `RegisterPipeTemplate.k_step`, `shape`, `geometry` | `register_pipeline.py` | shape instance | GEMM geometry, not physical allocation |
| `RegisterPipeTemplate.tc`, `contracts` | `register_pipeline.py`/`kernel_lds.py` | consumer adapter | WMMA descriptor and lane contract |
| `AMDStageBufferSpec.role`, `slots`, `fragments`, `lane_width` | `amd_register_allocator.py` | backend compatibility view | derives from logical A/B tiles; packing remains AMD-owned |
| `AMDStageBufferSpec.packed_vgpr_width`, `half_bytes` | `amd_register_allocator.py` | backend result | physical-width estimates, never logical fields |
| WMMA `TensorCore.dims`, `elements_per_thread`, `swizzle` | `tc.py` | WMMA consumer | lane remap and output ABI are not generic GEMM layout |
| `RegisterLogicalStagePlan` epoch/slot methods | `register_pipeline.py` | shared lifecycle | storage-independent ownership proof |

G1 introduces `LogicalRegisterTile` in `register_contracts.py` with role,
scalar dtype, tile extents, fragment/carrier widths, slot count/addressing,
layout identity, alignment, ownership, and lifetime labels.  The existing
AMD stage spec remains a compatibility projection, so current snapshots and
allocation behavior are unchanged.

## Proposed contract

### Logical register tile

Fields:

- `role`: stable operand identity such as A or B;
- `dtype` and scalar element size;
- `lane_width` and carrier width;
- logical tile extents and fragment count;
- `slot_count` and `slot_addressing` (`static`, `sequential`, or proven backend mode);
- logical slot/epoch ownership;
- byte/element span and alignment;
- producer/consumer lifetime labels;
- layout identity and source range ownership.

The descriptor reports logical bytes and packed-width requirements. It does not
report physical register numbers. A separate backend allocation result reports
physical spans, bank, alignment, spills, and overlap proof.

### Consumer adapter

Every consumer implements a small validation/lowering protocol:

1. validate logical tile dtype, carrier width, layout, and output contract;
2. validate operand lane mapping and accumulator shape;
3. translate logical carriers into the consumer's fragment ABI;
4. declare required waits and resource classes;
5. expose a stable consumer identity for artifacts and search.

Initial adapters:

- WMMA: existing RDNA3 descriptor and four/four/three binary-axis contract;
- MFMA: future adapter, only after its ISA lane map and ABI are measured;
- dot2/vector ALU: bounded small consumer proving the generic storage path;
- other GEMM consumers: rejected until a typed adapter exists.

## Phases

### G0 — Inventory and contract extraction

Map every field in `RegisterPipeTemplate`, `AMDStageBufferSpec`, WMMA
validation, and the lifecycle proof into logical, backend, or consumer-owned
categories. Identify duplicate fields and preserve serialized identities.

**Exit:** one field table and no proposed duplicate allocator/lifecycle.

### G1 — Add the compiler-neutral descriptor

Create the smallest neutral descriptor in the existing compiler-opt ownership
area. Add validation for logical sizes, static-addressing declarations, slot
count, alignment, and layout identity. Adapt `RegisterPipeTemplate` to consume
it without changing default output.

**Exit:** existing register/LDS suites pass; descriptor has no AMD/WMMA imports.

### G2 — Separate backend allocation results

Define the backend result contract: physical bank, spans, packed width,
alignment, reserved ranges, overlap/lifetime proof, spill/scratch status, and
target identity. Route the AMD implementation through the existing allocator.
Keep dynamic VGPR indexing fail-closed.

**Exit:** AMD snapshots prove logical-to-physical joins for the sequential
one-slot case; double-buffer dynamic slots remain explicitly rejected.

### G3 — Extract consumer adapter protocol

Move WMMA-specific ABI validation and fragment translation behind a consumer
adapter interface. Keep the current shared descriptor checks and ensure the
existing WMMA path produces byte-identical graph/ISA behavior where supported.

**Exit:** malformed WMMA contracts fail before devectorization; WMMA identity
and resource artifacts include the adapter identity.

### G4 — Prove a non-WMMA GEMM consumer

Implement the smallest useful dot2 or vector-ALU consumer using the same
logical descriptor, lifecycle, backend allocation result, and wait provenance.
Use a bounded shape with a complete correctness oracle. Do not build a second
pipeline or hand-write instruction lists.

**Exit:** non-WMMA normal lowering, correctness, and resource tests pass while
the existing WMMA route remains green.

### G5 — Wait and evidence integration

Make wait dependencies and final resource artifacts reference logical tile,
consumer, backend, source hash, binary hash, and candidate identity. Reject
consumer/backend mismatches, missing spans, dynamic addressing, spills, and
opaque route payloads.

**Exit:** pure-register admission accepts only complete generic evidence.

### G6 — Machine-search exposure

Expose only proven generic fields and consumer capabilities to machine search:
tile shape, carrier width, slot schedule, wait policy, consumer adapter, and
resource limits. Search emits exact identities and runs correctness/timing for
each consumer separately.

**Exit:** search cannot select WMMA-only fields for dot2/MFMA or apply a shape
contract to an unsupported consumer.

## Parallel Spark assignments

### Spark A — Neutral contract and extraction (G0-G1)

Own the field inventory, compiler-neutral descriptor, validation, and adapter
compatibility tests. Do not edit AMD isel or add consumer-specific fields.

### Spark B — AMD backend allocation (G2)

Own logical-to-physical allocation results, static sequential mapping, overlap
proof, and fail-closed dynamic-index tests. Reuse the existing AMD allocator;
do not add raw ISA to codegen or accept LDS fallback.

### Spark C — Consumer adapters and evidence (G3-G5)

Own the consumer protocol, WMMA adapter extraction, bounded dot2/vector
consumer, wait/resource identity joins, and negative tests. Do not duplicate
the lifecycle or register allocator.

### Sequenced integration

1. Review G0-G2 contracts together and resolve field ownership.
2. Merge G3 only after the neutral descriptor is stable.
3. Prove one non-WMMA consumer in G4 before machine-search changes.
4. Run G5 evidence gates, then expose G6 search fields.

## Non-goals and stop conditions

- Do not rewrite the compiler or create a second GEMM scheduler.
- Do not expose AMD register numbers in compiler-opt modules.
- Do not treat WMMA lane maps as universal GEMM layout rules.
- Do not lower dynamic VGPR indexing by silently using LDS or indirect memory.
- Do not promote any consumer without final binary resources and correctness.
- Stop if a generic field cannot be validated independently for each consumer;
  keep it consumer-specific rather than weakening the contract.

## Definition of done artifacts

The final handoff must include the descriptor schema, field ownership table,
consumer adapter identities, backend allocation snapshots, wait provenance,
source/binary/resource joins, correctness results, pinned timings, and machine
search candidate identities for every admitted consumer. Existing WMMA and LDS
routes must remain green throughout.
