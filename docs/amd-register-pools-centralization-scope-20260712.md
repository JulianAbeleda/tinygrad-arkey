# AMD register pools and register-pipe centralization scope

Date: 2026-07-12

## Objective

Centralize register allocation facts and register-resident pipeline interfaces
so every kernel section uses the same typed tools, while keeping AMD/RDNA3
physical register assignments in the AMD backend.

The goal is not to move every register number into generic Tinygrad code. The
goal is:

```text
logical register role
  -> target register-bank contract
  -> allocator reservation/lease
  -> physical register artifact
  -> resource/ABI proof
```

No route or kernel section should independently hard-code a physical range for
the same logical role.

## Current inventory

### Fixed ABI registers

Current owner: `tinygrad/renderer/isa/amd.py`.

- `s0:s1`: kernarg pointer;
- `s2:s4`: workgroup IDs;
- `v0`: packed workitem ID;
- `s5`: EXEC save/restore scratch in gated stores;
- `VCC`, `EXEC`, and `SCC`: implicit architectural state.

These remain target/backend-owned, but must be exposed through a typed target
ABI descriptor instead of repeated literals.

### SGPR pools

Current hard-coded pools:

- pointer pairs `s6:s38` (`SPTR_POOL`, even aligned);
- scalar loop counters `s40:s63` (`SCNT_POOL`);
- scalar address/math temporaries `s64:s103` (`SCALAR_TMP`).

These need one allocator-facing bank descriptor with width, alignment,
reserved intervals, and use class.

### VGPR pools and reservations

Current hard-coded regions:

- all virtual VGPRs `v0:v255` (`VBASE`);
- `v0` workitem ID;
- optional pinned accumulator window `v1:v16`;
- WMMA fragment window `v200:v237`;
- LDS packing scratch `v232:v235`;
- low resident A/B and C windows derived below the fragment window for
  multi-output WMMA.

The allocator currently spreads this policy across `_vpool`, `_accum_pin`,
`_frag_base`, `_acc_base`, `_ab_base`, `_n_ab_frags`, and `_pin`. Those helpers
must converge on one reservation/lease API.

### Logical roles already present in code

- `kernarg_ptr`;
- `workgroup_id`;
- `workitem_id`;
- `loop_counter`;
- `scalar_address_tmp`;
- `accumulator`;
- `wmma_fragment_a`;
- `wmma_fragment_b`;
- `wmma_accumulator_c`;
- `lds_pack_scratch`;
- `wait_state` / dependency metadata.

The register-pipe graph should use these logical roles. It must not request
`v200` or `s40` directly.

## Centralized architecture

### 1. Logical register contract

Add a compiler-owned, target-neutral register-role contract in the codegen
optimization layer. It should describe:

- role name;
- register bank (`SGPR`, `VGPR`, implicit state);
- width in scalar registers;
- alignment;
- lifetime (`entry`, `loop`, `stage`, `kernel`);
- sharing policy (`exclusive`, `stage_reuse`, `resident`);
- required ABI/resource facts.

Example roles:

```text
RegisterRole("wmma_a", VGPR, width=8, align=8, lifetime="stage", residency="resident")
RegisterRole("accumulator_c", VGPR, width=8, align=8, lifetime="loop", residency="pinned")
RegisterRole("kernarg", SGPR, width=2, align=2, lifetime="entry", residency="fixed")
```

The contract contains no AMD physical index.

### 2. Target register-bank descriptor

Add an AMD-owned target descriptor, for example
`tinygrad/renderer/isa/amd_registers.py`, containing:

- fixed ABI assignments;
- SGPR/VGPR bank limits;
- reserved intervals;
- alignment rules;
- wave/register accounting rules;
- target-specific opcode constraints.

`AMDISARenderer` consumes this descriptor. Other backends may provide their
own descriptor or reject the register policy.

### 3. Reservation/lease allocator interface

Replace independent helpers with one typed interface:

```text
reserve(role, width, align, lifetime, key) -> RegisterLease
lookup(key) -> RegisterLease
release(lease) -> None
available(bank) -> intervals
```

The allocator must reject overlap, misalignment, bank overflow, and lifetime
violations. A lease records logical role, physical interval, owner key,
provenance, and whether it is visible to ordinary linear-scan allocation.

Existing behaviors map as follows:

- `_accum_pin` -> pinned `accumulator_c` lease;
- `_frag_base` -> resident `wmma_a/b/c` lease;
- `_ab_base` -> resident multi-tile A/B lease;
- `_pin` -> constrained lease attachment;
- `_vpool` -> free-bank calculation after reservations;
- `SPTR_POOL`/`SCNT_POOL`/`SCALAR_TMP` -> SGPR bank leases.

### 4. Register artifact and resource proof

Every lowered register kernel emits a typed artifact containing:

- role-to-physical interval map;
- reserved/free intervals;
- VGPR/SGPR high-water marks;
- LDS/scratch/spill facts;
- ABI fixed-register usage;
- source/binary hash;
- target and wave size;
- candidate identity.

This is the only data consumed by resource gates and machine search. Search
does not inspect or generate physical register literals.

### 5. Register-pipe storage adapter

`tinygrad/codegen/opt/register_pipeline.py` owns logical producer/fragment
construction and uses register-role leases indirectly through the backend
artifact. It must remain free of `AMDOps`, raw instruction constructors, and
physical `vN`/`sN` values.

The adapter is responsible for:

- zero-LDS policy;
- A/B fragment carrier ABI;
- logical stage identity;
- producer/consumer dominance;
- wait dependency metadata.

The backend is responsible for realizing leases and proving final resources.

## Dependency-ordered implementation phases

### G0 - Inventory and freeze

Record every physical literal and helper in `amd.py`. Classify each as fixed
ABI, allocator pool, reservation, temporary, or instruction operand. Add a
snapshot test for the current no-flag renderer behavior.

Exit: no unexplained register literal remains in the inventory.

### G1 - Logical role types

Implement immutable `RegisterRole`, `RegisterBank`, `RegisterLifetime`, and
`RegisterLease` contracts in a compiler-neutral module. Add overlap, alignment,
width, and lifetime validation. No backend behavior changes.

Exit: logical register-pipe and existing policy tests can describe roles
without physical indices.

### G2 - AMD target descriptor

Move fixed ABI and pool facts into an AMD descriptor. Keep compatibility names
in `amd.py` temporarily, but derive them from the descriptor. Add target tests
for gfx1100 wave32 and unsupported target rejection.

Exit: one source of truth for AMD register banks and fixed reservations.

### G3 - Reservation allocator migration

Implement the lease API and migrate, in this order:

1. fixed ABI reservations;
2. SGPR pools;
3. pinned accumulators;
4. WMMA fragment A/B/C windows;
5. LDS pack scratch;
6. virtual pool calculation;
7. constrained `_pin` attachments.

Each migration keeps a byte-identical or artifact-equivalent no-flag output
test. Do not migrate all helpers in one unreviewable rewrite.

Exit: `_accum_pin`, `_frag_base`, `_acc_base`, `_ab_base`, and `_vpool` either
delegate to the lease allocator or are deleted.

### G4 - Register artifact/resource join

Emit the role-to-physical map and final resource facts from the backend. Join
them to source/binary/candidate identity. Reject unknown registers, overlaps,
spills, and resource overflow.

Exit: the existing native register path produces a complete artifact without
changing its route selection.

### G5 - Register-pipe adapter integration

Connect `RegisterPipeTemplate` to normal postrange only after G1-G4. Use the
logical role contract and shared WMMA validation. Preserve the existing LDS
adapter and stage1 semantics. Prove logical two-stage readiness, tails, and
typed wait dependencies.

Exit: `attn_qo` compile-only graph contains no local allocation or physical
register literals and produces a joined register artifact.

### G6 - Wait/resource backend gates

Lower dependency-derived waits through the typed wait seam. Ensure native AMD
physical wait analysis and AMDLLVM `WaitCount` lowering consume the same typed
metadata. Reject arbitrary wait markers and duplicate counter state.

Exit: source and binary prove wait placement, stage coverage, and final
resource identity.

### G7 - Role expansion and authority

Expand register roles to `ffn_down` and `attn_kv` only after `attn_qo` passes
correctness/resource gates. Fix whole-prefill role attribution and require
fallback-free pure evidence before machine search.

Exit: every selected role has a logical register plan, physical artifact,
correctness result, and pinned timing.

## Parallel work packets

The lowest-cost non-overlapping agents are:

### Agent A - Register inventory and target descriptor

Own G0-G2. Only touch the new logical-role module, AMD descriptor, and focused
tests. Do not modify register allocation behavior yet.

### Agent B - Lease allocator migration

Own G3. Work only in the AMD backend allocator helpers and allocator tests.
Migrate one reservation family at a time and preserve baseline artifacts.

### Agent C - Artifact/resource join

Own G4. Work only in backend manifests/resource extraction and fail-closed
evaluation gates. Do not change allocation or register-pipe graph generation.

After A completes, B and C can run in parallel. G5 depends on A+B. G6 depends
on G5 plus the existing wait seam. G7 is sequential and requires final binary
and correctness evidence.

## Hard boundaries

- Generic code never names `v200`, `v16`, `s40`, or other physical registers.
- AMD target code owns physical register numbers and instruction constraints.
- Register leases are the only interface between logical roles and physical
  intervals.
- The register pipe never constructs `Ops.INS` or raw ISA payloads.
- Existing LDS and no-flag native routes remain regression oracles.
- A register artifact with unknown VGPR/SGPR, spill, or missing identity is not
  searchable or promotable.

## Completion definition

Centralization is complete when all register consumers use logical roles and
the lease/artifact APIs, the AMD descriptor is the only physical-register
source of truth, existing routes preserve their baseline output, and the pure
register pipe can consume the same artifacts for resource, wait, correctness,
timing, and machine-search gates.

## Spark execution status (2026-07-12)

- **G0-G2 complete:** `b7d9d6e12` adds logical register roles/banks/leases and
  the gfx1100 descriptor. This is metadata-only and does not alter renderer
  output.
- **G3 complete:** `c50317551` moves the lease allocator into
  `tinygrad/renderer/isa/amd_register_allocator.py` and makes the research
  `wmma.py` layout reuse the canonical lease types and gfx1100 capacities.
  Legacy register windows remain unchanged.
- **G4 complete:** `ead74597e` plus `1e022a7ea` add the fail-closed physical
  register/resource artifact and reuse the shared register-bank enum. The
  artifact joins role intervals, resource facts, target/ABI, source/binary,
  and candidate identity.
- **G5-G7 remain pending:** the logical register-pipe graph is still a
  structural scaffold, and final postrange integration, executable waits,
  correctness, timing, and role expansion remain gated on the pure register
  lifecycle/WMMA ABI.

The centralization work is therefore complete through the allocator and
artifact boundary, but it does not claim an executable pure register GEMM.
