# 14B Q4_K/Q8_1 MMQ: Pure Machine-Search Scope

Status: scoped, research-only, default route unchanged

Date: 2026-07-14

Target: AMD gfx1100, Qwen3-14B prefill

## Decision

The MMQ effort is realigned to the same machine-search principles used by the
successful 8B generated routes.

No handwritten kernel schedule may become a production or promoted route.

Human-authored code may define the legal Q4_K/Q8_1 operation grammar, the
correctness reference, compiler lowering rules, and safety constraints. The
machine must generate, compile, validate, benchmark, and select the schedule
that becomes a route candidate.

The current bounded UOp atom remains a research probe only. It is not the final
implementation and must not be promoted by adding more hand-selected geometry
or route logic around it.

## Objective

Build a reusable, descriptor-driven Q4_K x Q8_1 MMQ primitive that follows the
existing 8B route lifecycle:

```text
Q4/Q8 primitive grammar
  -> machine-generated candidate descriptors
  -> generated UOp schedule
  -> tinygrad AMD lowering
  -> isolated correctness gate
  -> resource/ISA gate
  -> same-session benchmark
  -> machine-selected winner
  -> centralized route manifest
```

The first target is a Q4_K `ffn_gate_up` prefill role. The implementation must
remain shape- and role-agnostic in its interfaces so that additional 14B roles
and Q6_K can reuse the infrastructure later.

## Why the scope changed

The 8B path already had a complete lifecycle:

```text
search space -> candidate payload -> compiler lowering -> evidence -> winner
```

The MMQ path initially had only:

```text
llama reference -> no tinygrad MMQ grammar -> bounded UOp feasibility probe
```

The bounded probe was useful for answering feasibility and safety questions,
but it manually selected important schedule decisions: `16x16x256` geometry,
lane mapping, wave behavior, LDS layout, reduction structure, and writeback.
That is generated code, but it is not machine-searched code.

The implementation mistake was allowing a research probe to resemble a route
implementation. The probe must now be quarantined while the missing MMQ
grammar/search layer is built.

## Definitions

### Allowed human-authored content

- Q4_K packed-byte and scale/minimum semantics.
- Q8_1 and DS4 activation semantics.
- Legal WMMA/dot-product primitives.
- ABI and output-layout definitions.
- Correctness reference implementation.
- Search dimensions, bounds, and rejection rules.
- Compiler lowering and safety rules.
- Evidence and promotion policy.

### Machine-owned decisions

- Tile M/N/K sizes.
- Workgroup and wave geometry.
- Accumulator count and placement.
- Register versus LDS staging.
- Q4/Q8 panel layout.
- K-slice lifecycle.
- Barrier and wait placement among legal schedules.
- Writeback ownership strategy.
- Compiler options.
- Candidate promotion.

### Forbidden in a promoted route

- A fixed `16x16x256` schedule.
- A Python function whose final route body directly encodes one geometry.
- Manually selected lane ownership.
- Manually selected LDS layout or buffer count.
- Raw AMD ISA or inline assembly.
- A route branch that directly selects one kernel function.
- A human-selected winner that has no search artifact.
- A hidden direct-packed fallback while reporting MMQ execution.

The primitive grammar itself is not considered a handwritten kernel. It is the
search vocabulary. A candidate schedule becomes a kernel only after the
machine-generated descriptor is lowered and evaluated.

## Current repository state

### Reusable and retained

- `extra/qk/mmq_q4k_q8_reference.py`
  - canonical Q4_K/Q8_1 numerical reference;
  - layout and edge-case tests.
- `extra/qk/mmq_llama_oracle.py`
  - translated llama structural oracle;
  - owner and fragment mapping reference;
  - not a production backend.
- `extra/qk/mmq_owner_coverage.py`
  - owner-map and duplicate/missing-store validation.
- `extra/qk/mmq_coop_tile_harness.py`
  - isolated execution, guard, timeout, GPU-health, and provenance checks.
- `extra/qk/mmq_machine_search.py`
  - candidate/evidence search surface and fail-closed promotion checks.
- `extra/qk/q4k_prefill_route_spec.py`
  - existing descriptor-driven Q4 prefill pattern.
- `extra/qk/q6k_prefill_route_spec.py`
  - existing descriptor-driven Q6 prefill pattern.
- `extra/qk/generated_route_registry.py`
  - descriptor and emitter registry.
- `extra/qk/route_manifest.py`
  - centralized route identity, status, provenance, selector, and rollback.
- `tinygrad/llm/model_route_plan.py`
  - model-facts-to-primitive route planning.
- `tinygrad/llm/qk_primitives.py`
  - quantized primitive installation and execution boundary.

### Research-only and quarantined

- `extra/qk/mmq_q4k_q8_atom.py`
  - bounded UOp feasibility probe;
  - current shape and schedule assumptions are not promotion-eligible;
  - the latest isolated AMD execution exposed guard corruption and NaNs.
- `extra/qk/mmq_atom_boundary.py`
  - fail-closed boundary stub;
  - must not become the long-term MMQ route contract.
- `extra/qk/mmq_role_adapter.py`
  - temporary `ffn_gate_up` admission adapter;
  - must be generalized or replaced by the shared candidate contract.

### Current hard blocker

The current bounded emitted candidate dispatches but is not correct on real
AMD. The latest isolated run reported:

- dispatch completed within the timeout;
- GPU health passed before and after;
- owner geometry was statically complete;
- guard corruption occurred after dispatch;
- NaNs appeared in the output;
- direct-packed timing was correctly not attempted.

This is an ABI/launch/geometry correctness blocker, not a performance result.

## Reuse architecture

### Common primitive descriptor

Add or factor a small common descriptor layer, preferably in:

```text
extra/qk/prefill_primitive_spec.py
```

The common descriptor should own only fields shared by generated prefill
primitives:

```text
workload
profile
role
quant_format
activation_format
weight_layout
output_layout
M/N/K shape
parts
target
backend strategy
schedule options
```

It must provide:

- validation;
- stable JSON serialization;
- canonical identity input;
- launch/ABI description;
- route attribution metadata.

It must not contain Q4-specific decode logic.

### MMQ descriptor

Add:

```text
extra/qk/q4k_q8_mmq_prefill_spec.py
```

The MMQ specialization adds:

```text
q4k_group_size
q8_block_size
activation_layout
tile_x_layout
tile_y_layout
tile_m
tile_n
tile_k
wave_width
workgroup_size
accumulator_slots
staging_strategy
writeback_strategy
lds_bytes
```

The descriptor must describe both bounded research shapes and real model
shapes. It may reject unsupported candidates, but it may not silently replace
one candidate with another.

### Primitive emitter

Create a descriptor-driven emitter with the shape:

```python
emit_q4k_q8_mmq_kernel(spec: Q4KQ8MMQPrefillSpec)
```

The emitter may contain reusable lowering rules for:

- Q4_K unpacking;
- Q4 scale/min reconstruction;
- Q8_1 DS4 loads;
- legal dot/WMMA operations;
- reduction primitives;
- owner-map generation;
- bounds and synchronization.

It may not choose the final schedule internally. Every schedule choice must
come from `spec`.

### Candidate generator

The search layer must generate descriptors over a constrained space, for
example:

```text
tile_m             in legal candidates
tile_n             in legal candidates
tile_k             in legal candidates
staging_strategy   in {register, lds}
accumulator_slots  in legal candidates
writeback_strategy in legal candidates
workgroup_size     in legal candidates
```

The exact candidate sets are data, not constants embedded in the emitter.

The generator must reject candidates before compilation when:

- dimensions violate Q4/Q8 alignment;
- owner coverage cannot be complete;
- LDS exceeds the target budget;
- accumulator count exceeds the resource budget;
- wave/workgroup mapping is invalid;
- the ABI cannot represent the candidate;
- required synchronization cannot be uniform.

## Dataflow contract

The first route must preserve the existing Q8_1 contract:

```text
fp16 activation
  -> shared Q8_1 quantizer
  -> Q4_K x Q8_1 MMQ primitive
```

The first MMQ kernel consumes Q8_1. Fusing fp16-to-Q8 quantization into the
MMQ kernel is a later search axis and is not required for the initial route.

The timing contract must nevertheless include required Q8 preparation when
comparing end-to-end prefill. A kernel-only benchmark must be labeled as such.

The MMQ tile dataflow is:

1. Load packed Q4_K weights through the candidate-selected path.
2. Load or stage Q8_1 DS4 activation panels.
3. Decode Q4 values and scale/minimum metadata according to the reference.
4. Accumulate using a candidate-selected legal dot/WMMA primitive.
5. Iterate over K slices.
6. Apply corrections exactly once.
7. Write each output element through exactly one owner.
8. Record ABI, geometry, resource, and binary identity evidence.

The dataflow must support partial edge tiles and real 14B dimensions without
embedding model-specific constants in the emitter.

## Implementation phases

### M0 — Freeze the 8B contracts

Deliverables:

- tests documenting the existing 8B descriptor/registry/manifest flow;
- route identity and rollback invariants;
- no changes to 8B defaults.

Exit criteria:

- existing Q4/Q6 generated route tests pass;
- no second candidate identity system is introduced;
- no MMQ route is live.

### M1 — Extract common primitive metadata

Deliverables:

- common prefill primitive descriptor;
- stable JSON and canonical identity helpers;
- common ABI/launch metadata;
- Q4/Q6 compatibility tests.

Exit criteria:

- existing `Q4KPrefillRouteSpec` and `Q6KPrefillRouteSpec` behavior is unchanged;
- existing route manifest validation passes;
- MMQ remains research-only.

### M2 — Define the MMQ semantic grammar

Deliverables:

- Q4_K decode primitives;
- Q8_1/DS4 primitives;
- legal dot/WMMA primitive descriptions;
- reference-backed owner-map generation;
- synchronization legality rules;
- resource constraints.

Exit criteria:

- grammar tests cover Q4 groups, Q8 blocks, scale/minimum corrections, and
  owner coverage;
- no geometry is selected by the grammar itself;
- the llama source remains an oracle, not a copied kernel schedule.

### M3 — Build descriptor-driven candidate generation

Deliverables:

- `Q4KQ8MMQPrefillSpec`;
- candidate enumeration;
- precompile rejection rules;
- canonical candidate payloads;
- candidate provenance.

Exit criteria:

- two descriptors with different geometry produce distinct identities;
- no candidate identity can be forged from an adapter-generated replacement;
- candidates serialize and replay deterministically.

### M4 — Refactor the emitter

Deliverables:

- `emit_q4k_q8_mmq_kernel(spec)`;
- separation of grammar, lowering, and candidate schedule;
- no hidden shape constants;
- explicit ABI and launch geometry;
- reusable owner-only writeback lowering.

Exit criteria:

- emitter accepts candidate data only;
- fixed-shape assumptions are removed from the promoted path;
- existing bounded probes remain marked research-only;
- no raw ISA or inline assembly is introduced.

### M5 — Repair emitted-kernel correctness

Use isolated canary buffers and the existing guarded executor to resolve, in
order:

1. output pointer order;
2. `gidx0/gidx1` semantics;
3. global/local geometry;
4. Q4 row/column indexing;
5. Q8 DS4 indexing;
6. staged warp reduction;
7. owner-only writeback;
8. LDS barriers and wait ordering.

Exit criteria:

- no guard corruption;
- no NaNs;
- full output matches the canonical reference;
- GPU health passes before and after;
- candidate identity and binary identity are recorded.

### M6 — Automated compile/correctness/resource search

For each generated candidate:

1. compile in isolation;
2. capture source and binary identity;
3. capture geometry;
4. capture VGPR/SGPR/LDS/scratch/spill evidence;
5. reject resource failures;
6. dispatch under timeout protection;
7. validate guards and GPU health;
8. compare the full output;
9. validate owner coverage;
10. retain only passing candidates.

No candidate may proceed to timing if correctness or resource evidence is
missing.

### M7 — Same-session performance search

Compare passing candidates against:

- direct-packed Q4 prefill;
- direct DS4 comparator where useful;
- llama timing only as directional context.

The measurement must define whether it includes:

- Q8 preparation;
- required packing;
- output reduction;
- synchronization;
- all dispatches.

Exit criteria:

- a generated MMQ candidate beats direct packed on the target role;
- the result is reproducible across repeated sessions;
- no hidden fallback occurred;
- resource and timing evidence share the same candidate identity.

If no candidate wins, the route remains blocked and direct packed stays the
default. A correct but slower candidate is not a promotion.

### M8 — Centralized one-role integration

Bind only `ffn_gate_up` first through:

```text
ModelRoutePlan
  -> PrimitiveRouteEntry
  -> route_manifest
  -> generated_route_registry
  -> candidate admission
  -> MMQ primitive execution
```

Requirements:

- explicit payload;
- canonical identity;
- role and quantization validation;
- no environment-specific schedule logic;
- no hidden direct-packed fallback;
- direct packed remains the rollback route.

`tinygrad/llm/prefill_routes.py` must not gain a one-off MMQ branch.

### M9 — Q4 role expansion

After `ffn_gate_up` passes independently, search and validate:

- `ffn_down`;
- `attn_qo`;
- `attn_kv`.

Every role requires separate correctness, resource, timing, and route-identity
evidence.

### M10 — Q6 extension

Q6_K must reuse the common descriptor, candidate, evidence, and route layers,
but use a separate Q6 semantic grammar and decoder.

Q4 success does not prove Q6 success. Q6 is complete only when it avoids
full dequant materialization and passes its own correctness/performance gates.

### M11 — End-to-end 14B validation

Record:

- prefill tok/s;
- decode tok/s;
- per-role timing;
- route census;
- candidate identities;
- output/token parity;
- memory use;
- Q8 preparation cost;
- GPU health;
- fallback count.

Compare against the current direct-packed path and the separately measured
llama.cpp workload using the same measurement definition.

### M12 — Promotion and cleanup

Only after end-to-end evidence passes:

- mark the route promoted in `route_manifest.py`;
- add the descriptor/emitter to `generated_route_registry.py`;
- update the canonical candidate artifact;
- set provenance to `machine_authored_generated` or
  `tinygrad_scheduler_generated`;
- retain direct packed as rollback;
- remove duplicate MMQ-only admission/boundary logic;
- keep the reference and isolated harness as regression infrastructure.

## Evidence artifact

Every candidate result must contain at least:

```text
schema
candidate_id
canonical_identity
source_identity
binary_identity
profile
role
quant_format
activation_format
shape
tile_geometry
launch_geometry
backend_strategy
correctness_status
owner_coverage_status
guard_status
gpu_health_status
resource_status
timing_status
same_session_comparator
fallback_status
promotion_status
rollback_route
```

Missing evidence must produce `BLOCKED_FAIL_CLOSED`.

## Test matrix

### Reference and grammar

- Q4_K group decode.
- Q4 scale/minimum correction.
- Q8_1 quantization and DS4 layout.
- K-slice decomposition.
- partial M/N edges.
- owner coverage.
- malformed candidate rejection.

### Emitter and ABI

- candidate identity changes when schedule changes;
- output/global/input ABI order;
- global/local geometry;
- no hidden shape constants;
- no invalid synchronization;
- no silent fallback.

### Real AMD

- bounded correctness;
- guard canaries;
- NaN/Inf rejection;
- repeated dispatch;
- GPU health before/after;
- resource and spill checks;
- binary identity.

### Route integration

- route admission;
- role/quant/shape matching;
- default remains direct packed when MMQ is absent;
- default remains direct packed when evidence is incomplete;
- exact candidate identity reaches the compiled program;
- no fallback while claiming MMQ;
- rollback route works.

### End to end

- `ffn_gate_up`;
- `ffn_down`;
- `attn_qo`;
- `attn_kv`;
- Q4/Q6 separation;
- full 14B prefill;
- token/output parity;
- prefill and decode attribution.

## File ownership boundaries

### Grammar and specs

- `extra/qk/prefill_primitive_spec.py`
- `extra/qk/q4k_q8_mmq_prefill_spec.py`
- existing Q4/Q6 route specs

### Lowering

- new MMQ emitter module;
- reusable lowering primitives;
- no route policy in the emitter.

### Search and evidence

- `extra/qk/mmq_machine_search.py`
- `extra/qk/mmq_coop_tile_harness.py`
- `extra/qk/mmq_owner_coverage.py`
- candidate artifacts under `bench/`.

### Route policy

- `extra/qk/route_manifest.py`
- `extra/qk/generated_route_registry.py`
- `tinygrad/llm/model_route_plan.py`
- `tinygrad/llm/qk_primitives.py`

### Must remain unchanged until promotion

- 8B promoted route behavior;
- direct-packed default;
- decode route selection;
- ordinary fallback behavior;
- GPU safety boundary.

## Stop conditions

Stop and report a blocker when:

- no legal candidate can compile without scratch/spills;
- all correct candidates lose to direct packed and the remaining search space is
  exhausted;
- correctness depends on hard-coded model-specific constants;
- ABI or geometry cannot be proven with guards;
- synchronization cannot be proven uniform;
- candidate identity cannot be propagated to the compiled program;
- the only passing path is a handwritten schedule;
- route attribution cannot distinguish MMQ from direct packed;
- required Q8 preparation is excluded from the comparison.

In all blocker cases, keep the route research-only and preserve direct packed as
the default.

## Definition of 100 percent

The effort is complete only when:

- no handwritten schedule is promoted;
- Q4_K/Q8_1 semantics are represented as reusable primitives;
- candidate schedules are machine-generated;
- geometry/staging/writeback are machine-selected;
- generated candidates pass real-AMD correctness;
- no guard corruption, NaNs, spills, or unsafe synchronization remain;
- a generated candidate beats direct packed on a real 14B role;
- `ffn_gate_up` is integrated through the shared route authority;
- remaining Q4 roles are independently validated;
- Q6 has a separate valid path or an explicit documented limitation;
- full 14B prefill evidence is reproducible;
- route provenance is generated, not hand-authored;
- rollback and fallback are centralized and tested;
- duplicate MMQ boundary/adapter code has been removed or reduced to shared
  infrastructure.

Until these conditions hold, the MMQ route remains research-only and the
existing direct-packed route remains authoritative.
