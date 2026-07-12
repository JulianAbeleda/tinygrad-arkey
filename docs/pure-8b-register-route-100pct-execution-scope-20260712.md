# Pure 8B Register Route: 100% Execution Scope

Date: 2026-07-12

## Objective

Make the Qwen3-8B Q4_K_M ctx512 prefill route a real pure,
machine-searchable register-resident route on AMD gfx1100, then measure it
against the historical approximately 4.4k tok/s hybrid reference.

This scope uses the existing compiler, lifecycle, policy, descriptor, resource,
correctness, and benchmark owners. It does not create a second pipeline IR,
second allocator, second route selector, or route-owned raw ISA emitter.

## What 100% means

The effort is complete only when all of the following are true:

1. Every selected dense 8B prefill role uses the compiler-owned register route;
   no role silently falls back to a handwritten atom or LDS candidate.
2. Register stages are mapped to real AMD VGPR allocations with exact role,
   slot, fragment, lane, and lifetime ownership. No LDS staging, spills, or
   scratch are present unless explicitly measured and admitted.
3. Producer, wait, consume, overwrite, and drain edges are typed and lowered
   to the intended targeted `vmcnt` waits. A full barrier is not accepted as a
   disguised register-pipe implementation.
4. The generated full-K program lowers through the normal AMD path and emits a
   final resource artifact joined to the exact source, candidate, and binary.
5. Single-role and whole-model numerical correctness pass against the existing
   authority path.
6. Pinned-clock, synchronized whole-prefill timing is recorded for ctx512 and
   the required context set. The route either reaches the 4.4k reference or
   produces a measured, reproducible ceiling with residual attribution.
7. Machine search varies only typed compiler-owned policy fields, records exact
   candidate identity, and admits only candidates with complete evidence.

The 4.4k number is a performance target, not permission to weaken any proof or
fallback rule.

## Current measured state

- The persistent A/B two-slot register lifecycle exists in
  `tinygrad/codegen/opt/register_pipeline.py`.
- K=1, K=2, K=3, and K=256 lifecycle proofs pass.
- A valid full-K chained WMMA graph passes normal AMD rewrite.
- The earlier devectorizer crash was a malformed test ABI: `float.vec(8)` was
  paired with an empty C/output contract. It was corrected in `6476cb4f3`.
- AMD ISA now has a static, sequential one-slot mapping for the current WMMA
  stage contract. The mapping is fail-closed for dynamic or double-buffered
  VGPR indexing and uses the existing pinned fragment/allocator seam. The full
  graph still hits a devectorizer/stack-carrier failure before native AMD
  emission, so this is an isel proof, not a runnable binary.
- AMD has no general indirect VGPR addressing. The current symbolic
  `slot = rng % 2` index therefore cannot be lowered by simply assigning a
  physical base. R1 must choose either static slot expansion/unrolling or a
  proven sequential schedule that does not require indirect register indexing.
  Mapping the dynamic index to LDS, inventing an indirect VGPR instruction, or
  silently using one shared value is not an acceptable fix.
- The postrange candidate path now reuses `RegisterStorageAdapter` and the
  existing stage-1 graph builder for the sequential register policy. This is
  structural integration only: native AMD lowering still has no retained
  `Ops.WAIT` emission (the native regalloc pseudo-op path drops it), and final
  resource/correctness evidence is therefore not available.
- A sequential one-buffer register schedule now exists as a structurally proved
  fallback (`69253c3f7`): it uses compile-time slot zero and orders each
  overwrite after accumulator updates. It passes normal graph rewrite, but it
  still needs the same backend mapping from register-buffer values to pinned
  VGPR fragments and removes load/compute overlap, so it is a feasibility path,
  not yet a performance candidate.
- The current pure 8B whole-model authority is approximately 1.5k tok/s at
  ctx512; the generated gate/up two-buffer result is approximately 2.4k tok/s.
  The historical hybrid reference is approximately 4.4k tok/s.

## Existing owners to reuse

| Concern | Existing owner | Do not duplicate |
|---|---|---|
| Policy identity and storage/wait/resource composition | `tinygrad/codegen/opt/compiler_policies.py` | New route-specific policy schema |
| Descriptor, range, CONTRACT, and lane validation | `tinygrad/codegen/opt/kernel_lds.py` | Copied WMMA descriptor validators |
| Epoch/slot lifecycle and proof | `tinygrad/codegen/opt/kernel_pipeline.py` | A second lifecycle state machine |
| Register lease and gfx1100 descriptor | `tinygrad/renderer/isa/amd_register_allocator.py` | Hard-coded register windows in route code |
| AMD physical wait analysis | `tinygrad/renderer/isa/amd.py` | Route-owned `Ops.INS` wait instructions |
| Final resource artifact and evidence gates | `tinygrad/codegen/opt/amd_resource_artifact.py`, `extra/qk/prefill/pure_register_evaluation_gate.py` | Host-only estimates presented as final facts |
| Pinned benchmark authority | existing `extra/qk/prefill/*whole_synced.py` harnesses | A new unsynchronized benchmark script |

## Phases and acceptance gates

### R0: Freeze the baseline and failure

Capture the current valid full-K graph, the exact AMD ISA failure, register
window limits, and the pinned 8B baselines. Store source hashes and command
lines. This prevents agents from debugging the old malformed WMMA fixture.

**Exit:** reproducible graph-rewrite pass, reproducible ISA failure, pinned
baseline artifact, and no dirty unrelated changes.

### R1: Make register stage buffers a real VGPR resource

Trace the existing AMD isel path from `DEFINE_REG` through `isel_index`, WMMA
fragment selection, register leases, and final allocation. Define one typed
stage-buffer contract for A and B:

- role and stage slot;
- half-element width and byte width;
- physical VGPR span and alignment;
- fragment/lane mapping;
- producer and consumer lifetime;
- accumulator and stage-buffer non-overlap;
- spill/scratch prohibition.

Correct the role-specific buffer sizes. For the current 2x2 pipe, each role
needs `2 * pipe_role_fragments * 16` half elements, not one combined A+B
buffer. Route code must request leases through the existing AMD allocator;
physical register numbers remain backend-owned.

The current graph uses a symbolic alternating slot. Because gfx1100 cannot
address VGPRs indirectly, this phase must also select and prove one of:

- compile-time/static slot expansion with explicit physical slot ownership and
  loop ordering; or
- a one-buffer, no-prefetch schedule whose wait and overwrite proof is still
  valid and whose measured cadence is acceptable.

Do not lower the symbolic modulo expression as a memory address and call the
result register-resident.

**Exit:** full-K AMD isel emits VGPR-backed stage fragments for both roles;
no LDS instructions are emitted for stage buffers; allocator and overlap tests
pass; no raw ISA is introduced in `register_pipeline.py`.

### R2: Prove register-buffer and fragment ABI ownership

Add structural checks at the compiler boundary for exact A/B buffer widths,
slot offsets, half.vec(16) loads, CONTRACT axes, WMMA C output axes, and
fragment lifetime. Reject malformed output contracts before devectorization.
Keep the valid chained WMMA rewrite regression and add negative tests for
scalar/vec8 mismatches, wrong slot, missing producer readiness, and extra
register definitions.

**Exit:** malformed ABI fails with a typed compiler error; valid K=1/K=2/full-K
graphs lower identically through the normal rewrite; no generic devectorizer
workaround is needed.

### R3: Lower typed targeted waits

Use the existing `WaitDependency`, `WaitCount`, and AMD LLVM wait intrinsic
seams. Connect lifecycle producer/load-group/consumer edges to backend wait
selection. Preserve provenance through graph rewrites and reject untagged or
coverage-incomplete waits.

The generated graph may not claim targeted-vmcnt policy while emitting only a
full barrier. If the backend temporarily supports only a barrier, classify it
as a measured barrier ceiling and keep it out of pure promotion.

**Current status:** A/B per-stage wait edges and typed `WaitCount` nodes exist
in the graph and LLVM path. Native AMD still needs an explicit lowering or a
fail-closed rejection; until then the emitted mechanism does not match policy
for the native route.

**Exit:** A/B per-stage wait edges are present in the final graph and final
artifact, the emitted wait mechanism matches policy, and tests prove duplicate,
missing, and wrong-stage waits fail closed.

### R4: Connect the register adapter to postrange safely

The adapter is now connected at the existing
`build_stage1_uop_graph_with_storage` boundary. It resolves a typed
`PipelinePolicy` and logical register-stage plan and does not allocate a
zero-sized or fake LDS placeholder. Existing LDS postrange behavior remains on
its original branch.

Candidate context must carry geometry, policy, wait coverage, and exact identity
as typed data. Do not parse an opaque payload tuple in lowering. Keep the
current fail-closed error until R1-R3 evidence is available.

**Current status:** the register candidate reaches the common postrange graph
builder, but the full-K graph has not passed native AMD emission. The phase is
not complete until that binary path and its waits are proven.

**Exit:** an admitted register candidate reaches normal postrange, emits the
same valid full-K graph tested in R2, and LDS candidates remain unchanged.

### R5: Produce final resource and identity evidence

Capture final AMD source, binary, candidate identity, target, ABI, VGPR, SGPR,
LDS, scratch, spill, wait, and role facts in one `AMDResourceArtifact`.
Join source and binary hashes. Reject host estimates, missing roles, spills,
unknown register counts, stale binaries, and mixed fallback attribution.

**Exit:** `pure_register_evaluation_gate` admits one role only from a final
artifact and rejects every incomplete or mismatched fixture.

### R6: Single-role correctness and timing

Start with `attn_qo` at the exact 8B shape `(M,N,K)=(512,4096,4096)`, then run
the existing full-output correctness authority and pinned isolated timing.
Compare against the generated LDS route and the hybrid teacher only as
diagnostic references. Check clock pins, warmup/capture exclusion,
synchronization, and output parity.

**Exit:** one role has a valid binary, zero-error output, final resources, and
repeatable pinned timing. No whole-model claim is made from this role alone.

### R7: Roll out the remaining dense roles

Reuse the same adapter and backend mapping. Instantiate exact candidates for
`attn_kv`, `ffn_down`, and `ffn_gate_up` only where their shapes, quantized
loads, CONTRACTs, and resource budgets are independently proven. Keep role
policies and identities separate; do not broaden the attn_qo candidate by
shape substitution.

**Exit:** every selected role has independent correctness, wait, resource,
identity, and pinned timing evidence.

### R8: Assemble the strict pure 8B route

Bind all selected role candidates into the existing whole-prefill authority.
Require complete role attribution, no hybrid fallback, no hidden oracle
rollback, and matching source/binary identities. Run ctx512 first, then the
established context set.

**Exit:** the whole-model route is demonstrably pure and reproducible, even if
the 4.4k target is not yet reached.

### R9: Close the performance gap

Run pinned-clock synchronized measurements against the frozen baselines. Use
per-role timing and final resource facts to rank residuals: VGPR occupancy,
wait cadence, memory traffic, WMMA issue cadence, launch/epilogue overhead,
and quantized decode/unpack cost. Change one typed policy variable at a time.

**Exit:** 4.4k is reached, or a documented measured ceiling explains the
remaining delta with no unmeasured fallback or clock ambiguity.

### R10: Enable pure machine search

Expose only compiler-owned fields already proven by R1-R9: tile geometry,
logical stages, buffer widths, role mapping, wait policy, and supported
resource limits. Generate exact identities, compile every candidate, attach
final artifacts, run correctness and pinned timing, and admit only complete
pure candidates.

**Exit:** machine search can select a pure candidate for each role and the
whole-model authority can reproduce the selected result from its identity.

## Agent execution order

### Parallel low-cost investigations

- **Spark A — AMD VGPR mapping:** R0/R1. Own `amd.py` isel and allocator seam;
  no route-policy edits.
- **Spark B — ABI and wait contract:** R2/R3. Own typed validation, wait
  provenance, and negative tests; no physical register-number edits.
- **Spark C — evidence and harness:** R5/R6 scaffolding. Own artifact joins,
  correctness/timing fixtures, and baseline capture; no backend lowering edits.

### Sequenced integration

1. Review and merge A/B/C findings; reject duplicate abstractions.
2. Implement R4 only after R1-R3 contracts are explicit.
3. Run R6 on `attn_qo`; stop if binary/resource/correctness is not real.
4. Roll out R7 role-by-role, with independent artifacts.
5. Assemble R8, then R9 performance closure.
6. Enable R10 machine search last.

## Stop conditions

Agents must stop and report a concrete blocker when:

- a proposed fix requires route-owned raw ISA or a second allocator;
- a generic devectorizer or renderer change would weaken malformed-ABI checks;
- targeted waits cannot be proven from lifecycle provenance;
- final VGPR/SGPR/spill facts cannot be extracted from the binary;
- correctness or timing requires hidden fallback, unpinned clocks, or
  unsynchronized measurement.

Each agent must commit only scoped changes, include focused tests, report the
exact command and artifact paths, and leave unrelated work untouched.

## Definition of done

The final handoff must link the commits and artifacts for R0-R10, state the
selected pure candidate identities, show per-role and whole-model correctness,
show final AMD resources and waits, and include pinned ctx512 performance. If
the 4.4k line is missed, the handoff must contain a reproducible residual
attribution rather than a qualitative explanation.
