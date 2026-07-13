# Pure-register/direct-L2 completion scope

## Goal

Promote the `attn_qo` gfx1100 register-resident path only after one identity-joined
final binary proves zero LDS, scratch, and spills; then establish whether direct-L2
is better than the preserved LDS fallback under guarded hardware measurement.

## Phase 1 — final compile artifact (current)

The CPU-only `attn_qo` native compile is proven at commit `97434655b` (72 live
VGPRs in a 95-register pool; no spill path). The missing authority boundary is a
compiler-emitted record that joins final source, ELF, descriptor resources,
post-regalloc physical intervals, final instruction text, target/ABI, candidate
identity, and launch geometry.

Required implementation:

1. Make final ELF descriptor/resource metadata available from the AMD assembly
   boundary without creating an `AMDProgram`, allocating memory, or dispatching.
2. Export final allocator intervals from the post-regalloc linear program, with
   logical A/B/C/stage roles and physical VGPR/SGPR intervals.
3. Return final disassembly as data, or provide a deterministic pure disassembler
   with a recorded tool identity; do not accept printed/debug-only text.
4. Construct an `AMDCompileCapture` record at `do_assemble`, then adapt it through
   `capture_final_program_compile_only` and the existing evaluation gate.

### Phase 1 execution plan

1. **Renderer record.** `AMDISARenderer.asm` retains the exact scheduled,
   wait-inserted, label-resolved instruction list used for the emitted ELF.
   `compile_capture(prg, lin, binary)` consumes that exact `binary`; it must not
   assemble a second code object or create an `AMDProgram`.
2. **Descriptor facts.** Parse the exact ELF descriptor for VGPR, LDS,
   workgroup-thread, and wave facts. Scratch/spill facts are accepted only from
   final allocator state; descriptor absence is not evidence of zero spills.
3. **Allocator role evidence.** Export final fixed leases as role-bearing
   lifetime segments, not inferred register ranges. Required current roles are
   A and B one-slot register-stage leases and every C WMMA accumulator fragment.
   Each record includes bank, half-open range, purpose, owner, slot/fragment or
   subtile identity, fixed status, and lifetime segment. ABI registers (`v0`,
   kernarg SGPRs), alignment padding, and optional packing scratch are recorded
   separately and never attributed to A/B/C.
4. **Reuse rule.** A/B intervals may reuse physical registers at distinct times.
   Capture must preserve their segment boundaries; a union range alone is not
   proof of simultaneous ownership. Dynamic/multi-slot stage indexing is
   rejected for this one-slot milestone.
5. **Disassembly.** Return text and tool/failure identity from the exact ELF.
   A missing disassembler, failed disassembly, malformed stage provenance, or
   any LDS instruction blocks promotion.
6. **End-to-end compile-only gate.** Compile the production-shaped
   `(attn_qo, 512, 4096, 4096)` Tensor route, retrieve its capture attachment,
   and pass it through `capture_final_program_compile_only` and `evaluate`.
   The test may pass only with all final authorities; partial records must
   produce a precise fail-closed error.

Acceptance:

- source and binary hashes are joined to the candidate identity;
- descriptor authority is `final_code_object_descriptor`;
- allocator authority is `final_regalloc`;
- target is gfx1100 and ABI is amdgpu_kernel;
- LDS, scratch, VGPR spills, and SGPR spills are all zero;
- final instruction proof establishes global-load → vmcnt(0) → register-stage →
  WMMA and no LDS instruction;
- the capture path has no runtime program creation or dispatch.

Phase 1 exit produces one immutable artifact plus a negative-test matrix for
missing descriptor facts, missing A/B/C leases, overlapping or ambiguous reuse,
spills/scratch, stale binary identity, failed disassembly, malformed waits, and
LDS presence. Only this exit unlocks runtime route binding.

## Phase 2 — production route admission

Bind the exact artifact to the existing `global_register_resident` candidate route.
Admission must fail closed when the identity, artifact, route role/shape, or
resource proof differs. Existing LDS route selection remains the fallback.

Acceptance:

- production Tensor route reaches the candidate-specific postrange plan;
- no `DEFINE_LOCAL`, LDS allocation, residual staging pseudo-op, or untyped wait
  reaches the final program;
- a mismatched/stale/missing artifact cannot select the register route;
- the LDS route’s existing tests remain unchanged.

## Phase 3 — guarded hardware correctness

Only after Phases 1–2, use the promotion state machine to run bounded canaries:
one exact candidate binary, one exact shape, strict timeout/health handling, no
fallback promotion on failure, and identity-joined observations.

Acceptance:

- numerical parity against the trusted reference within predeclared tolerances;
- finite outputs, watchdog/health checks, reset/recovery report on failure;
- captured binary identity equals the dispatched binary identity;
- no automatic broad rollout.

## Phase 4 — direct-L2 versus LDS decision

Benchmark the proven direct-register/direct-L2 binary and its equivalent LDS
fallback under the same shape, clocks/environment, warm-up, repetitions, and
correctness binding.

Required measurements: latency distribution, variance, occupancy/resource facts,
and available L2/memory counters. Do not infer L2 residency from the absence of
LDS.

Promotion acceptance:

- direct path is correct and stable;
- direct path is materially faster under the declared decision threshold;
- no regression in timeout, variance, or required occupancy;
- otherwise retain LDS as the promoted route and record the direct path result.

### CPU-only paired decision and machine-search specification

The research-plane authority is `extra/qk/prefill/pure_register_direct_l2_decision.py`.
It validates captured records and computes a decision; it never imports a device,
creates a program, or dispatches hardware.

Each candidate is schema `pure-register-prefill-candidate.v1` and contains:
`role`, exact `shape={m,n,k}`, shared `canonical_identity`, distinct
`binary_sha256`, `storage` (`direct_l2` or `lds`), and passing `artifact` and
`correctness` authorities. A pair is admissible only when all identity fields
match except binary identity, both artifacts are final/passing, both correctness
records pass, and both records have at least 9 positive finite kernel-only
latency samples. Samples must be paired by warmup, clock/environment, repetition
protocol, commit, target, ABI, launch geometry, and input/output binding. Any
missing or unequal join is `status=blocked`, never a win.

The machine-search result records both medians, population coefficient of
variation, speedup, thresholds, and blockers. Counter evidence must be live and
identity-joined for `l2`, `memory`, and `compute`; unavailable counters are
blocked rather than zero. Direct-L2 is promoted only when speedup is at least
3% and its CV is no more than 1.25 times LDS CV. Otherwise a complete pair
returns `retain_lds`. Correctness, artifact, identity, sample, or counter gaps
return explicit `blocked` results. This is a decision artifact, not route
admission; LDS remains the fallback until a separately authorized guarded
runner exists.

## Phase 5 — role expansion and whole-prefill attribution

Repeat Phases 1–4 independently for `attn_kv`, `ffn_down`, and `ffn_gate_up`.
Whole-prefill attribution is allowed only when each active role has an
identity-joined artifact, correctness result, and measured route decision.

## Safety rules

- CPU-only compilation and static inspection are always permitted.
- GPU dispatch is prohibited until Phase 1 and Phase 2 gates pass.
- Every passing milestone is tested, committed, and pushed.
- A failed experiment is reverted or retained only as default-off diagnostics;
  it must not alter candidate admission or hardware dispatch.
