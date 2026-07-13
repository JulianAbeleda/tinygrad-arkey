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

Acceptance:

- source and binary hashes are joined to the candidate identity;
- descriptor authority is `final_code_object_descriptor`;
- allocator authority is `final_regalloc`;
- target is gfx1100 and ABI is amdgpu_kernel;
- LDS, scratch, VGPR spills, and SGPR spills are all zero;
- final instruction proof establishes global-load → vmcnt(0) → register-stage →
  WMMA and no LDS instruction;
- the capture path has no runtime program creation or dispatch.

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
