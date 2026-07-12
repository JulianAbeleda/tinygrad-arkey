# Pure Register Runtime Integration Scope

Date: 2026-07-12

## Current Boundary

The native compiler graph now lowers the sequential register candidate through
the normal AMD rewrite. The graph fix is committed in `37d7b3b46` and covers
both full-width WMMA A/B carriers and accumulator-contract reads.

This is not yet a runtime pure route. The current route path is:

1. `route_pf16_graph_gemm` resolves a `FullKernelAdmission`.
2. `runtime_specs.admit_full_kernel_candidate` always constructs an LDS
   `KernelStage1PipelinePlan` and `KernelCandidateContext`.
3. `_install_candidate_matmul` installs the LDS warmstart and returns ordinary
   matmul transport.
4. `postrange.py` sees only the LDS pipeline object, so its register adapter
   branch is not reached by the runtime candidate path.

The existing register implementation is reusable: `RegisterPipeTemplate`,
`RegisterStorageAdapter`, `build_stage1_uop_graph_with_storage`, the typed wait
contract, AMD VGPR mapping, and the existing resource/evidence gates are all
already present. Do not add a second scheduler, candidate admission system, or
runtime-specific register lifecycle.

## Minimal Integration Seam

Add a typed storage choice to the admitted candidate context while preserving
the existing candidate identity and geometry joins.

### Admission

- Reuse the existing payload schema and exact-shape/target validation.
- Derive the storage policy from a typed residency field, not an environment
  flag or route-name string.
- For `global_register_resident`, attach a `RegisterPipePlan` (sequential,
  static slot, targeted vmcnt) to `KernelCandidateContext.pipeline`.
- Do not allocate `KernelStage1PipelinePlan` LDS windows for this branch.
- Keep LDS candidates byte/graph-equivalent to the current path.
- Reject register candidates that request two-slot or dynamic VGPR addressing;
  gfx1100 has no indirect VGPR indexing. The first executable route is the
  one-slot sequential plan.

### Route Installation

- Split `_install_candidate_matmul` at the storage-policy boundary.
- Keep the current LDS installation unchanged.
- For register storage, install only the compiler-owned postrange options and
  candidate context. Do not call the raw `build_gemm_pipe` or LDS warmstart.
- The returned expression remains the ordinary `a @ bt.transpose()` graph so
  scheduling, `to_program`, correctness, and runtime invocation are reused.
- Record the same candidate census and canonical identity fields for both
  storage policies.

### Postrange / Compiler

- Reuse the existing `register_mode` branch in `postrange.py`.
- Verify that the admitted `RegisterPipePlan` exposes the required geometry,
  WMMA descriptor, A/B contracts, and role identity before constructing
  `RegisterPipeTemplate`.
- Preserve the static one-slot restriction and typed wait coverage.
- Run `full_rewrite_to_sink` and `to_program` with a production `KernelInfo`,
  not a synthetic graph with missing axis metadata.

## Evidence Sequence

The runtime integration is complete only in this order:

1. Candidate admission: register payload accepted; identity and exact shape
   joins pass; no LDS allocation is present.
2. Compile: production Tensor route reaches `postrange` register mode and
   emits a native AMD program. Final stream has no `Ops.VCAT`, `Ops.STACK`, or
   raw route-owned ISA pseudo-ops; typed waits lower to `s_waitcnt`.
3. Resource artifact: capture final binary/source hashes and final VGPR/SGPR,
   LDS, scratch, spill, workgroup, and wave facts. Join physical A/B mapping
   roles to the artifact.
4. Correctness: run nonconstant sampled and full-output parity against the
   existing reference authority, joined by candidate and binary hashes.
5. Timing: run kernel-only, compile-excluded, pinned-clock samples and compare
   against the established pure baseline.
6. Role rollout: repeat for `attn_qo`, `attn_kv`, `ffn_down`, and
   `ffn_gate_up`; emit explicit role attribution with no fallback.
7. Whole-model route: run the existing whole-prefill authority and verify the
   selected route census is pure for every hot role.
8. Machine search: expose only typed policy fields (`storage_kind`, wait kind,
   buffer count, static slot addressing, and generic GEMM consumer identity) to
   the existing pure-search gate.

## Hard Blockers

- No register candidate payload/admission branch exists today.
- No runtime compile artifact proves final register mapping/resources.
- Existing benchmark JSONs are LDS/hybrid or structural register evidence;
  they are not a pure register binary authority.
- Synthetic `to_program` probes can fail on incomplete `KernelInfo` metadata;
  that is not evidence against the native graph.
- GPU correctness, pinned timing, whole-model purity, and machine-search
  promotion remain unmeasured.

## Exit

This scope reaches 100% only when the existing pure-register evaluation gate
passes compile, final resources, correctness/timing, role attribution, and
machine-search stages for all selected roles. Until then, the native graph is
structurally executable but the runtime pure route remains unproven.
