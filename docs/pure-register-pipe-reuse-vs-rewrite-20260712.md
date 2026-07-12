# Pure register pipe: reuse versus rewrite decision

## Decision

Do **not** rewrite the compiler or the AMD backend. The reusable path is already
large enough to justify an incremental implementation. The correct change is a
new register-resident storage implementation behind the existing lifecycle,
descriptor, and resource boundaries, plus one backend wait-lowering seam.

This is not a claim that the register path is executable today. It is a scope
decision: the current code has the right reusable pieces, but they are joined
by an LDS-specific adapter and the targeted wait is only emitted by the native
AMD ISA renderer. The missing work is a coordinated extension, not a second
compiler.

## Evidence inventory

### Reusable without semantic rewrite

| Concern | Existing owner | Evidence | Reuse action |
|---|---|---|---|
| Epoch/slot ownership and prologue/body/drain | `tinygrad/codegen/opt/kernel_pipeline.py` | `stage1_lifecycle_events` and `prove_stage1_lifecycle` establish producer, ready, consume, release, overwrite, and complete-drain invariants | Generalize the plan's storage assumptions; keep the proof and event model |
| Typed producer/fragment callbacks | `Stage1StorageAdapter`, `KernelStage1ProducerStage`, `KernelStage1FragmentStage` | `build_stage1_uop_graph_with_storage` already routes LDS through typed callbacks | Add a register adapter implementing the same callback results |
| WMMA descriptor and lane remaps | `tinygrad/codegen/opt/kernel_lds.py`, shared tensor-core descriptor | `PrecontractPipelineTemplate.__post_init__` validates dimensions, ranges, four binary A/B axes, folded element IDs, and descriptor remaps | Extract descriptor checks from LDS allocation checks; do not duplicate them |
| Accumulator ownership and loop-carried state | `build_stage1_uop_graph` and postrange accumulator contract | Existing graph constructs `float.vec(8)` slices, updates, drain, and exact owner coverage | Reuse the graph and WMMA callback; retain the existing accumulator contract |
| Global b128 and WMMA instruction selection | `tinygrad/renderer/isa/amd.py` | `AMDOps.GLOBAL_LOAD_B128` and `AMDOps.V_WMMA` lower to the required RDNA3 instructions | Keep instruction selection; feed it compiler-owned values rather than route-owned instruction lists |
| Register allocation/resource gates | AMD ISA renderer and resource capture | Fragment windows, accumulator pins, no-spill behavior, and final resource extraction already exist | Add a register-pipe resource budget and fail closed on unknown/overflow |
| Candidate policy/cache separation | `PipelinePolicy`, `RegisterPipePlan`, candidate context | Register and LDS policies are now distinct and immutable | Keep canonical identity and policy fields; do not add another route schema |
| Correctness and proof artifacts | existing unit/authority tests | LDS precontract and lifecycle suites are passing | Clone the test shape for register storage, not the implementation |

### Missing or currently coupled

| Gap | Why it blocks a reusable register path | Smallest change |
|---|---|---|
| Storage callback is LDS-only | `PrecontractPipelineTemplate` requires `DEFINE_LOCAL`, LDS windows, and `active_lds_bytes` | Split shared operand/descriptor validation from `LDSStorageTemplate`; add `RegisterStorageTemplate` with no local allocation |
| Lifecycle plan encodes stage-1 LDS semantics | `KernelStage1PipelinePlan` requires `stage_count == 1` and derives slots from LDS bytes | Keep physical LDS slots separate from logical register stages; add an explicit register mapping (two logical stages, zero LDS slots) rather than changing the existing plan in place |
| Wait dependency is not a compiler UOp contract | Pure AMDLLVM supports full barriers, while targeted `vmcnt` is currently inserted in `AMDISARenderer._insert_waitcnt` after raw instruction selection | Thread typed `WaitDependency` through graph metadata; add a backend hook that either lowers it or rejects it before launch |
| Native targeted waits are renderer-local | `_insert_waitcnt` tracks physical register spans after post-regalloc; route code cannot reuse it without importing `AMDOps`/`Ops.INS` | Reuse the dependency algorithm as backend implementation, not its raw instruction representation; add a typed marker/source proof at the compiler boundary |
| LLVM path lacks targeted vmcnt | `amdllvm_wait_dependency` intentionally fails closed for `targeted_vmcnt` | Keep full-barrier compile/correctness diagnostics separate; do not promote them as the performance implementation. Add targeted lowering only in a backend that can prove it |
| Exact global-load-to-WMMA ABI | A synthetic direct global-load graph currently fails devectorization unless real range ownership, CONTRACT axes/remaps, half.vec(16), and float.vec(8) accumulator ABI are present | Reuse existing precontract contracts and fixture construction; add a real register producer that preserves those nodes |
| Mixed-role route attribution | The current authority can report selected candidate roles without recording fallback roles | Fix attribution before combined pure promotion; this is instrumentation, not a compiler rewrite |

## What “reuse” means in practice

The implementation should have one lifecycle and two storage policies:

```text
candidate policy
  -> shared descriptor/CONTRACT validation
  -> shared epoch/slot lifecycle + proof
  -> storage policy callbacks
       LDS: global b128 -> LDS window -> barrier -> LDS b128 fragments
       register: global b128 -> register fragment carrier -> typed wait -> WMMA
  -> shared WMMA/accumulator/store construction
  -> backend-specific wait/resource lowering
```

The register policy must return `KernelStage1ProducerStage` and
`KernelStage1FragmentStage`; it must not return an instruction list and must not
call `extra/qk/prefill/wmma.py::build_gemm_pipe`. The handwritten primitive is a
teacher for cadence and expected resource shape only.

The existing lifecycle has an important semantic mismatch that must remain
explicit: `KernelStage1PipelinePlan.stage_count=1` means one proved lifecycle
template with one or two physical LDS slots, while `RegisterPipePlan.stages=2`
means two logical register load stages. Mapping those fields directly would
silently misreport storage. Introduce an adapter/mapping rather than changing
the established LDS meaning.

## Rewrite threshold

A rewrite would be justified only if one of these facts is proven:

1. the existing UOp graph cannot represent a global b128 producer whose values
   retain row/K range ownership and the required CONTRACT/remap metadata;
2. the existing WMMA and accumulator ABI cannot consume the resulting
   `half.vec(16)`/`float.vec(8)` carriers; or
3. no backend can lower a typed wait dependency without route-owned raw ISA.

Current evidence proves none of these impossibilities. It shows an incomplete
adapter and a backend capability gap. Therefore a full rewrite would duplicate
the validated descriptor, lifecycle, and resource machinery and increase risk.

## Ordered implementation boundary

1. Extract descriptor/range/CONTRACT validation from
   `PrecontractPipelineTemplate` while keeping all LDS allocation checks in the
   LDS implementation.
2. Add a register storage callback implementation that emits only ordinary
   compiler UOps (global loads, typed carriers, and dependency edges).
3. Extend the lifecycle graph with typed wait dependencies and a fail-closed
   backend lowering interface.
4. Reuse AMD ISA's physical wait analysis behind that interface; do not expose
   `AMDOps` or raw `Ops.INS` to route code.
5. Add compile-only, resource, correctness, and pinned timing gates for one
   role (`attn_qo`) before expanding to `attn_kv` and `ffn_down`.

## Exit criteria

This decision is validated only when the register candidate has:

- no `DEFINE_LOCAL` or LDS window in the lowered graph;
- exact A/B CONTRACT axes, descriptor remaps, and `half.vec(16)` carriers;
- lifecycle proof for K=1, K=2, full K, and tail cases;
- a final backend artifact proving targeted waits (or a documented measured
  barrier-only ceiling, without promotion);
- joined source/binary identity, VGPR/SGPR, LDS, scratch, spill, and ABI facts;
- full-output correctness and pinned isolated timing.

Until then, the current LDS candidate remains the execution oracle and the
register policy remains a fail-closed compile/selection candidate.
