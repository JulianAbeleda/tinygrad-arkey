# 8B Prefill Generated Pipe Lowerer MVP Scope

## Status: Deferred by S10.5 (generated transport correct-but-slow; see docs/8b-prefill-s10_5-machine-search-over-backend-atom-scope.md)

Date: 2026-07-08.

## Goal

Build the first compiler-owned register-resident pipe lowerer without copying `extra/qk/prefill/wmma.py::build_gemm_pipe`.

The immediate deliverable is a **diagnostic generated lowering report**, not route-bound promotion. It must prove that
the existing tinygrad AMD ISA path can produce the core instruction family for the pipe candidate:

```text
global_load_b128 -> targeted/non-full waitcnt -> v_wmma_f32_16x16x16_f16 -> global_store
```

## Why This Is Split From Route-Bound E2E

The current `PREFILL_GRAPH_GEMM=1` route executes by wrapping an instruction list inside `Tensor.custom_kernel`:

```text
route_pf16_graph_gemm -> Tensor.custom_kernel -> UOp(Ops.INS, arg=inst)
```

The hand pipe instruction list is authored for that custom-kernel ABI. A compiler-generated matmul program has its own
kernarg order, prologue, grid semantics, and metadata. Dropping those generated instructions into the current hand
custom-kernel wrapper would risk an ABI mismatch and would still look like route-local raw `Ops.INS` transport.

So the lowerer MVP has two stages:

1. **Diagnostic lowerer:** compile a generated native ISA matmul from `WMMAPipeSpec` and report structure/provenance.
2. **Route transport:** add a route-bound generated execution path that uses the compiler program directly instead of
   the hand custom-kernel wrapper.

This doc scopes stage 1 and names stage 2 as the next blocker.

## Stage 1 Definition Of Done

The diagnostic lowerer is done when:

| Gate | Requirement |
|---|---|
| D0 API | `build_wmma_pipe_diagnostic_lowering_report(spec)` accepts a `WMMAPipeSpec`. |
| D1 Generated source | It compiles through tinygrad/codegen/AMD ISA, not through `wmma.py`. |
| D2 Structure | Report includes instruction counts for b128 loads, WMMA, waits, and stores. |
| D3 Wait quality | Report separates ordinary generated waits from pipe-quality `vmcnt(pipe_tm*2 + pipe_tn*2)` waits. |
| D4 Provenance | Report marks `uses_hand_pipe_oracle=false` and `transport=generated_program_diagnostic`. |
| D5 Safety | `lower_wmma_pipe_spec(spec)` remains fail-closed for route-bound execution until transport exists. |
| D6 Artifact | MVP artifact can embed the diagnostic report without claiming route-bound generated execution. |

## Stage 1 Non-Goals

- Do not execute the generated program inside `route_pf16_graph_gemm`.
- Do not claim `PREFILL_WMMA_PIPE_PRIMITIVE=1` is route-bound complete.
- Do not copy hand instruction lists.
- Do not touch LDS/DBUF.
- Do not alter default prefill behavior.

## Stage 2 Blocker

The next blocker after this MVP is route transport:

```text
How does route_pf16_graph_gemm execute a compiler-owned generated program directly,
without converting it back into a route-local full-kernel Ops.INS list?
```

Until that is solved, the generated lowerer can prove structure and provenance offline, but the E2E route remains
blocked at the execution boundary.

## Current Result

`build_wmma_pipe_diagnostic_lowering_report(WMMAPipeSpec(m=64,n=64,k=64,...))` now compiles through tinygrad/codegen
and produces:

- `global_load_b128`,
- `v_wmma_f32_16x16x16_f16`,
- `global_store_b16`,
- `s_waitcnt`,
- no call to `build_gemm_pipe`.

With the existing `PREFILL_WMMA_CHAIN_AB_RESIDENT=1` primitive enabled, the bounded generated stream now reports:

- `mvp_core_structure_ok=true`,
- `mvp_pipe_wait_ok=true`,
- `mvp_structure_ok=true`,
- expected `vmcnt(pipe_tm*2 + pipe_tn*2)` present (`vmcnt(8)` for `pipe_tm=2,pipe_tn=2`),
- diagnostic correctness on AMD:ISA: finite output, rel_rmse about `2.1e-4`.

The opt-in route transport also now has two route-bound proofs:

- bounded nonzero `512x64x64`: returns an ordinary generated matmul result, does not call the hand custom-kernel wrapper,
  and matches fp32 reference with rel_rmse about `2.1e-4`;
- real `attn_qo` shape with zero activations, `512x4096x4096`: returns finite all-zero output with shape
  `(1,512,4096)`.

Remaining blocker: record real nonzero correctness/performance artifacts for `attn_qo`, then run whole-prefill smoke.
