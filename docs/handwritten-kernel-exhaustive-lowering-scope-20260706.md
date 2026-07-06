# Handwritten Kernel Exhaustive Lowering Scope

Date: 2026-07-06

This scope answers whether every handwritten QK kernel surface can be lowered into pure tinygrad/codegen form, what
"exhaustive lowering" means, and what gates prove completion.

This document is subordinate to `docs/pure-machine-search.md` and
`docs/pure-machine-search-handwritten-kernel-scope-20260706.md`. The current mechanical authority is
`extra/qk/pure_kernel_surface_audit.py`, fed by the centralized generated-route and runtime-surface registries.

## Public Alignment

The repo definition aligns with public compiler/autoscheduling vocabulary, but intentionally uses a stricter shipping
rule than generic autotuning:

- TVM Ansor is described by TVM as taking tensor expressions and generating high-performance code without manual
  templates; this supports the repo distinction between generated schedule/topology and tuned handwritten kernels.
  Reference: https://tvm.apache.org/2021/03/03/intro-auto-scheduler
- The Ansor paper frames the system as tensor program generation that samples programs from a hierarchical search-space
  representation and refines them with search/cost models. Reference: https://arxiv.org/pdf/2006.06762
- TVM MetaSchedule is search-based autotuning over TIR schedules such as tiling, vectorization, and thread binding,
  measured on hardware. Reference: https://tvm.apache.org/docs/deep_dive/tensor_ir/tutorials/meta_schedule.html
- TVM's MetaSchedule RFC explicitly separates manual schedules, template-based spaces, and automatically generated
  design spaces. Reference: https://github.com/apache/tvm-rfcs/blob/main/rfcs/0005-meta-schedule-autotensorir.md
- MLIR documents progressive lowering through multiple abstraction levels/dialects before final codegen. That maps to
  this repo's "lowering level" language. Reference: https://mlir.llvm.org/docs/Tutorials/Toy/Ch-5/
- Triton matmul documentation shows a compiler-generated kernel model where users express block-level programs and the
  compiler lowers them to high-performance GPU code. This is useful context, but in this repo a route-local Triton-like
  custom kernel would still be considered handwritten unless topology is descriptor/generated-owned. Reference:
  https://triton-lang.org/main/getting-started/tutorials/03-matrix-multiplication.html

Conclusion: public systems support the idea of layered lowering and search-generated schedules. The repo's extra rule is
that tuning parameters on a human-written route-local kernel is not pure machine search; the executing topology must be
ordinary tinygrad lowering, descriptor-owned generated UOps, or backend-owned intrinsic lowering.

## Lowering Levels

| Level | Repo surface | Pure final default? | Meaning | Current examples |
|---|---|---:|---|---|
| L0 | handwritten external/raw | no | Source strings, inline asm, native ISA blobs, `Ops.INS`, `Ops.BINARY`. | `prefill_pipe_role_selective_generated` via raw WMMA substrate. |
| L1 | route-local custom UOp | no | A human-authored `Tensor.custom_kernel`/UOp body owns loops, lanes, loads, reductions, stores. | attention live-split/block-tile routes; direct-packed Q4_K prefill. |
| L2 | descriptor-wrapped hand kernel | no | A spec/search object picks shape or tile parameters, but execution calls L0/L1 code. | current prefill schedule selection over `extra/qk/prefill/wmma.py`. |
| L3 | descriptor-owned UOp codegen | yes, with audit | Structured route/search descriptor owns topology; shared emitter lowers to UOps; no route-local raw escape. | `decode_q4k_g3_generated`, `decode_q6k_coop_generated`. |
| L4 | ordinary tinygrad graph | yes | Runtime is expressed as normal Tensor/UOp graph and lowered by scheduler/codegen. | Q4_K int-WMMA research core when it remains `Tensor.matmul(dtype=int)`. |
| L5 | backend-owned intrinsic lowering | yes, with allowlist | Route uses normal IR; renderer/backend owns WMMA/dot/cross-lane lowering. | target state for WMMA/dot4/v_dot2/cross-lane ops. |

Exhaustive lowering means every default route moves from L0/L1/L2 to L3/L4/L5, and every remaining L0/L1/L2 surface is
non-default rollback/test debt with explicit audit classification.

## Current Mechanical State

Command:

```bash
python3 -m extra.qk.pure_kernel_surface_audit
```

Current verdict:

- `PURE_KERNEL_SURFACE_AUDIT_DEBT_FOUND`
- `STRICT_DEFAULT_PURITY_FAIL`

Current strict default blockers:

| Route | Current surface | Target level |
|---|---|---|
| `decode_flash_live_split_g4_8b_kvboth` | `route_local_custom_kernel` | L3 descriptor-owned flash specs or L4 ordinary graph. |
| `decode_flash_block_tile_g5_konly` | `route_local_custom_kernel` | L3 descriptor-owned flash specs or L4 ordinary graph. |
| `prefill_pipe_role_selective_generated` | `external_raw_or_binary` | L3/L5 generated LDS+WMMA substrate. |
| `prefill_q4k_direct_tile4x4_default` | `route_local_custom_kernel` | L3 generated Q4_K prefill/MMQ or L4 graph if performant. |

Current unmanifested runtime-capable surfaces:

- `prefill_q6k_direct_packed_default_capable`
- `decode_q4k_smallk_batched`
- `decode_q6k_smallk_batched`

These rows live in `extra/qk/runtime_surface_registry.py`. They must either gain manifest rows and strict guard
coverage or be blocked under `PURE_MACHINE_SEARCH_ONLY`.

## Exhaustive Lowering Algorithm

For every non-pure route surface:

1. **Specify semantics.** Write the exact tensor math, quant layout, shape domain, precision contract, and allowed
   reassociation/error bound.
2. **Extract topology facts.** Move lane ownership, tile sizes, split counts, staging mode, reduction shape, combine
   mode, barriers, and fallback policy into a data descriptor.
3. **Choose the highest viable pure target.**
   - Prefer L4 ordinary tinygrad graph when it can express the operation without graph explosion.
   - Use L3 descriptor-owned UOp codegen when topology must be explicit but can be generated from data.
   - Add L5 backend-owned intrinsic lowering when hardware operations are the missing vocabulary.
4. **Remove route-local execution ownership.** The route may select a descriptor, but it must not own an instruction
   list, source string, or custom kernel body.
5. **Bind policy to descriptors.** BubbleBeam/FutureSight may select candidates, but tinygrad runtime must consume only
   explicit route descriptors and rollback mappings.
6. **Gate generated-only execution.** The authority harness must prove correctness, route binding, no hidden fallback,
   no forbidden markers, and timing.
7. **Promote only after audit pass.** Update manifest provenance only after `pure_kernel_surface_audit` can classify the
   selected default as L3/L4/L5.

## Shared Compiler Vocabulary To Build

These capabilities should be built once and reused across routes:

| Capability | Needed by | Pure owner |
|---|---|---|
| Stable local-buffer allocation and bufferization | prefill WMMA, attention staging | scheduler/codegen |
| Cooperative global-to-LDS staging | prefill WMMA, flash attention | scheduler/codegen |
| Barrier and wait placement | prefill WMMA, multi-stage attention | backend/codegen |
| Renderer-owned WMMA/MFMA selection | fp16 prefill, int-WMMA research | backend intrinsic lowering |
| Renderer-owned dot4/v_dot2/fdot2 lowering | Q4/Q6 decode and prefill | backend intrinsic lowering |
| Cross-lane reduction lowering | GEMV/MMQ/attention reductions | backend intrinsic lowering |
| Descriptor registry and emitted-kernel attribution | all generated routes | route policy/gate stack |
| Shape/role policy schema | all generated routes | BubbleBeam policy plus tinygrad manifest |

Initial scaffolds:

- `extra/qk/generated_route_registry.py` records L3 descriptor-owned generated route rows, emitted-kernel patterns,
  authority artifacts, selector binding, shape/role policy, and manifest provenance.
- `extra/qk/runtime_surface_registry.py` records runtime-capable handwritten surfaces that are not manifest routes yet,
  so audits and future guards share one inventory.
- `extra/qk/backend_intrinsic_lowering_allowlist.py` records L5 backend-owned intrinsic categories and the route-local
  markers that remain non-pure.

The important constraint is reuse: adding a one-off generated emitter that only encodes the old hand kernel in a new
file does not complete the conversion unless the topology is represented as descriptor data and the lowering path is
shared enough for the audit to distinguish generator ownership from a renamed hand kernel.

## Route-Family Conversion Plan

### Phase 0: Audit Closure

Goal: make the inventory mechanically complete.

Actions:

- Keep `pure_kernel_surface_audit` as the strict surface authority.
- Keep unmanifested Q6_K direct prefill and small-K batched Q4_K/Q6_K paths centralized in
  `runtime_surface_registry` until they gain manifest rows or strict guard coverage.
- Add emitted-kernel-name binding to the audit artifact so "selected route" and "executed kernel" are tied together.
- Keep historical/non-runtime fixtures out of default blockers but classified as fixture debt.

Done:

- No default-capable non-pure route exists outside the audit.
- `PURE_MACHINE_SEARCH_ONLY=1` rejects every selected L0/L1/L2 default.

### Phase 1: Small-K Batched Q4_K/Q6_K

Goal: close the smallest runtime-capable handwritten surfaces.

Target:

- L3 descriptor-owned small-batch GEMM specs, or L4 ordinary graph fallback when `K<=32`.

Actions:

- Add `Q4KSmallBatchGEMMSpec` and `Q6KSmallBatchGEMMSpec` or explicitly route these shapes to ordinary tinygrad graph
  under strict mode.
- Gate correctness against current hand templates.
- Measure only enough to prevent catastrophic regression; these are not the main performance route.

Done:

- `decode_q4k_smallk_batched` and `decode_q6k_smallk_batched` disappear from unmanifested debt.

### Phase 2: Direct-Packed Q4_K/Q6_K Prefill

Goal: replace direct-packed prefill custom UOp templates.

Target:

- L3 generated quantized prefill/MMQ descriptors.

Actions:

- Create reusable packed-quant prefill descriptors for Q4_K and Q6_K.
- Move block layout, scale/min handling, tile geometry, lane ownership, and reduction shape into descriptor fields.
- Lower descriptors through shared generated UOp/codegen path.
- Retain hand route as rollback until generated path passes correctness and timing.

Done:

- `prefill_q4k_direct_tile4x4_default` is strict-pure or no longer a default.
- Q6_K direct prefill is manifest-covered and strict-mode safe.

### Phase 3: Decode Attention Live-Split / Block-Tile

Goal: make promoted attention routes descriptor-owned instead of route-local custom kernels.

Target:

- L3 `FlashDecodeTileSpec`, `LiveSplitGeometrySpec`, and `FlashCombineSpec`, or L4 ordinary graph if practical.

Actions:

- Serialize attention topology: `B/Hq/Hkv/Hd/MAXC/Tc`, split count, K/V ownership, score/PV lifecycle, combine mode.
- Move current hand-authored topology into data descriptors.
- Build a shared flash lowering emitter with generated-only marker gate.
- Add route-bound proof for both G4 8B KV-both and G5 live-split routes.

Done:

- `decode_flash_live_split_g4_8b_kvboth` and `decode_flash_block_tile_g5_konly` classify as L3/L4/L5.
- Existing hand flash kernels are rollback/reference or removed.

### Phase 4: Raw WMMA Prefill Substrate

Goal: replace the L0 raw RDNA3 WMMA instruction-list substrate.

Target:

- L3/L5 generated LDS+WMMA substrate, using backend-owned intrinsic lowering rather than `Ops.INS`.

Actions:

- Rebuild fp16 prefill through generated graph/codegen in this order:
  1. global-direct WMMA,
  2. single-buffer LDS staging,
  3. double-buffer/cooperative LDS staging,
  4. Q4_K fused dequant-to-LDS,
  5. role-complete Q4_K fused prefill.
- Add missing backend/codegen vocabulary for stable LDS buffers, barriers, waits, and WMMA selection.
- Use the current raw route as correctness/perf oracle only.

Done:

- `prefill_pipe_role_selective_generated` no longer touches `Ops.INS`, raw source, or route-local custom kernels.
- `extra/qk/prefill/wmma.py` is not selected by any default route.

### Phase 5: Rollback And Fixture Quarantine

Goal: make remaining handwritten code impossible to confuse with pure defaults.

Actions:

- Mark rollback/reference routes explicitly.
- Delete dead hand kernels when no longer needed.
- Keep microbench/proof custom kernels outside route manifest and default selectors.
- Ensure `docs/pure-machine-search.md`, route manifest, and audit agree.

Done:

- A reviewer can run the audit and see `PURE_KERNEL_SURFACE_AUDIT_PASS`, or see only non-default rollback/fixture debt.

## Non-Goals

- Do not reintroduce inherited tinygrad beam search.
- Do not move BoltBeam policy/search ledgers into tinygrad runtime.
- Do not claim purity for a generated selector over a handwritten implementation.
- Do not add per-route "generated" emitters that are just hand kernels renamed into descriptor files.
- Do not remove rollback before generated correctness and performance gates exist.

## Acceptance Gates

Every converted route needs:

- semantic correctness against mathematical reference or current oracle,
- route-bound emitted-kernel proof,
- no hidden fallback,
- strict audit classification as L3/L4/L5,
- forbidden marker scan for `Ops.INS`, `Ops.BINARY`, `asm volatile`, route-local `Tensor.custom_kernel`, `Ops.CUSTOM`,
  and `Ops.CUSTOMI`,
- W==D or role-level timing comparison,
- rollback retained until the generated route survives regression windows.

Repository-level completion requires:

```bash
python3 -m extra.qk.pure_kernel_surface_audit
```

to return `PURE_KERNEL_SURFACE_AUDIT_PASS`, or to return debt only for non-default rollback/test fixtures.
