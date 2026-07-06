# Handwritten Kernel Exhaustive Lowering Scope

Date: 2026-07-06

This scope answers whether every handwritten QK kernel surface can be lowered into pure tinygrad/codegen form, what
"exhaustive lowering" means, and what gates prove completion.

This document is subordinate to `docs/pure-machine-search.md` and
`docs/pure-machine-search-handwritten-kernel-scope-20260706.md`.

It defines completion in terms of centralized mechanical authorities, not in-page checklist rows:

- `extra/qk/lowering_phase_registry.py` (planned work items and target lowering level)
- `extra/qk/exhaustive_lowering_report.py` (single source of truth for route + runtime work queue)
- `extra/qk/lowering_done_criteria.py` (completion criteria by lowering level; the report exposes these as
  `done_criteria` on phase-backed work items)

The repository should source route-completion facts from those modules and keep docs focused on policy and scope.

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

The current blockers and pending items are authoritative in:

```bash
python3 -m extra.qk.exhaustive_lowering_report
```

This report is intentionally source-of-truth for:

- strict-default blockers currently failing mechanical audits,
- unmanifested runtime-capable surfaces still requiring manifest or strict guard coverage,
- `lowering_phase_registry` metadata and `lowering_done_criteria` gates joined to each work item.

Use this output instead of maintaining a duplicate route checklist in this document.

## When Is Something Done for Lowering?

Mark a route/surface as lower-complete only when all of these are true at the same time:

1. It is absent from the blocker lists in `extra/qk/exhaustive_lowering_report` and no longer needs a
   registry-only phase work item.
2. Its `route_id`/`surface_id` is represented in `extra/qk/lowering_phase_registry.py` with a target level.
3. Its target level is covered by `extra/qk/lowering_done_criteria.py`, which expresses the required proof for that
   lowering class (`L3`, `L4`, or `L5`).

If any one of these fail, keep the route in the registry, keep rollback/fixture classification explicit, and run a strict
gate until the report moves it out of blocking status.

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
- Keep unmanifested Q6_K direct prefill centralized in `runtime_surface_registry` until it gains manifest rows or strict guard coverage.
- Add emitted-kernel-name binding to the audit artifact so "selected route" and "executed kernel" are tied together.
- Keep historical/non-runtime fixtures out of default blockers but classified as fixture debt.

### Phase 1: Small-K Batched Q4_K/Q6_K (completed)

Goal: small-K Q4_K/Q6_K is out of active handwritten debt because runtime fallback now routes these shapes to graph execution.

Target:

- Keep route-level status and guards as proof while preserving a deprecation checkpoint for fallback behavior.

Actions:

- Remove the small-K phase/runtime debt rows and keep registry artifacts in sync with runtime route status.
- Continue strict fallback/route-level guard coverage while the small-K benchmark envelope remains non-default.

### Phase 2: Direct-Packed Q4_K/Q6_K Prefill

Goal: replace direct-packed prefill custom UOp templates. Q4_K and Q6_K direct-packed defaults are completed by
`Q4KPrefillRouteSpec` and `Q6KPrefillRouteSpec`; the refuted Q4_K generated-tile opt-in is retired fail-loud.

Target:

- L3 generated quantized prefill/MMQ descriptors.

Actions:

- Keep Q4_K/Q6_K defaults bound to descriptor-owned generated routes, with no unmanifested runtime surface rows.
- Move block layout, scale/min handling, tile geometry, lane ownership, and reduction shape into descriptor fields.
- Lower descriptors through shared generated UOp/codegen path.
- Retain hand route as rollback until generated path passes correctness and timing.

### Phase 3: Decode Attention Live-Split / Block-Tile

Goal: make promoted attention routes descriptor-owned instead of route-local custom kernels.

Target:

- L3 `FlashDecodeTileSpec`, `LiveSplitGeometrySpec`, and `FlashCombineSpec`, or L4 ordinary graph if practical.

Actions:

- Serialize attention topology: `B/Hq/Hkv/Hd/MAXC/Tc`, split count, K/V ownership, score/PV lifecycle, combine mode.
- Move current hand-authored topology into data descriptors.
- Build a shared flash lowering emitter with generated-only marker gate.
- Add route-bound proof for both G4 8B KV-both and G5 live-split routes.

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

### Phase 5: Rollback And Fixture Quarantine

Goal: make remaining handwritten code impossible to confuse with pure defaults.

Actions:

- Mark rollback/reference routes explicitly.
- Delete dead hand kernels when no longer needed.
- Keep microbench/proof custom kernels outside route manifest and default selectors.
- Ensure `docs/pure-machine-search.md`, route manifest, and audit agree.

## Non-Goals

- Do not reintroduce inherited tinygrad beam search.
- Do not move BoltBeam policy/search ledgers into tinygrad runtime.
- Do not claim purity for a generated selector over a handwritten implementation.
- Do not add per-route "generated" emitters that are just hand kernels renamed into descriptor files.
- Do not remove rollback before generated correctness and performance gates exist.

## Acceptance Gates

Every converted route is done only if the centralized artifacts are clear:

- route-classification and blocker status in `pure_kernel_surface_audit` and `exhaustive_lowering_report`,
- route-specific target in `lowering_phase_registry`,
- target-level proof criteria from `lowering_done_criteria` for `L3`/`L4`/`L5`,
- rollback/fixture separation and W==D or role-level timing policy as enforced by route-specific gates.

Repository-level completion requires:

```bash
python3 -m extra.qk.exhaustive_lowering_report
```

with `audit_verdict` equal to `PURE_KERNEL_SURFACE_AUDIT_PASS` and no non-fixture work items left blocking lowering completion.
