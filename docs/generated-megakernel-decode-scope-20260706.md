# Generated Megakernel Decode — Scope

Date: 2026-07-06

Goal: fuse the **entire decode step** (all layers, one token) into **one persistent GPU kernel**, **generated and
machine-searched — never hand-written**. This is the batch-1 decode-latency frontier (kills launch overhead + between-op
HBM round-trips + pipeline bubbles). See `knowledge_base/notes/arkey-inference-100-percent-audit.md` (gap #3) and the megakernel landscape
(Hazy "No Bubbles"; Mirage/MPK; ETC dynamic megakernel; Ada-MK searched megakernel; AMD MI300X monokernel).

This document is **subordinate to `docs/pure-machine-search.md`** (the generated-vs-handwritten contract) and consistent
with `docs/handwritten-kernel-exhaustive-lowering-scope-20260706.md` (the megakernel is an L3 descriptor-owned / L5
backend-lowered artifact, NOT an L0/L1 hand template).

## Non-negotiables

1. **No handwritten megakernel.** The kernel body must be **emitted from a descriptor by shared infrastructure**, the
   same way `emit_q6k_gemv_kernel` lowers `Q6KGEMVRouteSpec`. A route-local `Ops.INS` / `Tensor.custom_kernel` hand
   template for the megakernel is **forbidden** and must fail the audit. Authoring a *spec + emitter + primitive*
   (the compiler) is allowed; authoring a *kernel body* (the output) is not.
2. **Reuse the substrate — zero duplication.** The megakernel **composes existing generated op-emitters**; it does not
   re-derive GEMV/attention/FFN math. It reuses the existing kernel emission, ISA backend, runtime, search, and
   correctness harness (see "Substrate to reuse").
3. **Machine-searched.** The schedule degrees of freedom (grid size, tile assignment, warp roles, pipeline depth,
   barrier placement, LDS/reg budget) are searched by BubbleBeam/FutureSight, not hand-picked.
4. **Decode-only, ring/static context** for this scope. Prefill (compute-bound) is out of scope — that's the
   fused-dequant->WMMA work. Dynamic-context-length beyond the ring is out of scope (revisit with an ETC-style dynamic
   megakernel later).

## Substrate to reuse (do NOT rebuild any of this)

| Need | Existing asset | Role in the megakernel |
|---|---|---|
| Op bodies (the math) | `extra/qk/gemv_g3_codegen_lowering.py` (Q4_K G3 GEMV), `extra/qk/q6k_route_spec.py` `emit_q6k_gemv_kernel` (Q6_K), `extra/qk/live_split_geometry.py` (attention), FFN GEMVs | Emitted as **tasks/tiles inside** the persistent kernel, not as separate launches |
| Instruction-stream emission | the gen_sched asm-substrate builder + `extra/qk/asm_scheduler.py` + `Ops.INS` | The mechanism that emits the single fused kernel body |
| ISA backend | `tinygrad/renderer/isa/amd.py` + `renderer/amd/dsl.py` + `autogen/amd/rdna3` | Lowers the emitted stream to gfx1100 |
| Runtime launch/replay | `tinygrad/runtime/ops_amd.py` HCQ + `runtime/graph/hcq.py` | Launches the one persistent kernel; already single-submit |
| Static context | ring-full path in `tinygrad/llm/decode_routes.py` (`_tc = MAXC`) | Gives a **concrete `Tc`** so the megakernel grid is static — sidesteps dynamic-length |
| Search | `extra/qk/bubblebeam_futuresight.py` | Searches the megakernel schedule DOF |
| Correctness | the tiny-Transformer `DEV=PYTHON` forward harness | Bit-close verification vs the per-op decode path |
| Measurement | native PMC sampler in `ops_amd.py` | Latency / launch-count / bandwidth |
| Provenance | `extra/qk/pure_kernel_surface_audit.py`, `extra/qk/generated_route_registry.py`, `route_manifest.py` | Proves the result is generated, not a renamed hand kernel |

## Architecture (descriptor-owned, generated)

- **`MegakernelDecodeSpec` (descriptor / data):** the decode op-DAG as data — ordered tasks with dependencies, per-task
  tile shapes, grid size, barrier points (layer boundaries), warp roles (producer/consumer), LDS/register budget,
  staging mode, and which existing op-emitter produces each task body. Serializable (`to_json`), search-tunable.
- **`emit_megakernel_decode(spec)` (shared emitter):** lowers the spec to **one persistent kernel** by composing the
  existing op-emitters into a single instruction stream + inserting the barrier/scheduler scaffolding. This is the only
  new "compiler" code; it emits, it is not a hand kernel.
- **Search:** BubbleBeam explores the spec's DOF; the audit + harness gate each candidate.

## New primitives to build ONCE (as generated/emitted infra, not hand kernels)

1. **Grid-wide barrier** — an emitted HBM-counter spin-barrier (atomic inc + spin to grid size) so layer N+1 sees all of
   layer N. Emitted by the substrate as a reusable fragment; not a route-local hand kernel.
2. **Persistent-grid launcher + in-kernel task loop** — launch a fixed resident grid once; the kernel walks the spec's
   task list with barriers at boundaries.
3. **DAG -> persistent-schedule lowering** — take the captured decode graph (the tinygrad UOp graph / HCQ command list)
   and lower it into the `MegakernelDecodeSpec` task order. This is the Mirage-task-graph / ETC-event-tensor equivalent,
   built on the graph you already capture.

## Phases (boil the ocean, but each phase is verifiable)

### Phase 0 — reuse map + spec + barrier primitive
- Enumerate exactly which existing emitters produce each decode op (fill the Substrate table with concrete symbols).
- Define `MegakernelDecodeSpec` (data only).
- Build + unit-test the **grid barrier** in isolation (throwaway: N workgroups write -> barrier -> cross-read), emitted
  by the substrate, verified on gfx1100.
Done: spec type exists; grid barrier proven; zero op-math duplicated.

### Phase 1 — single fused layer (generated)
- `emit_megakernel_decode` composes ONE transformer layer (norm->QKV->attn->O->norm->FFN) into one persistent kernel,
  activations in LDS within the layer, HBM between. Emitted from the spec, composing existing op-emitters.
Done: one fused layer is **bit-close** to the per-op path (tiny-forward harness); not slower; audit sees it generated.

### Phase 2 — full-decode megakernel (generated, ring-static context)
- Extend the task loop to all layers + embed + final norm + lm_head + sample: the whole decode step in one launch.
Done: full decode is bit-close vs the current N-launch path; **one launch/token**; latency measured.

### Phase 3 — search the schedule
- BubbleBeam searches warp roles / tiling / pipeline depth / barrier granularity over `MegakernelDecodeSpec`.
Done: searched schedule beats the Phase-2 default latency; winner registered.

### Phase 4 — warp-specialized on-chip residency (perf frontier)
- Producer warps stream weights/KV while consumer warps compute; keep activations resident across ops.
Done: bandwidth/latency approaches the no-bubble ceiling; still generated.

### Phase 5 — provenance + promotion
- Register in `generated_route_registry`; add a `route_manifest` row (provenance `machine_authored_generated`);
  `pure_kernel_surface_audit` classifies the decode default as L3/generated; guard **forbids** a handwritten megakernel
  (forbidden-marker scan: no route-local `Ops.INS`/`custom_kernel` megakernel body).
Done: audit PASS for the megakernel decode default; it is the promoted decode route.

## Non-goals / forbidden

- A hand-written megakernel (Ops.INS / custom_kernel template). It must be emitted from the spec by shared infra.
- Rebuilding the runtime, ISA backend, op-math, or search — compose what exists.
- Prefill fusion (separate compute-bound track: fused-dequant->WMMA).
- Dynamic context length beyond the ring (future ETC-style dynamic megakernel).
- Any "generated" emitter that is just the old hand kernel renamed into a descriptor file (per pure-machine-search.md).

## Acceptance gates (every phase)

- Semantic correctness: bit-close vs the per-op decode path (tiny-forward harness) or a math reference.
- Route-bound: the emitted kernel name proves the megakernel executed (no hidden fallback to per-op launches).
- Forbidden-marker scan: no route-local handwritten megakernel body.
- Strict audit: `pure_kernel_surface_audit` classifies it L3 descriptor-owned / generated.
- Latency: measured below the multi-launch baseline on the PMC.
- Reproducible: the emitted kernel regenerates from the spec (REGEN-style), it is not a committed hand artifact.
