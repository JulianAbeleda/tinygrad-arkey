# Pure Machine Search Handwritten Kernel Scope

Date: 2026-07-06

This document applies the strict reviewer contract in `docs/pure-machine-search.md` to the current QK route surface. The
goal is to make every non-pure executing kernel explicit, then define the path to pure machine search.

## Classification

Definitions used here:

- `external_handwritten_kernel`: raw instruction emitters, source strings, inline asm, precompiled ISA, or `Ops.BINARY`
  injection. These are never pure final defaults.
- `hand_authored_uop_template`: a human-authored `Tensor.custom_kernel` / UOp body. These can be transitional runtime
  debt, rollback oracles, or research routes, but are not pure unless converted to descriptor-owned generated emitters.
- `descriptor_owned_generated`: a route whose topology is owned by structured data/search descriptors and lowered through
  shared generator/codegen infrastructure. This can be pure when the runtime path has no raw-ISA/source-string escape.

The audit question is: who owns the executing kernel body/topology?

## Summary

Current non-pure surface by severity:

| Priority | Surface | Runtime status | Classification | Required outcome |
|---|---|---|---|---|
| P0 | Prefill raw WMMA graph GEMM substrate | live for fp16 graph prefill and opt-in fused Q4_K | `external_handwritten_kernel` | Replace with generated tinygrad/codegen LDS+WMMA substrate or reclassify manifest debt. |
| P0 | Q4_K direct-packed prefill default | live default for memory-safe 14B/32B packed prefill | `hand_authored_uop_template` | Replace with generated quantized prefill/MMQ substrate. |
| P1 | Decode attention live-split / block-tile UOp templates | promoted default routes | manifest alignment required | Prove descriptor ownership or reclassify as hand-authored UOp debt. |
| P1 | Q4_K/Q6_K decode rollback oracles | rollback/reference, not default | `rollback_oracle` / `hand_authored_uop_template` | Keep gated as rollback only or convert/delete. |
| P2 | Q4_K/Q8_1 sdot4/vdot prefill experiments | default-off research | mixed hand UOp / inline asm | Convert to generated descriptor/codegen or retire. |
| P2 | Native ISA block-tile injection | research/probe | `external_handwritten_kernel` | Keep non-runtime only or delete after generated route covers it. |
| P3 | Microgates/bench/proof custom kernels | non-runtime | test fixture debt | Keep out of route manifest; do not promote without conversion. |

## Inventory

### 1. Raw WMMA Prefill Instruction Emitters

Files:

- `extra/qk/prefill/wmma.py`
- `extra/qk/prefill_graph_gemm_route.py`
- `extra/qk/asm_scheduler_proofs.py`

Functions/builders:

- `build_tile_kernel`
- `build_lds_tile`
- `build_gemm`
- `build_gemm_pipe`
- `build_gemm_lds`
- `build_gemm_lds2`
- `build_gemm_lds2_q4k`
- `_run_insts`, `_run_insts_lds`
- `route_pf16_graph_gemm`
- `route_q4k_graph_gemm`

Why non-pure:

- Builds explicit RDNA3 instruction lists.
- Wraps instructions with `UOp(Ops.INS, ...)`.
- Manually owns VGPR/LDS/barrier/waitcnt/branch/store behavior.

Runtime reachability:

- `route_pf16_graph_gemm` is selected by prefill graph GEMM paths.
- `route_q4k_graph_gemm` is selected by `PREFILL_Q4K_WMMA_FUSED=1`.
- `prefill_pipe_role_selective_generated` currently describes spec-generated schedule selection but still lowers through
  `ref.build_gemm_pipe / ref.build_gemm_lds2`; under the strict rule this is generated selection over a handwritten
  substrate, not final pure machine search.

Gameplan:

1. Introduce a generated `PrefillWMMAScheduleSpec` lowering that emits ordinary UOps/tinygrad codegen, not `Ops.INS`.
2. Add codegen support for the missing primitives:
   - cooperative global-to-LDS staging,
   - `DEFINE_LOCAL` allocation that survives scheduling,
   - barrier placement,
   - ds-load/ds-store expression,
   - WMMA/tensor-core selection from normal matmul/reduce structure,
   - wait scheduling or safe backend-owned wait lowering.
3. Reproduce the hand substrate in phases:
   - fp16 global-direct generated WMMA,
   - fp16 LDS single-buffer,
   - fp16 LDS double-buffer/cooperative staging,
   - Q4_K fused dequant-to-LDS,
   - Q4_K fused dequant-to-LDS plus WMMA role coverage.
4. Gate each phase against the raw-ISA route:
   - correctness,
   - route-bound/no hidden fallback,
   - emitted code has no `Ops.INS`, `Ops.BINARY`, source string, or inline asm,
   - performance reaches the documented decider threshold before replacing default.
5. After replacement, reclassify `prefill_pipe_role_selective_generated` only if the executing substrate is generated.

Done means:

- No prefill default route calls `extra/qk/prefill/wmma.py`.
- `PREFILL_Q4K_WMMA_FUSED` is either removed or routes to generated codegen.
- `q4k_wmma_tiled_no_hand_kernel_gate` or successor forbids `Ops.INS`, `Ops.BINARY`, `asm volatile`, route-local
  `__builtin_amdgcn_wmma`, and route-local `.custom_kernel(` for the promoted path.

### 2. Q4_K Direct-Packed Prefill UOp Templates

Files:

- `tinygrad/llm/prefill_routes.py`
- `extra/qk/quant/q4_k_gemv_primitive.py`
- `extra/qk/prefill_packed_tile_spec.py`

Runtime functions:

- `q4k_gemm_kernel`
- `q4k_gemm_packed_load_kernel`
- `q4k_gemm_packed_load_direct_out_kernel`
- `q4k_gemm_packed_load_reduce_out_kernel`
- `q4k_q8_1_gemm_kernel`
- `q4k_q8_1_sdot4_gemm_kernel`
- `q4k_q8_1_sdot4_coop_gemm_kernel`
- `q4k_q8_1_sdot4_coop_direct_out_kernel`
- `q8_signed_pack_u32_kernel`
- `emit_q4k_packed_prefill_tile`

Why non-pure or provisional:

- `q4_k_gemv_primitive.py` templates are hand-authored UOp custom kernels.
- `prefill_q4k_direct_tile4x4_default` is explicitly manifest debt: `hand_authored_uop_template`.
- `prefill_q4k_reduce_out_research` and `prefill_q4k_mmq_direct_out_research` are also manifest debt.
- `emit_q4k_packed_prefill_tile` is descriptor-shaped, but it still shares hand-authored Q4_K UOp helper topology; it
  should remain research until generated-only provenance is proved.

Runtime reachability:

- Default memory-safe Q4_K prefill routes through direct-packed templates when resident fp16 is unavailable or not chosen.
- Q4_K/Q8_1 modes are selected by `PREFILL_Q4K_Q8`.
- Generated packed tile is selected by `PREFILL_QK_GENERATED_TILE=1`.

Gameplan:

1. Promote Q4_K format facts into data descriptors:
   - block bytes/words,
   - group layout,
   - scale/min extraction,
   - nibble mapping,
   - activation format.
2. Lower Q4_K packed prefill from descriptors through a generated emitter:
   - no hand-authored `*_kernel` per route,
   - no source strings,
   - no `Ops.CUSTOM`/`Ops.CUSTOMI` except renderer-owned primitives with explicit allowance.
3. Choose the pure target:
   - short-term: descriptor-owned UOp generator for packed dequant dot,
   - stronger target: normal tinygrad tensor/reduce graph that codegen lowers to tiled code,
   - performance target: fused Q4_K dequant + matmul that removes current dequant VALU bottleneck.
4. Replace current default in order:
   - direct-packed default parity,
   - reduce-out correctness foundation,
   - generated tile/direct-warp if it beats current default,
   - final fused MMQ/WMMA only if it beats direct-packed.
5. Update manifest only after authority:
   - `prefill_q4k_direct_tile4x4_default` leaves default,
   - replacement route has `machine_authored_generated`,
   - old direct-packed templates become rollback or deleted.

Done means:

- No default Q4_K prefill path calls `q4k_gemm_packed_load_*` or `q4k_q8_1_sdot4_*` from
  `q4_k_gemv_primitive.py`.
- The replacement route has generated-only binding audit evidence and 14B pp512 timing authority.

### 3. Q6_K Direct-Packed Prefill UOp Templates

Files:

- `tinygrad/llm/prefill_routes.py`
- `extra/qk/quant/q6_k_gemv_primitive.py`

Runtime functions:

- `q6k_gemm_kernel`
- `q6k_gemm_packed_load_kernel`
- `q6k_gemm_packed_load_direct_out_kernel`

Why non-pure:

- Human-authored UOp templates.
- Unlike decode, there is no equivalent promoted generated Q6_K prefill descriptor path.

Runtime reachability:

- Direct-packed prefill supports Q6_K through the same `route_direct_packed_prefill` surface.

Gameplan:

1. Reuse the successful decode `Q6KGEMVRouteSpec` pattern.
2. Add `Q6KPrefillRouteSpec` with token/tile axes and packed-load/direct-out layout.
3. Lower from descriptor through shared generated UOp infrastructure.
4. Gate against current Q6_K direct-packed prefill:
   - correctness,
   - no hidden fallback,
   - no hand template calls,
   - W==D or better on Q6_K prefill roles.

Done means:

- Q6_K prefill no longer imports `q6_k_gemv_primitive.py` kernels on the promoted path.

### 4. Q4_K Decode Rollback and Legacy UOp Templates

Files:

- `tinygrad/llm/decode_routes.py`
- `extra/qk/quant/q4_k_gemv_primitive.py`
- `extra/qk/q4k_lane_partition_gemv.py`
- `extra/qk/gemv_g3_codegen_lowering.py`

Runtime/template functions:

- rollback/legacy: `q4k_gemv_kernel`, `q4k_gemv_partial_kernel`, `q4k_gemv_warp_kernel`, `q4k_coop_partial_kernel`,
  `q4k_lane_partition_gemv_kernel`
- Q4_K/Q8_1 experimental: `q4k_q8_1_vdot_builtin_partial_kernel`, `q8_1_bias_pack_u32_kernel`
- generated/default candidates: `q4k_g3_lanemap_gemv_kernel`, `q4k_g3_lanemap_gemv_splitk_kernel`,
  `q4k_g3_lanemap_gemv_inkernel_combine_kernel`

Why mixed:

- The owned warp/coop/partial routes are rollback or hand-authored UOp templates.
- The `Q4K_VDOT` path still builds custom source strings through `_q4k_q8_1_vdot_source`; the non-builtin variant contains
  inline asm and must stay non-pure.
- G3 is intended to be descriptor-owned via `Q4KGateUpLaneMap`; it is acceptable only if the generated provenance gate
  remains authoritative. If a reviewer finds route-specific human topology outside the LaneMap/spec, reclassify it.

Runtime reachability:

- G3 is the default for eligible Q4_K decode.
- Owned warp/coop/partial paths are reachable by rollback flags or unsupported shapes.

Gameplan:

1. Keep G3 as the positive-control generated route, but harden its provenance audit:
   - route must derive from `Q4KGateUpLaneMap`,
   - no owned warp function on selected path,
   - no inline asm/source string,
   - emitted program name proves G3 binding.
2. For rollback templates:
   - keep as `rollback_oracle`,
   - block from `PURE_MACHINE_SEARCH_ONLY=1`,
   - delete only when generated route has enough shape coverage and stable rollback alternatives.
3. For Q4_K/Q8_1 vdot experiments:
   - replace source-string vdot with renderer-owned dot lowering or generated tensor/reduce lowering,
   - retire the inline-asm variant.

Done means:

- Default Q4_K decode remains generated and all hand templates are rollback-only or removed.
- `Q4K_VDOT` cannot select inline asm on a pure path.

### 5. Q6_K Decode Rollback UOp Templates

Files:

- `tinygrad/llm/decode_routes.py`
- `extra/qk/quant/q6_k_gemv_primitive.py`
- `extra/qk/q6k_route_spec.py`

Runtime/template functions:

- generated/default: `emit_q6k_gemv_kernel`
- rollback/legacy: `q6k_gemv_warp_kernel`, `q6k_halfwarp_partition_kernel`, `q6k_gemv_partial_kernel`,
  `q6k_coop_partial_kernel`
- batched/decode support: `q6k_gemm_kernel`

Why mixed:

- Decode Q6_K default is spec-driven via `Q6KGEMVRouteSpec`.
- Shipped hand templates remain rollback/reference via `DECODE_Q6K_GENERATED=0`.
- `Q6K_DIRECT_ROUTE` is a refuted hand-authored route.

Gameplan:

1. Preserve generated decode default.
2. Keep rollback oracles out of pure path.
3. Delete or quarantine `Q6K_DIRECT_ROUTE` unless a new topology appears.
4. Port prefill Q6_K to the same spec-driven pattern.

Done means:

- Q6_K decode default has generated-only proof.
- Q6_K prefill has a matching generated descriptor route.

### 6. Decode Attention UOp Kernel Families

Files:

- `tinygrad/llm/decode_routes.py`
- `extra/qk/flash_decode.py`
- `extra/qk/flash_kernels.py`
- `extra/qk/live_split_geometry.py`
- `extra/qk/flash_decode_fused_combine.py`
- `extra/qk/fdot2_lowering.py`
- `extra/qk/amd_warp_reduce.py`

Runtime functions include:

- `flash_decode_attention`
- `flash_decode_attention_whole_cache`
- `flash_decode_live_split_block_tile`
- `flash_decode_g5_block_tile`
- `flash_decode_fused_combine`
- `flash_block_tiled_xlane_score_pv_tile_whole_cache_kernel`
- `flash_fused_gmax_combine_kernel`
- `flash_state_gmax_kernel`
- `flash_state_combine_kernel`
- `flash_fused_state_combine_kernel`
- many older `flash_*_kernel` variants in `flash_kernels.py`

Why this needs alignment:

- The route manifest marks live-split/block-tile attention as `machine_authored_generated`.
- The executing implementation is still Python UOp custom kernels in `flash_kernels.py` / `live_split_geometry.py`.
- Under the strict rule, this is pure only if the route topology is demonstrably owned by a descriptor/search space rather
  than by route-specific handwritten functions.
- `Ops.CUSTOMI __builtin_amdgcn_fdot2` and `ds_bpermute` helpers are special intrinsic surfaces. They can be acceptable
  only if treated as renderer/backend-owned primitives, not route-local source escapes.

Runtime reachability:

- 8B live-split KV_BOTH is promoted default.
- 14B/32B G=5 live-split routes are promoted for validated shapes.
- Generic flash fallback is selected by `DECODE_LIVE_SPLIT=0`.

Gameplan:

1. Write the missing descriptor layer:
   - `FlashDecodeTileSpec`,
   - `LiveSplitGeometrySpec`,
   - `FlashCombineSpec`,
   - explicit fields for `B`, `Hq`, `Hkv`, `Hd`, `S`, staging, rope, KV quant, combine style, and split policy.
2. Move route-specific constants and topology choices out of hand functions and into specs.
3. Make `flash_kernels.py` a shared emitter from specs, not a set of route-local hand kernels.
4. Add a provenance gate:
   - selected route has a serialized spec,
   - emitted kernel name includes spec identity,
   - no retired owned HIP route,
   - no hidden generic fallback,
   - no raw `Ops.INS`/`Ops.BINARY`/source string.
5. Decide whether `Ops.CUSTOMI` intrinsics are allowed:
   - allowed only for renderer-owned `fdot2`, `ds_bpermute`, `exp2` helpers with backend coverage,
   - disallowed if route-local strings directly select hardware instructions outside a shared lowering.

Done means:

- Attention defaults either have descriptor-owned generated proof or are reclassified as `hand_authored_uop_template`.
- `PURE_MACHINE_SEARCH_ONLY=1` can reject any non-proven attention path.

### 7. Native ISA / Precompiled Program Injection

File:

- `extra/qk/native_isa_block_tile_graph_node.py`

Why non-pure:

- Compiles with `AMDISARenderer`, extracts `Ops.BINARY`, and injects a precompiled native program into HIP runtime.
- This is `external_handwritten_kernel` by the new contract even when the source AST was generated.

Runtime reachability:

- Research/probe path, not a promoted default in the manifest.

Gameplan:

1. Keep it non-runtime or remove after generated HIP/codegen route covers the same tile.
2. If native ISA backend becomes the shared compiler backend, define a separate backend-owned rule:
   - no route-local binary injection,
   - normal scheduler/codegen path owns ISA emission,
   - manifest records backend as compiler-generated.

Done means:

- No route injects `Ops.BINARY` directly.

### 8. Inline ASM / Source-String Kernels

File:

- `extra/qk/quant/q4_k_gemv_primitive.py`

Known functions/surfaces:

- `_vdot4_q4_q8_accum`
- `_q4k_q8_1_vdot_source`
- `q4k_q8_1_vdot_partial_kernel`
- `q4k_q8_1_vdot_builtin_partial_kernel` uses generated source string with `_dp4a`

Why non-pure:

- The non-builtin source contains `asm volatile("v_dot4_u32_u8 ...")`.
- Source-string kernels are explicitly not pure.

Gameplan:

1. Delete or quarantine the inline-asm source-string variant.
2. Replace the builtin source-string variant with renderer-owned dot4 lowering or a generated UOp/tensor expression.
3. Add a static guard that fails if a selected route contains `asm volatile`.

Done means:

- No pure route can reach a source-string kernel.

## Non-Runtime Custom Kernel Surface

These files contain custom kernels or synthetic kernel functions used as gates, probes, diagnostics, or benches. They are
not route debt unless wired into `tinygrad/llm/decode_routes.py`, `tinygrad/llm/prefill_routes.py`, or
`tinygrad/llm/route_ops.py` as a selected runtime path.

Examples:

- `extra/qk/asm_scheduler_proofs.py`
- `extra/qk/decode_physical_tile.py`
- `extra/qk/decode_physical_tile_score_broadcast_kernels.py`
- `extra/qk/decode_attention_*_microgate.py`
- `extra/qk/fused_*_gate*.py`
- `extra/qk/tg_p*_*.py`
- `extra/qk/block_tile_one_case.py`
- `extra/qk/canonical_recurrence_check.py`
- `extra/qk/quant/q4_k_bench.py`
- `extra/qk/quant/q6_k_gemv_primitive.py` `__main__` bench paths

Policy:

- Allowed as tests/probes.
- Must not be promoted to a default route without route-manifest provenance and generated-only binding audit.
- If a probe becomes a runtime route, reclassify it using the main inventory rules.

## Phased Plan

### Phase 0: Make The Audit Mechanical

Tasks:

1. Extend `generated_quant_binding_audit` or add `pure_kernel_surface_audit`.
2. Scan runtime-bound files, not every test fixture:
   - `tinygrad/llm/decode_routes.py`
   - `tinygrad/llm/prefill_routes.py`
   - `tinygrad/llm/route_ops.py`
   - manifest-attributed writer files.
3. Emit per-route classification:
   - `pure`,
   - `hand_uop`,
   - `external_raw_isa`,
   - `rollback_oracle`,
   - `unknown`.
4. Fail on manifest contradictions:
   - route says `machine_authored_generated` but selected writer contains raw `Ops.INS`,
   - default route is hand UOp without `replacement_scope`,
   - pure route reaches `asm volatile` or `Ops.BINARY`.

Acceptance:

- One JSON artifact lists every selected runtime route, writer file, marker evidence, classification, and replacement
  scope.

### Phase 1: Fix Manifest Truth

Tasks:

1. Reclassify `prefill_pipe_role_selective_generated` or split it:
   - schedule selection can remain generated,
   - executing raw-ISA substrate must be explicit debt until Route B lands.
2. Audit attention routes against the strict rule:
   - either prove descriptor-owned generated topology,
   - or reclassify as `hand_authored_uop_template` with replacement scope.
3. Keep decode Q4_K G3 and Q6_K generated defaults only if their provenance gates prove descriptor ownership.

Acceptance:

- `default_purity_report()` and `pure_search_guard.py` reflect the strict definition, not just historical route names.

### Phase 2: Delete Raw WMMA Substrate From Defaults

Tasks:

1. Build generated LDS+WMMA substrate in tinygrad/codegen.
2. Port fp16 prefill graph GEMM first.
3. Port fused Q4_K dequant-to-LDS/WMMA second.
4. Remove default dependency on `extra/qk/prefill/wmma.py`.

Acceptance:

- Same or better correctness.
- Performance meets the decider threshold.
- No `Ops.INS` on selected default prefill routes.

### Phase 3: Replace Direct-Packed Q4_K/Q6_K Prefill Templates

Tasks:

1. Convert Q4_K direct-packed route to descriptor-owned generated emitter or normal tensor/codegen lowering.
2. Convert Q6_K direct-packed route using Q6 decode spec pattern.
3. Retain old templates as rollback only until W==D confidence is high.

Acceptance:

- `prefill_q4k_direct_tile4x4_default` is no longer a default.
- Q4_K/Q6_K prefill defaults are generated-only and route-bound.

### Phase 4: Attention Provenance Closure

Tasks:

1. Add explicit attention specs and generated binding audit.
2. Split generic flash fallback, live-split geometry, and combine into descriptor-owned emitted kernels.
3. Gate all promoted attention routes under `PURE_MACHINE_SEARCH_ONLY=1`.

Acceptance:

- Attention default route classification is mechanically pure or explicitly debt.

### Phase 5: Retire Or Quarantine Research Hand Kernels

Tasks:

1. Remove inline-asm vdot source-string kernels or fence behind non-pure debug flags.
2. Keep native ISA binary injection as proof-only or delete.
3. Move microgate kernels out of route-import surfaces.

Acceptance:

- No research-only hand kernel can be accidentally selected by model runtime.

## Final Done State

The repo reaches strict pure machine search when:

1. `default_purity_report()["verdict"] == "TINYGRAD_DEFAULT_PURITY_PASS"` under the strict implementation audit.
2. `PURE_MACHINE_SEARCH_ONLY=1` permits all default hot routes and rejects all rollback/source-string/raw-ISA routes.
3. Runtime-bound writer files for promoted defaults contain no selected:
   - `Ops.INS`,
   - `Ops.BINARY`,
   - `asm volatile`,
   - source-string kernels,
   - route-local handwritten `Tensor.custom_kernel` bodies outside descriptor-owned generated emitters.
4. Every default route has:
   - serialized descriptor/spec or ordinary tinygrad graph provenance,
   - route-bound proof,
   - correctness proof,
   - timing authority,
   - rollback/reference classification.
