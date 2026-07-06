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
| P1 | Small-K batched Q4_K/Q6_K verify/prefill via decode primitives | runtime-capable for `K<=32` | `hand_authored_uop_template` | Add manifest coverage or block under pure-search mode until generated. |
| P2 | Q4_K/Q8_1 sdot4/vdot prefill experiments | default-off research | mixed hand UOp / inline asm | Convert to generated descriptor/codegen or retire. |
| P2 | Native ISA block-tile injection | research/probe | `external_handwritten_kernel` | Keep non-runtime only or delete after generated route covers it. |
| P3 | Microgates/bench/proof custom kernels | non-runtime | test fixture debt | Keep out of route manifest; do not promote without conversion. |

## Review-Incorporated Manifest Alignment Actions

An xhigh review on this scope found several manifest/documentation mismatches that must be fixed before any final purity
claim:

1. `prefill_pipe_role_selective_generated` is currently marked `machine_authored_generated`, but the executing path still
   lowers through `route_pf16_graph_gemm -> UOp(Ops.INS)`. Split generated schedule selection from raw WMMA substrate, or
   reclassify the row as non-pure with `replacement_scope`.
2. Q6_K direct-packed prefill is default-capable because `PREFILL_DIRECT_QUANTS` defaults to `Q4_K,Q6_K`, but the manifest
   debt row is Q4_K-specific. Add an explicit Q6_K direct-packed prefill row or broaden the direct-packed debt row.
3. `decode_attention_generic_flash_generated` is not ordinary tinygrad scheduler output under the strict rule; it calls
   hand-authored flash `Tensor.custom_kernel` UOp templates. Reclassify as hand-authored/unknown until descriptor proof
   exists.
4. Promoted live-split/block-tile attention routes must not rely on names alone. They need serialized
   `FlashDecodeTileSpec` / `LiveSplitGeometrySpec` / `FlashCombineSpec` artifacts and a generated-only binding gate, or
   they should be downgraded to `hand_authored_uop_template`.
5. Small-K batched Q4_K/Q6_K routes (`K<=32`) in `decode_routes.py` call hand UOp GEMM templates and need explicit
   manifest or guard coverage.
6. `prefill_q4k_generated_tile_research` is descriptor-shaped, but its emitter still returns hand-written UOp bodies; keep
   it research/provisional until generated provenance is mechanically proven.
7. Non-runtime flash source-string helpers such as `flash_partial_src` / `flash_reduce_src` must be listed as fixtures so
   audits can distinguish them from selected route debt.

## Conversion Contract

Every handwritten surface in this document is convertible to codegen in principle because each one computes a finite,
well-defined tensor program. The conversion is not complete until the selected runtime route satisfies one of these
targets:

| Target | Allowed as pure? | Meaning | Examples |
|---|---:|---|---|
| `ordinary_tinygrad_graph` | yes | The route is expressed as normal Tensor/UOp graph operations and tinygrad scheduler/codegen lowers it. | Tensor matmul/reduce/dequant graph, ordinary flash graph. |
| `descriptor_owned_uop_codegen` | yes, with audit | A structured spec/search descriptor owns topology and a shared emitter lowers it to UOps consumed by normal codegen. | `Q6KGEMVRouteSpec` pattern, future `FlashDecodeTileSpec`, future `Q4KPrefillTileSpec`. |
| `backend_owned_intrinsic_lowering` | yes, with allowlist | The route uses normal IR and the renderer/backend owns hardware intrinsic lowering. | WMMA selected by tensor-core matcher, renderer-owned dot4/fdot2. |
| `descriptor_wrapped_hand_kernel` | no | A spec selects parameters but calls a route-local handwritten UOp/raw-ISA kernel. | Current prefill graph GEMM schedule over `prefill/wmma.py`. |
| `route_local_custom_kernel` | no | Runtime calls a hand-authored `Tensor.custom_kernel` body directly. | Current direct-packed Q4_K/Q6_K templates. |
| `external_raw_or_binary` | no | Runtime injects raw `Ops.INS`, source strings, inline asm, or `Ops.BINARY`. | Current raw WMMA emitters, native ISA injection. |

The conversion rule:

1. Move semantic facts into data specs.
2. Move topology choices into searchable descriptor fields.
3. Lower through shared emitters or ordinary tinygrad graph/codegen.
4. Make hardware operations backend-owned, not route-local strings or instruction lists.
5. Prove with a generated-only binding gate before changing manifest provenance.

## Cross-Cutting Codegen Workstreams

These are shared dependencies for converting all families, not per-route one-offs.

### A. Provenance And Binding Audit

`extra/qk/pure_kernel_surface_audit.py` is the strict route-surface audit. It is wired into `extra/qk/gate_registry.py`
as `pure_kernel_surface_audit` and intentionally reports `PURE_KERNEL_SURFACE_AUDIT_DEBT_FOUND` until the selected
default surfaces are strict pure machine search. Keep extending that audit so it can answer, per route:

- selected route id,
- selected writer function/file,
- manifest provenance,
- runtime env selector,
- raw markers found (`Ops.INS`, `Ops.BINARY`, `asm volatile`, source strings),
- custom-kernel markers found (`Tensor.custom_kernel`, `Ops.CUSTOM`, `Ops.CUSTOMI`),
- descriptor/spec artifact id when present,
- emitted kernel names observed by the authority harness,
- verdict: `pure`, `descriptor_wrapped_hand_kernel`, `hand_uop`, `external_raw_or_binary`, `rollback_oracle`, `unknown`.

This audit must be route-aware. A marker in a test fixture is not default debt; the same marker in a manifest-attributed
writer is route debt.

### B. Descriptor Registry

Create a single registry for generated route specs:

- decode GEMV specs,
- prefill quant specs,
- prefill WMMA/LDS schedule specs,
- attention tile/combine/live-split specs,
- rollback mapping,
- authority artifacts.

The registry should serialize enough data that a reviewer can reconstruct why a generated kernel shape exists without
reading route-local Python control flow.

### C. Backend Intrinsic Allowlist

Define one allowlist for backend-owned intrinsic lowering:

- WMMA through tinygrad tensor-core matcher / renderer,
- dot4 through renderer-owned helper,
- fdot2 through renderer-owned helper,
- cross-lane shuffle/reduce through backend-owned lowering,
- exp2 or fast math through backend-owned lowering.

Route-local string emission of those same operations remains non-pure.

### D. Shape And Role Policy

Move shape guards and role policy into descriptors:

- model role,
- quant format,
- `M/N/K` or attention `B/Hq/Hkv/Hd/MAXC/Tc`,
- tile sizes,
- split count,
- staging mode,
- combine mode,
- fallback and rollback route.

This prevents generated policy from being scattered across environment checks that call different hand kernels.

### E. Authority Harness Requirements

Every conversion must ship with:

- correctness against current route or mathematical reference,
- route-bound proof from emitted kernel names,
- no-hidden-fallback proof,
- generated-only surface audit,
- W==D or role-level timing comparison,
- rollback route retained until the generated route survives periodic regression.

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

### 6. Small-K Batched Q4_K/Q6_K Primitive GEMM Routes

Files:

- `tinygrad/llm/decode_routes.py`
- `extra/qk/quant/q4_k_gemv_primitive.py`
- `extra/qk/quant/q6_k_gemv_primitive.py`

Runtime functions:

- Q4_K: `q4k_gemm_kernel`
- Q6_K: `q6k_gemm_kernel`

Why non-pure:

- `q4k_primitive_linear_call` routes non-decode `K<=32` batched calls through `q4k_gemm_kernel`.
- `q6k_primitive_linear_call` routes non-decode `K<=32` batched calls through `q6k_gemm_kernel`.
- Both are human-authored UOp templates.

Runtime reachability:

- These are not the main pp512 prefill path, but they are runtime-capable verify/small-batch paths through
  `decode_routes.py`.

Gameplan:

1. Add manifest rows or a manifest sub-row for small-K batched primitive routes.
2. Decide policy under `PURE_MACHINE_SEARCH_ONLY=1`:
   - block these paths,
   - force ordinary tinygrad fallback,
   - or require a generated small-K descriptor route.
3. Convert by reusing Q4_K/Q6_K generated linear specs:
   - `Q4KSmallBatchGEMMSpec`,
   - `Q6KSmallBatchGEMMSpec`,
   - token axis extent `K<=32`,
   - generated partial/reduction layout.
4. Gate against existing hand template for correctness and timing.

Done means:

- Small-K batched routes either have generated-only provenance or are explicitly forbidden under pure-search mode.

### 7. Decode Attention UOp Kernel Families

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

### 8. Native ISA / Precompiled Program Injection

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

### 9. Inline ASM / Source-String Kernels

File:

- `extra/qk/quant/q4_k_gemv_primitive.py`
- `extra/qk/flash_decode.py`

Known functions/surfaces:

- `_vdot4_q4_q8_accum`
- `_q4k_q8_1_vdot_source`
- `q4k_q8_1_vdot_partial_kernel`
- `q4k_q8_1_vdot_builtin_partial_kernel` uses generated source string with `_dp4a`
- `flash_partial_src`
- `flash_reduce_src`

Why non-pure:

- The non-builtin source contains `asm volatile("v_dot4_u32_u8 ...")`.
- Source-string kernels are explicitly not pure.
- Flash source-string helpers are currently treated as non-runtime/probe debt, but must remain excluded from pure routes.

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

## Conversion Backlog

This is the implementation backlog to convert every runtime-relevant handwritten family to codegen.

| Workstream | Replaces | New generated surface | Codegen substrate needed | First gate | Promotion gate |
|---|---|---|---|---|---|
| `prefill_fp16_wmma_codegen` | `build_gemm_pipe`, `build_gemm_lds2`, `route_pf16_graph_gemm` | `PrefillWMMAScheduleSpec -> generated UOps/Tensor graph` | LDS allocation, cooperative stores/loads, barriers, WMMA from codegen, wait lowering | single role fp16 parity + no `Ops.INS` | whole-prefill W==D + throughput near raw-ISA decider |
| `prefill_q4k_fused_codegen` | `build_gemm_lds2_q4k`, `route_q4k_graph_gemm` | `Q4KPrefillFusedSpec -> generated dequant-to-LDS + WMMA` | Q4_K unpack/dequant lowering, LDS staging, WMMA tile scheduler | ffn_gate_up correctness + no raw markers | 14B pp512 beats direct-packed baseline |
| `prefill_q4k_direct_codegen` | `q4k_gemm_packed_load_*`, `q4k_q8_1_*` templates | `Q4KPrefillTileSpec/Q4KMMQSpec -> generated UOps or Tensor graph` | packed nibble decode, grouped reductions, direct output, optional dot4 lowering | direct-packed parity on role shapes | replaces `prefill_q4k_direct_tile4x4_default` |
| `prefill_q6k_direct_codegen` | `q6k_gemm_packed_load_*` templates | `Q6KPrefillRouteSpec -> generated UOps` | Q6_K block decode, token tiling, direct output | Q6 prefill parity | Q6 prefill W==D |
| `smallk_primitive_codegen` | `q4k_gemm_kernel`, `q6k_gemm_kernel` small-K runtime paths | `Q4KSmallBatchGEMMSpec`, `Q6KSmallBatchGEMMSpec` | small token-axis tiling, generated packed decode reductions | K<=32 parity | pure-mode allowed small-batch route |
| `decode_q4k_cleanup` | `q4k_gemv_warp`, `q4k_coop_partial`, lane partition rollback | hardened `Q4KGateUpLaneMap/G3` + optional split/combine specs | descriptor audit, split-K/in-kernel combine codegen | generated-only G3 audit | rollback-only or delete owned templates |
| `decode_q6k_cleanup` | `q6k_*` shipped rollback templates | existing `Q6KGEMVRouteSpec` plus coverage | existing generated UOp emitter | generated/rollback byte identity | delete or quarantine refuted direct route |
| `attention_codegen_specs` | `flash_kernels.py`, `live_split_geometry.py`, `flash_decode_fused_combine.py` hand UOps | `FlashDecodeTileSpec`, `LiveSplitGeometrySpec`, `FlashCombineSpec` | descriptor-owned tile/softmax/PV/combine lowering, cross-lane reductions | one-shape route-bound parity + no fallback | 8B/14B/32B attention W==D |
| `intrinsic_lowering_cleanup` | inline asm/source-string vdot, route-local fdot2/ds_bpermute strings | backend-owned dot4/fdot2/shuffle/exp lowering | renderer/codegen intrinsic allowlist | static no-source-string gate | no pure route reaches inline asm/source string |
| `native_isa_cleanup` | `native_isa_block_tile_graph_node.py` binary injection | normal backend compiler path or deleted probe | backend path emits ISA under codegen, not route injection | no `Ops.BINARY` in route path | no runtime route injects binaries |

### Workstream Details

#### `prefill_fp16_wmma_codegen`

Objective: replace fp16 graph GEMM raw instruction emitters while preserving the schedule search space.

Steps:

1. Define `PrefillWMMAScheduleSpec` with tile shape, wave layout, `BK`, padding, double-buffering, PLR/PLRAB,
   role-selective policy, and relocation policy.
2. Implement a generated lowering that expresses:
   - cooperative global load,
   - LDS write/read,
   - barrier,
   - tiled matmul shape that codegen tensorizes to WMMA,
   - epilogue layout.
3. Add codegen tests that generated kernels contain backend-owned WMMA but no `Ops.INS`.
4. Port `route_pf16_graph_gemm` to select the generated lowering behind a flag.
5. Flip default only after authority beats or matches the raw-ISA substrate.

Blockers:

- current scheduler/codegen may not preserve the intended LDS tile lifetime,
- cooperative partitioning may fail to expose enough parallelism,
- wait scheduling must become backend-owned or conservatively correct.

#### `prefill_q4k_fused_codegen`

Objective: replace the fused Q4_K raw WMMA route with generated dequant-to-LDS plus generated WMMA.

Steps:

1. Define `Q4KPrefillFusedSpec` with Q4_K layout, role shape, `BM/BN/BK`, group decode mode, staging layout, and output
   dtype.
2. Reuse the fp16 generated WMMA substrate for the compute phase.
3. Add generated Q4_K decode-to-fp16-LDS lowering:
   - scale/min extraction,
   - nibble unpack,
   - fp16/fp32 dequant choice,
   - LDS B-tile layout matching WMMA B operand expectations.
4. Validate one role at a time:
   - `attn_qo`,
   - `ffn_gate_up`,
   - `ffn_down`,
   - `attn_kv`.
5. Remove or quarantine `PREFILL_Q4K_WMMA_FUSED` raw route after generated parity/perf.

Blockers:

- dequant VALU scheduling can dominate if not overlapped with LDS/WMMA,
- register pressure can erase WMMA gains,
- role shapes must not graph-explode.

#### `prefill_q4k_direct_codegen`

Objective: replace Q4_K direct-packed UOp templates with descriptor-owned generated prefill code.

Steps:

1. Split Q4_K semantics from topology:
   - `Q4KLayoutSpec`,
   - `Q4KActivationSpec`,
   - `Q4KPrefillTileSpec`,
   - `Q4KReductionSpec`.
2. Convert current direct-packed schedules into descriptor rows.
3. Move helper functions such as scale/min extraction and packed-load dot into shared generated lowering modules.
4. Build two generated targets:
   - lossless direct-packed generated UOp route,
   - fused Q4_K/Q8_1 MMQ route using backend-owned dot4 or normal int matmul.
5. Compare against current direct-packed pp512 baseline, not only synthetic parity.

Blockers:

- current direct-packed template is simple and memory-safe, so replacement must not regress 14B memory behavior,
- `sdot4`/dot4 lowering must be backend-owned to pass purity.

#### `prefill_q6k_direct_codegen`

Objective: give Q6_K prefill the same provenance conversion decode Q6_K already has.

Steps:

1. Generalize `Q6KGEMVRouteSpec` into `Q6KLinearRouteSpec` or add `Q6KPrefillRouteSpec`.
2. Encode batch/token axes and direct output layout.
3. Emit generated UOps from the spec.
4. Replace `q6k_gemm_packed_load_*` runtime calls in `prefill_routes.py`.

Blockers:

- Q6_K prefill may be less urgent than Q4_K but must still be classified accurately,
- generated path must handle large `N` and token tiling without partial-buffer blowup.

#### `attention_codegen_specs`

Objective: make promoted attention routes mechanically descriptor-owned.

Steps:

1. Define `AttentionProblemSpec`: `B`, `Hq`, `Hkv`, `Hd`, `MAXC`, live `Tc`, quant/rope flags.
2. Define `FlashDecodeTileSpec`: split size, staging mode, score/PV fusion mode, lane/wave mapping.
3. Define `LiveSplitGeometrySpec`: live-context split count, occupancy cap, ring-buffer behavior.
4. Define `FlashCombineSpec`: state layout, gmax/lse combine, fused or staged combine.
5. Refactor `flash_decode.py`, `flash_kernels.py`, and `live_split_geometry.py` so runtime selects specs and shared
   emitters lower those specs.
6. Add a generated provenance gate for 8B, 14B/G5, and 32B-style shapes.

Blockers:

- online softmax lifecycle is harder to represent than GEMV/GEMM,
- combine is sensitive to reduction codegen and register scalarization,
- dynamic live-context/ring-buffer semantics must be explicit in the spec.

#### `intrinsic_lowering_cleanup`

Objective: remove route-local hardware strings from pure paths.

Steps:

1. Move dot4/fdot2/ds_bpermute/exp2 decisions into renderer/codegen allowlisted helpers.
2. Replace source-string `_q4k_q8_1_vdot_source` paths with normal UOps or Tensor expressions.
3. Fail selected-route audit on `asm volatile`.
4. Fail selected-route audit on route-local `__builtin_amdgcn_*` unless file/function is in the backend intrinsic
   allowlist.

Blockers:

- some AMD intrinsics may not have first-class UOps today,
- performance may depend on preserving exact dot/reduce lowering.

## Dependency Order

Recommended order:

1. `pure_kernel_surface_audit`: needed first so progress is measurable.
2. manifest truth split: prevent false passes while conversions are underway.
3. Q6 decode audit hardening: cheapest positive-control generated route.
4. Q4 decode G3 audit hardening: second positive control.
5. Q6 prefill spec conversion: easier prefill conversion using known pattern.
6. small-K primitive conversion or pure-mode blocking: closes runtime-capable verify/prefill hand paths.
7. Q4 direct-packed generated conversion: removes live 14B prefill default debt.
8. fp16 prefill LDS+WMMA codegen: foundation for raw WMMA deletion.
9. Q4 fused WMMA codegen: hardest prefill route, depends on fp16 substrate.
10. attention spec conversion: broad but separable once audit conventions are stable.
11. delete/quarantine research-only hand kernels.

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
