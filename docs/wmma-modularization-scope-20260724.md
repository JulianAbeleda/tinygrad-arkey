# wmma.py Modularization + Centralization — Scope (2026-07-24)

## Goal
`tinygrad/schedule/wmma.py` (886 lines, ~35 functions) is a monolith mixing 6 concerns and is the single
coupling point 5 modules import from ("too much power"). Split it into a cohesive package with a clear
concern-per-module layout, and CENTRALIZE the duplicated loop-state/fragment emission that 5 kernels
copy-paste. Behavior MUST be byte-identical (same emitted UOps → same ISA); this is a structural refactor,
not a rewrite.

## Purpose
- Modular: each concern in its own module so a change (e.g. the fragment ABI) touches one place.
- Centralized: the `wr`/`rd`/`fr`/`state_write`/`state_read`/`fragment` closures are re-implemented in
  amd_gfx1100_q16_kv64/q32/grid/qk_stats/pv_slice with only the owner-id (9604/9704/9804) differing — one
  shared emitter instead of 5 copies. This is the concrete "power reduction".
- Enables the pure-search direction: once the loop-state/fragment/softmax primitives are shared, generic,
  and composable, a `FlashPrefillAttentionSpec` can compose them as DATA (the real A' lift) instead of the
  current fixed hand functions. This refactor is the prerequisite for that.

## Target layout — package `tinygrad/schedule/wmma/`
Keep `from tinygrad.schedule.wmma import <name>` working for ALL 5 importers (rangeify.py, postrange.py,
fused_attention.py, extra/qk/phase_abi_v1_resource_probe.py, extra/qk/benchmark_split_shared_attention.py)
by re-exporting the full public API from `wmma/__init__.py`. No importer changes.

- `wmma/fragments.py` — tile gather / fragment index-map / WMMA-shaped emission primitives:
  grouped_tile_load, tile_gather, build_owned_fragment_index_map, lower_tile_gather,
  lower_attached_tile_gather, emit_tile_gather_shaped_wmma, adapt_wmma_fragment, shaped_wmma.
- `wmma/softmax.py` — online-softmax primitives: row_softmax_lds_repack, amd_gfx1100_row_softmax_repack,
  amd_gfx1100_row_softmax_state, amd_gfx1100_row_softmax_initial, OnlineSoftmaxBlockTransition,
  online_softmax_block_transition, amd_gfx1100_broadcast_row_state, amd_gfx1100_pv_c_lane.
- `wmma/loop_state.py` — THE CENTRALIZED emitters (see below): the shared loop-state read/write and packed
  fragment loader currently duplicated as wr/rd/fr/state_write/state_read/fragment across 5 kernels.
- `wmma/kernels.py` — the fixed-geometry hand kernels (assembled from the primitives above):
  amd_gfx1100_q16_attention, _q16_kv32_attention, _q16_kv32_hd128_attention, _q16_kv64_hd128_loop_attention,
  _q32_hq4_hkv2_kv64_hd128_loop_attention, _q16_grid_hd128_loop_attention, _q16_grid_qk_stats_stage,
  _q16_grid_pv_slice_stage.
- `wmma/composite.py` — composite tile-carrier glue + reports: construct_hd16_tile_carriers,
  composite_reduce_hd16_carriers, emit_hd16_dual_tile_wmma, adapt_composite_tile_fragments,
  composite_reduce_tile_report, amd_tile_wmma_boundary_report, OnlineSoftmaxTile, online_softmax_tile.
- `wmma/__init__.py` — re-export every currently-public name (the union above) so external imports are
  unchanged. Verify the re-export set == the current `dir()` public surface.

(Exact function→module assignment may shift once dependency edges are mapped; the INVARIANT is the public
import surface and emitted UOps are unchanged.)

## The centralization (the real work)
5 kernels define near-identical closures differing only by owner id + fixed args:
- amd_gfx1100_q16_kv64_hd128_loop_attention: state_write/state_read/fragment (owner via AMDLoopStateSpec)
- amd_gfx1100_q32_...: state_write/state_read/fragment
- amd_gfx1100_q16_grid_hd128_loop_attention: wr(...,owner=9604)/rd(...,owner=9604)/fr(grid=grid)
- amd_gfx1100_q16_grid_qk_stats_stage: wr(...,owner=9704)/rd/fr
- amd_gfx1100_q16_grid_pv_slice_stage: wr(...,owner=9804, role="acc" fixed)/rd/fr/stat
Design (`wmma/loop_state.py`), parametrized so each call site is byte-identical to its current closure:
- `loop_state_write(reg, value, *, role, owner, rng?, block=0, offset=0, access="write", lanes=8)`
  -> the tuple of `UOp(Ops.AMD_ATTENTION_LOOP_STATE, void, (reg.index(offset+i).store(value.gep(i)),),
     arg=AMDLoopStateSpec(role, access, block, lane=i, owner))` for i in range(lanes).
- `loop_state_read(reg, init, *, role, owner, rng, block=0, final=False, lanes=8)`
  -> the `Ops.STACK(float.vec(8), tuple(UOp(AMD_ATTENTION_LOOP_STATE, float, (reg,init[,rng]),
     arg=AMDLoopStateSpec(role, "final_read"/"read", block, lane=i, owner))))`.
- `packed_fragment_load(owner_uop, *, role, head_block, grid, lane, col, rng, group)`
  -> `UOp(Ops.AMD_PACKED_FRAGMENT_LOAD, half.vec(16), (owner_uop,lane,col,rng,group),
     arg=AMDPackedFragmentLoopSpec(role, head_block, grid))`.
Each kernel replaces its local closure with a thin lambda binding owner/rng/grid to the shared emitter.
The emitted UOp for every call site must be identical (verify by AST/ISA equality, below). The `stat`
closure in pv_slice is stage-specific -> leave local (do not force-share).

## INVARIANTS (hard)
1. Byte-identical emitted kernels. Extraction/centralization must not change any produced UOp. These
   kernels are gated by unit tests + isolated captures + the pure-search authority; any UOp change is a
   regression.
2. External import surface unchanged: `from tinygrad.schedule.wmma import X` works for all 5 importers and
   every X that resolves today still resolves.
3. No logic changes, no "improvements", no signature changes to public functions. Pure move + de-dup.
4. Do NOT touch the AMD*Spec definitions in uop/ops.py or the renderer.

## VALIDATION (gates — run after each step)
- Import surface: for the CURRENT wmma.py, capture `sorted(n for n in dir(module) if not n.startswith('_'))`;
  after refactor the package's public dir must be a SUPERSET (same names resolve). Plus explicit import of
  every symbol the 5 importers use.
- Unit: `pytest -q test/unit/test_online_softmax_tile.py test/unit/test_shared_attention_compiler_capture.py`
  -> 6 fail / 93 pass baseline UNCHANGED (compare set, not count).
- Isolated capture: `python -m extra.qk.generate_shared_attention_captures --output-dir <d>` -> all 4 routes
  254 vgpr / 0 spills / 0 scratch (byte-for-byte kernel identity is the real proof).
- Injection numerics unaffected: a4_numerics (8B+14B) still PASS 6.1e-5; varkv still PASS.
- Pure-search guard still passes (this refactor must not change any route provenance).

## Execution order (incremental, gate each)
1. Create the package skeleton + move functions into modules with `__init__.py` re-exporting all public
   names. NO de-dup yet. Gate (imports + unit + capture). This is the pure "modular" step.
2. Centralize the loop-state/fragment emitters into `wmma/loop_state.py`; rewrite the 5 kernels' closures as
   thin bindings. Gate again (esp. isolated capture byte-identity). This is the "centralized" step.
3. (Later, separate scope) the pure-search A' lift: a FlashPrefillAttentionSpec composing these now-shared
   primitives as data.

## Out of scope
- No behavior/perf changes, no new kernels, no spec/descriptor work (that's the follow-on A' lift).
- Do not delete any kernel variant (even if it looks unused — several are AST-swap/stage targets).
