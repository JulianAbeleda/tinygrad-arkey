# Owned-vs-generated ISA diff gate — scope (2026-06-26)

Continues `docs/decode-fused-xlane-score-pv-tile-wd-result.md`: W==D refuted the fused tile and localized
the wall to generated-codegen **code quality** (compute-bound on inefficient ISA, ~1665× over roofline at
matched occupancy). This gate pins the *specific instructions* where the generated tile bleeds vs the
owned hand-written AMDGCN tile, so the next work targets concrete renderer/lowering patterns instead of
another attention layout.

## Build on (no new tooling needed)

- `extra/qk_decode_attention_fused_score_state_pv_attribution.py`: `_disasm` (llvm-objdump on captured lib
  bytes), `_hist` (valu/vmem_load/vmem_store/ds/cross_lane/fma_dot/exp/scratch/branch/s_inst/total),
  `_parse_desc` (VGPR/SGPR/LDS/scratch). Reused by import.
- `extra/qk_amdgpu_isa_primitive_audit.py` (general .co auditor) — reference for the flag patterns.
- `bench/qk-prefill-schedule-diff-oracle/static_isa_diff.json` — the `key_diff` narrative format to mirror.
- llvm-objdump/readelf confirmed at `/opt/rocm/llvm/bin` (`isa_tooling_inventory.json`).

## Method

`extra/qk_decode_attention_isa_diff_gate.py`, artifact `bench/qk-decode-attention-isa-diff/latest.json`.
Captures two tiles via the runtime hook (subprocess per arm, env selects the route):
- `owned` (baseline route) → `owned_flash_tile_gqa_whole`
- `xlane` (DECODE_ATTN_FUSED_XLANE_SCORE_PV_TILE) → `flash_fused_xlane_score_pv_tile_whole_cache_32_128`

Disassembles both, computes `_hist` + `_parse_desc` + structural markers (global_load width, s_barrier,
s_waitcnt, v_dot2, cross_lane, scratch), then a normalized diff (delta + ratio xlane/owned per category)
and a `key_diff` narrative. The static ISA is ctx-independent (same compiled kernel), so the diff is too.

## Hypotheses to confirm/refute (from the W==D bleed)

| W==D hypothesis | ISA signal |
|---|---|
| per-token cross-lane reduction (warp_reduce per token×head) | `cross_lane` count xlane ≫ owned |
| per-token LDS barriers | `ds` + `s_barrier` xlane ≫ owned |
| scalar / uncoalesced V loads | `global_load_dword` (xlane) vs `global_load_dwordx4` (owned) |
| register spill | `scratch` > 0 / `has_spill` xlane, owned 0 |
| no native packed dot in the hot path | `v_dot2` / `fma_dot` density |

## Verdicts

- `ISA_DIFF_PINNED` — one or more bleeders identified with concrete ratios (the expected outcome).
- `ISA_DIFF_INCONCLUSIVE` — no dominant bleeder (would push toward dynamic profiling / SQTT).

## What it unlocks

A concrete, prioritized codegen worklist (e.g. "emit `global_load_dwordx4` for the V tile", "hoist the
cross-lane reduction out of the token loop", "eliminate the scratch spill on the acc registers") — the
renderer/lowering frontier, replacing speculative kernel rewrites. The right next step after this is a
renderer change targeting the top bleeder, re-diffed here, not another W==D on a new route.
