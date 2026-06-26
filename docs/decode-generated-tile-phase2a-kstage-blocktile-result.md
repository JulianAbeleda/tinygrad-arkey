# Decode generated tile Phase 2A result: K/V UPCAST + block-tiling decision

## Goal

Execute the combined path requested after the REG-store devectorize scope:

1. Finish the K-stage/coalescer side enough to avoid the full-route verifier wall.
2. Decide whether the next useful action is still incremental load coalescing or the owned-style block tile.

## What landed

- `REG_STORE_DEVEC=1` now hooks an AMD-only, default-off REG-store devectorize pass after codegen devectorize.
- The cache identity gate is REG-store aware and includes K-stage dynamic LDS cases.
- The generated fused-xlane route enables V/PV UPCAST and K-stage UPCAST only under `REG_STORE_DEVEC=1`.
- K-stage masked UPCAST still fails with `PTRCAT(SHRINK)` verification, but the generated attention tile can safely use an unguarded K-stage store with `t_safe`; invalid tail scores are masked before state update.

## Gate results

| Gate | Result |
|---|---|
| `extra/qk_decode_cache_identity_index_gate.py` | `CACHE_5D_REG_STORE_DEVEC_PASS` |
| `dynamic_v_sum_upcast_5d` | PASS |
| `k_upcast_lds_dynamic_unguarded_5d` | PASS |
| `k_upcast_lds_dynamic_masked_5d` | still FAIL, expected/out of route |
| fused-xlane microgate | `FUSED_XLANE_SCORE_PV_MICROGATE_PASS` |
| fused-xlane route gate | `FUSED_XLANE_SCORE_PV_ROUTE_CLEAN__ECONOMICS_NEXT` |
| ISA diff | still `ISA_DIFF_PINNED`; generated `global_load_d16=0`, `global_load_dwordx4=0`, LDS 256 B, cross-lane 20 |
| W==D | `82.7 / 7.2 / 4.1 / 1.1` tok/s @ ctx `128 / 512 / 1024 / 4096` |

## Interpretation

The REG-store and K-stage verifier blockers are no longer the main stop. They are solved enough for a clean, correct generated route.

The remaining gap is structural:

- generated tile is still per-token,
- generated LDS is still 256 B, not owned tile's 8192 B,
- cross-lane count is still 20, not owned tile's 5,
- no owned-style `global_load_d16` marker appears,
- W==D is still GPU-bound and far from baseline at ctx >= 512.

So the next label is not the old REG-store blocker. The useful blocker is:

`SEARCH_BLOCKED_BY_CODEGEN__BLOCK_TILED_MULTI_WARP_TILE_NOT_EXPRESSED`

## Next scope

Build the owned-style generated block tile, not another scalar per-token coalescer tweak:

- TK=16 token block.
- 4 warps / 128-thread workgroup.
- 8 KB LDS staging for K/V.
- One barrier per token block.
- Inner loop reads K/V from LDS.
- Keep `REG_STORE_DEVEC=1`, fdot2, cross-lane reduce, online softmax, raw 5D `cache_kv`, and route-clean lifecycle.

Acceptance for the next step:

- microgate PASS,
- route gate clean,
- ISA `lds` moves toward 8192 B,
- cross-lane count moves down from 20 toward 5,
- generated load shape improves,
- W==D materially improves at ctx 512+.

Do not promote the current Phase 2A route. It is a useful codegen unblock and a small speed improvement, but it is not competitive.
