# Decode Attention Online-State+PV Tile P6 Lowering Bind Scope

## Goal

Decide whether the existing cross-lane and packed-dot lowerings can bind directly to the P5 online-state+PV tile.

P5 created the first generated route where per-split `m` and `l` live in the tile lifecycle:

```text
flash_score_whole_cache_32_128
flash_online_state_pv_tile_whole_cache_32_128
flash_state_gmax_32_128
flash_state_combine_32_128
```

P6 asks whether this is already enough for codegen lowerings, or whether the tile still needs a lane-sharded dataflow rewrite.

## Current P5 dataflow

Inside `flash_online_state_pv_tile_whole_cache_32_128`:

| State | Current ownership |
|---|---|
| PV `acc[D]` | per local `d` lane, register array `c[G]` |
| per-split `l` | computed redundantly per local `d` lane, stored from `d == Hd` |
| per-split `m` | computed redundantly per local `d` lane, stored from `d == Hd+1` |
| token loop `j` | serial reduce loop inside each lane |
| score dot | still external in `flash_score_whole_cache_32_128` |

This is structurally correct, but it does not yet shard token work across lanes. A cross-lane reduction lowers work that is split across lanes; it has no useful site if every lane still does the full serial `j` loop.

## Lowering targets

| Target | Required site | Current P5 site exists? |
|---|---|---|
| cross-lane max for `m` | lane-sharded score partial max across token/dot lanes | no |
| cross-lane add for `l` | lane-sharded denominator partials | no |
| cross-lane add for `acc[D]` | lane-sharded PV partials for same output D | no |
| packed-dot / `v_dot2` | score production inside/directly fused with tile | no, score still external |

## Expected decision

If the tool confirms no lane-sharded reduction site exists, P6 verdict should be:

```text
ONLINE_STATE_PV_TILE_P6_NEEDS_TOKEN_SHARDED_REWRITE
```

That is not a failure. It means P5 successfully moved state into the tile, and the next step is to split token/dot work across lanes so the existing lowerings have something to lower.

## Next implementation if P6 chooses token-sharded rewrite

Create a P7 candidate:

```text
flash_online_state_pv_tile_xlane_whole_cache_32_128
```

Required structural change:

- keep `Hkv*S` workgroups;
- keep whole-cache identity and no `E_49152`;
- introduce a lane axis that owns token or dot shards;
- compute partial `m/l/acc[D]` per lane shard;
- use cross-lane reduction to merge those partials;
- only then test packed-dot / `v_dot2` direct score fusion.

## Non-goals

- Do not rerun standalone score `v_dot2` as the main path.
- Do not add `WARP_REDUCE_LOWERING=1` globally and hope it binds.
- Do not promote P5/P6 without W==D.
