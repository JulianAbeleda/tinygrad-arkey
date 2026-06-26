# Decode Attention Online-State+PV Tile P6 Lowering Bind Result

## Verdict

`ONLINE_STATE_PV_TILE_P6_NEEDS_TOKEN_SHARDED_REWRITE`

P6 checked whether the existing cross-lane and packed-dot lowerings can bind directly to the P5 online-state+PV tile.

Artifact:

- `bench/qk-decode-attention-online-state-pv-p6-lowering-bind/latest.json`

Tool:

```bash
PYTHONPATH=. python3 extra/qk_decode_attention_online_state_pv_p6_lowering_bind.py
```

## Current P5 Route

```text
flash_score_whole_cache_32_128
flash_online_state_pv_tile_whole_cache_32_128
flash_state_gmax_32_128
flash_state_combine_32_128
```

## Lowerings Available

| Lowering | Available |
|---|---|
| `extra/qk_fdot2_lowering.py` | yes |
| `extra/qk_warp_reduce_lowering.py` | yes |
| `extra/qk_lane_partition_reduce.py` | yes |

## Bind Decision Matrix

| Target | Required site | Current P5 site | Bindable now? |
|---|---|---|---|
| cross-lane `m` | lane-sharded partial max for same `(h,s)` | full serial `j` loop per `d` lane | no |
| cross-lane `l` | lane-sharded partial denominator | full serial `j` loop per `d` lane | no |
| cross-lane `acc[D]` | multiple lanes contribute partial PV to same output D | each `d` lane owns one D and does full token loop | no |
| packed-dot inside tile | score production inside/directly fused with tile | score still external | no |

## Interpretation

P5 moved `m/l` state into the tile, but it did not shard token or dot work across lanes. Every local `d` lane still performs the full serial token loop.

That means the lowerings exist, but there is no useful site for them to lower:

- cross-lane reduction needs lane-sharded partials;
- packed-dot needs score production inside or directly fused with the tile;
- global `WARP_REDUCE_LOWERING=1` would not fix this by itself.

## Decision

Next implementation is P7:

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

Do not do next:

- global `WARP_REDUCE_LOWERING` without a lane-sharded site;
- standalone score `fdot2` rerun as the main path;
- metadata fusion;
- combine-only optimization.
