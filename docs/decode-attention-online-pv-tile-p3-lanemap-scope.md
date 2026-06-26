# Decode Attention Online-Softmax+PV Tile P3 LaneMap Scope

## Goal

Make the P2 structural online-PV tile's lane/work ownership explicit before changing reduction lowering or packed-dot codegen.

P2 proved this generated route identity is clean:

```text
flash_score_whole_cache_32_128
flash_max_32
flash_online_pv_tile_whole_cache_32_128
flash_gmax_32
flash_den_32
flash_combine_32_128
```

P3 answers whether this structural route preserves decode `T=1` parallelism and where the next missing primitive lives.

## Current P2 ownership

The generated online-PV tile currently maps:

| Axis | Owner | Meaning |
|---|---|---|
| `kvh` | global workgroup axis | KV head, 8 groups |
| `s` | global workgroup axis | split-KV chunk, `S = ceil(Tc/L)` |
| `d` | local lane axis | output dimension plus denominator column, `Hd+1 = 129` lanes |
| `j` | reduce axis | token positions inside a split, `L` |
| `g` | register loop | GQA query heads per KV head, `G = Hq/Hkv = 4` |

So the current tile has workgroups:

```text
Hkv * S
```

and each workgroup has local lanes:

```text
Hd + 1
```

This is structurally useful because it preserves split-KV decode parallelism and coalesced V/D-lane access. It is not yet the owned route's full primitive because `m/l/acc[D]` online state and reduction ownership are still split across separate score/max/global metadata stages.

## Required P3 artifact

Create a machine-readable artifact that records:

- profile shape: `Hq`, `Hkv`, `Hd`, `G`, `L`, ctx ladder;
- tile workgroups per ctx: `Hkv*S`;
- local lanes per workgroup: `Hd+1`;
- GQA register accumulators: `G`;
- which state is owned by the tile and which state is still external;
- whether cross-lane reduction is emitted or still absent;
- whether P2 route evidence is present.

## Gate

P3 passes only if:

- P2 latest artifact verdict is `ONLINE_PV_TILE_STRUCTURAL_ROUTE_CLEAN`;
- `flash_online_pv_tile_whole_cache_32_128` is present;
- owned attention programs are absent in the P2 route;
- `E_49152` is absent;
- lane map preserves `Hkv*S` workgroups for all ctx points;
- local lane count is `Hd+1`;
- missing state/reduction pieces are named explicitly.

## Kill

Do not proceed to reduction/codegen changes if:

- route identity is not clean;
- `Hkv*S` parallelism is not preserved;
- denominator/PV lanes are not represented;
- the artifact cannot say where `m`, `l`, and `acc[D]` live.

## Expected verdicts

```text
ONLINE_PV_TILE_P3_LANEMAP_READY
ONLINE_PV_TILE_P3_FAIL__P2_ROUTE_NOT_CLEAN
ONLINE_PV_TILE_P3_FAIL__PARALLELISM_COLLAPSED
ONLINE_PV_TILE_P3_FAIL__STATE_UNATTRIBUTED
```

## Next step after pass

P4 should change codegen/reduction behavior, not just documentation:

- choose a reduction target: per-split max, denominator, or PV accumulation;
- add a generated lane-owned reduction lowering or explicitly classify `SEARCH_BLOCKED_BY_CODEGEN`;
- preserve the P3 lane map and P2 route hygiene.
