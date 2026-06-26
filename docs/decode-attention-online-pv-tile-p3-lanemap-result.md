# Decode Attention Online-Softmax+PV Tile P3 LaneMap Result

## Verdict

`ONLINE_PV_TILE_P3_LANEMAP_READY`

P3 records the lane/work ownership for the P2 structural online-PV tile before changing reduction or dot lowering.

Artifact:

- `bench/qk-decode-attention-online-pv-lanemap/latest.json`

Tool:

```bash
PYTHONPATH=. python3 extra/qk_decode_attention_online_pv_lanemap.py
```

## Shape

| Field | Value |
|---|---:|
| `Hq` | 32 |
| `Hkv` | 8 |
| `Hd` | 128 |
| `G` | 4 |
| `L` | 256 |
| `W=Hd+1` | 129 |

## Axis Ownership

| Axis | Owner | Meaning |
|---|---|---|
| `kvh` | global axis | KV-head workgroup owner |
| `s` | global axis | split-KV chunk workgroup owner |
| `d` | local axis | V/PV output dimension plus denominator lane |
| `j` | reduce axis | token positions inside split |
| `g` | register loop | GQA query heads per KV head |

## Parallelism Table

| ctx | split count `S` | tile workgroups `Hkv*S` | local lanes `Hd+1` | GQA register accumulators |
|---:|---:|---:|---:|---:|
| 512 | 2 | 16 | 129 | 4 |
| 1024 | 4 | 32 | 129 | 4 |
| 2048 | 8 | 64 | 129 | 4 |
| 4096 | 16 | 128 | 129 | 4 |

## Current State Attribution

Tile-owned now:

| State | Location |
|---|---|
| PV accumulator `acc[D]` | register array `c[G]` inside `flash_online_pv_tile_whole_cache_32_128` |
| denominator contribution | `d == Hd` lane inside tile output width `W=Hd+1` |
| V/PV dimension ownership | `d` local lane axis |

Still external to the tile:

| State | Program |
|---|---|
| score `score[h,t]` | `flash_score_whole_cache_32_128` |
| per-split max `m[h,s]` | `flash_max_32` |
| global max `gm[h]` | `flash_gmax_32` |
| global denominator `den[h]` | `flash_den_32` |
| final rescale/combine | `flash_combine_32_128` |

Missing for the primitive-complete tile:

| Missing piece | Meaning |
|---|---|
| lane-owned online update of `m` | max must move into the tile lifecycle |
| lane-owned online update of `l` | denominator must move into the tile lifecycle |
| cross-lane or equivalent reduction schedule | needed for `m/l/acc[D]` ownership without scalar duplicate work |
| packed-dot score production inside/directly fused with tile lifecycle | needed to stop depending on separated scalar score program |

## Decision

P3 is complete.

Proceed to P4 only by changing reduction/dot codegen or explicitly classifying the missing lowering. P3 itself is structural attribution, not speed promotion.
