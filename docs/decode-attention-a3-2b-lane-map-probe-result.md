# Decode Attention A3.2b Lane-Map Probe Result

## Verdict

`A3_2B_ATTENTION_LANE_MAP_NOT_WIRED`

The generated A2 attention route is clean, and the cross-lane/LanePartition building blocks exist, but attention does
not yet have an explicit x-lane score program.

## Artifact

- `bench/qk-decode-attention-a3-2b-lane-map/latest.json`
- Tool: `extra/qk_decode_attention_a3_2b_lane_map_probe.py`

## Checks

| Check | Result |
|---|---:|
| A2 route clean | yes |
| score program present | `flash_score_whole_cache_32_128` |
| x-lane score program present | no |
| `extra/qk_lane_partition_reduce.py` exists | yes |
| `extra/qk_warp_reduce_lowering.py` exists | yes |
| `extra/amd_warp_reduce.py` exists | yes |
| GEMV G3 LaneMap example exists | yes |
| global cross-lane blocker artifact exists | yes |

## Interpretation

The next wall is not absence of cross-lane primitives. It is that generated attention has not been given an explicit
lane-owned score kernel.

The required next implementation is:

```text
DECODE_ATTN_SCORE_XLANE=1
flash_score_whole_cache_xlane_32_128
```

This should follow the working GEMV G3 pattern:

- `UOp.special(32, "lidx0")`
- lane owns a subset of `Hd=128`
- per-lane partial accumulation
- `lane_partition_reduce_sum`
- same score buffer output contract

## Decision

Proceed to implementation of the scoped x-lane score kernel.
