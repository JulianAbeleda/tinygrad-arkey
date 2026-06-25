# P3 beam coalesce route result

Date: 2026-06-25
Verdict: `P3_BEAM_COALESCE_ROUTE_PASS`

## What changed

`BEAM_COALESCE=1` now owns the q4k FFN gate/up lane-partition route selection through the static COALESCE scorer.
The old `Q4K_GEMV_SCHEDULER=4` route remains as a comparator, but is no longer required to select the lane-partition candidate.

## W==D result

Artifact: `bench/qk-scheduler-gemv-vs-owned/coalesced_dequant_mE_20260625-160623.json`

| ctx | owned tok/s | beam_coalesce tok/s | route |
|---:|---:|---:|---|
| 512 | 103.6 | 103.7 | 72 lane-partition kernels, 0 owned q4k warp |
| 1024 | 101.9 | 101.8 | 72 lane-partition kernels, 0 owned q4k warp |
| 2048 | 99.2 | 99.2 | 72 lane-partition kernels, 0 owned q4k warp |
| 4096 | 94.7 | 94.6 | 72 lane-partition kernels, 0 owned q4k warp |

Tokens matched at every ctx. The harness verdict remained `PROCEED_P3_SEARCH_GENERALIZATION` with `beam_route=True`.

## Current boundary

This is search-owned route selection, not full generic upstream beam synthesis. The route is still narrow to the proven q4k FFN gate/up shape, and generic `add_gpudims` REDUCE substitution remains unchanged.
