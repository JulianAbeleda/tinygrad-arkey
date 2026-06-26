# Decode Attention A3.9 Tile Partial-PV Result

## Verdict

`A3_9_TILE_PARTIAL_PV_NO_TRANSFER`

A3.9 targeted the next meaningful stage after A3.8 attribution: partial PV production.

The candidate `flash_tile_partial_pv_whole_cache_32_128` replaces `flash_partial_coop_vec_whole_cache_32_128` while preserving the A2 generated whole-cache lifecycle.

## Files

- Gate: `extra/qk_decode_attention_a3_9_tile_partial_pv_gate.py`
- Artifact: `bench/qk-decode-attention-a3-9-tile-partial-pv/latest.json`

## Command

```bash
DEV=AMD JIT=1 PYTHONPATH=. .venv/bin/python extra/qk_decode_attention_a3_9_tile_partial_pv_gate.py
```

## Gate Result

| Check | Result |
|---|---|
| A2 generated whole-cache route clean | pass |
| owned tile/combine absent | pass |
| `E_49152` absent | pass |
| token sample matches owned baseline | pass |
| `flash_tile_partial_pv_whole_cache_32_128` present | pass |
| old `flash_partial_coop_vec_whole_cache_32_128` absent | pass |
| full TILE+COMBINE bundle bound | pass |
| material W==D transfer over A2 | fail |

## W==D Result

| ctx | owned tok/s | A2 generated tok/s | A3.9 tile-partial-PV tok/s | A3.9 vs A2 | A3.9 vs owned |
|---:|---:|---:|---:|---:|---:|
| 512 | 105.0 | 78.3 | 78.2 | 99.9% | 74.5% |
| 1024 | 103.2 | 75.7 | 75.6 | 99.9% | 73.3% |
| 2048 | 100.7 | 69.8 | 69.7 | 99.9% | 69.2% |
| 4096 | 95.8 | 60.6 | 60.6 | 100.0% | 63.3% |

## Captured A3.9 Bundle Signature

| Program class | Programs | Status |
|---|---|---|
| score | `flash_score_whole_cache_32_128` | present |
| metadata | `flash_max_32`, `flash_prob_32`, `flash_gmax_32`, `flash_den_32` | present |
| partial PV | `flash_tile_partial_pv_whole_cache_32_128` | present |
| old partial PV | `flash_partial_coop_vec_whole_cache_32_128` | absent |
| combine | `flash_combine_32_128` | present |

## Implementation Note

The first A3.9 gate classified any positive delta as `TRANSFERS`. That was too loose because the observed deltas were within measurement spread. The gate was tightened to require a delta larger than the larger of the A2/A3.9 spread estimates. The corrected canonical artifact reports `A3_9_TILE_PARTIAL_PV_NO_TRANSFER`.

## Interpretation

A3.9 proves that simply renaming/replacing the partial PV stage with an equivalent tile-named generated program does not move W==D.

This is still useful: the route can bind a tile partial-PV program cleanly, but equivalent stage replacement is not enough. The remaining gap requires a real lifecycle fusion, not a one-for-one stage substitution.

The next candidate should combine responsibilities, for example:

- probability + partial PV, removing both `flash_prob_32` and old partial PV in one payload
- score/max/prob/partial online-softmax+PV tile
- LDS/cross-lane tile implementation if the fused generated payload cannot be represented cleanly

## Decision

Promote nothing from A3.9.

Next executable step: build a fused probability+partial-PV tile payload or stop and add deeper per-kernel timing if we want exact stage cost before attempting fusion.
