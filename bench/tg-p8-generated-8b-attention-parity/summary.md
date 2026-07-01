# TG-P8.0 8B Attention Baseline

Verdict: **TG_P8_0_PASS_BASELINE_PINNED**

| ctx | owned tok/s | gen tok/s | % owned | owned attn us/fwd | gen attn us/fwd | token_match | route_bound |
|---|---|---|---|---|---|---|---|
| 512 | 107.4 | 94.4 | 87.9% | 592.92 | 2184.52 | True | True |
| 4096 | 97.7 | 93.7 | 95.9% | 1577.84 | 2245.88 | True | True |

## Per-kernel attention wall split (us per forward, summed over layers)

### ctx 512
owned:
- owned_flash_combine: 224.48us (36x, 6.236us/occ)
- owned_flash_tile_gqa_whole: 368.44us (36x, 10.234us/occ)
generated:
- flash_block_tiled_xlane_score_pv_tile_whole_cache_32_128: 1402.64us (36x, 38.962us/occ)
- flash_state_combine_32_128: 584.8us (36x, 16.244us/occ)
- flash_state_gmax_32_128: 197.08us (36x, 5.474us/occ)

### ctx 4096
owned:
- owned_flash_combine: 224.16us (36x, 6.227us/occ)
- owned_flash_tile_gqa_whole: 1353.68us (36x, 37.602us/occ)
generated:
- flash_block_tiled_xlane_score_pv_tile_whole_cache_32_128: 1466.08us (36x, 40.724us/occ)
- flash_state_combine_32_128: 583.28us (36x, 16.202us/occ)
- flash_state_gmax_32_128: 196.52us (36x, 5.459us/occ)

