# TG-P8.0 8B Attention Baseline

Verdict: **TG_P8_0_PASS_BASELINE_PINNED**

| ctx | owned tok/s | gen tok/s | % owned | owned attn us/fwd | gen attn us/fwd | token_match | route_bound |
|---|---|---|---|---|---|---|---|
| 512 | 107.6 | 104.0 | 96.7% | 591.84 | 1165.92 | True | True |
| 4096 | 97.9 | 93.3 | 95.3% | 1579.56 | 2273.2 | True | True |

## Per-kernel attention wall split (us per forward, summed over layers)

### ctx 512
owned:
- owned_flash_combine: 223.44us (36x, 6.207us/occ)
- owned_flash_tile_gqa_whole: 368.4us (36x, 10.233us/occ)
generated:
- flash_block_tiled_xlane_score_pv_tile_whole_cache_32_128: 382.12us (36x, 10.614us/occ)
- flash_state_combine_32_128: 586.2us (36x, 16.283us/occ)
- flash_state_gmax_32_128: 197.6us (36x, 5.489us/occ)

### ctx 4096
owned:
- owned_flash_combine: 224.52us (36x, 6.237us/occ)
- owned_flash_tile_gqa_whole: 1355.04us (36x, 37.64us/occ)
generated:
- flash_block_tiled_xlane_score_pv_tile_whole_cache_32_128: 1493.24us (36x, 41.479us/occ)
- flash_state_combine_32_128: 583.92us (36x, 16.22us/occ)
- flash_state_gmax_32_128: 196.04us (36x, 5.446us/occ)

