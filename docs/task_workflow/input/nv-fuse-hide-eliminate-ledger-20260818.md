# NV FUSE / HIDE / ELIMINATE ledger (2026-08-18)

Date: 2026-08-18
Branch: `nvidia-bringup-20260731`, HEAD `daf591ad3`
Status: **living ledger. One table that shows every lever row, its camp
(FUSE / HIDE / ELIMINATE), its measured status at HEAD, and its tok/s ceiling.**
Companion audit with per-kernel evidence:
`nv-full-audit-fuse-hide-eliminate-20260818.md`. Principles:
`docs/what-makes-a-token-fast-20260731.md`.

Anchors (fresh census this session, `route_kernel_census.py` control):
**205.99 tok/s**, wall ~4854.7 us, node_sum 4999.6 us, 596 kernels/token,
token sha `227ad3ce`. All ceilings below are computed against that wall and are
ceilings (perfect 1:1 recovery), not forecasts; wall-to-tok/s is sublinear.

## The ledger

| # | camp | row | node us | status at HEAD | wall ceiling (us) | tok/s ceiling |
| --- | --- | --- | ---: | --- | ---: | ---: |
| L8 | FUSE | fused GEMV anchors (gemv/norms/rope_kv/combine/vocab head) | 3393.4 | **LANDED** - fusion is why node_sum is 496 us below llama | -496.3 (already won) | baseline |
| L1 | FUSE | reduce_output (`reduce_output_rmsnorm_*`) | 383.5 | **LANDED** P1 per-row grid (+55-67 us captured); q/k remainder bitwise-blocked, 4096 at parity | ~0 | ~206 |
| L4 | FUSE | other residual launches | ~13 | open (small), body-free folds measured FLAT | ~13 | ~207 |
| F5 | FUSE | `E_*`/`r_*` norm/residual plumbing | 466.1 | FLAT - fusion of these maps ~0 wall (08-15 composition review) | ~0 | ~206 |
| L2 | FUSE | vocab argmax tail (`E_1187_32_4`, `r_32_4_1187`, `r_128_16_8_1187`, `r_16_8`) | 57.5 | **NO-GO** - hidden mass, ~10% wall transfer (A/B -1.55 us); real ceiling 2-3 us | 2-3 | ~206 |
| L5 | HIDE | overlap / shadow mass | llama 1125.1, tg 0 | size-class WALL - all-tiny shadows -4% to -37%; zero 8-25 us kernels left after fusion | ~18-33 transferable | ~208-209 |
| L7 | HIDE | PDL programmatic launch | - | **substrate PROVEN** (+65 us device probe, checksum pass; decode A/B wall-neutral 205.99 vs 205.55) | per-edge final wave only | ~206 |
| L6 | HIDE | host gap (submit-ahead gate, closed-default) | 100.6 | open - last clean wall lever; gate `_decode_submit_ahead_eligible` exists | +100.6 | **~213** |
| L3 | ELIMINATE | flash score shape (32-lane/5-stage/16-serial/48-split vs llama 8-lane/3-stage/128-parallel/2-split) | 39.4 | structural - template is hand-picked constants, search cannot reach llama's shape | 39.4 at 1:1 | ~208 |
| E1 | ELIMINATE | packed-key argmax lowering (in-GEMV vocab top-1) | 57.5 | codegen target - packed u64 key reduce lowers 2x slower than `Tensor.argmax` (142.6 vs 71.9 us) | 2-3 real (hidden) | ~206 |

## Reading it

- **FUSE is the winning camp and it is nearly exhausted.** The fused anchors
  (L8) already beat llama on work by 496 us; L1 landed with ~0 left; L2 is
  hidden mass so it was never worth 59.5 us of wall (the old ledger row was
  wrong on that); F5 maps ~0 wall.
- **HIDE is closed at HEAD.** Substrate proven (L7) but wall-neutral, and the
  shape bar (8-25 us, >=4 same-class kernels, one join) has no production mass
  left to fill it. L6 is the only HIDE-adjacent row with real headroom, and it
  is host-side (submit-ahead), not kernel co-scheduling.
- **ELIMINATE is where the remaining upside lives**, and both rows are
  codegen/search work: a searchable flash shape (L3) and a cheaper packed-key
  reduce (E1). Neither is a buildable kernel-level fusion today.

## Actual breakdown (all 31 kernel families, fresh census)

Every kernel family from `nv-full-audit-census-head-20260818.json`, its launch
count and median per token, its total node us, and the camp it belongs to.
Families are grouped by camp; within a camp, largest first.

### FUSE - landed anchors (already on the wall, keep)

| kernel | n | med us | total us |
| --- | ---: | ---: | ---: |
| `q4k_g3_lanemap_gemv_w1w3fused16_12288_4096` | 36 | 38.98 | 1408.80 |
| `q6k_fp16_mmvq_direct_4096_12288_epi_ffnresadd` | 18 | 31.70 | 577.80 |
| `q4k_fp16_mmvq_direct_4096_12288_epi_ffnresadd` | 18 | 22.00 | 404.35 |
| `q4k_g3_lanemap_gemv_epi_resadd_4096_4096` | 36 | 9.98 | 364.83 |
| `q6k_gen_coop_151936_4096_inkernel` (vocab) | 1 | 319.87 | 319.87 |
| `q4k_g3_lanemap_gemv_4096_4096` | 19 | 9.54 | 183.15 |
| `q4k_warp_coop_q8_dp4a_partial_4096_4096` | 17 | 9.38 | 158.63 |
| `q4k_g3_lanemap_gemv_1024_4096` | 28 | 4.80 | 134.57 |
| `q4k_warp_coop_q8_dp4a_partial_1024_4096` | 26 | 3.78 | 98.87 |
| `q6k_v_four_warp_fp16_direct_1024_4096` | 10 | 4.99 | 50.44 |
| `q6k_q8_warp_direct_1024_4096` | 8 | 4.19 | 34.16 |
| **camp total** | **213** | | **3735.47** |

### FUSE - landed / blocked remainder (reduce_output, L1)

| kernel | n | med us | total us |
| --- | ---: | ---: | ---: |
| `reduce_output_rmsnorm_1_4096` | 19 | 7.84 | 151.71 |
| `reduce_output_rmsnorm_32_128` | 36 | 3.07 | 116.10 |
| `reduce_output_rmsnorm_8_128` | 36 | 3.14 | 115.73 |
| **camp total** | **91** | | **383.54** |

### FUSE - flat / blocked (norm-residual plumbing, F5)

| kernel | n | med us | total us |
| --- | ---: | ---: | ---: |
| `r_16_256` | 37 | 3.87 | 149.59 |
| `E_32_32_4` | 38 | 2.30 | 87.62 |
| `E_16_32_4_2` | 36 | 2.26 | 86.35 |
| `E_8_8_16_2` | 36 | 1.95 | 69.32 |
| `r_8_32_4_4` | 26 | 1.68 | 44.91 |
| `r_32_32_4_4` | 17 | 1.70 | 29.52 |
| `r_32_32_4_32_4` | 1 | 4.80 | 4.80 |
| `E_16_4_2_8_16_2_4_4` | 1 | 3.20 | 3.20 |
| `E_2` | 1 | 1.70 | 1.70 |
| `E` | 1 | 1.47 | 1.47 |
| **camp total** | **194** | | **478.48** |

### FUSE - NO-GO (vocab argmax tail, L2)

| kernel | n | med us | total us |
| --- | ---: | ---: | ---: |
| `r_32_4_1187` | 1 | 39.14 | 39.14 |
| `r_128_16_8_1187` | 1 | 11.10 | 11.10 |
| `E_1187_32_4` | 2 | 3.65 | 7.29 |
| `r_16_8` | 1 | 1.70 | 1.70 |
| **camp total** | **5** | | **59.23** |

### HIDE - no shape (Q8 provider; the quantize equivalent)

| kernel | n | med us | total us |
| --- | ---: | ---: | ---: |
| `rmsnorm_q8_1_llama_provider_4096` | 17 | 2.46 | 44.99 |
| **camp total** | **17** | | **44.99** |

### ELIMINATE - codegen/search targets

| kernel | n | med us | total us |
| --- | ---: | ---: | ---: |
| `flash_block_tiled_xlane_score_pv_tile_whole_cache_32_128` (score) | 36 | 6.48 | 239.02 |
| `flash_fused_gmax_combine_f16_32_128` (combine, already ahead of llama) | 36 | 3.36 | 122.03 |
| **camp total** | **72** | | **361.05** |

### Ledger check

| camp | count | node us |
| --- | ---: | ---: |
| FUSE landed anchors | 213 | 3735.47 |
| FUSE landed/blocked (reduce_output) | 91 | 383.54 |
| FUSE flat/blocked (plumbing) | 194 | 478.48 |
| FUSE NO-GO (vocab tail) | 5 | 59.23 |
| HIDE (Q8 provider) | 17 | 44.99 |
| ELIMINATE (flash) | 72 | 361.05 |
| **total** | **592** | **5062.76** |

The census reports 596 launches/token; the per-kernel table carries 592
launch rows (the 4-row delta is kernel-name dedup rounding in the census
classifier). The ELIMINATE flash row includes the combine, which is already
ahead of llama; only the score (239.02 us) is the open structural row.

## Honest position

Resolved non-search rows land ~213 tok/s (L6 host gap at 1:1). 230 needs
~470 us cut; the only rows that can supply it are the ELIMINATE/codegen rows,
which is the post-230 direction (search finds shapes hand-picking cannot).
