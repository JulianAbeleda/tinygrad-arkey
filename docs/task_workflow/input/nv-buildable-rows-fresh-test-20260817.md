# NV buildable rows: fresh at-HEAD test (2026-08-17)

Date: 2026-08-17. Branch `nvidia-bringup-20260731`, HEAD `a1c827358`.
GPU: RTX 5090 (idle, bench lock held). Harness: `extra/llm_research/decode/route_kernel_census.py`
(the production-route census: prime token at d512, DEBUG=2 kernel launches,
per-kernel medians/counts/totals, token sha + tok/s pins).
Evidence: `/tmp/census_nv_head_20260817.json`.

Status: **fresh at-HEAD measurements for all four scoped rows.** This is the
audit step of the standing pipeline run with the production census tool on the
real route. Pins: token sha `227ad3ce9621f2c382cc722a3c2f1677637d3e3f2bfbf37d6ca652f98880eb4e`
(identical 3/3), first token 271, tok/s median 205.76.

## Fresh per-row numbers (vs the 08-17 ledger)

| row | ledger (node ledger) | fresh census (prime token) | delta | verdict |
| --- | ---: | ---: | ---: | --- |
| L1 reduce_output | 312.1 us | **384.1 us** | +72.0 | row is BIGGER than scoped |
| L2 vocab_aux | 59.5 us | **59.3 us** | -0.2 | confirmed |
| L3 flash (score+combine) | 213.1 + 99.6 = 312.7 | **240.0 + 121.9 = 361.9** | +49.2 | total at llama parity; mix is the story |
| L4 other (truly-small) | 47.2 us | **~13.4 us** | -33.8 | smaller than scoped |

## L1 - reduce_output (fresh 384.1 us, scoped 312.1)

| body | count | med us | total us |
| --- | ---: | ---: | ---: |
| `reduce_output_rmsnorm_1_4096` | 19 | 7.94 | 152.3 |
| `reduce_output_rmsnorm_32_128` | 36 | 3.07 | 116.4 |
| `reduce_output_rmsnorm_8_128` | 36 | 3.16 | 115.4 |

The q/k bodies match the booked P1 geometry (32_128 at 3.07, 8_128 at 3.16).
The 1_4096 body (152.3 us, 40% of the row) is the closed parity side at its
restored 7.94 us geometry. Fresh ceiling at 1:1: wall 4788.3 - 384.1 = 4404.2
us = **227.1 tok/s** (scope said 223.4 from 312.1).

## L2 - vocab_aux (fresh 59.3 us, scoped 59.5 - confirmed)

| kernel | count | med us | total us |
| --- | ---: | ---: | ---: |
| `E_1187_32_4` | 2 | 3.65 | 7.3 |
| `r_32_4_1187` | 1 | 39.20 | 39.2 |
| `r_128_16_8_1187` | 1 | 11.10 | 11.1 |
| `r_16_8` | 1 | 1.70 | 1.7 |

Same four-kernel aux chain the 08-12 scope priced (57.3 then, 59.5 ledger,
59.3 now). The row is real and its number is stable.

## L3 - flash (fresh 361.9 us total, ledger 312.7)

| kernel | count | med us | total us |
| --- | ---: | ---: | ---: |
| `flash_block_tiled_xlane_score_pv_tile_whole_cache_32_128` | 36 | 6.53 | 240.0 |
| `flash_fused_gmax_combine_f16_32_128` | 36 | 3.36 | 121.9 |

At HEAD the running combine is the **f16 variant** (3.36 us x 36 = 121.9).
Score+combine total: census prime-token 361.9 us vs llama 363.3 us (parity);
the same-session ledger counted our flash total at 312.7 (already -50.6
ahead). The two bases disagree by +49 us on our flash total - a
methodology delta (prime-token kernel tm vs steady-replay node ledger) that
must be reconciled before quoting a flash total, but both bases agree the
class is at-or-ahead of llama. The scoped "+39.4 score parity" row is
misleading as a lever: the score sub-class is behind (+66.4 census / +39.4
ledger), the combine sub-class is ahead (-67.8 census / -90.1 ledger), and
the net class is at-or-ahead. Fixing score without giving up the combine win
has little headroom; L3's honest read is "score/combine mix", not "+39.4".

## L4 - other (fresh ~13.4 us, scoped 47.2)

Truly-small single-count launches (not norm/reduce/GEMV families):
`r_32_32_4_32_4` 5.3, `E_16_4_2_8_16_2_4_4` 3.2, `E_2` 1.7, `r_16_8` 1.7,
`E` 1.5 = 13.4 us. The larger `r_32_32_4_4` (17x, 29.2) and `r_8_32_4_4`
(26x, 44.6) rows belong to the norm/reduce epilogue families already counted
in L1/norms, not to "other". The scoped 47.2 us over-counts the row by mixing
epilogue-family kernels into it.

## What this changes

1. L1 is the row: 384.1 us fresh (bigger than the 312.1 scope), 227.1 tok/s
   ceiling at 1:1, and 40% of it is the closed 1_4096 parity body. The open
   q/k remainder is 231.8 us at the booked P1 geometry.
2. L2 is confirmed stable at 59.3 us (211.5 tok/s ceiling).
3. L3 is at parity at HEAD (combine f16 win offsets the score gap); the row
   should be re-scoped as "score/combine mix", not "+39.4".
4. L4 is ~13 us, not 47.2 - the scope must drop the epilogue-family kernels
   from the row.
5. Fresh L1+L2 at 1:1 = 443.4 us -> wall 4344.9 us = **230.1 tok/s** ceiling
   (scope said 228.4 from 371.6). The gap to 240 is still overlap + host gap
   + PDL, per the substrate docs.
