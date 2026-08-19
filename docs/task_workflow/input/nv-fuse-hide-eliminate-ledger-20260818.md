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
| L5 | HIDE | overlap / shadow mass | llama 1125.1, tg 0 | **LANDED 2026-08-19** reuse-lane arena coloring + native two-GPFIFO readiness placement reproduce the NO_MEMORY_PLANNER ceiling (decode census 21/11 matches planner-off; +6.1 tok/s this session, token sha identical) | ~140 us this session (~81 us in the prior session) | ~212 |
| L7 | HIDE | PDL programmatic launch | - | **substrate PROVEN** (+65 us device probe, checksum pass; decode A/B wall-neutral 205.99 vs 205.55) | per-edge final wave only | ~206 |
| L6 | HIDE | host gap (submit-ahead gate, closed-default) | 100.6 | open - last clean wall lever; gate `_decode_submit_ahead_eligible` exists | +100.6 | **~213** |
| L3 | ELIMINATE | flash score shape (32-lane/5-stage/16-serial/48-split vs llama 8-lane/3-stage/128-parallel/2-split) | 39.4 | structural - template is hand-picked constants, search cannot reach llama's shape | 39.4 at 1:1 | ~208 |
| E1 | ELIMINATE | packed-key argmax lowering (in-GEMV vocab top-1) | 57.5 | codegen target - packed u64 key reduce lowers 2x slower than `Tensor.argmax` (142.6 vs 71.9 us) | 2-3 real (hidden) | ~206 |

## Reading it

- **FUSE is the winning camp and it is nearly exhausted.** The fused anchors
  (L8) already beat llama on work by 496 us; L1 landed with ~0 left; L2 is
  hidden mass so it was never worth 59.5 us of wall (the old ledger row was
  wrong on that); F5 maps ~0 wall.
- **HIDE overlap is now landed, but it is only the modest slice the current
  two-GPFIFO substrate can expose.** L5's reuse-lane fix moves
  `reduce_output_rmsnorm_8_128` and `E_8_8_16_2` to the auxiliary queue and
  matches the planner-off ceiling, worth ~6 tok/s (~140 us) this session, not
  llama's full ~946 us shadow. L6 remains the only other HIDE row with real
  headroom and is host-side.
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

## 2026-08-19 P3 update: measured flash geometry search

The L3 (ELIMINATE flash score shape) row now has a measured, correctness-cleared
winner population. Evidence: `docs/task_workflow/evidence/nv-flash-geometry-search-20260819.json`.

- Population: 432 legal candidates over `lane_width x token_block x stage_width x
  reduce_structure x dot_pair_width x split_count`; `score_group_width` and
  `warps` are pinned because sub-lane groups and `warps < query_group_size` are
  numerically invalid in the current emitter (now rejected by validation).
- Correctness: all 432 candidates match the production fused tile+combine output
  on CUDA (`matches_control=True`), max observed abs diff `1.49e-08`.
- Authoritative body (`nsys --trace=cuda`, 400 back-to-back launches + 20 warmup,
  control bracketed): control `4.289 us` in-session (pinned 4.19 us), and 11
  candidates beat the pinned 4.19 us. Best tile bodies:
  `flash_block_tiled_xlane_score_pv_tile_whole_cache_32_128_sw4_ri_dpw4` and
  `..._sw4_dpw4` at `3.968 us` median (420 instances each).
- Isolated llama matched body was pinned at `4.10 us` (grid.y=2), so the best
  candidates clear the llama isolated body as well.
- Promotion wiring landed but **not booked**: `_flash_decode_tile_geometry_lease`
  now threads the searched tile geometry through
  `flash_decode_attention_route -> flash_decode_live_split_block_tile`, and
  `extra/llm_research/decode/nv_flash_geometry_ab.py` is the control/candidate/
  control wall bracket.
- P4 wall bracket (2026-08-19, fresh process per arm, reverse
  control/candidate/control): the bitwise-identical `stage_width=2` candidate
  preserves the exact token stream but improves wall by only `~39.6 us/token`,
  below the standing `+50 us` promotion bar. The larger isolated-body winners
  (`dot_pair_width=4` / inline variants) change fp reduction order
  (`max_abs 8.8e-04`) and therefore fail the exact-token gate, so they are not
  promotable. Evidence: `docs/task_workflow/evidence/nv-flash-geometry-ab-20260819.json`.

Status: **P3 gate met** (isolated body beat 4.19 us cold-discipline). **P4
wall promotion NO-GO**: best bitwise-identical candidate is sub-bar
(~39.6 us < 50 us) and the faster bitwise-breaking candidates cannot be
promoted under the exact-logits gate.

## 2026-08-19 clean-GPU re-run (27B server unloaded)

The earlier P4 bracket ran while the 27B llama-server was resident. After it
was unloaded (`nvidia-smi` 32071 MiB free), the reverse bracket was re-run on
the idle GPU: fresh process per arm, interleaved
control/candidate/control/candidate/control, settled windows with the first
per-arm window dropped as prefill-capture, `DEV=CUDA`, under
`flock -w 600 /tmp/gpu-bench.lock`. Evidence:
`docs/task_workflow/evidence/nv-flash-geometry-ab-clean-20260819.json`.

- Token identity: control and candidate are bitwise identical in the clean
  session (`sha256 e3f81cdb...`, first token 271). The contended-session sha
  was `bf1dc829...`; the drift confirms the token stream is
  memory/state-adaptive across sessions even though it is stable within a
  session. Absolute tok/s stayed ~177 after the unload, so freeing VRAM did not
  recover the ~206 historical anchor.
- Pooled settled samples: control n=18 median 176.549 tok/s (5664.13 us/token);
  candidate (`stage_width=2`) n=10 median 176.482 tok/s (5666.29 us/token).
- Median delta: candidate -0.07 tok/s, i.e. ~2.2 us/token slower. Bootstrap
  95% CI: [-1.36, +1.19] tok/s -> wall delta CI [-43.9, +37.8] us/token
  (positive = candidate faster). The +50 us promotion bar is above the upper
  bound.

**Clean verdict: the bitwise-identical `stage_width=2` flash-shape candidate is
statistically neutral and NOT promotable.** The prior +39.6 us reading was
contended-session noise, not a real wall gain. The faster isolated-body
winners (`dot_pair_width=4` / inline variants) still change fp reduction order
and fail exact-logits identity, so the flash-shape codegen lever is closed at
the P3 gate with a measured-negative P4. This completes the scoped
flash-shape search end to end.

## 2026-08-19 overlap-substrate proof (HIDE/L5)

The remaining HIDE question was whether the planner-on multi-queue route can
recover the `NO_MEMORY_PLANNER` overlap ceiling without disabling the planner.
The blocker was a false WAR/WAW edge: `reduce_output_rmsnorm_8_128` reused the
arena slot of `reduce_output_rmsnorm_32_128`, whose last reader is a sibling
norm, so the runtime serialized the two independent branches.

Fix landed in `tinygrad/schedule/memory.py`:

- `_fanout_lanes` gives overlapping sibling fan-out outputs distinct arena
  lanes.
- `_independent_reuse_lanes` colors a dead buffer's slot only to a
  dependency-reachable successor; otherwise it hands the new writer a fresh
  lane. This removes the planner-introduced false dependency.
- The lane key is now `(device, copy_flag, fanout_lane, reuse_lane)`.
- Native NV construction is two compute GPFIFOs by default
  (`HCQ_NUM_COMPUTE=2`) with generic readiness placement on
  (`HCQ_NV_READY_PLACEMENT=1`).

Measured on `DEV=NV`, depth 512, fresh process per arm, token sha
`1d299b89...` identical across arms:

| arm | tok/s | wall us/token | decode queue census |
| --- | ---: | ---: | ---: |
| serial control (1q) | 205.99 | 4854.8 | 32/0 |
| planner-off ceiling (2q + placement) | 211.50 | 4728.1 | 21/11 |
| landed reuse lanes (2q + placement) | 212.12 | 4714.2 | 21/11 |

Evidence:
`docs/task_workflow/evidence/nv-overlap-substrate-reuse-lanes-20260819.json`.
The landed decode-graph queue census exactly matches planner-off and moves the
target `reduce_output_rmsnorm_8_128` / `E_8_8_16_2` nodes to queue 1.

**Verdict: the overlap substrate is proven and landed.** It is a real but
modest lever (~6 tok/s, ~140 us this session; ~81 us in the earlier session),
not llama's full ~946 us shadow. The remaining path to 254 tok/s is still
codegen/search plus any deeper multi-queue construction beyond the currently
qualified two GPFIFOs.

## 2026-08-19 route-to-parity theory sweep (T1/T2/T3)

Fresh landed control re-confirmed at `210.8 tok/s / 4744.5 us/token`, token sha
`1d299b89...` (`nv-fresh-critical-path-audit-20260819.json`). Four candidate
levers were tested in same-session flocked A/Bs on `DEV=NV`; none clears the
`+50 us/token` bar.

| theory | measured | booking |
| --- | --- | --- |
| T3 flash coarse split S=4/S=2 | S=4 +763 us, S=2 +1672 us slower; tokens bitwise identical to S=48 | NO-GO, S=48 already optimal (`nv-flash-coarse-split-ab-20260819.json`) |
| T1 early PDL trigger | START vs END `griddepcontrol.launch_dependents` = 0.24 us noise; only the landed QMD latch overlaps (+99.8 us) | NO-GO, no new lever (`nv-pdl-early-trigger-20260819.json`) |
| T2 cross-layer anchor+shadow | 3 us support hides ~1-5 us behind 100 us anchor on the landed 2-GPIFO substrate | NO-GO, re-measures landed substrate, not new wall (`nv-anchor-shadow-probe-20260819.json`) |
| deeper queues / critical path | queue-count sweep bound +26.4 us beyond 2q; fresh PROFILE=1 capture has replay-boundary inflation | closed (`nv-fresh-critical-path-audit-20260819.json`) |

Bottom line: the 685.6 us gap to llama's same-session anchor is llama's own
overlap of support mass that tinygrad has already fused away (tinygrad node_sum
is ~496 us below llama's), plus ~100 us host gap. The remaining levers are
structural (llama-style multi-stream pipelining of work tinygrad no longer
emits, and the 5-replay host-gap structure), not flash/vocab/PDL codegen.
