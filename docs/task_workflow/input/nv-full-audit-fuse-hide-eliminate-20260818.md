# NV decode full audit: FUSE / HIDE / ELIMINATE (2026-08-18)

Date: 2026-08-18
Branch: `nvidia-bringup-20260731`, HEAD `daf591ad3`
Status: **audit record, read-only. Exhaustive classification of every kernel
family in the steady decode token at HEAD into the three camps (FUSE / HIDE /
ELIMINATE) with measured evidence and a verdict per row. No runtime change.**

Principles anchor: `docs/what-makes-inference-fast.md` (the canonical
"what makes a token fast" doc). In particular its wall decomposition:
`T_token >= T_bulk + T_boundary,critical`, its rule that "fewer kernels is not
faster unless a boundary actually leaves the non-overlapped critical path",
and its promotion order (correctness -> emitted ISA -> isolated kernel ->
same-session route -> endpoint). This audit applies that frame to the current
NV decode route and assigns every kernel to the camp that can actually remove
it from the wall.

Evidence (fresh, this session, RTX 5090 idle):
`docs/task_workflow/evidence/nv-full-audit-census-head-20260818.json`
(`route_kernel_census.py` control arm at HEAD: 205.99 tok/s, token sha
`227ad3ce`, 596 kernels/token, 5 graph groups).

## 1. The frame: two ways work leaves the wall, and a priority order

The wall equation (re-derived in `nv-llama-full-trace-lever-ledger-20260817.md`,
residual 0.0 at the time; node_sum now exceeds wall at HEAD, see 1.1):

```text
wall = node_sum - overlap + host_gap
```

Work leaves the wall by exactly two mechanisms, and they are not equal:

| mechanism | how | cost | status of the substrate |
| --- | --- | --- | --- |
| **FUSE** | fold work into a kernel already on the wall (GEMV epilogue). The work rides inside an anchor that is running anyway; if the anchor has spare issue capacity it is free. Removes node mass and needs no scheduling substrate. | body added to the anchor (register/ILP/occupancy pressure) | built and proven on NV (vocab head -85us, reduce_output P1 +55-67us, norm/rope/kv fusions) |
| **HIDE** | keep the work as a separate kernel and co-schedule it so it runs inside another kernel's shadow (llama's PDL). | needs same-size-class kernels (8-25us, >=4, one join), a latch substrate, and the native latch only fires at the last CTA so the window is a final wave | proven on the exec path this session (+65us device probe) but wall-neutral on the real route (A/B 205.99 vs 205.55) |

**ELIMINATE** is the special case of FUSE where the work never exists (algebraic
removal, or better lowering so a fused epilogue is cheaper to emit). It is the
same camp in spirit: it shrinks node_sum without paying a scheduling substrate.

The decision rule, therefore, is a priority order, not a menu:

```text
1. FUSE into a wall kernel    (free - exhaust this first)
2. ELIMINATE algebraically    (work never exists)
3. HIDE as a shadow kernel    (last resort - only if independent AND 8-25us class)
```

"Hide what we can't fuse" is correct only when the remainder clears the shape
bar. Otherwise hiding is negative: all-tiny shadows measured -4% to -37%
overlap (each tiny kernel pays the runqueue-switch penalty on native;
`nv-overlap-substrate-build-scope-20260817.md` 2b/2c).

### 1.1 Fresh position at HEAD

| quantity | value | source |
| --- | ---: | --- |
| tok/s (census control) | 205.99 | fresh this session |
| node_sum | 4999.6 us | fresh this session |
| wall implied | ~4855 us (1000/205.99) | census median |
| node_sum - wall | +144.6 us | implied slack / overlap (not yet verified through the wall-account identity) |
| kernels/token | 596 | fresh |
| graph groups/token | 5 | fresh |

node_sum > wall at HEAD implies some slack exists; the S4 PDL landing may have
created it. This is implied, not re-verified through the exact wall-account
identity (`nv-240-exact-wall-account-20260817.md`). The audit does not depend on
which way that resolves: every row below is classified on its own measured wall
transfer.

## 2. Complete kernel inventory at HEAD (all 596 launches/token)

Class sums from the fresh census (count, node us):

| class | count | node us |
| --- | ---: | ---: |
| q4k_gemv (fused lanemap GEMVs) | 119 | 2078.4 |
| other (FFN-down fp16 GEMVs, q8 providers, reduce_output, warp-coop partials) | 206 | 1723.5 |
| elementwise_fusion (E_/r_ norm/residual plumbing) | 194 | 466.1 |
| flash_decode_attention (score + combine) | 72 | 354.2 |
| vocab_head (fused vocab GEMV) | 1 | 319.9 |
| scatter (vocab argmax tail, 1187-family) | 4 | 57.5 |
| **total** | **596** | **4999.6** |

Per-kernel table (count x median = total us), the exhaustive set:

| kernel | count | med us | total us | class |
| --- | ---: | ---: | ---: | ---: |
| `q4k_g3_lanemap_gemv_w1w3fused16_12288_4096` | 36 | 38.98 | 1408.80 | q4k_gemv (Q anchor) |
| `q6k_fp16_mmvq_direct_4096_12288_epi_ffnresadd` | 18 | 31.70 | 577.80 | FFN-down GEMV |
| `q4k_fp16_mmvq_direct_4096_12288_epi_ffnresadd` | 18 | 22.00 | 404.35 | FFN-down GEMV |
| `q4k_g3_lanemap_gemv_epi_resadd_4096_4096` | 36 | 9.98 | 364.83 | q4k_gemv (K/V anchor) |
| `q6k_gen_coop_151936_4096_inkernel` | 1 | 319.87 | 319.87 | vocab_head (fused) |
| `flash_block_tiled_xlane_score_pv_tile_whole_cache_32_128` | 36 | 6.48 | 239.02 | flash score |
| `q4k_g3_lanemap_gemv_4096_4096` | 19 | 9.54 | 183.15 | q4k_gemv |
| `q4k_warp_coop_q8_dp4a_partial_4096_4096` | 17 | 9.38 | 158.63 | q8 partial |
| `reduce_output_rmsnorm_1_4096` | 19 | 7.84 | 151.71 | reduce_output (L1) |
| `r_16_256` | 37 | 3.87 | 149.59 | input norm reduce |
| `q4k_g3_lanemap_gemv_1024_4096` | 28 | 4.80 | 134.57 | q4k_gemv |
| `flash_fused_gmax_combine_f16_32_128` | 36 | 3.36 | 122.03 | flash combine |
| `reduce_output_rmsnorm_32_128` | 36 | 3.07 | 116.10 | reduce_output (L1) |
| `reduce_output_rmsnorm_8_128` | 36 | 3.14 | 115.73 | reduce_output (L1) |
| `q4k_warp_coop_q8_dp4a_partial_1024_4096` | 26 | 3.78 | 98.87 | q8 partial |
| `E_32_32_4` | 38 | 2.30 | 87.62 | residual eltwise |
| `E_16_32_4_2` | 36 | 2.26 | 86.35 | residual eltwise |
| `E_8_8_16_2` | 36 | 1.95 | 69.32 | residual eltwise |
| `q6k_v_four_warp_fp16_direct_1024_4096` | 10 | 4.99 | 50.44 | V GEMV |
| `rmsnorm_q8_1_llama_provider_4096` | 17 | 2.46 | 44.99 | Q8 provider |
| `r_8_32_4_4` | 26 | 1.68 | 44.91 | norm reduce |
| `r_32_4_1187` | 1 | 39.14 | 39.14 | vocab argmax tail |
| `q6k_q8_warp_direct_1024_4096` | 8 | 4.19 | 34.16 | V GEMV (q8) |
| `r_32_32_4_4` | 17 | 1.70 | 29.52 | norm reduce |
| `r_128_16_8_1187` | 1 | 11.10 | 11.10 | vocab argmax tail |
| `E_1187_32_4` | 2 | 3.65 | 7.29 | vocab argmax tail |
| `r_32_32_4_32_4` | 1 | 4.80 | 4.80 | misc reduce |
| `E_16_4_2_8_16_2_4_4` | 1 | 3.20 | 3.20 | misc eltwise |
| `E_2` | 1 | 1.70 | 1.70 | misc eltwise |
| `r_16_8` | 1 | 1.70 | 1.70 | vocab argmax tail |
| `E` | 1 | 1.47 | 1.47 | misc eltwise |

## 3. Camp 1: FUSE (fold into a wall kernel - the free hiding)

Everything here is either already fused (landed, keep) or a candidate whose
remaining wall value is measured, not assumed.

| row | node us | wall transfer (measured) | status | blocker / next |
| --- | ---: | --- | --- | --- |
| GEMV anchors with fused norm/rope/quant/residual epilogues (`q4k_g3_lanemap_*`, `*_epi_ffnresadd`, `*_epi_resadd`, vocab `_inkernel`; + q8-partial / V GEMVs brings the family to 3735.5) | 3393.4 | landed (fusion is the reason node_sum is 496 us below llama) | **LANDED** | none - this is the baseline that already wins the work ledger |
| `reduce_output_rmsnorm_*` (L1) | 383.5 | P1 per-row grid landed (+55-67 us captured); q/k remainder bitwise-blocked (tree reduce flips token sha, P2/phase6/M1 all NO-GO); 4096 side at parity | **LANDED, remainder ~0** | `nv-reduce-output-site-absorption-scope-20260812.md`, `30cd1c54d` |
| vocab argmax tail (L2: `E_1187_32_4`, `r_32_4_1187`, `r_128_16_8_1187`, `r_16_8`) | 57.5 | A/B: -1.55 us wall for -16.5 us node (~10% transfer - already hidden mass); ceiling ~2-3 us | **NO-GO as wall row** | fused lease only renames the tail 4->2 (`5c30155dd`); real fix is a cheaper packed-key reduce (see 5) |
| Q8 provider (`rmsnorm_q8_1_llama_provider_4096`) | 45.0 | feeds 43 warp-coop consumers; folding into each consumer would re-quantize N times | **keep separate** | this is llama's quantize equivalent; hide-candidate (camp 3), not fuse |
| `E_*` / `r_*` norm/residual plumbing | 466.1 | input norm `r_16_256` + `E_32_32_4` measured body-free FLAT (08-15); remaining rows are 0.5-5 us each | **FLAT / BLOCKED** | `nv-220-composition-review-outcome-20260815.md` - fusion of these maps ~0 wall |
| flash score + combine | 354.2 | combine ahead of llama (-90 us); score +39.4 us structural | **structural** | score shape is a search/codegen target (see 5) |

## 4. Camp 2: HIDE (co-schedule as a shadow kernel - paid hiding)

The shape contract from the exhaustive overlap/substrate work
(`nv-overlap-substrate-build-scope-20260817.md`,
`nv-shadow-size-class-unfuse-substrate-scope-20260817.md`): a shadow that
co-schedules on native must be same-size-class as the anchor (8-25 us), at
least ~4 kernels or one producer-continuation pipeline, and joined once. Tiny
shadows are strictly negative.

| candidate | shape test | measured | verdict |
| --- | --- | --- | --- |
| kv/rope/norm support as shadow behind the GEMV anchor | all-tiny (0.5-1.5 us) | -4% to -37% overlap | **NO** (size-class wall) |
| 1 big + tiny shadow (q6k_v + rope/norm) | mixed | +0.8 to +5.3% (head-wait), -0.1 to -9.4% (rejoin, the real DAG) | **NO** (rejoin is the real topology and it is flat-negative) |
| same-size-class shadow (synthetic) | 1024-class, >=4 kernels | +6.6 to +15.9% | **YES in isolation, no production mass** - we have zero 8-25 us kernels left after fusion |
| PDL latch (producer arrive / consumer wait, in-kernel griddepcontrol) | exec-path wired (S4) | device probe +65 us overlap, checksum pass; decode A/B wall-neutral 205.99 vs 205.55 | **substrate PROVEN, economics flat** - native latch fires at last CTA, so per-edge window is a final wave |

Verdict: HIDE is not a lever at HEAD. The substrate exists (this session's
device probe: control 0 overlap, latch +65 us, checksum correct) but there is
no kernel mass that satisfies the shape bar, and the native latch window is a
final wave rather than llama's whole-kernel window (CUDA
`cudaTriggerProgrammaticLaunchCompletion` fires at kernel start;
`nv-pdl-substrate-verdict-20260817.md` 5). Unfusing work just to hide it adds
the node_sum back and is priced +3.7 to +11 tok/s max, below the honest ceiling
(`nv-overlap-substrate-arithmetic`).

## 5. Camp 3: ELIMINATE (work never exists / cheaper codegen)

The two rows that are structurally fuseable but blocked on emission cost. Both
are **codegen/search targets**: if the lowering gets cheaper, previously
NO-GO fusions become profitable.

| row | node us | root cause | measured | status |
| --- | ---: | --- | --- | --- |
| in-GEMV vocab top-1 (fold the 4-kernel argmax tail into the vocab GEMV epilogue) | 57.5 | the packed u64 (max,index) key construction + MAX reduce lowers 2x slower than `Tensor.argmax` (142.6 vs 71.9 us, microgate 08-05); custom `emit_q6k_vocab_top1_reduce_kernel` cost ~0.89 ms/token | NO-GO at current lowering | **codegen/search target** - a cheaper key-pack or reduction shape flips the economics; `nv-packed-argmax-microgate-record-20260805.md`, `nv-vocab-aux-chain-fusion-scope-20260812.md` |
| flash score shape (8-lane reduce, 3 shuffle stages, 128-column parallel, 2 KV splits vs our 32-lane/5-stage/16-serial/48-split) | 39.4 | the flash template is hand-picked above the search (LANES=32, WARPS=4, TK=16, S=48, stage_width=1 are constants, not levers); search cannot reach llama's shape | structural NO-GO today | **codegen/search target** - make the shape searchable (the "generic flash kernel search" task) |

## 6. What the audit says about the 230 path

The honest ceiling with resolved non-search rows stays ~213-215 tok/s
(`nv-llama-full-trace-lever-ledger-20260817.md` L6 + L1). The two open rows
that could beat it are both ELIMINATE/codegen rows (5), not FUSE or HIDE rows:
cheaper packed-key lowering for in-GEMV argmax, and a searchable flash shape.
That is exactly the direction the user wants for the post-230 audit: our
fusions already win the work ledger (-496 us vs llama); the remaining lever is
making the fusions cheaper to emit and letting search find shapes hand-picking
cannot.

Not relitigated (closed with evidence): multi-stream overlap (FLAT at HEAD,
width-4 bandwidth-bound), all-tiny shadow (size-class wall), vocab two-program
chain (NO-GO), packed-argmax route (2x slower), DP4A Q4 down (NO-GO +45), F3
norm folds (FLAT), flash single-stage (NO-GO +82.5), QMD dependence-counter
linkage (WEDGED), hold_membar-only latch (checksum fail).

## 7. Standing next steps (unchanged by this audit)

1. L6 host-gap A/B: open the closed-default submit-ahead gate
   (`_decode_submit_ahead_eligible`, `model.py:2117`), measure the real delta
   against the ~100 us ceiling (~+4-5 tok/s). Last clean wall lever.
2. Codegen/search audit (the ELIMINATE rows above): cheaper packed-key reduce
   lowering, searchable flash shape.
