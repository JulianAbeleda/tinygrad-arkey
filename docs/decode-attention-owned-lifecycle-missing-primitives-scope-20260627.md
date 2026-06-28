# Decode-attention: exhaustive missing-primitives map (owned ASM lifecycle) — 2026-06-27

Built from a line-level disasm read of BOTH tiles (owned `owned_flash_tile_gqa_whole` vs generated
`flash_block_tiled_xlane_score_pv_tile_whole_cache_32_128`). The generated route is route-bound + token-correct +
harness-measured at **35.0/6.7 tok/s = 33.7%/7.1% of owned (103.8/94.6)**. This names every owned discipline the
generated path lacks, across the whole lifecycle, with evidence and a build plan.

## Root cause (one line)
The generated tile runs the softmax recurrence **per-token, unrolled 8-wide and reduced per-token**, where owned
**batches the reduction and defers the rescale**; and the combine is a **redundant gmax+combine pair** instead of a
single fused normalize. Every marker inflation traces to those two.

## The missing primitives (whole lifecycle, leverage order)

### P1 — Batched cross-lane reduction  (ds_bpermute 40 → 5)  [HIGHEST leverage; hardest]
Owned packs all token partials into the lanes and runs the 5-step XOR butterfly **once** (masks hoisted to the
prologue). Generated runs a full 5-step butterfly **per token × 8** = 40, each forcing an `lgkmcnt` drain → the
dominant inner-loop stall (waitcnt 48 vs 21). **Build:** restructure the `tt` accumulation to compute all TK token
partials first, then one batched reduce; hoist the permute masks. Touches the tile's lane mapping (`qk_flash_decode.py:985-1000`).

### P2 — Deferred accumulator rescale + batched prob-exp  (v_exp 16 → 2, v_mul 40 → 6)
Owned issues 2 exps for the whole block and rescales the accumulator once at the block boundary; generated re-runs
`max→sub→mul→exp→cndmask` and the 4-element `acc*corr` rescale **every token**. **Build:** defer the `*corr` rescale
to the block epilogue (apply the product of corrections once), pre-scale Q by `1/sqrt(Hd)` and fold `log2e` once.
Tile source `:1002-1011`.

### P3 — Lazy/streamed V conversion  (vgpr 88 → 64)
Generated eagerly converts the whole V tile f16→f32 (`v41-v76`, ~36 vgprs live) → occupancy loss. Owned streams V
in half, converts the lane's 4 dims lazily per token. **Build:** convert `vsh` per-token at use, don't materialize
the tile. Tile source `:1008`. Closes the occupancy delta (the resource.vgpr parity row).

### P4 — Fused K+V single LDS read  (ds_read 12 → 1)
Owned interleaves K and V in LDS so ONE `ds_load_2addr_stride64_b64` pulls both. Generated issues separate K/V
loads. **Build:** interleave the `ksh`/`vsh` staging layout so one 2-addr load serves both. Staging `:982-984`.

### P5 — Fused combine  (3 dispatches → 2; redundant gmax + double-M-read removed)  [MOST TRACTABLE; building now]
Combine is `flash_state_gmax_kernel` + `flash_state_combine_kernel` (`:469-496`, wired `:1356-1357`): the partial
buffer is written by the tile, the M column read by gmax, then re-read in full by combine. **Build (this commit):** a
single `flash_fused_state_combine_kernel` that computes `gm` inline (pass-1 max over splits) then rescales
(pass-2) — drops the gmax dispatch + the `gm` buffer round-trip. Default-off `DECODE_ATTN_FUSED_COMBINE=1`.
This is the dominant **long-context** delta (the combine grows with split count → the ctx4096 cliff).

## Gates (all on the route-bound harness `qk_decode_route_attribution_wd.py`)
Each primitive: in-model **token-match preserved** (the recurrence/combine reorder must stay greedy-identical) →
the targeted **parity row moves toward owned** (re-measure markers) → **W==D** rises at the targeted ctx without
regressing the other → default-off, owned stays shipped default.

## Honest priority
P5 (combine) attacks the ctx4096 cliff and is self-contained → build+run first. P1+P2 (batched reduce + deferred
rescale) are the highest-leverage for the base tile cost but are real tile restructures. P3 (lazy V) directly closes
the vgpr/occupancy row. If P1+P2+P3+P5 still leave the tile far from owned, that names the **abstraction limit** —
tinygrad's UOp/custom-kernel path cannot express the owned hand-scheduled lifecycle, the honest hard lower bound.
