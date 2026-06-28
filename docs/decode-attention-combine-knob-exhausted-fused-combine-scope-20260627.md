# split_kv_combine: knob search exhausted → fused-combine primitive (2026-06-27)

Closes the loop's knob phase on the dominant long-context delta and escalates to a real codegen capability, on
route-bound ground truth.

## Decision (per the closure loop)
The parity matrix's biggest open delta is `split_kv_combine` (`COMBINE_TAX_DOMINATES`) driving the long-context
cliff (block tile 33.7% of owned @ctx512 → 7.1% @ctx4096). Its only search axis is the split count
`DECODE_ATTN_FUSED_XLANE_SCORE_PV_S`. That axis is now **exhausted** on route-bound, token-correct, harness-measured
data:

| split S | ctx512 | ctx4096 | outcome |
|---|---|---|---|
| 48 (base) | 35.0 | 6.7 | the canonical block-tile route |
| 64 | — | — | TOOLING_BUG (route fell back; pre-route-binding) |
| **96** | **7.6** | **7.0** | **REFUTED_WD** — *craters* ctx512 (35.0→7.6, 4.6× worse), ctx4096 within ±21ms noise |

The generator now returns `NO_UNTRIED_CANDIDATE_TARGETS_A_FAILED_ROW` for `split_kv_combine` → `SEARCH_SPACE_BUG`:
**stop knob search; add the missing primitive.**

## Refreshed combine evidence (route-bound, replaces the stale 2026-06-21 owned-route economics)
S=96's crater at ctx512 is direct, route-bound proof the combine tax dominates: **more splits = more combine work**,
and the combine is a larger fraction when KV is small (short ctx). Split-count is the *wrong* lever — it trades away
short context for nothing at long context. The combine itself must get cheaper, not more numerous.

## The fused-combine primitive (the next capability — scope)
**Target:** the block-tile route's TWO-kernel combine — `flash_state_gmax_kernel` + `flash_state_combine_kernel`
(`extra/qk_flash_decode.py:469/480`, wired at `:1356-1357`): it materializes per-split partial `(m, den, PV[Hd])`
into `po`, then a global-max kernel, then a log-sum-exp merge. Owned fuses/cheapens this (its
`DECODE_ATTN_AMDGCN_COMBINE` "base"/"hd64" is a single tight combine).

**Approaches (in order of leverage):**
1. **Fuse gmax into the combine** — one kernel instead of two (drop a full launch + the `gm` materialization).
2. **Cheaper combine** — match owned's combine occupancy/bytes (the B5 "hd64" shape): fewer workgroups, coalesced
   partial reads, no redundant `PV` round-trip.
3. **Epilogue fusion** — fold the cross-split merge into the tile's last pass where feasible (bounded by the
   cross-split reduction that forces a separate stage).

**Gates (authority order, all on the route-bound harness):**
- correctness: in-model **token-match** preserved (any reordered log-sum-exp must stay greedy-identical — gate #6).
- the **`split_kv_combine` parity row must move toward owned** (re-measure combine fraction; else `SEARCH_SPACE_BUG`/`TOOLING_BUG`).
- **W==D** via `qk_decode_route_attribution_wd.py` (route_bound + token_match + tok/s); accept only if ctx4096 % of
  owned rises materially without regressing ctx512.
- default-off, cache-keyed; owned stays the shipped default.

**Pre-work (cheap, do first):** regenerate split-KV economics (`qk_split_kv_economics_audit.py`) from a measured
tile_us/combine_us breakdown of the *route-bound* block tile (DEBUG=2 per-kernel timing of `flash_block_tiled` vs
`flash_state_combine`), so the combine fraction target is quantified before building.

## Could-it-close caveats (the user's 6, on real data now)
The owned kernel proves the hardware can hit 94 tok/s @ctx4096, so the distance is closable *if tinygrad can express
the same lifecycle*. The live risks: (3) UOps may not express "keep state local + avoid materializing split outputs
+ fused combine"; (4) the vgpr 88-vs-64 gap may be structural (data-ownership/lane-map, not schedule); (5) latency
hiding (waitcnt 50 vs 21) may need real scheduler control. If the fused combine + a work-removal pass on vgpr cannot
move the row, that names a concrete **abstraction limit** — the honest hard lower bound the loop is meant to prove.
