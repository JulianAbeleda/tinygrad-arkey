# Block tile route-bound W==D — ground truth (2026-06-27)

Closes the route-binding arc. The generated decode block tile is now **route-bound, token-correct, and
harness-measured in-model** — the phantom W==D is replaced by verified data.

## Harness
`extra/qk_decode_route_attribution_wd.py` (harness-as-primitive). Two bugs the first runs exposed, both fixed:
1. DEBUG=2 on a JIT replay shows graph nodes, not kernel names → attribution uses an **eager** forward.
2. tinygrad `getenv` **memoizes**, so two route configs can't coexist in one process (the candidate inherited
   owned's cached flags and fell back) → each route measured in a **fresh subprocess** (clean lifecycle).

## Result (full: ctx 512/4096, NMEAS=40)

| route | ctx512 | ctx4096 | attn kernel | token_match |
|---|---|---|---|---|
| owned (oracle) | 103.8 | 94.6 | `owned_flash_tile_gqa_whole` | — |
| **block tile (full stack)** | **35.0** | **6.7** | **`flash_block_tiled_xlane_score_pv_tile_whole_cache_32_128`** | **true** |
| % of owned | 33.7% | 7.1% | route_bound ✓ | correct ✓ |

`verdict = ROUTE_BOUND__TOKEN_MATCH__WD_BELOW_THRESHOLD`. The block tile is **correct** in-model (greedy
token-identical to owned) and **route-bound**, but ~3× off owned @ctx512 and **~14× @ctx4096** — far below the
90%-of-owned promotion threshold.

## What this changes
- The session-reported snapshot (32.8/6.2) was *approximately right* but **never verified or proven route-bound**.
  Deleted; replaced by `transfer_snapshot_20260627-224000.json` (`authority=harness_measured`, `route_bound=true`).
- `bench/qk-owned-oracle-parity/route_attribution.json` (harness-measured) → the parity `route_bound` row flips to
  **MATCH** and `wd_tok_s` to a **real MISMATCH** (owned 103.8/94.6 vs 35.0/6.7). Matrix: 8 MATCH / 5 MISMATCH /
  1 UNKNOWN (`reduce_placement`).
- The loop now reasons from **ground truth**: the open failed rows (`vgpr` 88 vs 64, `waitcnt` 50 vs 21,
  `shadow_fill` 3.75 vs 0.2, `split_kv_combine`, `wd_tok_s`) are the *real* frontier, and `token_match=true` means
  the gap is **pure performance**, not correctness.

## Honest bottom line
No performance win — the block tile is **3–14× off owned** in verified in-model W==D. But the foundational gap is
closed: the route binds, fires, and is token-correct, the harness measures the kernel that would ship, and the
phantom can no longer reach the loop. The next work is the named failed rows (resource/schedule/lifecycle), now on
real data.
