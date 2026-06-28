# Decode-attention tile-vs-combine breakdown — the combine is a red herring (2026-06-27)

Route-bound, eager DEBUG=2 per-kernel timing of the generated block-tile route @ctx4096. **Overturns the
"combine dominates" thesis** (which came from the stale 2026-06-21 *owned-route* split-KV economics).

## The measurement (per layer, ctx4096)
| kernel | time | share |
|---|---|---|
| **`flash_block_tiled...` (the TILE)** | **~3711 µs** | **~99%** |
| `flash_state_combine` | ~19.7 µs | 0.5% |
| `flash_state_gmax` | ~10.7 µs | 0.3% |

**The tile is ~124× the combine.** At ~36 layers, the tile alone (~3.7ms × 36 ≈ 133ms) is essentially the whole
149ms/token decode step (6.7 tok/s). Owned's attention is ~290µs/layer → the **tile is the entire 12.8× gap**.

## Consequences
- **`split_kv_combine` is NOT the bottleneck** for the block-tile route. The parity row's `COMBINE_TAX_DOMINATES`
  came from the 2026-06-21 owned-route economics; on the route-bound block tile the real economics is
  `COMBINE_SMALL_AMDAHL_LIMIT` (combine ≈0.8%, even a free combine cannot move W==D).
- **P5 (fused combine) is correctly REFUTED** — it removed a ~10µs dispatch from a ~3711µs step.
- The S=96 crater @ctx512 was the combine growing *relative to a smaller tile at short ctx*; at ctx4096 the tile
  dominates absolutely.

## Re-prioritized frontier — attack the TILE, not the combine
The tile's 3711µs is the **per-token serial recurrence** (the disasm analysis: reduce-per-token + rescale-per-token,
8-wide unrolled). The leverage order is now:
1. **P1 batched cross-lane reduce** (bpermute 40→5) — owned reduces all tokens' partials once; generated pays the
   full 5-step butterfly + lgkmcnt drain **per token**. This is the dominant tile stall.
2. **P2 deferred rescale + batched exp** (exp 16→2, mul 40→6) — defer the `*corr` accumulator rescale to the block
   boundary; pre-scale Q once. Removes the per-token serial `max→exp→rescale` chain.
3. **P3 lazy V conversion** (vgpr 88→64) — closes occupancy.

These are real tile restructures (the tile's lane mapping + accumulation), not knobs and not the combine. Each
gated on the route-bound harness (token-match + tile-time drops + W==D). This is the honest next build; if P1+P2+P3
cannot move the 3711µs materially, that names the abstraction limit (tinygrad's UOp path cannot express owned's
batched/deferred per-token schedule).
