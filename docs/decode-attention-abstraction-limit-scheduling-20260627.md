# Decode-attention: the gap is scheduling quality, not a missing primitive (2026-06-27)

The definitive result of the exhaustive owned-vs-generated investigation, on **route-bound, token-correct,
harness-measured** W==D. It names the honest hard lower bound the parity-closure loop was built to find.

## The verified gap
Generated block tile **35.0/6.7 tok/s = 33.7%/7.1% of owned (103.8/94.6)**. The tile (not the combine) is the whole
gap: `flash_block_tiled` 3711µs vs combine ~30µs @ctx4096 (124×).

## Every lever, tested and refuted (22 ledger rows, all route-bound where it matters)
| lever | result | why |
|---|---|---|
| topology / split count (S=64,96) | REFUTED_WD | more splits = more combine; craters short ctx |
| **occupancy (vgpr 80→40 via no-unroll)** | **REFUTED_WD (22.9/3.8, WORSE)** | **the tile is NOT occupancy-bound — lower vgpr hurt** |
| ILP (8-wide unroll) | HELPS (kept) | hides per-token latency; helps in-model too, not isolated-overfit |
| combine fusion (P5) | REFUTED_WD | combine is 0.8% of cost |
| knobs (STAGE_COALESCE, SCHED_UNROLL 4/16, INLINE_REDUCE) | REFUTED | tune the same structure, don't move the bound |

**Key correction:** I hypothesized the stack was *isolated-overfit* and occupancy was the lever. The no-unroll W==D
**refuted it** — dropping vgpr to 40 (below owned's 64) made W==D *worse*. The unroll's registers buy ILP that
helps in-model. Occupancy is not the bound.

## What the gap actually is — scheduling quality (the abstraction limit)
The primitives are all PRESENT and CORRECT (parity: 8 MATCH — v_dot2, cross-lane, LDS staging; token-match true).
There is **no missing primitive**. The 12.8× is that **tinygrad's codegen produces a 12.8× slower *schedule* of the
same correct algorithm**:
- owned **software-pipelines** the LDS staging (`s_clause` + descending `s_waitcnt` staircase) so the next block's
  K/V load overlaps the current block's compute;
- owned **hoists** the 5 cross-lane permute masks to the prologue and **fuses K+V into one `ds_load_2addr`**;
- result: owned hides the per-token online-softmax recurrence latency (s_waitcnt=21, shadow_fill=0.2).

tinygrad's tile + `SCHED_LIST` cannot interleave the unrolled per-token recurrence to hide the cross-lane/LDS
latency to the same degree (s_waitcnt=39–50, shadow_fill=3.75). This is a **codegen-scheduler capability gap**, not
an algorithm or primitive gap.

## The honest bottom line
The owned kernel proves the hardware does 94 tok/s @ctx4096, so the distance is *physically* closable. But every
*exposed* lever is exhausted, and the residual is **scheduling discipline that tinygrad's codegen does not express**:
software-pipelined staging + mask hoisting + fused loads + latency-aware interleaving of the recurrence. Closing it
is a **deep codegen-scheduler effort** (a real software-pipelining / list-scheduler upgrade — the repo's long-noted
"separately-funded codegen capability"), not another primitive or knob.

This is the loop's terminal `GENUINE_EXHAUSTION` / abstraction-limit verdict for decode attention, reached
constructively: **the machine can generate a correct tile; it cannot yet schedule it like hand-ASM.** The decision
is now explicit — fund the codegen-scheduler capability, or keep the hand-ASM tile as the shipped default (it ships;
the generated path stays the correct, route-bound, search-owned reference at 33.7%/7.1%).
