# Decode-attention outer-b split — codegen lowering BUILT + REFUTED (2026-06-27)

Closes the breaking point in `docs/decode-attention-outer-b-split-breaking-point-result-20260627.md`
(`SEARCH_BLOCKED_BY_CODEGEN__OUTER_B_LDS_SPLIT_COMBINE_LOWERING_NOT_BUILT`). The lowering is now **built and
correct** — and **constructively refuted on speed**. Scope:
`docs/decode-attention-outer-b-lds-split-combine-scope-20260627.md`.

## What was built

`extra/qk_codegen_outer_b_lds_split.py` — a codegen UOp lowering (`DECODE_OUTER_B_SPLIT=<K>`, default-off,
cache-keyed, byte-identical when unset; hooked in `tinygrad/codegen/__init__.py` beside the recurrence-unroll). It
splits the serial outer-`b` REDUCE loop into K **independent** partitions over disjoint block sub-ranges — each with
private online-softmax state, private inner ranges, and a private K/V LDS tile — then reconstructs the flash
log-sum-exp combine (`M=max_k mx_k ; acc=Σ acc_k·exp(mx_k−M) ; den=Σ den_k·exp(mx_k−M)`). It detects the block-tile
structure (b REDUCE END with 3 reg post-reads; `mx`=the MAX-fed reg; `acc`=the range-indexed array) and **declines**
(byte-identical) on anything it does not positively recognize.

## Gate results (K=2, full best-stack)

| gate | result |
|---|---|
| correctness `BLOCK_TILE_MICROGATE` | **PASS** — max_abs ≤ 1.14e-05 (tighter than baseline 1.53e-05), all 4 cases |
| isolated timing ctx512 | 0.402 → **0.471 ms (+17%)** |
| isolated timing ctx4096 | 2.840 → **3.385 ms (+19%)** |
| gen/owned ratio | 51.6→61.5 (ctx512), 91.9→110.4 (ctx4096) — **worse** |
| VGPR | 88 → **176** (occupancy guardrail ceiling 88) |
| LDS | 8192 → **16384** |

## Verdict: `OUTER_B_SPLIT_COMBINE_LOWERING_BUILT__REFUTED_ON_SPEED__OCCUPANCY_TAX`

The split is **correct but ~18% slower at both contexts; it does not bend the slope.** Mechanism: partition
independence (the whole point — overlapping the K long-latency chains) requires privatizing state. That doubles
**VGPR (88→176)** and **LDS (8192→16384)** → an occupancy tax that outweighs the ILP/latency-overlap gain on this
**occupancy-bound, HBM-leaning** tile. This is the **same failure class as the refuted `SCHED_UNROLL_SPLIT`**
(ILP-via-state), now confirmed on the `b` axis with a fully correct implementation — exactly what **diagnostic
truth #2** predicted ("levers must REMOVE work, not add ILP-via-state").

**LDS-staging the partials does not escape it:** the hot online-softmax accumulators must be register-resident
during the inner `tt` loop (LDS read-modify-write every iteration would be far slower); independence still requires
private registers, so the VGPR tax stands either way. Sequential partitions sharing registers remove the tax but
also remove the overlap → only combine overhead remains (also slower).

## Where the real lever is (next layer)

Within-warp `b`-split is the wrong axis. `b`-parallelism must come from **more workgroups** (more `s`-splits /
smaller `L`), which the existing split-KV combine already merges — and that lever is **capped by the split-KV
combine economics** (already characterized: `COMBINE_TAX_DOMINATES` / `COMBINE_SMALL_AMDAHL_LIMIT`,
`docs/split-kv-economics-audit-result-20260621.md`). The within-tile lever that *can* win is **work-removal**
(no new state), like `DECODE_FAST_EXP2` — not a split.

## Disposition

The lowering is **kept** (default-off, byte-identical when unset, microgate-gated) as a correct, documented codegen
capability and a recorded refutation asset — not a promotion candidate. The pure-search audit reflects this: the
outer-b contract reports `lowering_built=true, bends_slope=false, refuted=true`, and the gap-audit withholds the
20-pt slope component (decode-attention score stays an honest **60/100**; overall **67/100**). Do not re-chase the
within-warp split as a speed lever.
