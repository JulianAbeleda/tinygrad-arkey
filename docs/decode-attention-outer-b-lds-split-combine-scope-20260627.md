# Scope — outer-b LDS-staged split-combine codegen lowering (2026-06-27)

The codegen primitive that closes the breaking point in
`docs/decode-attention-outer-b-split-breaking-point-result-20260627.md`. Default-off, cache-keyed,
microgate-gated, revert-clean. Builds on the recurrence-unroll machinery (`extra/qk_codegen_recurrence_unroll.py`).

## Lever (from the diagnoses — do not re-derive)

The generated block tile is **occupancy-bound** (vgpr 88/88) and the ctx-slope is the **serial outer-`b` carry**.
Recurrence-unroll cannot reach `b` (no `AFTER(_,b)` edge) and a serial b-unroll adds VGPRs (refuted
`SCHED_UNROLL_SPLIT`). The only slope-bending move is to split `b` into K **independent** partitions whose
online-softmax partials live in **LDS** (not VGPR), combined once — independence lets the scheduler overlap the
per-partition long-latency chains without raising register pressure.

## Graph reality (probed `flash_block_tiled_xlane_score_pv_tile_whole_cache_kernel`, toposort=231)

| element | uop | note |
|---|---|---|
| outer block loop `b` | `RANGE axis=3 REDUCE size=NB` const-bound | END(b) single-range (`nsrc=2`); only AFTER-carry is `DEFINE_REG(235)`=dotp reinit |
| inner token loop `tt` | `RANGE axis=5 REDUCE size=TK` | carries the true online-softmax state |
| online-softmax state | `DEFINE_REG 232=acc, 233=den, 234=mx` | `mx` is the unique `MAX`-fed carry; `acc/den` rescaled by `exp(old_m-new_m)` |
| K/V LDS tile | `DEFINE_LOCAL 230/231` | the existing 8 KB staged tile |
| combine ops present | 1×`MAX`, 2×`EXP2` | online-softmax max + corr/p exps — recoverable |

## Transformation (`DECODE_OUTER_B_SPLIT=<K>`, K in {2,4}, K | NB)

For the recognized block-tile sink only (decline otherwise, like `axis_stride`):

1. **Locate** the `b` REDUCE range: const-bound REDUCE whose END is single-range, directly under the workgroup
   END, wrapping the `tt`-END that carries `{acc,den,mx}` (REG 232/233/234, `mx` MAX-fed).
2. **Split** `b` (size NB) into K disjoint sub-ranges `b_k` of size NB/K (offset `k*NB/K`). Reuse the unroll's
   range/inner-range/reinit-reg duplication, but **do NOT thread the carry serially** — each partition gets
   PRIVATE `acc_k/den_k/mx_k` + private init, so the K chains are independent.
3. **Combine** (once, before the final PV store, math = `flash_*combine` at `qk_flash_decode.py:1032-1047`):
   `M = max_k mx_k ; acc = Σ_k acc_k·exp(mx_k−M) ; den = Σ_k den_k·exp(mx_k−M)`. Stage `(acc_k,den_k,mx_k)` in
   LDS to keep VGPR flat (the occupancy guardrail rejects a register-resident K-fold).
4. **Redirect** the final `pout` PV/den/max store (`dd2`, axis 8) to read the combined `(acc,den,M)`.

## Gates (authority order)

1. `extra/qk_decode_attention_block_tile_microgate.py` → `BLOCK_TILE_MICROGATE_PASS` (max_abs ≤ 5e-3) — correctness first.
2. `extra/qk_decode_occupancy_guardrail.py` → `OCCUPANCY_GUARDRAIL_PASS` (vgpr ≤ 88, scratch 0, wg/CU ≥ 4) — no pressure regression.
3. `extra/qk_decode_hotloop_schedule_diff.py` — the selected outer-`b` loop counters/shadow-fill must MOVE (proof the lever hit `b`, not `tt`).
4. Isolated timing `extra/qk_decode_block_tile_isolated_timing.py` — must bend the ctx4096 slope.
5. Only then W==D (`extra/qk_decode_runtime_overhead.py`, `QK_CKPTS=512,4096`) + token-match.

## Stop conditions

- Correctness fails → revert clean, classify the failing layer (split mechanics vs combine math vs LDS staging).
- Correct but occupancy regresses → the partials must move REG→LDS; if already LDS and still over, the lever is
  capacity-walled.
- Correct + occupancy OK + slope does NOT bend → the b-serialization is NOT the latency source; **refute the lever**
  and record (research-mode stop).
- Default-off discipline: flag unset ⇒ byte-identical; add `DECODE_OUTER_B_SPLIT` to the `to_program` cache key.
