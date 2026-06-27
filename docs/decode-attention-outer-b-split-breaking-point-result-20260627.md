# Decode-attention outer-b split — diagnose→solve loop breaking point (2026-06-27)

Result of running the measure-first diagnose→solve loop on the next pure-search decode-attention lever, using the
pure-search gap audit as the diagnostic authority. Authoritative live state: `docs/pure-machine-search-roadmap.md`.

## The loop

1. **Diagnose (audit tool).** `extra/qk_pure_search_gap_audit.py` (now derived-from-live-artifacts, commit
   `4c58406d3`) reports decode-attention pure-search score **60/100**, `wd_pct_of_owned_avg=19.2%` (≈15× off at
   ctx4096), rank-1 next action = *"implement LDS-staged outer-b split-combine lowering (must bend ctx4096 slope +
   pass occupancy guardrail)."* Live sub-instruments:
   - `qk_decode_hotloop_schedule_diff.py`: the **outer `b` loop is the bound** — generated selected loop is
     `outer_b_or_main_ctx_loop`, `ds_bpermute` 40 vs owned 5, shadow-fill 3.75 vs 0.2.
   - `qk_decode_occupancy_guardrail.py`: **PASS but pinned at the ceiling** — vgpr 88/88, wg/CU 4.0/4.0,
     cross_lane 40/40, waitcnt 50/50. Any pressure increase fails.
2. **Solve attempt (codegen tool).** `SCHED_UNROLL=8` (recurrence-unroll) selects `axis=5` (the inner `tt` loop,
   size 16), **never** the outer `b` loop (`axis=3`). Confirmed live with `SCHED_UNROLL_DEBUG=1`: true carries
   `DEFINE_REG(232/234/233)` (acc/mx/den) threaded on `axis=5`.
3. **Breaking point.** The recurrence-unroll **structurally cannot reach `b`.** In
   `flash_block_tiled_xlane_score_pv_tile_whole_cache_kernel` (`extra/qk_flash_decode.py:958-1011`) the
   online-softmax state (`acc`/`den`/`mx`) is carried via `.after(tt)` and merely **closed** by the `b`-END
   (`mxu = ...end(tt).end(b)`, line 1011). There is **no `AFTER(_, b)` true-carry edge**, so `_true_carry_afters`
   returns empty for `b` and `b` is never a candidate (`extra/qk_codegen_recurrence_unroll.py:90-97`).

## Classification

`SEARCH_BLOCKED_BY_CODEGEN__OUTER_B_LDS_SPLIT_COMBINE_LOWERING_NOT_BUILT`

- Even if the unroller *could* select `b`, a **serial** b-unroll re-threads the carry into private registers → adds
  VGPRs onto a tile already at **88/88** → the occupancy guardrail rejects it. This is the exact failure mode of the
  already-refuted `SCHED_UNROLL_SPLIT` (see `docs/decode-attention-pure-search-state-and-learnings-20260627.md`,
  diagnostic truth #2).
- The only slope-bending path is the **outer-`b` LDS-staged split-combine**: split the within-workgroup `b`-range
  into K *independent* block partitions, each keeping its own online-softmax partial `(m, den, pv)` in **LDS** (not
  VGPR), combined once. That lowering does not exist — `bench/qk-decode-outer-b-split-combine/latest.json` stands at
  `OUTER_B_SPLIT_COMBINE_SEARCH_VOCAB_PRESENT__LOWERING_NOT_BUILT`.

## What is proven (don't re-derive)

- The audit tool is now measurement-grounded: built-lowering + search-owned + W==D parity drives it 60→100
  (`PURE_SEARCH_PROMOTABLE`); missing inputs mark it `DEGRADED`. So re-running the loop after building the lowering
  WILL register the change (the old literal score could not).
- The lever is `b`, not `tt` (recurrence-unroll already hides the tt-carry); partials must be LDS, not VGPR
  (occupancy ceiling).

## Next step

Build the codegen primitive — see `docs/decode-attention-outer-b-lds-split-combine-scope-20260627.md`. Then re-run
this loop and record the new state or the next wall.
