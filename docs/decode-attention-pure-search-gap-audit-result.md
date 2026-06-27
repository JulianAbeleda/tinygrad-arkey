# Decode attention pure-search gap audit result

Canonical tool: `extra/qk_pure_search_gap_audit.py`.

Canonical artifact: `bench/qk-pure-search-gap/latest.json`.

## Verdict

`PURE_SEARCH_PARTIAL__TIME_DELTA_EXPLAINED__VOCAB_GAPS_IDENTIFIED__NOT_PROMOTABLE_YET`

The existing fine-tuning audit (`extra/qk_decode_hotloop_schedule_diff.py`, commit `3158b6677`) is the timing/schedule oracle, but it is not the full pure-machine-search oracle. The missing layer is the wrapper that joins timing deltas to primitive/search-vocabulary status. This result adds that wrapper.

## What the audit answers

| Axis | Answer |
|---|---|
| Time delta | Generated stack transfers in-model, but remains below owned: 32.8 vs 103.2 tok/s at ctx512 and 6.2 vs 93.8 tok/s at ctx4096. |
| Transfer | Full generated stack is +72.6% at ctx512 and +77.1% at ctx4096 versus the block-tile route without the stack. |
| Prior generated gap | Full stack is 5.05x / 6.89x faster than the prior fused-xlane route at ctx512 / ctx4096. |
| Remaining gap | Owned is still 3.15x faster at ctx512 and 15.13x faster at ctx4096. |
| Schedule verdict | `HOTLOOP_SCHEDULE_DIFF__GENERATED_CROSSLANE_OVERHEAD_BOUND` from the split-aware hotloop tool. |
| Current hotloop JSON status | `SPLIT_AWARE_HOTLOOP_READY`; the extractor now selects the real outer loop on both owned and generated disassembly. |
| Primitive/vocab verdict | `VOCAB_PARTIAL__FOUNDATION_PRIMITIVES_VISIBLE__OUTER_B_SPLIT_AND_OCCUPANCY_SEARCH_MISSING`. |
| Pure-search score | 60 / 100 for decode attention. |

## Present or refuted primitives

| Primitive | Status | Meaning |
|---|---|---|
| `CrossLane.ds_bpermute_reduce` | present, refuted as gap | Owned and generated use the same reduce family; do not build a new ds_permute primitive. |
| `TileMemory.lds_tile` | present, generated default-off | LDS block tile exists in generated path. |
| `DotLowering.v_dot2` | present, generated default-off | `v_dot2` is visible in generated block-tile ISA. |
| `LaneMap.cooperative_stage` | present, generated default-off | Cooperative staging composes with coalesced-load lowering. |
| `Math.fast_exp2_valid_domain` | present but manual flag | `DECODE_FAST_EXP2` closes a real +8-9% delta, but BubbleBeam does not own it yet. |
| `Sched.recurrence_unroll_list` | present but manual flag | `SCHED_UNROLL` / `SCHED_LIST` transfer, but are manually selected. |

## Missing or not search-owned vocabulary

| Missing item | Status | Why it matters |
|---|---|---|
| `OuterBlockLoop.lds_staged_split_combine` | missing search vocab | The ctx slope is the outer `b`-block online-softmax carry; current unroll targets inner `tt` only. |
| `ResourceModel.occupancy_guardrail` | missing search scoring | The tile is VGPR/occupancy-bound; pressure-increasing changes regress. |
| `Scheduler.pressure_aware_latency_hiding` | partial, not search-owned | Generated code still has the scheduling/pipelining residual versus owned hand-shaped code. |
| `Audit.split_aware_hotloop_oracle` | present | Backward branches are target-parsed, loop candidates are enumerated, and owned/generated selected loops are classified. |

## Next actions

| Rank | Action | Gate |
|---:|---|---|
| 1 | Build occupancy guardrail gate | Abort candidates that raise VGPR or lower waves/CU versus the best stack. |
| 2 | Use the split-aware hot-loop audit as the preflight for new split candidates | Require selected-loop class and counters to move before implementation. |
| 3 | Add/search LDS-staged outer-`b` split-combine primitive | Must bend ctx4096 slope without increasing VGPR occupancy cost. |
| 4 | Bind manual winning flags into BubbleBeam/FutureSight | Candidate provenance must change from manual flags to search-owned selection. |

## Why this is the right boundary

The project no longer needs another prose-only audit to say "generated is slower." The useful audit is now two-dimensional:

| Dimension | Required output |
|---|---|
| Time explanation | What measured delta remains and which loop/resource explains it. |
| Vocabulary attribution | Whether the missing move is present, refuted, manual-only, search-owned, or absent. |

This result makes `3158b6677` one input to the pure-search verdict, not the verdict itself.
