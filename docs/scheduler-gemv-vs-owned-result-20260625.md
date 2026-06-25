# Scheduler-generated GEMV vs the owned warp GEMV — Milestone 6 result (2026-06-25)

## Verdict: **`M6_SCHEDULER_GEMV_TRAILS_OWNED`** (cross-lane works in-model; a scheduler GEMV is ~2× off the oracle; the gap is NOT the reduce)

The first end-to-end "how close is a search-generated kernel to the hand oracle" measurement for the decode
weight-GEMV. Follows the mixed-reduce lowering (`ada720375`) and the cross-lane lowering (`70b43a8f0`). Honest,
expected outcome — the *measurement and the quantified gap* are the deliverable.

## Setup

Three arms, only the FFN gate/up GEMV (Q4_K 4096×12288) differs; standalone clock-pinned in-process interleaved
A/B (`extra/qk_scheduler_gemv_vs_owned_wd.py`), real per-token `.item()` W==D, NMEAS=30, REPEATS=3:
- **owned** — the hand warp `custom_kernel` (oracle): `Q4K_GEMV_SCHEDULER=0`.
- **sched_lds** — scheduler-generated fp matvec (`_fallback`, lazy Q4_K→fp16 dequant fused), LDS-tree group reduce:
  `Q4K_GEMV_SCHEDULER=1 WARP_REDUCE_LOWERING=0 MV_ROWS_PER_THREAD=1`.
- **sched_xlane** — same scheduler matvec, but the group reduce auto-lowered to the `ds_bpermute` ladder:
  `Q4K_GEMV_SCHEDULER=1 WARP_REDUCE_LOWERING=1 MV_ROWS_PER_THREAD=1`.

(`MV_ROWS_PER_THREAD=1` gives the scheduler GEMV a scalar lane reduce; the cross-lane ladder declines vectorized
UPCAST>1 reduces in this first pass — see below.) Route-fire is verified per arm.

## Results (whole-decode tok/s, gate/up routed)

| ctx | owned | sched_lds | sched_xlane | xlane vs lds | sched_xlane vs owned |
|----:|----:|----:|----:|----:|----:|
| 512  | 103.4 | 50.6 | 50.5 | −0.08% | −104.5% |
| 1024 | 101.6 | 50.2 | 50.0 | −0.22% | −103.1% |
| 2048 | 99.1  | 49.6 | 49.4 | −0.31% | −100.6% |
| 4096 | 94.3  | 48.3 | 48.2 | −0.30% | −95.8% |

- `tokens_match` True across all 3 arms, all ctx (scheduler GEMV is correct).
- Route-fire: owned arm = 72 owned `q4k_gemv_warp_12288` kernels / 0 scheduler; sched arms = 0 owned / 18 scheduler
  gate/up kernels (sched_xlane: emitting `ds_bpermute`).

## Findings

1. **The cross-lane capability holds in-model.** A scheduler-generated model GEMV does route through `ds_bpermute`
   (Milestone 5 confirmed beyond the toy matvec). Route fires, output correct.
2. **A scheduler GEMV trails the owned kernel ~2×** (50 vs ~100 tok/s). Expected: the owned kernel reads packed
   Q4_K words with coalesced access + a 4-way block-group-K split; the scheduler arm fuses a lazy Q4_K→fp16 dequant
   (recomputed per token) and lacks those.
3. **Cross-lane is ~neutral for this GEMV** (−0.08…−0.31%). For a bandwidth-bound GEMV the group-reduce is a tiny
   fraction of the work; replacing the LDS-tree + `s_barrier` with `ds_bpermute` saves almost nothing against the
   weight-read cost. **The reduce is not the bottleneck** — so cross-lane, which only fixes the reduce, is not the
   lever for closing the search-vs-oracle gap. This matches the DNR-arc caveat: exposing an instruction makes the
   primitive *searchable*, but reaching owned-kernel perf is a separate problem.

## Implication for the search-vs-hand-tuned goal

The bottleneck a search must close to approach the owned GEMV is the **Q4_K dequant lifecycle + packed-word memory
pattern + block-group-K**, not the reduce. The cross-lane lowering remains a correct, now-in-model-proven primitive
in the search space, but it is low-leverage for this role. Recorded as the next named search-space levers in
`bench/qk-search-spaces/decode_ffn_gemv_gfx1100_v1.json`:
- a fused packed-Q4_K dequant in tinygrad ops with coalesced word loads (vs the lazy fp16 dequant);
- block-group-K / occupancy via opts so a scheduler GEMV gets the owned K-split;
- per-component cross-lane (UPCAST>1) — lower priority, since the reduce isn't the bottleneck.

## Status

Research measurement only; **no default change**. `Q4K_GEMV_SCHEDULER` and `WARP_REDUCE_LOWERING` stay default-off.
The owned warp GEMV remains the shipped default. Artifacts: `bench/qk-scheduler-gemv-vs-owned/{decision.json,wd.json}`.
