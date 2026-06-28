# Phase 1 EXPOSE result: the scheduling wall is NOT authoritatively confirmed (2026-06-27)

"Do it again" = re-ran the diagnose→build loop on the codegen-scheduler capability. Step 1 (Phase 1 EXPOSE,
`extra/qk_decode_hotloop_schedule_diff.py`) came back **inconclusive** — and per measure-first discipline that
**gates the Phase-2 modulo-scheduler build**, exactly as the occupancy hypothesis was gated and refuted.

## What Phase 1 EXPOSE found (route-bound hot loops, owned vs generated)
| metric (selected hot loop) | owned | generated | ratio |
|---|---|---|---|
| ds_bpermute (cross-lane) | 5 | 10 | 2.0× |
| global_load | 22 | 34 | 1.5× |
| s_waitcnt | 21 | 34 | 1.6× |
| ds_bpermute latency-shadow-fill | 0.2 | **0.6** | gen hides *more* |

**Verdict: `HOTLOOP_SCHEDULE_DIFF__SPLIT_AWARE_PARITY_OR_STRUCTURAL`** — not a clean `SCHEDULING_BOUND`.

## Why this qualifies the scheduling thesis
- The generated loop's latency-shadow-fill is *higher* (0.6 vs 0.2): the unroll's ILP already overlaps independent
  work into the reduce-wait. The gap does **not** read as *exposed* latency that a modulo scheduler removes.
- It reads as **structural**: ~2× cross-lane reduces + ~1.5× loads per loop. But **2× static ops cannot explain the
  12.8× wall time** — so the static gate is genuinely inconclusive, not pointing cleanly at either scheduling or
  structure.
- Both tiles are far above the HBM-bound floor (~17.5µs/layer @ctx4096 vs owned 290µs / gen 3711µs), so both are
  compute/stall-bound — but **which** (stall = scheduling, busy = structural) is unresolved statically.

## The decision point (honest)
The exhaustive reference-grounded scope (`decode-codegen-scheduler-capability-scope-v2-references-20260627.md`) is
sound and ready. But its **Phase 1 verdict does not justify committing to the Phase-2 build**: a modulo scheduler
is the right tool *only if* the tile is stall-bound on exposed reduce/recurrence latency, and the static gate does
not show that. Building it on an unconfirmed thesis is the exact mistake the occupancy refutation taught us to avoid.

**Prerequisite to the build (the real next step):** a **dynamic** cycle attribution — stall (memory/LDS/cross-lane)
vs VALU-busy — on the generated tile @ctx4096.
- `rocprof` is **not installed** in this env.
- `extra/sqtt/roc.py` (SQ thread trace) exists and is the repo's profiling path; wiring it to capture
  `flash_block_tiled...` and bucket cycles is the next build task.
- Outcome routing: **stall-bound → build Arm A (P-SWP modulo scheduler)** as scoped; **VALU-busy/structural →**
  the lever is reducing the 2× cross-lane (warp-reduce efficiency, `qk_warp_reduce_lowering.py`) + 1.5× loads
  (P-FUSE), *not* a scheduler.

## Net
Scope: delivered, reference-grounded, ready. Phase 1 EXPOSE: run, **inconclusive** — the wall is not authoritatively
a scheduling wall on the purpose-built static gate. The honest path forward is the **dynamic profile (sqtt)** to
attribute the 12.8×, then build the arm the data names — not the modulo scheduler on faith.
