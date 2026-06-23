# Prefill ASM Instruction Scheduler — Inc 2 Scope + Result (2026-06-23)

## Verdict: `ASM_SCHED_CROSS_MOTION_SOUND` + `LATENCY_REORDER_PERF_NEUTRAL` + `INC1_HAZARD_MISDIAGNOSIS_CORRECTED`
Inc 2 completes the third correctness gate and measures a real latency-aware reorder. The headline is a **correction**:
Inc 1's "RDNA3 hardware-spacing hazard" was a misdiagnosis. The actual missing piece was the **loop-entry
(branch-target) control-flow boundary** — a static modeling gap, not a hardware hazard. With it, fence_only
cross-motion is byte-identical-correct across the route config space. A latency-aware (critical-path) reorder is then
**perf-neutral** on the hand-tuned kernel: pure instruction reordering does not recover the prefill→Tensile residual;
that needs waitcnt-relocation / cross-iteration pipelining (Inc 3 scope).

## Complete Inc 2 scope (as planned, with outcomes)
| phase | plan | outcome |
|---|---|---|
| **2a** sound cross-motion | model the missing gate so fence_only memory/compute motion is correct | **DONE** — it was the loop-entry boundary, not a HW hazard |
| **2b** latency-aware schedule | critical-path list scheduler over fence_only regions; measure whole-prefill | **DONE** — built; clean clock-pinned timing = **NEUTRAL** (+/−<1%) |
| **2c** waitcnt relocation | strip full-drain waits, interleave load/compute, reinsert per-consumer waits | **SCOPED, not built** — the only remaining reorder-class lever (see below) |

## 2a — The corrected diagnosis (the real Inc 2 fix)
Inc 1 found a fence_only reorder that was register-legal (Inc-0 DAG, 0 missing edges) AND wait-correct yet computed
wrong, and (wrongly) blamed an RDNA3 hardware-spacing hazard. Inc 2 found the true cause:
- `build_gemm_lds2`'s **`LOOP` label** (the backward-branch target = loop entry) sits *inside* a fence_only region
  (between the prologue and `coop_load`). It is a byte-offset marker, not an instruction, so `_is_fence` never saw it.
- The reorder therefore moved instructions **across the loop entry** — between the prologue and the loop body —
  changing what executes each iteration. Wrong values, not a fault; only the prologue region (the one containing the
  loop entry) broke, which is exactly what the per-region bisect showed.
- **Fix:** `branch_target_indices(insts)` computes loop-entry indices from the branches' signed simm16 + byte offsets;
  `build_regions(..., boundaries=...)` starts a fresh region at each. `schedule(fence_only=True)` auto-applies them.
- **Result:** fence_only cross-motion is now **byte-identical-correct** across `default_PLRA / kv_halved / DBUF1 /
  8wave_PLRAB`, both `asap` (167–310 mem ops moved) and `critical` modes — `extra/qk_asm_scheduler_inc2_test.py` R1/R2.
- Inc 0 never hit this because its memory-delimited regions made the loop entry (a `global_load`) a boundary already.

**ISA corroboration (independent research).** On RDNA3/gfx11, `s_delay_alu` is a *performance* hint, not a correctness
mechanism: the hardware interlocks VALU/VMEM register dependencies (RDNA3 ISA §5.6/§5.7/§10.8; LLVM
`AMDGPUInsertDelayAlu` is "avoid stalls", VMEM waits for `VA_VDST==0`). So a register-legal + wait-correct reorder
**cannot** corrupt values via spacing — confirming the hazard theory was wrong and the loop-entry boundary is the cause.
(The one genuine VALU↔VMEM correctness hazard, `checkVMEMHazards` = VALU→SGPR→VMEM 5 wait states, does not apply here:
the loads' SGPRs come from `s_load`/SALU, never from a VALU result.)

## 2b — Latency-aware schedule: built, measured NEUTRAL
A critical-path list scheduler (`mode="critical"`): per-instruction RDNA3 latency weights, schedule the highest
critical-path-height ready node first (hoist long-latency VMEM/WMMA producers, keep the longest chain moving).
- Correctness: byte-identical across all configs (R1/R2).
- **Timing (clean, clock-pinned, isolated, copies excluded, DBUF1 512×4096×4096):** identity ≈ 287 µs / **59.8
  TFLOPS**; critical ≈ 288 µs / 59.6 TFLOPS → **−0.3% (within noise)**. Repeated runs land within ±1%.
- **Why neutral:** the hardware scoreboard already hides in-region latency; the regions are bounded by full-drain
  `s_waitcnt`s so there is no cross-region overlap for a *within-region* reorder to exploit; and the hand-tuned
  DBUF/PLRA already captured the cross-iteration prefetch. So reordering a *fixed* instruction set within the existing
  wait structure cannot beat the hand schedule.

> Measurement note: per project rule, isolated timing is a SIGNAL, not promotion authority. Since the signal is
> neutral, no whole-prefill route wiring was pursued (nothing to promote).

## 2c — The only remaining reorder-class lever (scoped, not built)
The full-drain `s_waitcnt lgkmcnt(0)` between the `ds_load`s and the `wmma`s is a hard barrier: every WMMA waits for
*all* fragment loads. Tensile-class cadence (SIA1/PLR1) instead lets `wmma_i` start once *its* fragments are ready.
Realizing this needs **waitcnt relocation**, not reordering: merge the ds_load and wmma regions (remove the full-drain
fence), interleave them, and **insert per-consumer `lgkmcnt` waits** before each WMMA (the wait model already computes
the minimal counts; branch offsets are now recomputable via `branch_target_indices`). This *changes the instruction
set* (inserts/moves waits), so it is beyond "reorder a fixed list." Honest ROI: uncertain — the hardware may already
overlap some of this via scoreboarding; the bounded upside remains ≤~2–3% and partly a `beta` work confound. This is
Inc 3.

## Honest standing
- The asm scheduler now has all THREE correctness gates: **register DAG (Inc 0) + wait-counter model (Inc 1) +
  loop-entry/branch-target boundaries (Inc 2)**. Cross-motion is sound and verified across the config space.
- **Pure instruction reordering is perf-neutral** on this hand-tuned kernel. The prefill→Tensile residual is NOT
  recoverable by reordering a fixed instruction set; it needs structural change (waitcnt relocation / cross-iteration
  pipelining — Inc 3) or the vendored-Tensile path. This sharpens the original audit's "needs an asm scheduler": the
  *reorder* part of an asm scheduler does not move the needle here; the *schedule-structure* part (waits + pipelining)
  is where any remaining win lives.
- No `tinygrad/` source, no production path, no default flip, no whole-prefill speed claim.

## Files
New: `extra/qk_asm_scheduler_inc2_test.py`, this doc. Modified (additive): `extra/qk_asm_scheduler.py`
(`branch_target_indices`, `boundaries` in `build_regions`, `critical` schedule mode + latency model). Inc 1 test
narrative corrected (Q6 + summary point to this correction). +1 ledger.
