# RESULT — Prefill boost-state resolution (gfx1100, 2026-06-19)

Executes `prefill-boost-resolution-scope-20260619.md` (P1/P2/P3). Driver `extra/qk_prefill_boost_probe.py` +
`extra/qk_tensile_ab_measure.py`. Artifacts `bench/qk-prefill-boost/{p1a,p1b,p2}.txt`. Real GFXCLK via
`rocm-smi --showgpuclocks` sampled from a SEPARATE process (DPM-nominal `pp_dpm_sclk` is NOT the real clock).

## Headline
tinygrad WMMA prefill is **bimodal per process: ~1438 tok/s (~47% llama) or ~2674 (~87% llama)** — latched at process
init, stable within a process. **It is NOT a clock/power/thermal state (P3): the slow runs sit at a HIGHER clock than
the fast ones.** It is an uncontrolled per-process GPU **execution-efficiency** state (occupancy / wave-scheduling /
SMU power-grant). **No tested lever forces the fast state.** Tensile reliably lands fast (~2640 every run) — that is
its only "advantage," not kernel efficiency.

## P1a — the lottery is real and (currently) cold-stuck
15 fresh launches (10 + 5), WMMA-only, real clock verified: **15/15 STUCK at ~1435–1439 tok/s, each at real sclk
~2333 with 30–32/32 busy samples.** Earlier in the session, after sustained back-to-back GPU load, runs landed at
~2674 (`ab_measure` ×6, etc.). So the fast state exists but is **not reliably reachable on a fresh/cold process**; the
default cold launch is stuck at ~47% llama.

## P1b — no lever forces it (each ≥3 fresh, real-clock-verified)
| lever | WMMA tok/s | real sclk | busy | verdict |
|---|---:|---:|---:|---|
| none | 1435–1441 | 2333 | 31–32/32 | STUCK |
| L1 `profile_peak` | 1501–1503 | 2194 (lower!) | 30/30 | STUCK (+4%, lower clock, more power) |
| L2 manual sclk2/mclk3 | 1436–1438 | 2334 | 31–32/32 | STUCK |
| L3 boost primer (3 s dense matmul) | 1436 | 2333 | 32/32 | STUCK |
| L4 queue saturation | — | — | — | dropped (unsynced-forward hangs) |
None reach the boosted band. Clock pinning, power profile, and a dense pre-warm all fail.

## The disambiguation (what it is NOT)
- **NOT clock.** Slow = 1438 @ sclk **2333**; fast = 2668 @ sclk **2315**. The slow state is at the *higher* clock,
  fully busy (32/32). Definitively clock-independent → **answers P3**.
- **NOT thermal/temporal.** Back-to-back, same thermal state: `ab_measure` WMMA = 2672/2676, `boost_probe` WMMA =
  1436 (@ verified sclk 2332). Same conditions, different result.
- **NOT idle-gapping.** (The earlier "idle-gap" claim was a broken-sampler artifact — time-base mismatch gave n=0
  samples. The fixed sampler shows 32/32 busy in the stuck state.)
- **NOT simply the harness code.** Bisection: `ab_measure` stripped to joff-only (no Tensile jit) = 1438 (slow); a
  fresh inline script that builds the Tensile jit = 2668 (fast); but "build+run Tensile then measure WMMA alone" =
  1437 (slow). The trigger does not reduce to a clean code switch — it is a per-process GPU init state that only
  loosely correlates with workload history.
- **Power signature.** Stuck WMMA draws **~55 W median** at 2333 MHz / 32-32 busy — absurdly low for a 7900 XTX
  doing real WMMA (which should pull 200 W+). So the stuck kernel is **stalling / under-occupied**, not
  compute-saturated. (Clean fast-vs-slow power separation was defeated by model-load contamination in the sampling
  window; the 55 W stuck figure stands.)

## P2 — realistic generate-path number
A fresh, cold process (what a real `model.generate` prefill is) lands **STUCK ~1437 tok/s = ~47% of llama (3070)**
in 15/15 launches this session. The ~87% (2674) state is achievable but not on demand. **So the honest realistic
dependency-free prefill number is ~47% llama**, with an unreliable ~87% upside.

## P3 — clock-independent verdict
Settled by the disambiguation above: the WMMA-vs-itself 1.85× swing is **clock-independent** (slow runs at higher
clock). Therefore the WMMA-vs-Tensile comparison is confounded by WMMA's execution-state lottery, not by clock. At
WMMA's fast state it matches Tensile (~2674 vs ~2640) → **Tensile is not a kernel win**; it merely lands in the fast
execution reliably. (PMC `SQ_BUSY_CYCLES` capture was not needed — the clock+power evidence already isolates it to a
GPU occupancy/power-grant state, not the kernel algorithm.)

## Bottom line / what changed
- **Dependency-free prefill is bimodal ~47%↔~87% llama, latched per process, and currently cold-stuck at ~47%.**
- **No clock/power lever forces the fast state.** profile_peak/manual/primer all fail.
- **Tensile (~87% llama, stable) is a dependency that reliably lands fast — but it is NOT faster than WMMA's fast
  state.** Vendoring it buys reliability of the fast execution, not a better kernel.
- The real lever is **whatever sets the per-process occupancy/power-grant** — an RDNA3/ROCm-driver-level question
  (CU masks, power profiles, the SMU's power grant, or tinygrad's launch geometry/occupancy for the WMMA kernel),
  NOT clock and NOT the matmul algorithm.

## Open (next, if pursued)
1. **Why the per-process latch?** Inspect launch geometry/occupancy of the WMMA FFN kernel (grid/wg/VGPR) in a fast
   vs stuck process; check whether tinygrad picks a low-occupancy schedule sometimes. Likely the real handle.
2. **Force max occupancy/power grant** at the driver level (ROCm power profile via the gpu_metrics binary interface;
   CU mask; or a kernel launch-geometry change to guarantee high occupancy).
3. **Realistic end-to-end generate timing** including decode, to see how much the prefill state actually matters
   for a full request (prefill is one-shot; decode dominates long generations).

## Methodology bankables
- `pp_dpm_sclk` DPM-level nominal ≠ real GFXCLK; use `--showgpuclocks`, separate process, time-base = wall-clock
  (a perf_counter window can't correlate to a `date +%s.%N` sampler — that bug produced the false "idle-gap" story).
- Low power draw at "full clock + fully busy" is the tell for a stalled/under-occupied kernel.
- A prefill claim needs N≥5 fresh launches + the state spread; one process is meaningless here.
