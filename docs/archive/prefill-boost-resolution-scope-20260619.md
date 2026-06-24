# SCOPE — Prefill boost-state resolution (gfx1100, 2026-06-19)

Follow-up to `prefill-clock-dpm-authority-result-20260619.md` (the corrected, multi-run verdict). That work proved
the prefill WMMA-vs-Tensile comparison is confounded by a **per-process GPU boost lottery** (ROCm #6289): the same
WMMA prefill path is bimodal — **~1437 tok/s (SMU stuck, GPU idle-gaps) OR ~2674 (sustained boost @ real sclk 2315,
~87% llama = matches Tensile)** — stable within a process, varying across launches. Tensile is stable ~2640. So the
"Tensile 1.76× win" is an artifact of comparing stuck-WMMA vs boosted-Tensile; at WMMA's best there is no Tensile win.

This scope resolves the three open questions. **Measure-only: no kernel/route/default changes. Multi-run mandatory
(N≥5 fresh launches per claim). Real clock via `rocm-smi --showgpuclocks` sampled from a SEPARATE process (a
per-sample subprocess INSIDE the timing loop poisons tok/s 2674→740). `pp_dpm_sclk` DPM-level nominal is NOT the
real GFXCLK.**

## Purpose
1. **P1 — Force/sustain the WMMA boost** (the dependency-free path to ~87% llama). Characterize the lottery, then
   find a lever that makes WMMA reliably (≥5/5 fresh launches) land in the boosted state, telemetry-verified.
2. **P2 — Realistic generate-path prefill.** Measure prefill the way `model.generate` actually runs it (single cold
   prefill, not a tight 30× loop) — which state does the realistic user path land in, and does the P1 lever lift it?
3. **P3 — Clock-independent metric.** Compare WMMA vs Tensile work with a metric independent of the boost confound
   (GPU cycles via native PMC `SQ_BUSY_CYCLES` / `GRBM_GUI_ACTIVE`), to settle "at equal clock, is WMMA == Tensile?"

## P1 — Boost characterization + forcing
### P1a — Characterize the lottery
- N≥15 fresh process launches, each: build WMMA jit, warm, measure tok/s (best-of-25), classify stuck(<~1800) vs
  boosted(>~2400). Sample real GFXCLK from a separate process. Record: tok/s, state, real sclk med/min/max, n-busy
  samples (zero-busy = idle-gapping = stuck signal), time-since-prev-launch.
- Question: is it random per-launch, or does it correlate with cold-start / idle-gap / warmup-amount?
### P1b — Forcing levers (each tested N≥5 fresh, real-clock-verified)
- **L1 `profile_peak`** (ROCm #6289 documented fix) — verify real sclk sustained AND WMMA tok/s ≥2400.
- **L2 manual DPM** sclk=2/mclk=3.
- **L3 boost primer** — run a dense sustained workload (big fp16 matmul loop, ~2–3 s) BEFORE the prefill to latch
  boost, then measure prefill. Dependency-free.
- **L4 queue saturation** — reduce/remove inter-forward idle gaps (e.g., no per-forward host sync, or batched
  back-to-back dispatch) so the SMU never drops clock.
- Gate: a lever that yields boosted WMMA (≥2400 tok/s) in ≥5/5 fresh launches, real-clock-verified sustained.

## P2 — Realistic generate-path prefill
- Measure a SINGLE prefill of 512 tokens through the real `model.generate` path (cold, one shot), N≥10 fresh
  launches, default (auto) and with the best P1 lever. Sample real clock.
- Report the realistic distribution (likely stuck-low for a one-shot bursty prefill) and whether the lever lifts it.
- Cross-check vs llama pp512 measured in the same session.

## P3 — Clock-independent WMMA-vs-Tensile
- Use the native PMC tooling (`extra/qk_pmc_capture.py` / `PMC=1 PROFILE=1`) to capture **GPU cycle counters**
  (`SQ_BUSY_CYCLES`, `GRBM_GUI_ACTIVE`) for a WMMA-FFN forward vs a Tensile-FFN forward. Cycles are clock-INDEPENDENT
  work — if WMMA cycles ≈ Tensile cycles → same efficiency (the tok/s gap is purely clock/boost); if WMMA ≫ Tensile
  → Tensile genuinely does less GPU work.
- Fallback if PMC perturbs/blocks: pin the clock via the P1 lever (verified sustained real sclk for BOTH), then the
  interleaved tok/s A/B IS clock-fair → compare there.

## Gates (hard, carried from the clock-authority lessons)
- Never trust 1 process: every claim = N≥5 fresh launches + state spread.
- Real GFXCLK only (`--showgpuclocks`, separate process). `pp_dpm_sclk`-nominal is banned as authority.
- A lever is "works" only if telemetry proves the real clock sustained AND tok/s in the boosted band, ≥5/5.
- Distinguish boost-state effects from kernel effects: P3's clock-independent metric is the tiebreaker.

## Deliverables
- `docs/prefill-boost-resolution-result-20260619.md`.
- `extra/qk_prefill_boost_probe.py` (P1/P2 multi-run driver + separate-process clock sampler + classifier).
- `bench/qk-prefill-boost/*` artifacts (lottery table, lever matrix, generate-path table, PMC cycles).
- README pointer. NO route/default changes.

## Definition of done
- **P1:** either a dependency-free lever that reliably forces boosted WMMA (~87% llama) — a real shippable prefill
  win — OR proof no lever reliably forces it (then the realistic number is the stuck-low band and that's the honest
  ceiling).
- **P2:** the realistic generate-path prefill tok/s + state, with and without the lever.
- **P3:** a clock-independent verdict on whether WMMA == Tensile in GPU work (settling whether Tensile has ANY real
  advantage, boost aside).
