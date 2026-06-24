# RESULT — Prefill clock/DPM authority (gfx1100, 2026-06-19) — CORRECTED after multi-run testing

**⚠ This doc was first published with a WRONG verdict ("clock not a confound, Tensile 1.76× robust, SOLVED"). That
rested on ONE process and a faulty clock reader. Multi-run testing (below) overturns it.** The honest verdict:
**the prefill WMMA-vs-Tensile comparison is confounded by a per-process GPU boost-state lottery, and at WMMA's
achievable best it MATCHES Tensile (no real Tensile win).**

Drivers: `extra/qk_prefill_clock_experiment.py` (initial, flawed clock reader), then `/tmp/{localize,ramp,ab_instr}.py`
and `extra/qk_tensile_ab_measure.py` (the multi-run cross-check). NO kernel/route/default changes.

## What multi-run testing actually found
| harness / condition | WMMA tok/s | Tensile tok/s | ratio | note |
|---|---:|---:|---:|---|
| `qk_tensile_ab_measure.py` (×6 fresh, late session) | 2672–2676 | ~2664 | **0.997×** | WMMA boosted → matches Tensile |
| `qk_prefill_clock_experiment.py` (early session) | 1445–1548 | 2638–2735 | **1.75–1.83×** | WMMA stuck low |
| `ramp.py` / 3 fresh trials (early) | 1434–1440 | — | — | WMMA stuck low, stable across 320 forwards |
| PHASE test (mid) | 2630 | ~2640 | ~1.00× | WMMA boosted |

**The same WMMA path measures either ~1437 OR ~2674 tok/s** — a binary, ~1.85× split. Tensile is stable ~2640
every run. So the ratio swings 0.997×↔1.83× depending purely on **which state WMMA lands in**, not on the kernels.

## The mechanism (evidenced, not inferred)
1. **It is a per-process GPU boost-state lottery, not the harness.** Injecting my exact measurement *inside*
   `ab_measure`'s process gave 2675; my standalone scripts gave 1437 — with byte-identical build/run code. The
   variable is the process's GPU power/clock state, not the timing method (3 timing styles agreed within a process)
   nor the JIT trace (identical) nor Tensile-interleaving (ruled out — WMMA-only warmup also boosts in a boosted
   process).
2. **Stuck-low = the GPU idle-gaps; boosted = saturated.** A separate-process `rocm-smi --showgpuclocks` sampler
   caught **37 continuous samples at 2315 MHz** during a 2630-tok/s WMMA run, but **zero busy samples** during
   1437-tok/s runs (it only ever caught idle gaps). So the stuck-low state is the GPU dropping out of boost in the
   gaps between bursty WMMA forwards; the boosted state keeps it pinned at ~2315. This is ROCm #6289 territory
   (SMU doesn't reliably sustain GFXCLK for compute).
3. **Within a process it's stable; across launches it varies, and it warms over a session.** Early-session / cold
   launches got stuck at 1437; after sustained load the GPU stayed boosted (6/6 late runs at 2674). `profile_peak`
   did NOT reliably force the boost (gave 1511 at higher power earlier).
4. **My first clock reader was wrong.** `qk_prefill_clock_experiment.py` read the `pp_dpm_sclk` *DPM-level nominal*
   (2304/2331) and reported "max clock", but the real instantaneous GFXCLK (`--showgpuclocks`) in a stuck run is
   idle-gapping ~1500-ish. So the "WMMA at verified max clock = 1450" claim was an artifact of reading the wrong
   field — invalidating the original "clock is not the confound" conclusion.

## Corrected verdict
- **tinygrad WMMA prefill CAN reach ~2674 tok/s (~87% of llama 3070) dependency-free** — when the GPU sustains
  boost. At that state it MATCHES Tensile (~2664) → **Tensile provides NO real speedup** (0.997×).
- **The apparent "Tensile 1.76–1.83× win" is an artifact** of comparing a *stuck-low* WMMA (~1437) against a
  *stable-boosted* Tensile (~2640). It is NOT a kernel-efficiency win; Tensile's only edge is that it more reliably
  sits in the boosted state.
- **Both prior verdicts were one-state snapshots:** the old `0.997×` (ab_measure/tensile-land) = boosted-WMMA state;
  the reconciliation's `1.76×` and my first `1.83×` = stuck-WMMA state. Neither is "canonical"; the system is
  bimodal.
- **The real prefill lever is dependency-free and NOT Tensile:** make WMMA prefill reliably sustain GPU boost /
  keep the GPU saturated (eliminate the inter-forward idle gaps that let the SMU drop clock). If solved, WMMA hits
  ~87% llama with no external `.co` dependency. This reconnects to the earlier (prematurely retracted) host-gap /
  busy-wait findings.

## Status of claims
- RETRACTED: "clock is not the prefill confound", "Tensile 1.76× is robust/clock-independent", "auto is a
  controlled-authority lane", "2675 is a measurement artifact". (2675 is real — it's WMMA's boosted throughput.)
- STANDING: Tensile is clock/boost-stable ~2640; WMMA is bimodal 1437/2674; the comparison is confounded by the
  per-process SMU boost state; at WMMA's best there is no Tensile win.
- OPEN (the actual next work): (a) find what forces/sustains the WMMA boost reliably (power profile? queue
  saturation? a warmup that latches boost?); (b) measure prefill in the realistic `model.generate` path (single
  cold prefill), which is what a user actually hits — likely the stuck-low end unless boost is forced; (c) a
  clock-INDEPENDENT metric (GPU cycles / counters) to compare WMMA vs Tensile without the boost confound.

## Methodology lessons (for the claim checklist)
- `pp_dpm_sclk` DPM-level nominal ≠ real GFXCLK. Use `rocm-smi --showgpuclocks` (sclk level *frequency*), sampled
  from a SEPARATE process (a per-sample `rocm-smi` subprocess inside the timing loop poisoned tok/s 2674→740).
- A bursty workload defeats clock sampling (the sampler catches idle gaps) — n-busy-samples is itself a signal
  (zero busy = the workload idle-gaps = stuck-low state).
- One process is never enough for a prefill claim on this card. Report N≥5 fresh launches and the state spread.
