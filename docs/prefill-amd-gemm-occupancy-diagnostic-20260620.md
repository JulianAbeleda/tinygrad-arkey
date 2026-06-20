# Prefill AMD GEMM — Occupancy Diagnostic ("why are we stuck?")

Date: 2026-06-20

## Purpose

The Tensile-source audit *predicted* our SIA0 kernel is latency-bound and hides memory latency only via
occupancy (no PLR/PGR software prefetch). This **measures** that prediction by isolating occupancy. The result
**corrects the audit** — and surfaces an actionable lift.

Probe: `extra/qk_amd_gemm_occupancy_diagnostic.py` → `bench/.../amd_gemm_occupancy_diagnostic_result.json`.

## Method

Vary **only** occupancy via LDS padding. `_run_insts_lds` allocates a `DEFINE_LOCAL` of `lds_bytes`; the BK32
kernel only *uses* 16384 B, so padding to 24576/32768/49152/65536 leaves compute + correctness identical while
cutting LDS-limited workgroups/CU (64 KB / lds) from 4 → 2 → 1. Interleaved one-clock, correctness-gated,
power-witnessed. A rising throughput-vs-occupancy curve ⇒ latency-bound; flat/falling ⇒ not.

## Result (reproduced 4×, interleaved, all correct)

| config | LDS B | wg/CU | best TFLOPS | median |
|---|---:|---:|---:|---:|
| bk32 default | 16384 | **4** | 48.8 | 45.0 |
| bk32 pad | 24576 | 2 | 57.3 | 55.8 |
| **bk32 pad** | **32768** | **2** | **57.7** | **56.1** |
| bk32 pad | 49152 | 1 | 49.6 | 48.5 |
| bk32 pad | 65536 | 1 | 49.7 | 48.7 |
| authority_llvm | — | — | 52.9 | 51.6 |

Verdict: `CONFIRMED_CONTENTION_LIMITED_OCC_OPTIMUM_INTERIOR`. **Interior optimum at wg=2** (57.7); higher
occupancy (wg4 = 48.8) **and** lower (wg1 = 49.7) are both worse. peak/full = **1.18×**, peak/authority =
**1.09×**.

## What this means — the audit's mechanism is REFUTED, and corrected

- **Refuted**: "latency hidden only via occupancy → we lack waves." If that were true, *more* occupancy (wg4)
  would be faster. It is **slower**. The full→occ1 curve is flat-to-falling, not rising.
- **Corrected mechanism — contention, not latency starvation**: the curve has an interior peak. Too-high
  occupancy (wg4 = 4 workgroups/CU) **thrashes** the shared LDS bandwidth / barrier / L2; too-low (wg1) loses
  the modest 2-workgroup overlap. The kernel is **contention-limited at high occupancy**, not waiting on memory
  latency for lack of waves. (VGPR-limited occupancy is ~6 waves/SIMD at 234 VGPR, so LDS is the binding
  resource here.)

This is exactly why we measure instead of asserting: a clean, source-grounded hypothesis was wrong about the
*mechanism*, and one controlled experiment caught it.

## Actionable lift (banked)

The BK32 frontier we banked at ~55 was run at its **default wg4 occupancy, which is contention-limited**. The
occupancy sweet spot **wg2 (pad LDS to 32768) reaches ~57.7 — ~18% over the wg4 default and ~9% over the LLVM
authority**. So the dependency-free frontier is really **~57–58 (wg2), not 55**, clearly beating the authority.
(LDS padding to control occupancy is a legitimate, free technique — the kernel uses 16384 and we reserve 32768
purely to cap concurrent workgroups.)

## Why we're stuck (corrected) and the residual to Tensile

We are not stuck for lack of occupancy. At the wg2 optimum we already pass the authority (~57.7 vs ~53). The
residual to **Tensile (~66)** is **consistent with the contention story, not the occupancy one**: Tensile's
`SIA1` + `SLW1` + `PLR1` scheduling **reduces LDS/barrier pressure** (interleaves local writes/reads, prefetches
one iteration ahead), so it stays efficient *without* having to drop occupancy to avoid contention. Our coarse
SIA0 phase-block forces an occupancy/contention tradeoff (hence the wg2 compromise); Tensile sidesteps it.

So the audit's *what Tensile does* (scheduling/prefetch) still stands — but its *why we're slow* (occupancy
latency-hiding) was wrong. The real binding constraint is **LDS/barrier contention**, which scheduling
mitigates and occupancy-tuning only partially dodges.

## Honesty

- One prefill shape; best-of-N (median tells the same story: wg2 56 > authority 51.6 > wg4 45). Clock-volatile
  absolute TFLOPS; the *interleaved curve shape* is the robust claim, reproduced 4×.
- The wg2 sweet spot is a free occupancy cap via LDS padding; not a new kernel.
- This refutes only the audit's *mechanism* (occupancy/latency), not its decoding of Tensile's features.

## Verdict & next

`CONFIRMED_CONTENTION_LIMITED_OCC_OPTIMUM_INTERIOR`. Next, to pin the contention and the true ~58→66 gap — and
**not** another tile/depth/occupancy sweep:

1. **PMC** the BK32-wg2 kernel: LDS-bandwidth / `s_barrier` stall cycles / L2 hit — name the contention the
   curve implies (the audit's `instruction_scheduling` lever should show up as barrier/LDS-wait time).
2. Re-bank the frontier at **wg2 ~57.7** (beats authority), superseding the wg4 ~55 number.
3. Only a PMC finding justifies a scheduling change (a minimal `SLW`/`PLR`-style interleave to cut barrier/LDS
   pressure) — measured under the same gate, no BEAM.
